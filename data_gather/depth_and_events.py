#!/usr/bin/env python

import rospy
import rosbag
from sensor_msgs.msg import Image
# from dvs_msgs.msg import Event, EventArray
# from dv_ros_msgs.msg import Event, EventArray
# from prophesee_event_msgs.msg import Event, EventArray
from event_array_msgs.msg import EventArray
from cv_bridge import CvBridge, CvBridgeError
import cv2, numpy as np
import sys
from scipy.ndimage import gaussian_filter
import message_filters
import os
import numpy.lib.recfunctions as rf

sys.path.append('/home/anish/evfly_ws/src/event_array_py')
from event_array_py import Decoder

sys.path.append('/home/anish/evfly_ws/src/evfly/utils')
from ev_utils import form_eventframe, visualize_evim

from calibration_tools.rectify_bag import Aligner

class DepthEvsToImageNode:
    def __init__(self, args):
        rospy.init_node('depthevs_to_image_node', anonymous=True)

        self.decoder = Decoder()

        self.bridge = CvBridge()

        self.args = args

        self.aligner = Aligner('/home/anish/evfly_ws/src/evfly/utils/calib_7-28/K.yaml')

        # bagfile = os.path.basename(self.args.bagfile)

        if self.args.output_dir is not None:
            self.saveto_dir = self.args.output_dir # os.path.join(self.args.output_dir if self.args.output_dir is not None else os.getcwd(), bagfile.split('.')[0])
        else:
            print('No output directory specified. --output_dir must be specified as an argument. Exiting.')
            exit()

        if not os.path.exists(self.saveto_dir):
            os.makedirs(self.saveto_dir)

        self.evs = None
        self.last_evs_msg_time = None
        self.evframe = None
        self.evframes = []
        self.receiving_msgs = False
        self.evim = None
        self.depth = None
        self.output_im = None

        self.depth_msg = None
        self.evs_msg = None

        self.depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image) #, self.depth_process)
        self.events_sub = message_filters.Subscriber('/event_camera/events', EventArray) #, self.events_process)
        self.output_evs_pub = rospy.Publisher('/output/evs', Image, queue_size=10)
        self.output_depth_pub = rospy.Publisher('/output/depth', Image, queue_size=10)
        self.output_im_pub = rospy.Publisher('/output/image', Image, queue_size=10)

        # make a separate rospy type subscriber to maintain a queue of events batches and corresponding timestamps
        self.evs_queue = []
        self.evs_ts = []
        self.last_batch_ts = None
        self.events_continuous_sub = rospy.Subscriber('/event_camera/events', EventArray, self.events_continuous_callback)

        timesync = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.events_sub], queue_size=20000, slop=0.005)
        timesync.registerCallback(self.observation_callback)

        self.n_evframes = 0
        self.n_depthims = 0

        print('End of initialization.')

    def observation_callback(self, depth_msg, evs_msg):

        # if there is not last batch timestamp, we do not have a beginning timestamps to form the time window for the event batch construction
        if self.last_batch_ts is None:
            self.last_batch_ts = evs_msg.header.stamp.to_nsec()/1e9
            return

        print(f'depth_msg timestamp {depth_msg.header.stamp.to_nsec()/1e9:.3f}')
        print(f'evs_msg timestamp   {evs_msg.header.stamp.to_nsec()/1e9:.3f}')
        # print()

        # return

        self.depth_process(depth_msg)
        self.events_process(evs_msg)
        print()

        # publish events on top of depth image
        # downsample depth image from 480x640 to self.args.ev_heightxself.args.ev_width
        # depth_im = cv2.resize(self.depth, (self.args.ev_width, self.args.ev_height), interpolation = cv2.INTER_AREA)
        depth_im = np.clip(self.depth/30e3, 0.0, 1.0) if 'depth' in self.depth_sub.topic else np.clip(self.depth/255.0, 0.0, 1.0) # 65535.0
        
        if self.evframe is not None:

            # # align both
            # out_dict = self.aligner.align(depth_im, self.evframe)
            # depth_im = out_dict['depth']
            # self.evframe = out_dict['davis']

            if self.args.publish:

                # align both
                out_dict = self.aligner.align(depth_im, self.evframe)
                depth_im = out_dict['depth']
                self.evframe = out_dict['davis']

                # publish aligned depth image
                depth_im_pub = np.stack(((depth_im*255.0).astype(np.uint8),)*3, axis=-1)
                depth_im_msg = self.bridge.cv2_to_imgmsg(depth_im_pub, encoding="bgr8")
                self.output_depth_pub.publish(depth_im_msg)

                # publish depth/events overlay image
                depth_and_events = depth_im_pub.copy()
                # depth_and_events[np.where(np.abs(self.evframe) > 0.1)] = self.evim[np.where(self.evframe != 0.0)]
                depth_and_events[np.where(self.evframe != 0.0)] = self.evim[np.where(self.evframe != 0.0)]
                # depth_im = cv2.cvtColor(depth_im, cv2.COLOR_GRAY2BGR)
                depth_and_events = cv2.cvtColor(depth_and_events, cv2.COLOR_RGB2BGR)
                depth_and_events_msg = self.bridge.cv2_to_imgmsg(depth_and_events, encoding="bgr8")
                self.output_im_pub.publish(depth_and_events_msg)

            if self.args.save:
                cv2.imwrite(os.path.join(self.saveto_dir, f'{depth_msg.header.stamp.to_nsec()/1e9:.3f}_depth.png'), (depth_im*255.0).astype(np.uint8))
                self.n_depthims += 1

                self.evframes.append(self.evframe)
                self.n_evframes += 1

    def depth_process(self, data):
        self.depth_msg = data
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth = np.array(cv_image, dtype=np.float32)
            # print(f'max/min of self.depth = {np.max(self.depth)}/{np.min(self.depth)}')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        
        # # refine depth image
        # self.refine_depth()

    def decode_prophesee_events(self, msg):
        self.decoder.decode_bytes(msg.encoding, msg.width, msg.height, msg.time_base, msg.events)
        cd_events = self.decoder.get_cd_events()
        events = rf.structured_to_unstructured(cd_events)
        return events

    def events_process(self, data):
        self.receiving_msgs = True
        # Assuming form_eventframe(msg) is implemented elsewhere
        # and it returns a uint8 image
        self.evs_msg = data

        # for prophesee event camera, using metavision_ros_driver, need to decode the .events field of the ros msg using event_array_py
        if self.args.is_metavision_ros_driver:
            self.evs = self.decode_prophesee_events(data)
        else:
            self.evs = data.events

        self.last_evs_msg_time = rospy.Time().now().to_sec()

        # get all EventArray messages that are between two timestamps by indexing the queue appropriately
        curr_timestamp = data.header.stamp.to_nsec()/1e9
        self.evs_ts_np = np.array(self.evs_ts)
        time_window_mask = np.logical_and(self.evs_ts_np > self.last_batch_ts, self.evs_ts_np <= curr_timestamp)

        time_window_mask = time_window_mask.tolist()
        evarray_batch = [self.evs_queue[i] for i in range(len(time_window_mask)) if time_window_mask[i]]        
        
        # evarray_batch is a list of EventArray messages, from which I need to form an evframe
        all_evs = []
        for evlist in evarray_batch:
            all_evs.extend(evlist)
        print(f'Number of events packets found in this batch: {len(all_evs)}')
        
        # # print details of all_evs
        # print(f'all_evs type = {type(all_evs)}')
        # print(f'all_evs[0] type = {type(all_evs[0])}')
        # print(f'shape of all_evs[0] = {all_evs[0].shape}')
        # print(f'len of all_evs = {len(all_evs)}')
        # exit()

        if self.args.is_metavision_ros_driver:

            if not self.args.is_flipped:
                evs_np = np.array([[ev[3], ev[0], ev[1], ev[2]] for ev in all_evs])
            else:
                evs_np = np.array([[ev[3], self.args.ev_width - ev[0], self.args.ev_height - ev[1], ev[2]] for ev in all_evs])

        else:

            if not self.args.is_flipped:
                evs_np = np.array([[ev.ts.to_sec(), ev.x, ev.y, ev.polarity] for ev in all_evs])
            else:
                evs_np = np.array([[ev.ts.to_sec(), self.args.ev_width - ev.x, self.args.ev_height - ev.y, ev.polarity] for ev in all_evs])
        
        self.evframe = form_eventframe(evs_np, self.args.ev_height, self.args.ev_width, all_events=True)

        # print shape of evframe
        # print(f'evframe shape: {self.evframe.shape}')

        if self.args.publish:
            self.evim = visualize_evim(self.evframe)
            # publish visualized events
            try:
                cv_image = cv2.cvtColor(self.evim, cv2.COLOR_RGB2BGR)
                event_frame_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.output_evs_pub.publish(event_frame_msg)
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))

        # remove all event batches prior to the current timestamp
        remaining_future_evs = self.evs_ts_np > curr_timestamp
        if remaining_future_evs.sum() > 0:
            remaining_future_evs = remaining_future_evs.tolist()
            self.evs_queue = [self.evs_queue[i] for i in range(len(remaining_future_evs)) if remaining_future_evs[i]]
            self.evs_ts = [self.evs_ts[i] for i in range(len(remaining_future_evs)) if remaining_future_evs[i]]
        else:
            self.evs_queue = []
            self.evs_ts = []

    # queue event batches for use in the time sync'd functions
    def events_continuous_callback(self, data):
        if self.args.is_metavision_ros_driver:
            events = self.decode_prophesee_events(data)
            # print(f'Type of events = {type(events)}')
            # print(f'Shape of events = {events.shape}')
            self.evs_queue.append(events)
        else:
            self.evs_queue.append(data.events) # a list of list of Event messages
        self.evs_ts.append(data.header.stamp.to_nsec()/1e9)
        # print(self.evs_ts[-1])

    def refine_depth(self):
        # image = self.depth
        # Define the threshold values for pixels to interpolate
        low_threshold = 0.01 # 0.1  # Value close to 0.0
        high_threshold = 0.99 # 65534.9  # Value close to 65535.0

        # Create a mask for pixels to interpolate
        mask = np.logical_or(self.depth <= low_threshold, self.depth >= high_threshold)

        # Create a copy of the image to work on
        interpolated_image = np.copy(self.depth)

        # Apply a Gaussian filter to smooth the image (helps in interpolation)
        smoothed_image = gaussian_filter(interpolated_image, sigma=2)

        # Get the indices of the pixels to interpolate
        blob_indices = np.transpose(np.where(mask))

        # Interpolate each blob
        for blob_index in blob_indices:
            row, col = blob_index

            # Extract the local region around the pixel
            local_region = smoothed_image[max(0, row-2):min(row+3, self.depth.shape[0]), max(0, col-2):min(col+3, self.depth.shape[1])]

            # Calculate the mean value of the local region
            mean_value = np.mean(local_region)

            # Assign the mean value to the pixel
            interpolated_image[row, col] = mean_value

        # return interpolated_image
        self.depth = interpolated_image

    def cleanup(self):
        if self.args.save:
            # print number of evims and depth images saved
            print(f'Number of depth images saved: {self.n_depthims}')
            print(f'Number of evframes saved: {self.n_evframes}')
            # save self.evframes as a npy file in the output directory
            np.save(os.path.join(self.saveto_dir, 'evframes.npy'), np.array(self.evframes))

    def spin(self):
        rate = rospy.Rate(100)
        time_cutoff = 30.0
        while not rospy.is_shutdown():
            if self.evs_msg is not None and self.receiving_msgs and self.last_evs_msg_time is not None and rospy.Time().now().to_sec() - self.last_evs_msg_time > time_cutoff:
                print(f'curr time is {rospy.Time().now().to_sec()} and last evs msg time is {self.evs_msg.header.stamp.to_sec()}')
                self.receiving_msgs = False
                print(f'No messages received for {time_cutoff} second. Shutting down node.')
                self.cleanup()
                rospy.signal_shutdown(f'No messages received for {time_cutoff} second.')
            else:
                rate.sleep()
                # rospy.spin()

if __name__ == '__main__':
    
    # parse command line arguments via arg parser
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output_dir', type=str, help='output directory to store images', default='data')
    parser.add_argument('--bagfile', type=str, help='bagfile to process', default='none')
    parser.add_argument('--publish', action='store_true', help='publish the visualized events and depth image')
    parser.add_argument('--save', action='store_true', help='save the visualized events and depth image')
    parser.add_argument('--ev_height', type=int, help='height of the event frame resolution', default=260)
    parser.add_argument('--ev_width', type=int, help='width of the event frame resolution', default=346)
    parser.add_argument('--is_flipped', action='store_true', help='whether the image is flipped or not (prophesee camera is mounted upside down)')
    parser.add_argument('--is_metavision_ros_driver', action='store_true', help='whether using prophesee cam metavision ros driver instead of davis or prophesee driver)')
    args = parser.parse_args()


    
    try:
        depth_to_image_node = DepthEvsToImageNode(args)
        depth_to_image_node.spin()
    except rospy.ROSInterruptException:
        pass
