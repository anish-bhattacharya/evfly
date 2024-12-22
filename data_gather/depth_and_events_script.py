#!/usr/bin/env python

# this script reads depth and events from a rosbag and saves images as pngs and and events in a npy, which is the default dataset format

import rospy
import rosbag
from sensor_msgs.msg import Image
# from dvs_msgs.msg import Event, EventArray
#from dv_ros_msgs.msg import Event, EventArray
# from prophesee_event_msgs.msg import Event, EventArray
from cv_bridge import CvBridge, CvBridgeError
import cv2, numpy as np
import sys
from scipy.ndimage import gaussian_filter
import os
import time
import numpy.lib.recfunctions as rf

sys.path.append('/root/evfly_ws/src/evfly/utils')
from ev_utils import form_eventframe

np.set_printoptions(suppress=False, formatter={'float': lambda x: "{:.5f}".format(x)})

class DepthEvsToImage:
    def __init__(self, args):
        self.bridge = CvBridge()

        self.args = args

        if self.args.is_metavision_ros_driver:
            sys.path.append('/root/evfly_ws/src/event_array_py')
            from event_array_py import Decoder
            self.decoder = Decoder()

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

        self.evs_topic = self.args.ev_topic
        self.depth_topic = '/camera/depth/image_rect_raw'

        # make a separate rospy type subscriber to maintain a queue of events batches and corresponding timestamps
        self.evs_queue = []
        self.evs_ts = []
        self.last_batch_ts = None

        # timesync = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.events_sub], queue_size=20000, slop=0.005)
        # timesync.registerCallback(self.observation_callback)

        self.n_evframes = 0
        self.n_depthims = 0

    def read_bag(self):

        print(f'Reading from bagfile {self.args.bagfile} of size {os.path.getsize(self.args.bagfile)/1e9:.3f} GB')
        start_time = time.time()
        self.bag = rosbag.Bag(self.args.bagfile, 'r')
        # with rosbag.Bag(self.args.bagfile, 'r') as bag:
        # print('Done; reading messages...')
        evs_ts = np.array([msg.header.stamp.to_sec() for _, msg, _ in self.bag.read_messages(topics=[self.evs_topic])])
        depth_ts = np.array([msg.header.stamp.to_sec() for _, msg, _ in self.bag.read_messages(topics=[self.depth_topic])])
        depth_ims_msgs = [msg for _, msg, _ in self.bag.read_messages(topics=[self.depth_topic])]
        
        # print(f'First evs timestamp: {evs_ts[0]:.3f}, Last evs timestamp: {evs_ts[-1]:.3f}, Number of evs messages: {len(evs_ts)}')
        # print(f'First depth timestamp: {depth_ts[0]:.3f}, Last depth timestamp: {depth_ts[-1]:.3f}, Number of depth messages: {len(depth_ts)}')

        # print('====')
        # print(evs_ts[:5])
        # print(evs_ts[-5:])
        # print('====')
        # print(depth_ts[:5])
        # print(depth_ts[-5:])
        # print('====')

        # find first depth timestamp that is greater than the first evs timestamp
        first_depth_idx = np.argmax(depth_ts > evs_ts[0])
        
        # find last depth timestamp that is less than the last evs timestamp
        last_depth_idx = np.nonzero(depth_ts < evs_ts[-1])[0][-1]

        # print(f'First depth idx: {first_depth_idx}, Last depth idx: {last_depth_idx}')

        if first_depth_idx >= last_depth_idx:
            print('No overlapping timestamps between depth and evs messages. Exiting.')
            exit()
        if last_depth_idx - first_depth_idx < 2:
            print('Too few depth messages overlapping with evs messages to process. Exiting.')
            exit()

        # given the time window (depth_ts[first_depth_idx], depth_ts[last_depth_idx]), batch evs messages according to the depth image message frequency
        # for each depth image message, find the corresponding evs messages that are within the time window
        num_batches = last_depth_idx - first_depth_idx
        for evs_batch_i in range(num_batches):

            # print updates
            if evs_batch_i % 50 == 0:
                print(f'[BAG {os.path.basename(self.args.bagfile)}] Processing batch {evs_batch_i} of {num_batches} | elapsed time {time.time()-start_time:.1f} s')

            start_t = depth_ts[first_depth_idx + evs_batch_i]
            end_t = depth_ts[first_depth_idx + evs_batch_i + 1]

            # convert batch of events to an evframe
            # for prophesee event camera, using metavision_ros_driver, need to decode the .events field of the ros msg using event_array_py
            if self.args.is_metavision_ros_driver:
                # need full messages for decoding
                evs_batch_msgs = [msg for _, msg, _ in self.bag.read_messages(topics=[self.evs_topic], start_time=rospy.Time.from_sec(start_t), end_time=rospy.Time.from_sec(end_t))]
                evframe = self.decode_prophesee_events_list(evs_batch_msgs)
            else:
                evs_batch = [msg.events for _, msg, _ in self.bag.read_messages(topics=[self.evs_topic], start_time=rospy.Time.from_sec(start_t), end_time=rospy.Time.from_sec(end_t))]
                evframe = self.evarray_batch_to_evframe(evs_batch)

            if self.args.save:
                depth_im_i = first_depth_idx+evs_batch_i+1
                cv_image = self.bridge.imgmsg_to_cv2(depth_ims_msgs[depth_im_i], desired_encoding="passthrough")
                self.depth = np.array(cv_image, dtype=np.float32)

                self.depth = np.clip(self.depth/30e3, 0.0, 1.0) # 65535.0
                self.refine_depth()

                cv2.imwrite(os.path.join(self.saveto_dir, f'{depth_ims_msgs[depth_im_i].header.stamp.to_nsec()/1e9:.3f}_depth.png'), (self.depth*255.0).astype(np.uint8))
                self.n_depthims += 1

                self.evframes.append(evframe)
                self.n_evframes += 1

                # break

        self.bag.close()
        self.cleanup()

    def decode_prophesee_events_list(self, events_msgs):
        all_events = []
        for events_msg in events_msgs:
            self.decoder.decode_bytes(events_msg.encoding, events_msg.width, events_msg.height, events_msg.time_base, events_msg.events)
            cd_events = self.decoder.get_cd_events()
            # x, y, p, t
            events = rf.structured_to_unstructured(cd_events)
            # swap columns to be t, x, y, p
            events = events[:, [3, 0, 1, 2]]
            all_events.extend(events)
        evs_np = np.array(all_events)

        # if there are any events
        if evs_np.shape[0] > 0 and evs_np.ndim == 2:
            evs_np[:, 0] = evs_np[:, 0].astype(float) / 1e6
        else:
            print(f'[BAG {os.path.basename(self.args.bagfile)}] No events found in a batch.')
        # evs_np[:, 0] = evs_np[:, 0].astype(float) / 1e6
        return form_eventframe(evs_np, self.args.ev_height, self.args.ev_width, all_events=True)

    def evarray_batch_to_evframe(self, evarray_batch):
        all_evs = []
        for evlist in evarray_batch:
            all_evs.extend(evlist)
        # print(f'Number of events packets found in this batch: {len(all_evs)}')
        if not self.args.is_flipped:
            evs_np = np.array([[ev.ts.to_sec(), ev.x, ev.y, ev.polarity] for ev in all_evs])
        else:
            evs_np = np.array([[ev.ts.to_sec(), self.args.ev_width - ev.x, self.args.ev_height - ev.y, ev.polarity] for ev in all_evs])
        return form_eventframe(evs_np, self.args.ev_height, self.args.ev_width, all_events=True)

    def refine_depth(self):
        
        neighborhood_radius = 2
        gaussian_sigma = 2

        # image = self.depth
        # Define the threshold values for pixels to interpolate
        if self.depth.mean() > 1.0:
            low_threshold = 0.1  # Value close to 0.0
            high_threshold = 65534.9  # Value close to 65535.0
        else:
            low_threshold = 0.01
            high_threshold = 0.99

        # Create a mask for pixels to interpolate
        mask = np.logical_or(self.depth <= low_threshold, self.depth >= high_threshold)

        # Create a copy of the image to work on
        interpolated_image = np.copy(self.depth)

        # Apply a Gaussian filter to smooth the image (helps in interpolation)
        smoothed_image = gaussian_filter(interpolated_image, sigma=gaussian_sigma)

        # Get the indices of the pixels to interpolate
        blob_indices = np.transpose(np.where(mask))

        # Interpolate each blob
        for blob_index in blob_indices:
            row, col = blob_index

            # Extract the local region around the pixel
            local_region = smoothed_image[max(0, row-neighborhood_radius):min(row+neighborhood_radius+1, self.depth.shape[0]), max(0, col-neighborhood_radius):min(col+neighborhood_radius+1, self.depth.shape[1])]

            # Calculate the mean value of the local region
            mean_value = np.mean(local_region)

            # Assign the mean value to the pixel
            interpolated_image[row, col] = mean_value

        # return interpolated_image
        self.depth = interpolated_image

    def cleanup(self):
        if self.args.save:
            # print number of evims and depth images saved
            print(f'[BAG {os.path.basename(self.args.bagfile)}] Number of depth images saved: {self.n_depthims}')
            print(f'[BAG {os.path.basename(self.args.bagfile)}] Number of evframes saved: {self.n_evframes}')
            # save self.evframes as a npy file in the output directory
            np.save(os.path.join(self.saveto_dir, 'evframes.npy'), np.array(self.evframes))

if __name__ == '__main__':
    
    # parse command line arguments via arg parser
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output_dir', type=str, help='output directory to store images', default='data')
    parser.add_argument('--bagfile', type=str, help='bagfile to process', default='data.bag')
    parser.add_argument('--publish', action='store_true', help='publish the visualized events and depth image')
    parser.add_argument('--save', action='store_true', help='save the visualized events and depth image')
    parser.add_argument('--ev_height', type=int, help='height of the event frame resolution', default=260)
    parser.add_argument('--ev_width', type=int, help='width of the event frame resolution', default=346)
    parser.add_argument('--is_flipped', action='store_true', help='whether the image is flipped or not (prophesee camera is mounted upside down)')
    parser.add_argument('--is_metavision_ros_driver', action='store_true', help='whether using prophesee cam metavision ros driver instead of davis or prophesee driver)')
    parser.add_argument('--ev_topic', type=str, help='topic name for events', default='/event_camera/events')
    args = parser.parse_args()

    depth_to_image = DepthEvsToImage(args)

    depth_to_image.read_bag()
