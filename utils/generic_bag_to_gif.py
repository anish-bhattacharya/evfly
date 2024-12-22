# a generic script to read images from a bagfile and convert to a gif, while saving images to a directory

import os
import sys
import cv2
import rosbag, rospy
from cv_bridge import CvBridge
from tqdm import tqdm
import imageio
from ev_utils import *
from std_msgs.msg import UInt8MultiArray

def extract_images(bag_path, output, topic_input, times_input=None, fps=None, save_ims=False):

    print(f'Extracting images from {bag_path} to {output} on topic {topic_input}')

    # Create the output directory if it does not exist
    # if output then set output_dir
    if os.path.isdir(output):
        output_dir = output
    else:
        output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the CvBridge
    bridge = CvBridge()
    
    timestamps = []
    frames = []

    # Open the bag file
    with rosbag.Bag(bag_path, 'r') as bag:
        
        # Initialize the progress bar
        # with tqdm(total=bag.get_message_count(topic_filters=[topic_input]), desc="Extracting images", unit="image") as pbar:
        print(f"Extracting images from topic {topic_input}")
        # print available topics
        # print(f"Available topics: {bag.get_type_and_topic_info()}")
        # correct times by adding first timestamp in bag
        if times_input is not None:
            times = (times_input[0] + bag.get_start_time(), times_input[1] + bag.get_start_time())
        else:
            times = (bag.get_start_time(), bag.get_end_time())
        for topic, msg, t in bag.read_messages(topics=[topic_input], start_time=rospy.Time(times[0]), end_time=rospy.Time(times[1])):

            if 'proc_evs' in topic_input:

                try:
                    # use simple_evim() to create a viewable image
                    proc_evs = np.frombuffer(msg.data, dtype=np.uint8)
                    proc_evs = proc_evs.reshape(480, 640)
                    evframe = proc_evs.copy().astype(np.float32)
                    evframe -= 128
                    evframe *= 0.2
                    evim, _ = simple_evim(evframe, scaledown_percentile=.99, style='redblue-on-white')
                    us_t = int(t.to_nsec() / 1e3)
                    filename = os.path.join(output_dir, f"{us_t}.png")

                    # print stats of evim
                    # print(f'shape of evim {evim.shape}')
                    # print(f"evim stats: {np.min(evim)}, {np.max(evim)}")

                    # Save the image
                    # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)  # ensure the image is in BGR format before saving as PNG
                    if save_ims:
                        cv2.imwrite(filename, evim)
                    frames.append(evim)

                except Exception as e:
                    print(f"Failed to process image from topic {topic} at time {t}: {e}")

            elif msg._type == 'sensor_msgs/Image':

                try:
                    # Convert the ROS Image message to an OpenCV image
                    # determine if msg is of a grayscale mono image or 3 channel and load it as RGB
                    # print(f'msg encoding: {msg.encoding}')
                    # exit()
                    if msg.encoding == 'mono8':
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                    elif msg.encoding == '8UC1':
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='8UC1')
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                    elif msg.encoding == 'bgr8':
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    else:
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    
                    # Generate the filename with the ROS timestamp
                    us_t = int(msg.header.stamp.to_nsec() / 1e3)
                    if us_t == 0:
                        us_t = int(t.to_nsec() / 1e3)
                    timestamps.append(us_t)
                    filename = os.path.join(output_dir, f"{us_t}.png")
                    
                    # Save the image
                    if save_ims:
                        cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # ensure the image is in BGR format before saving as PNG
                        cv2.imwrite(filename, cv_image_bgr)
                    frames.append(cv_image)
                    
                except Exception as e:
                    print(f"Failed to process image from topic {topic} at time {t}: {e}")

    # create gif and save to same output directory
    print(f'Forming gif from {len(frames)} frames')
    if os.path.isdir(output):
        gif_filename = os.path.join(output_dir, 'output.gif')
    elif output[-3:] == 'gif':
        gif_filename = output
    else:
        print('Output must be a directory or a .gif filename! Exiting.')
        exit()
    if fps is not None:
        imageio.mimsave(gif_filename, frames, fps=fps)
    else:
        imageio.mimsave(gif_filename, frames, fps=len(frames)/(times_input[1] - times_input[0]))

if __name__ == '__main__':
    # Example usage
    bag_path = sys.argv[1]
    output_dir = sys.argv[2]
    topic = sys.argv[3]
    # get two float timestamps from bag
    if len(sys.argv) > 4:
        times = (float(sys.argv[4]), float(sys.argv[5]))
    else:
        times = None
    if len(sys.argv) > 6:
        save_ims = sys.argv[6] == 'True'
    else:
        save_ims = False
    if len(sys.argv) > 7:
        fps = int(sys.argv[7])
    else:
        fps = None
    extract_images(bag_path, output_dir, topic, times, fps, save_ims)
