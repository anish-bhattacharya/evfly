# extract images from a rosbag
# this script is usually used when calibrating events+depth via bagging the infrared stream from a d435
# so we extract the infra images and their timestamps (to use for events->image e2vid reconstruction) to use with kalibr later

import os
import sys
import cv2
import rosbag
from cv_bridge import CvBridge
from tqdm import tqdm

def extract_infra1_images(bag_path, output_dir):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the CvBridge
    bridge = CvBridge()
    
    timestamps = []

    # Open the bag file
    with rosbag.Bag(bag_path, 'r') as bag:
        # Get the total number of messages in the desired topic for progress bar

        topic = '/grayscale_camera/image_raw'
        
        # Initialize the progress bar
        with tqdm(total=bag.get_message_count(topic_filters=[topic]), desc="Extracting images", unit="image") as pbar:
            for topic, msg, t in bag.read_messages(topics=[topic]):
                try:
                    # Convert the ROS Image message to an OpenCV image
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    
                    # Generate the filename with the ROS timestamp
                    us_t = int(msg.header.stamp.to_nsec() / 1e3)
                    if us_t == 0:
                        us_t = int(t.to_nsec() / 1e3)
                    timestamps.append(us_t)
                    filename = os.path.join(output_dir, f"{us_t}.png")
                    
                    # Save the image
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)  # ensure the image is in BGR format before saving as PNG
                    cv2.imwrite(filename, cv_image)
                    
                except Exception as e:
                    print(f"Failed to process image from topic {topic} at time {t}: {e}")
                finally:
                    pbar.update(1)

    # Save timestamps to a txt file with each timestamp on a new line
    with open(os.path.join(output_dir, 'timestamps.txt'), 'w') as f:
        for ts in timestamps:
            f.write(f"{ts}\n")

if __name__ == '__main__':
    # Example usage
    bag_path = sys.argv[1]
    output_dir = sys.argv[2]
    extract_infra1_images(bag_path, output_dir)
