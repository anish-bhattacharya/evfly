# Marco Cannici, 2024

import rosbag
import rospy
import argparse
import os
import numpy as np
import glob
import cv2
from cv_bridge import CvBridge
import tqdm

bridge = CvBridge()

TIMESTAMPS_FILE = "timestamps.txt"

def parse_image_filenames(f):
    return int(os.path.basename(f).split(".")[0])

def parse_folder(folder):
    images = sorted(glob.glob(os.path.join(folder, "*.png")))

    if TIMESTAMPS_FILE in os.listdir(folder):
        timestamps_ns = 1000 * np.genfromtxt(os.path.join(folder, TIMESTAMPS_FILE))
    else:
        timestamps_ns = np.array([parse_image_filenames(f) for f in images])

    return timestamps_ns, images

def flip_image(image, flip):
    if flip == "n":
        return image
    elif flip == "h":
        return image[:,::-1]
    elif flip == "v":
        return image[::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Create bag from a sequence of images""")
    parser.add_argument("--image_folders", nargs="+", default=[])
    parser.add_argument("--topics", nargs="+", default=[])
    parser.add_argument("--flips", nargs="+", default=[])
    parser.add_argument("--output_bag", default="")
    args = parser.parse_args()

    if len(args.flips) == 0:
        args.flips = ["n"] * len(args.topics)

    with rosbag.Bag(args.output_bag, "w") as bag:
        pbar_tot = tqdm.tqdm(total=len(args.image_folders))
        for image_folder, topic, flip in zip(args.image_folders, args.topics, args.flips):
            pbar_tot.set_description("Writing images from " + image_folder + " to " + topic)
            timestamps_ns, images_files = parse_folder(image_folder)
            pbar = tqdm.tqdm(total=len(images_files))
            for t_ns, f_img in zip(timestamps_ns, images_files):
                image = cv2.imread(f_img, cv2.IMREAD_GRAYSCALE)
                image = flip_image(image, flip)
                msg = bridge.cv2_to_imgmsg(image, encoding="mono8")
                t_ros = rospy.Time.from_sec(float(t_ns)/1e9)
                msg.header.stamp = t_ros
                bag.write(topic, msg, t_ros)
                pbar.update(1)

            pbar_tot.update(1)