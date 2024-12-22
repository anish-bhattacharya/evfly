# given a large bag, splice it to smaller "trajectory" bags given pairs of timestamps

import rosbag
import rospy
import os, sys

def splice_bag(input_bag_path, output_dir, timestamps):
    """
    Splice a ROS bag file into smaller bags given some timestamps.

    Args:
    - input_bag_path (str): Path to the input ROS bag file.
    - output_dir (str): Directory to save the output spliced bags.
    - timestamps (list of tuples): List of (start_time, end_time) tuples in seconds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # extract filename from input_bag_path
    input_bag = rosbag.Bag(input_bag_path, 'r')
    bag_start_time = input_bag.get_start_time()

    start_timestamps = []

    for i, (start_time, end_time) in enumerate(timestamps):
        output_bag_path = os.path.join(output_dir, f'spliced_{i}.bag')
        with rosbag.Bag(output_bag_path, 'w') as output_bag:
            for topic, msg, t in input_bag.read_messages(start_time=rospy.Time.from_sec(bag_start_time+start_time), end_time=rospy.Time.from_sec(bag_start_time+end_time)):
                if len(start_timestamps) < i+1:
                    print(f'Adding start timestamp {t.to_sec()} to timestamps list')
                    start_timestamps.append(t.to_sec())
                output_bag.write(topic, msg, t)
        
        print(f"Created bag: {output_bag_path}")

    input_bag.close()

    return start_timestamps

# Example usage
if __name__ == "__main__":
    input_bag_path = sys.argv[1]
    output_dir = sys.argv[2]
    end_time = int(sys.argv[3])
    timestamps = [(i, i+10) for i in range(0, end_time, 10)]

    start_ts = splice_bag(input_bag_path, output_dir, timestamps)

    # rename bags according to each start timestamp, with milliseconds
    for i, ts in enumerate(start_ts):
        print(f'Renaming spliced_{i}.bag to {ts:.3f}.bag')
        os.rename(os.path.join(output_dir, f'spliced_{i}.bag'), os.path.join(output_dir, f'{ts:.3f}.bag'))
