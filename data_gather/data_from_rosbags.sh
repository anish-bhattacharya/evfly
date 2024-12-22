# this script runs through multiple rosbags in a directory,
# uses a python script to run through them and do synchronizer/alignment/collection

# usage: bash data_from_rosbags.sh /home/anish/evfly_ws/data/BAGS/2-27 /home/anish/evfly_ws/data/DATASETS/2-27

# directory to search for rosbag files in
data_dir=$1

# directory to save the output data in
output_dir=$2

# array to store PIDs
pids=()

# loop over each rosbag found in this data_dir
for bag in $data_dir/*.bag; do
    # get the filename without the path
    filename=$(basename $bag)
    # print the filename
    echo $filename
    # get the filename without the extension
    filename_noext=${filename%.*}

    # run the data synchronizer/alignment/collection node
    python3 /root/evfly_ws/src/evfly/data_gather/depth_and_events_script.py --bagfile $bag --output_dir "$2/$filename" --save --ev_height 480 --ev_width 640 --is_metavision_ros_driver &

    # store the PID of the last background command
    pids+=($!)

done

# Wait for all processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All processes have finished."
