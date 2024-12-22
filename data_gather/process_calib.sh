# script to run all processing scripts on calibration data

calib_dir=$1
# if second argument provided, else 0
GPU_ID=0
if [ $# -eq 2 ]; then
    GPU_ID=$2
fi

echo "Processing calibration data in $calib_dir"

# use calib bag to extract infra1 images and corresponding timestamps
echo "Extracting infra1 images and timestamps"
python3 /home/anish1/evfly_ws/src/evfly/data_gather/ims_from_rosbag.py \
 $calib_dir/evs*.bag \
 $calib_dir/infrared1 \
 &
# get pid of last process
infra1_pid=$!

# e2calib convert dvs events bag to h5
echo "Converting dvs events to h5"
python3 /home/anish1/e2calib/python/convert.py \
 $calib_dir/dvs_rosbag.bag \
 -o $calib_dir/events.h5 \
 -t /capture_node/events \
 &
# pid
dvs_pid=$!

# if both are done then run reconstruction
wait $infra1_pid
wait $dvs_pid

conda activate e2calib

# run reconstruction from events to images at infra1 timestamps
echo "Running events -> images reconstruction"
CUDA_VISIBLE_DEVICES=$GPU_ID python /home/anish1/e2calib/python/offline_reconstruction.py \
 --h5file $calib_dir/events.h5 \
 --height 480 \
 --width 640 \
 --timestamps_file $calib_dir/infrared1/timestamps.txt \
 --output $calib_dir/reconstructions \
 --upsample_rate 1

# merge reconstructions and infrared1 into a new rosbag
echo "Merging reconstructions and infrared1 into a new rosbag"
python3 /home/anish1/evfly_ws/src/evfly/data_gather/images_to_bags.py \
 --image_folders $calib_dir/reconstructions/e2calib $calib_dir/infrared1 \
 --topics /cam0/image_raw /cam1/image_raw \
 --output_bag $calib_dir/kalibr_rosbag.bag

# now enter kalibr docker and run calibration
echo "Running calibration in docker"
bash /home/anish1/evfly_ws/src/evfly/data_gather/run_kalibr.sh $calib_dir


