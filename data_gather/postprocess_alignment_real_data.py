# postprocess real data collected from bags, usually done with data_from_rosbags.sh and depth_and_events_script.py

import sys, os, glob
import numpy as np
import cv2

sys.path.append('/home/anish1/evfly_ws/src/evfly/utils')
from calibration_tools.rectify_bag import Aligner

data_dir = sys.argv[1] # folder containing original data read from rosbag
output_dir = sys.argv[2] # folder to save processed data in same format but processes (aligned)
if len(sys.argv) > 3:
    crop_size = (int(sys.argv[3]), int(sys.argv[4]))
    print(f'Doing center-crop of size {crop_size}')
else:
    crop_size = None
    print('No center crop size specified.')

# if output dir doesn't exist, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

aligner = Aligner(calib_file='/home/anish1/evfly_ws/src/evfly/utils/calib_7-28/K.yaml')

# loop over all directories in the data_dir
folders = sorted(glob.glob(os.path.join(data_dir, '*/')))
for folder_i, folder in enumerate(folders):

    folder = folder[:-1]

    print(f'Processing {folder} ({folder_i+1}/{len(folders)})')

    # make corresponding folder in output_dir
    output_folder = os.path.join(output_dir, os.path.basename(folder))

    os.makedirs(output_folder, exist_ok=False)

    evframes_aligned = []

    # load all *_depth.png files in this folder, sorted
    depth_files = sorted(glob.glob(os.path.join(folder, '*_depth.png')))

    # load evframes.npy file
    evframes = np.load(os.path.join(folder, 'evframes.npy'))

    # since the checking of depth & evframe file number matching is done later (utils/convert_realdata_to_datasetformat.py),
    # here just loop over each set independently and slowly to process
    for df in depth_files:
            
        # load depth image and convert to single channel if needed
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH)
        if len(depth.shape) > 2:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        
        out = aligner.align(depth=depth, davis=None)
        depth = out['depth']

        # print(f'depth shape 1 = {depth.shape}')

        # NOTE optional center-crop here of size 260, 346
        if crop_size is not None:
            center = (depth.shape[0]//2, depth.shape[1]//2)
            depth = depth[center[0]-crop_size[0]//2 : center[0]+crop_size[0]//2,
                        center[1]-crop_size[1]//2 : center[1]+crop_size[1]//2]

        # print(f'depth shape 2 = {depth.shape}')

        # save depth image
        depth_filename = os.path.basename(df)
        cv2.imwrite(os.path.join(output_folder, depth_filename), depth)

    for evframe in evframes:
        out = aligner.align(davis=evframe, depth=None)
        evframe_aligned = out['davis']
        if crop_size is not None:
            # NOTE optional crop again
            center = (evframe_aligned.shape[0]//2, evframe_aligned.shape[1]//2)
            evframe_aligned = evframe_aligned[center[0]-crop_size[0]//2 : center[0]+crop_size[0]//2, center[1]-crop_size[1]//2 : center[1]+crop_size[1]//2]
        evframes_aligned.append(evframe_aligned)
    np.save(os.path.join(output_folder, 'evframes.npy'), np.array(evframes_aligned))

print(f'Done processing dataset in {data_dir} and saved to {output_dir}')
