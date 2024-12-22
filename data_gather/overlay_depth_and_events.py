# overlay depth images and event frames and save to a gif for viewing

import os, sys, glob
import numpy as np
import cv2, imageio

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from ev_utils import simple_evim

frames = []

if len(sys.argv) == 2:

    depth_directory = sys.argv[1] # directory containing depth images pngs and event frame npy
    evframes_path = os.path.join(sys.argv[1], 'evframes.npy') # directory containing event frames npy

elif len(sys.argv) == 3:

    depth_directory = sys.argv[1]
    evframes_path = sys.argv[2]

# load all depth images from given directory
depth_images = sorted(glob.glob(os.path.join(depth_directory, '*_depth.png')))

# load all event frames from given directory
evframes = np.load(os.path.join(evframes_path), allow_pickle=True)

if len(sys.argv) == 3:

    # find index of given depth folder in directory
    upper_dir = os.path.dirname(depth_directory)
    all_depth_dirs = sorted(glob.glob(os.path.join(upper_dir, '17*/')))
    # find index of given depth folder in all depth folders
    for index, directory in enumerate(all_depth_dirs):
        if os.path.basename(depth_directory) in directory:
            chosen_traj = index
            break

    print(f'Chose trajectory {chosen_traj} from {len(all_depth_dirs)} trajectories')

    evframes = evframes[chosen_traj]

# there should be exactly an equal number of depth images and event frames
for frame_i in range(len(evframes)):
    depth = cv2.imread(depth_images[frame_i+1 if len(depth_images) == len(evframes)+1 else frame_i], cv2.IMREAD_ANYDEPTH)
    evframe = evframes[frame_i]

    # overlay depth and event frame
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    # depth = cv2.addWeighted(depth, 0.5, evframe, 0.5, 0)

    # FOR CONVENIENT EVENT-DEPTH COMPARISON
    # where evim has positive values, replace depth pixel with red value pixel
    # where evim has negative values, replace depth pixel with blue value pixel
    # where evim has zero values, keep depth pixel as is
    pos_ids = (evframe > 0).nonzero()
    neg_ids = (evframe < 0).nonzero()

    # subsample the ids by 1/5
    # downsample_factor = 5
    # pos_ids = (pos_ids[0][::downsample_factor], pos_ids[1][::downsample_factor])
    # neg_ids = (neg_ids[0][::downsample_factor], neg_ids[1][::downsample_factor])

    # depth[pos_ids] = [0, 0, 255]
    # depth[neg_ids] = [255, 0, 0]

    # FOR BETTER EVENT VIS
    evim, _ = simple_evim(evframe, style='redblue-on-white')    
    zero_ids = (evim == [255, 255, 255]).all(axis=-1)
    depth[~zero_ids] = evim[~zero_ids]

    # # print num positive, negative, zero pixels in evim
    # print(f'shape of evim = {evframe.shape}')
    # print(f'pos: {len(pos_ids[0])}, neg: {len(neg_ids[0])}, zero: {evframe.shape[0]*evframe.shape[1] - len(pos_ids[0]) - len(neg_ids[0])}')

    frames.append(depth)

# save frames as gif
imageio.mimsave('~/output.gif', frames, fps=30)

