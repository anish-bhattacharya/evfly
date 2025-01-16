# loads and formats data from a folder of images and events, or a h5 file
# includes some legacy formats
# Anish Bhattacharya, 2023

DEPLOYMENT = False

import glob, os, sys, time
from os.path import join as opj
import csv
import numpy as np
import torch
import random
import getpass
uname = getpass.getuser()
import cv2
from PIL import Image
import re
if not DEPLOYMENT:
    import h5py

def find_unmatched_indices(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    unmatched_indices_1 = [i for i, val in enumerate(list1) if val not in set2]
    unmatched_indices_2 = [i for i, val in enumerate(list2) if val not in set1]

    return unmatched_indices_1, unmatched_indices_2

def dataloader(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None, do_transform=True, events='', keep_collisions=False, return_unmatched=False, logger=None, do_clean_dataset=False, use_h5=True, resize_input=None, split_method='train-val', rescale_depth=0.0, rescale_evs=0.0, traj_ids=None, evs_min_cutoff=None):

    # NOTE use_h5 argument exists for purposes like cleaning datasets, forming h5, etc, which requires loading of original dataset in trajectory-folder format

    # # complete the data_dir to an absolute path if needed
    # if '/' not in data_dir:
    #     data_dir = opj(f'/home/{uname}/evfly_ws/data/datasets', data_dir)

    # given a relative path, complete it to an absolute path
    if not os.path.isabs(data_dir):
        data_dir = opj(os.getcwd(), data_dir)

    # assign logger function
    if logger is None:
        logger = print

    cropHeight = 60
    cropWidth = 90

    # if events filename has not been completed with tf or extension, do so
    if events != '' and '.' not in events:
        if do_transform:
            events += '_tf.npy'
        else:
            events += '.npy'

    # using h5 dataset?
    found_h5 = False
    # check if data_dir.h5 exists
    h5_filename = data_dir+f"{'_tf' if (do_transform and '_tf' not in data_dir) else ''}"+'.h5'
    if os.path.exists(h5_filename) and use_h5:
        logger(f'[DATALOADER] Found {h5_filename}, loading dataset')
        dataset_h5 = h5py.File(h5_filename, 'r')
        found_h5 = True

    # choose traj_folders
    # train_val_dirs may include multiple datasets, so only take the folders and ids that correspond to the current dataset
    dataset_name = os.path.basename(data_dir)
    train_val_dirs_is_invalid = True
    if train_val_dirs is not None:
        for folder_i in range(len(train_val_dirs[0])):
            if dataset_name in train_val_dirs[0][folder_i]:
                train_val_dirs_is_invalid = False
                break
        for folder_i in range(len(train_val_dirs[1])):
            if dataset_name in train_val_dirs[1][folder_i]:
                train_val_dirs_is_invalid = False
                break

    # print whether train val dirs was invalid
    logger(f'[DATALOADER] train_val_dirs_is_invalid={train_val_dirs_is_invalid}')

    if train_val_dirs is not None and not train_val_dirs_is_invalid:

        # convert elements to list if they are not (mistakenly ndarray sometimes)
        train_val_dirs = [el.tolist() if isinstance(el, np.ndarray) else el for el in train_val_dirs]

        # directly edit train_val_dirs to only be the current dataset's folders
        train_val_dirs_new = [[], [], [] ,[]]
        for folder_i in range(len(train_val_dirs[0])):
            if dataset_name in train_val_dirs[0][folder_i]:
                train_val_dirs_new[0].append(train_val_dirs[0][folder_i])
                train_val_dirs_new[2].append(train_val_dirs[2][folder_i])
        for folder_i in range(len(train_val_dirs[1])):
            if dataset_name in train_val_dirs[1][folder_i]:
                train_val_dirs_new[1].append(train_val_dirs[1][folder_i])
                train_val_dirs_new[3].append(train_val_dirs[3][folder_i])
        train_val_dirs = train_val_dirs_new

        # set traj_folders to the inputted train_val_dirs
        traj_folders = train_val_dirs[0] + train_val_dirs[1]
        val_split = len(train_val_dirs[1]) / len(traj_folders)

        # set traj_folders_ids to the inputted train_val_dirs
        traj_folders_ids = np.hstack((train_val_dirs[2], train_val_dirs[3]))

    else:

        if not found_h5:

            # get list of folders in data_dir that are just a number
            # NOTE not doing this ^ since to-events requires looking at all folders. revisit?
            # traj_folders = sorted(folder for folder in glob.glob(opj(data_dir, '*')) if os.path.basename(folder).isdigit())
            traj_folders = sorted(glob.glob(opj(data_dir, '*')))
            traj_folders_ids = np.arange(len(traj_folders)) # to keep track of folder idxs
        
        else:

            # get list of folders in data_dir directly from the h5
            # each group in the dataset is a folder
            traj_folders = list(dataset_h5.keys())
            traj_folders = [opj(data_dir, folder) for folder in traj_folders]
            traj_folders_ids = np.arange(len(traj_folders))
            # print(traj_folders)
            # exit()

        # if traj_ids is not None, use it to select a subset of traj_folders
        if traj_ids is not None:
            traj_folders = traj_folders[traj_ids[0]:traj_ids[1]]
            traj_folders_ids = traj_folders_ids[traj_ids[0]:traj_ids[1]]

        if seed > -2:
            seed = int(time.time()*1e3) if seed==-1 else seed
            random.seed(seed)
            random.shuffle(traj_folders)
            random.seed(seed)
            random.shuffle(traj_folders_ids)

    if short > 0:
        assert short <= len(traj_folders), f"short={short} is greater than the number of folders={len(traj_folders)}"
        traj_folders = traj_folders[:short]
        traj_folders_ids = traj_folders_ids[:short]

    # special case of selecting specific folders (used for 2-26 dataset)
    elif short == -1:
        logger(f'[DATALOADER] short=-1; Using special case for 2-26 dataset')
        tf = []
        tf.append(traj_folders[0:30])
        tf.append(traj_folders[70:100])
        tf.append(traj_folders[115:145])
        tf_ids = []
        tf_ids.append(traj_folders_ids[0:30])
        tf_ids.append(traj_folders_ids[70:100])
        tf_ids.append(traj_folders_ids[115:145])
        traj_folders = tf[0] + tf[1] + tf[2]
        traj_folders_ids = np.hstack((tf_ids[0], tf_ids[1], tf_ids[2]))

    if not found_h5:

        # load events
        # event frames are object arrays of variable length sequences of frames, where frames contain floats at each pixel
        # if frames is in the last part of the path events
        if 'frames' in events:

            evframes = np.load(opj(data_dir, events), allow_pickle=True) # this should be a .npy file, but may be large

            evframes = evframes[traj_folders_ids] if not short == -1 else evframes
            logger(f'[DATALOADER] Loaded event frames of length {len(evframes)} from {events}')

        else:

            evframes = None
            logger(f'[DATALOADER] No event frames loaded.')

    desired_vels = []
    traj_ims_full = []
    traj_depths_full = []
    traj_meta_full = []
    traj_evs_full = []

    # npy or png
    is_png = len(glob.glob(opj(traj_folders[0], '*.png'))) > 0
    logger(f"[DATALOADER] Image files are {'png' if is_png else 'npy'}")

    start_dataloading = time.time()
    do_transform_time = 0

    unmatched_ids_ims = []
    # unmatched_ids_meta = []

    num_collision_trajs = 0

    kept_folder_ids_shuffled = []

    for traj_i, traj_folder in enumerate(traj_folders):

        # logger(f'[DATALOADER] Loading folder {os.path.basename(traj_folder)}')

        corruptImg = 0
        if (len(traj_folders)//10 > 0 and traj_i % (len(traj_folders)//10) == 0) or (len(traj_folders) < 10):
            logger(f'[DATALOADER] Loading folder {os.path.basename(traj_folder)}, folder # {traj_i+1}/{len(traj_folders)}, time elapsed {time.time()-start_dataloading:.2f}s')
        
        # load metadata
        if not found_h5:

            csv_file = 'data.csv'
            try:
                traj_meta = np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype=np.float64)[1:]
            except:
                # Open the CSV file and read its contents
                with open(opj(traj_folder, csv_file), 'r') as file:
                    # Read the CSV file line by line
                    lines = file.readlines()
                    
                    traj_meta = []
                    # Iterate through the lines and exclude the line with the wrong number of columns
                    for line_num, line in enumerate(lines[1:]):

                        num_columns = len(line.strip().split(','))
                        if is_png and num_columns != 21:
                            continue
                        elif not is_png:
                            raise NotImplementedError('This try-except for data.csv reading code is not yet implemented for non-png datasets.')

                        traj_meta.append([float(x) for x in line.strip().split(',')])  # Split the line and convert values to float

                    traj_meta = np.array(traj_meta, dtype=np.float64)

        else:

            traj_meta = dataset_h5[traj_folder.split('/')[-1]]['data'][()]

        # check for nan in metadata
        if np.isnan(traj_meta).any():
            logger(f'[DATALOADER] NaN in {os.path.basename(traj_folder)}, skipping.')
            if do_clean_dataset and not found_h5:
                logger(f'[DATALOADER] Deleting folder {os.path.basename(traj_folder)}')
                os.system(f'rm -r {traj_folder}')
            continue

        # check for collisions in trajectory
        if traj_meta[:,-1].sum() > 0:
            num_collision_trajs += 1
            logger(f"[DATALOADER] {traj_meta[:,-1].sum()} collisions in {os.path.basename(traj_folder)}, {num_collision_trajs}th so far, {'skipping!' if not keep_collisions else 'keeping!'}")
            if not keep_collisions:
                continue

        # load images, depths, evs from dataset folder if not using h5
        traj_depths = None
        traj_ims = None
        if not found_h5:

            # find image and depths filenames
            # newer datasets have both depth images *_depth.png and gray images *_im.png
            depth_files = sorted(glob.glob(opj(traj_folder, '*_depth.png')))
            if len(depth_files) > 0:
                if traj_i == 0:
                    logger(f'[DATALOADER] Found images and depths in {os.path.basename(data_dir)}')
                im_files = sorted(glob.glob(opj(traj_folder, '*_im.png')))
            else:
                im_files = sorted(glob.glob(opj(traj_folder, '*.png' if is_png else '*.npy')))

            # check for empty folder
            if len(im_files) == 0:
                logger(f'[DATALOADER] No images in {os.path.basename(traj_folder)}, skipping.')
                if do_clean_dataset:
                    logger(f'[DATALOADER] Deleting empty folder {os.path.basename(traj_folder)}')
                    os.system(f'rm -r {traj_folder}')
                continue

            # read png files and scale them by 255.0 to recover normalized (0, 1) range
            # for npy files, manually normalize them by a set value (0.09 for "old" dataset)
            if is_png:
                traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in im_files], dtype=np.float32) / 255.0
                if len(depth_files) > 0:
                    traj_depths = np.asarray([cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE) for depth_file in depth_files], dtype=np.float32) / 255.0
            else:
                traj_ims = np.asarray([np.load(im_file, allow_pickle=True) for im_file in im_files]) / 0.09

            ### Deleting erroneous images and metadata ###

            # try to check for non-matching pairs of images/metadata
            # loop through images and check if the timestamp in the filename matches the timestamp in the metadata
            # if not, remove the image
            traj_ims_ts = []
            for im_i in range(traj_ims.shape[0]):
                if not is_png:
                    im_timestamp = os.path.basename(im_files[im_i])[6:-4]
                else:
                    # im_timestamp = os.path.basename(im_files[im_i])[:-4]
                    match = re.search(r'(\d+(\.\d+)?)', os.path.basename(im_files[im_i]))
                    # Check if there is exactly one match
                    # match groups contains "the whole match" and then each subcomponent, so index 1 is the first relevent one
                    if match is not None and len(match.groups()) == 2:
                        im_timestamp = match.group(1)
                        # logger(f'[DATALOADER] Found timestamp {im_timestamp} in image filename {im_files[im_i]}')
                        # logger('THIS WAS A DEBUG MSG, QUITTING NOW!')
                        # exit()
                    else:
                        logger(f'[DATALOADER] More than one numeric component in image filename {im_files[im_i]} with basename {os.path.basename(im_files[im_i])}. Exiting.')
                        # if do_clean_dataset:
                        #     logger(f'[DATALOADER] Deleting folder {os.path.basename(traj_folder)}')
                        #     os.system(f'rm -r {traj_folder}')
                        exit()
                traj_ims_ts.append(float(im_timestamp))
            
            # check for duplicates in metadata timestamps
            # since collisions force logging, there may be duplicate timestamps
            unique_elements, unique_indices, counts = np.unique(traj_meta[:, 1], return_index=True, return_counts=True)
            duplicate_indices = unique_indices[counts > 1]
            traj_meta = np.delete(traj_meta, duplicate_indices, axis=0)

            # find indices of mismatched timestamps
            st_match = time.time()
            unmatched_indices_1, unmatched_indices_2 = find_unmatched_indices(traj_ims_ts, list(traj_meta[:, 1]))
            if len(unmatched_indices_1) > 0 or len(unmatched_indices_2) > 0:
                logger(f'[DATALOADER] Unmatched timestamps in {os.path.basename(traj_folder)}: (deleting these!)')
                logger(f'[DATALOADER]\tIdxs of images: {unmatched_indices_1}')
                logger(f'[DATALOADER]\tIdxs of metadata: {unmatched_indices_2}')
                traj_ims = np.delete(traj_ims, unmatched_indices_1, axis=0)
                if do_clean_dataset:
                    for im_idx in unmatched_indices_1:
                        logger(f'[DATALOADER] Deleting image {im_files[im_idx]}')
                        os.system(f'rm {im_files[im_idx]}')
                if len(depth_files) > 0:
                    traj_depths = np.delete(traj_depths, unmatched_indices_1, axis=0)
                    if do_clean_dataset:
                        for depth_idx in unmatched_indices_1:
                            logger(f'[DATALOADER] Deleting depth {depth_files[depth_idx]}')
                            os.system(f'rm {depth_files[depth_idx]}')
                traj_meta = np.delete(traj_meta, unmatched_indices_2, axis=0)
                logger(f'[DATALOADER]\tTime to find and delete unmatched indices: {time.time()-st_match:.3f}s')

            # logger(f'traj {traj_i}')
            # logger(unmatched_indices_1)
            unmatched_ids_ims.append(unmatched_indices_1)
            # logger(type(duplicate_indices))
            # logger(duplicate_indices[0]) if len(duplicate_indices) > 0 else logger('no duplicates')
            # logger(unmatched_indices_2)
            # unmatched_ids_meta.append(list(duplicate_indices)+unmatched_indices_2)

            # convert metadata to np.float32 after matching timetamps and ims/depths is done
            # 0-start all timestamps in column 1
            traj_meta[:, 1] -= traj_meta[0, 1]
            # convert to np.float32
            traj_meta = np.array(traj_meta, dtype=np.float32)

            if events != '':
                traj_evs_full.append(torch.from_numpy(evframes[traj_i]))

        # else found_h5, load ims, depths, evs from h5 file
        else:

            traj_depths = dataset_h5[traj_folder.split('/')[-1]]['depths']
            traj_ims = dataset_h5[traj_folder.split('/')[-1]]['ims']

            if 'frames' in events:
                evframes = dataset_h5[traj_folder.split('/')[-1]]['evs'][()]
                traj_evs_full.append(torch.from_numpy(evframes))
            else:
                evframes = None

        # fill list of desired velocities
        for ii in range(traj_meta.shape[0]):
            if is_png or found_h5:
                desired_vels.append(traj_meta[ii, 2])
            else:
                desired_vels.append(np.max(traj_meta[:, 12])) # approximate desired vel from max vel in x vel cmd

        # try:
        traj_ims_full.append(torch.from_numpy(np.array(traj_ims, dtype=np.float32)))
        traj_meta_full.append(torch.from_numpy(traj_meta))
        if traj_depths is not None:
            traj_depths_full.append(torch.from_numpy(np.array(traj_depths, dtype=np.float32)))
        
        # except:
        #     logger(f'[DATALOADER] {traj_ims.shape}')
        #     logger(f"[DATALOADER] Suspected empty image, folder {os.path.basename(traj_folder)}")
        #     if do_clean_dataset:
        #         logger(f'[DATALOADER] Deleting folder {os.path.basename(traj_folder)}')
        #         os.system(f'rm -r {traj_folder}')

        kept_folder_ids_shuffled.append(traj_i)

    # maintain record of traj folders that were not skipped
    traj_folders = [traj_folders[i] for i in kept_folder_ids_shuffled]
    traj_folders_ids = [traj_folders_ids[i] for i in kept_folder_ids_shuffled]

    im_h = cropHeight if do_transform else traj_ims.shape[1]
    im_w = cropWidth if do_transform else traj_ims.shape[2]
    logger(f'[DATALOADER] Images are of size {im_h, im_w} (do_transform={do_transform})')
    logger(f'[DATALOADER] Time to load dataset: {time.time()-start_dataloading:.3f}s')
    logger(f'[DATALOADER] Time to do_transform: {do_transform_time:.3f}s')

    # concatenation of trajectory data with high resolution data takes a long time,
    # so let's downsample the data first according to resize_input

    # optional downsampling
    if resize_input is not None and \
        (traj_ims_full[0].shape[-2:] != torch.Size(resize_input) or \
         traj_depths_full[0].shape[-2:] != torch.Size(resize_input) or 
         (evframes is None or traj_evs_full[0].shape[-2:] != torch.Size(resize_input))):

        logger(f'[DATALOADER] Resizing input images to {resize_input}')
        resize_st_t = time.time()
        # ims
        for im_i, im in enumerate(traj_ims_full):
            traj_ims_full[im_i] = torch.nn.functional.interpolate(im.unsqueeze(1), size=resize_input, mode='bilinear', align_corners=False).squeeze()
        # depths
        for depth_i, depth in enumerate(traj_depths_full):
            traj_depths_full[depth_i] = torch.nn.functional.interpolate(depth.unsqueeze(1), size=resize_input, mode='bilinear', align_corners=False).squeeze()
        # evs
        if evframes is not None:
            for ev_i, ev in enumerate(traj_evs_full):
                traj_evs_full[ev_i] = torch.nn.functional.interpolate(ev.unsqueeze(1), size=resize_input, mode='bilinear', align_corners=False).squeeze()
        im_h, im_w = resize_input
        logger(f'[DATALOADER] Time to resize input images: {time.time()-resize_st_t:.3f}s')

    else:
        print(f'[DATALOADER] No resizing of input images to {resize_input} needed!')

    # make into numpy arrays
    logger(f'[DATALOADER] Concatenating data')
    start_concat = time.time()
    traj_lengths = np.array([traj_ims.shape[0] for traj_ims in traj_ims_full])
    traj_ims_full = torch.concatenate(traj_ims_full).reshape(-1, im_h, im_w)
    if traj_depths is not None:
        traj_depths_full = torch.concatenate(traj_depths_full).reshape(-1, im_h, im_w)
    traj_meta_full = torch.concatenate(traj_meta_full).reshape(-1, traj_meta.shape[-1])
    desired_vels = torch.Tensor(desired_vels)
    logger(f'[DATALOADER] Time to concatenate data: {time.time()-start_concat:.3f}s')

    # we need a numpy array of torch tensors since each trajectory of events is of variable length, and only np arrays can be of dtype object
    if events != '':
        traj_evs_full_array = np.empty(len(traj_evs_full), dtype=object)
        for ev_i, ev in enumerate(traj_evs_full):
            traj_evs_full_array[ev_i] = ev
        traj_evs_full = traj_evs_full_array

    # make train-val split (relies on earlier shuffle of traj_folders to randomize selection)
    if split_method == 'train-val':
        num_train_trajs = int((1.-val_split) * len(traj_lengths))
        train_trajs_st = 0
        train_trajs_end = num_train_trajs
        val_trajs_st = num_train_trajs
        val_trajs_end = len(traj_lengths)
        train_idx_st = 0
        train_idx_end = np.sum(traj_lengths[:num_train_trajs], dtype=np.int32)
        val_idx_st = train_idx_end
        val_idx_end = np.sum(traj_lengths, dtype=np.int32)
    elif split_method == 'val-train':
        num_val_trajs = int(val_split * len(traj_lengths))
        val_trajs_st = 0
        val_trajs_end = num_val_trajs
        train_trajs_st = num_val_trajs
        train_trajs_end = len(traj_lengths)
        val_idx_st = 0
        val_idx_end = np.sum(traj_lengths[:num_val_trajs], dtype=np.int32)
        train_idx_st = val_idx_end
        train_idx_end = np.sum(traj_lengths, dtype=np.int32)
    else:
        raise ValueError(f'split_method={split_method} not implemented!')

    traj_meta_train = traj_meta_full[train_idx_st:train_idx_end]
    traj_meta_val = traj_meta_full[val_idx_st:val_idx_end]

    traj_ims_train = traj_ims_full[train_idx_st:train_idx_end]
    traj_ims_val = traj_ims_full[val_idx_st:val_idx_end]
    
    if traj_depths is not None:

        # rescale depths to 0-1
        if rescale_depth > 0.0:
            # # DEBUG save a sample depth image
            # sample_depth = traj_depths_full[0].numpy()
            # np.save('sample_depth.npy', sample_depth)
            # exit()
            max_depth_val = torch.max(traj_depths_full)
            min_depth_val = torch.min(traj_depths_full)
            logger(f'[DATALOADER] Rescaling depth by {rescale_depth}\tNOTE max/min of dataset depth is {max_depth_val}/{min_depth_val}.')
            traj_depths_full = torch.clamp(traj_depths_full/rescale_depth, 0, 1.0)

        traj_depths_train = traj_depths_full[train_idx_st:train_idx_end]
        traj_depths_val = traj_depths_full[val_idx_st:val_idx_end]

    else:
        traj_depths_train = None
        traj_depths_val = None
    
    desired_vels_train = desired_vels[train_idx_st:train_idx_end]
    desired_vels_val = desired_vels[val_idx_st:val_idx_end]
    
    traj_folders_train = traj_folders[train_trajs_st:train_trajs_end]
    traj_folders_val = traj_folders[val_trajs_st:val_trajs_end]

    traj_folders_ids_train = traj_folders_ids[train_trajs_st:train_trajs_end]
    traj_folders_ids_val = traj_folders_ids[val_trajs_st:val_trajs_end]

    traj_lengths_train = traj_lengths[train_trajs_st:train_trajs_end]
    traj_lengths_val = traj_lengths[val_trajs_st:val_trajs_end]
    
    if evframes is not None:

        # compute max/min by list comprehension of traj_evs_full since each element is a torch array
        max_evs_val = max([torch.max(ev) for ev in traj_evs_full])
        min_evs_val = min([torch.min(ev) for ev in traj_evs_full])
        logger(f'[DATALOADER] Rescaling evs = {rescale_evs}\tNOTE max/min of dataset evs is {max_evs_val}/{min_evs_val}.')

        if rescale_evs > 0.0:
            traj_evs_full = [torch.clamp(ev/rescale_evs, -1.0, 1.0) for ev in traj_evs_full]

        # rescale by maximum of each frame
        elif rescale_evs == -1.0:

            # event frames can have wildly large values for a single pixel (2-12_0-99 dataset has max/min around 22.0/-22.0), throwing scaling off
            # so, clamp frames to values set by the 97th percentile of the data
            percentile_vals = []
            for ev_i, ev in enumerate(traj_evs_full):
                maxvals = torch.quantile(torch.abs(ev).view(ev.shape[0], -1), 0.97, dim=1) # per-frame 97th percentile
                percentile_vals.append(maxvals.mean())
                maxvals = maxvals.view(ev.shape[0], 1, 1)
                traj_evs_full[ev_i] = torch.clamp(ev / maxvals, -1.0, 1.0)

            logger(f'[DATALOADER] Rescaling evs by 97th percentile of each frame, first and last traj 97th mean percentile values are {percentile_vals[0]:.2f} and {percentile_vals[-1]:.2f}')

            # for ev_i, ev in enumerate(traj_evs_full):
            #     maxvals = torch.max(torch.abs(ev).view(ev.shape[0], -1), dim=1).values
            #     maxvals = maxvals.view(ev.shape[0], 1, 1)
            #     traj_evs_full[ev_i] = ev / maxvals

        # remove low-value events
        if evs_min_cutoff is not None:
            for ev_i, ev in enumerate(traj_evs_full):
                traj_evs_full[ev_i][ev.abs() < evs_min_cutoff] = 0.0

    traj_evs_train = traj_evs_full[train_trajs_st:train_trajs_end] if evframes is not None else None
    traj_evs_val = traj_evs_full[val_trajs_st:val_trajs_end] if evframes is not None else None

    unmatched_ids_ims_train = unmatched_ids_ims[train_trajs_st:train_trajs_end]
    unmatched_ids_ims_val = unmatched_ids_ims[val_trajs_st:val_trajs_end]
    
    # unmatched_ids_meta_train = unmatched_ids_meta[:num_train_trajs]
    # unmatched_ids_meta_val = unmatched_ids_meta[num_train_trajs:]

    # release h5 dataset if used
    if found_h5:
        dataset_h5.close()

    # Note, we return the is_png flag since it indicates old vs new datasets, which indicates how to parse the metadata
    # We also return the traj_folder names for train and val sets, so that they can be saved and later used to specifically generate evaluate plots on each set
    # For easy events loading from a previously loaded dataset, we now also include the shuffled trajectory indices in train_val_dirs.npy
    if not return_unmatched:
        return \
                (traj_meta_train, (traj_ims_train, traj_depths_train), traj_lengths_train, desired_vels_train, traj_evs_train, traj_folders_train, traj_folders_ids_train), \
                (traj_meta_val, (traj_ims_val, traj_depths_val), traj_lengths_val, desired_vels_val, traj_evs_val, traj_folders_val, traj_folders_ids_val), \
                is_png or found_h5
    else:
        return \
                (traj_meta_train, (traj_ims_train, traj_depths_train), traj_lengths_train, desired_vels_train, traj_evs_train, traj_folders_train, traj_folders_ids_train, unmatched_ids_ims_train), \
                (traj_meta_val, (traj_ims_val, traj_depths_val), traj_lengths_val, desired_vels_val, traj_evs_val, traj_folders_val, traj_folders_ids_val, unmatched_ids_ims_val), \
                is_png or found_h5

def parse_meta_str(meta_str):

    meta = torch.zeros_like(meta_str)

    meta_str

    return meta

def preload(items, device='cpu'):

    out_items = []
    for item in items:
        if item is not None:
            if isinstance(item, list) or (isinstance(item, np.ndarray) and item.dtype == object):
                out_items.append([torch.from_numpy(ite).to(device).float() if type(ite) == np.ndarray else ite.to(device).float() for ite in item])
            else:
                out_items.append(torch.from_numpy(np.array(item, dtype=item.dtype)).to(device) if type(item) == np.ndarray else item.to(device))
        else:
            out_items.append(None)

    return out_items
