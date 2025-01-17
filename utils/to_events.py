# to_events.py
# modified from EvDNeRF/datagen/genevents.py
# This script runs Vid2E on a given image directory to convert RGB images to events.
# Via argument acc_scheme it can either compile some number of fixed-time frames
# or return a continuous stream of events per viewpoint.

import subprocess
import numpy as np
import torch
import numpy as np
import glob
import cv2
from PIL import Image
import os, sys, json, argparse, shutil
from os.path import join as opj
import time

import getpass
uname = getpass.getuser()
sys.path.append(f'/home/{uname}/evfly_ws/src/evfly/learner')

from dataloading import dataloader

# this function manually loads the image and converts to grayscale via averaging the RGB channels
# conversion from BGR to RGB is not necessary since we are averaging channels, but keeping for clarity
def load_im_as_gray(f):
    return cv2.cvtColor(cv2.imread(f, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)[...,:3].mean(-1).astype(np.uint8)

parser = argparse.ArgumentParser(description='This script will generate events via vid2e/torch_esim. Note you should be in conda environment vid2e.')
parser.add_argument('--dataset', type=str, default='.', help='path to dataset of trajectories')
parser.add_argument('--upsamp_dir', type=str, default='', help='path to store upsamp_input and upsamp_output directories')
parser.add_argument('--output_path', type=str, default='./output', help='output path to deposit evs.npy of ev frames')
parser.add_argument('--output_suffix', type=str, default='', help='output filename suffix, usually used for manually batchifying trajectories to limit RAM usage')
parser.add_argument('--acc_scheme', type=str, default='continuous', help='accumulation scheme for events')
parser.add_argument('--path_evs_cont', type=str, default='', help='(optional) path to existing continuous-time events file, so esim is not run again')
parser.add_argument('--des_n_evframes', type=int, default=31, help='(optional) desired number of event frames to generate, if acc_scheme is time')
parser.add_argument("--upsample", action='store_true', help='whether to upsample the images before generating events; generally this should always be true')
parser.add_argument('--num_views', type=int, default=-1, help='number of views')
parser.add_argument('--upsampling_done', action='store_true', help='whether upsampling was done already')
parser.add_argument('--also_time', action='store_true', help='messy way to do continuous and time-based accumulation')
parser.add_argument('--thresh', type=float, default=0.2, help='threshold for positive and negative events')
parser.add_argument('--upsample_starting_traj', type=int, default=0, help='number of views')
parser.add_argument('--scale_up_fps', type=int, default=1, help='Manually scale up fps of image sequences for upsampling so SuperSloMo does not interpolate as much. Messy.')
parser.add_argument('--keep_collisions', action='store_true', help='keep trajectories with collisions (just calculating test events usually)')
parser.add_argument('--cpu', action='store_true', help='cpu version (esim_py, not esim_torch)')
parser.add_argument('--do_transform', action='store_true', help='transform images prior to calculating events via auto gimbaling')
parser.add_argument('--only_difflog', action='store_true', help='only calculate difflog approximation, not the vid2e events')
parser.add_argument('--short', type=int, default=0, help='shorten the dataset for testing')
parser.add_argument('--traj_ids', nargs='+', type=int, default=None, help='trajectory indices to process')

args = parser.parse_args()

print(args)

# if output_path doesn't exist, make it
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

pos_thresh = args.thresh
neg_thresh = args.thresh

# load data
train_data, val_data, is_png = dataloader(args.dataset, do_transform=args.do_transform, seed=-2, keep_collisions=args.keep_collisions, return_unmatched=False, use_h5=False, short=args.short, traj_ids=args.traj_ids)

train_meta, (train_ims, train_depths), train_trajlength, train_desvel, train_evs, train_dirs, train_dirs_ids = train_data

# for this script make data of type numpy array from torch tensor
train_meta = np.array(train_meta)
train_ims = np.array(train_ims)
train_depths = np.array(train_depths)
train_trajlength = np.array(train_trajlength)
train_desvel = np.array(train_desvel)
train_evs = np.array(train_evs)

# # if traj_ids is given, filter the data
# if args.traj_ids is not None:
#     train_meta = train_meta[args.traj_ids[0]:args.traj_ids[1]]
#     train_ims = train_ims[args.traj_ids[0]:args.traj_ids[1]]
#     train_depths = train_depths[args.traj_ids[0]:args.traj_ids[1]]
#     train_trajlength = train_trajlength[args.traj_ids[0]:args.traj_ids[1]]
#     train_desvel = train_desvel[args.traj_ids[0]:args.traj_ids[1]]
#     train_evs = train_evs[args.traj_ids[0]:args.traj_ids[1]]
#     train_dirs = train_dirs[args.traj_ids[0]:args.traj_ids[1]]
#     train_dirs_ids = train_dirs_ids[args.traj_ids[0]:args.traj_ids[1]]

num_trajs = len(train_trajlength)

print(f'Meta infomation of shape {train_meta.shape}')
print(f'Images of shape {train_ims.shape}')

if not args.upsample:
    print('WARNING: Not upsampling images before generating events.')

    # # get all timestamps for consecutive entries in transforms json
    # timestamps_s = np.array([frame['time'] for frame in metas['frames']])
    # num_timesteps = int(len(metas['frames']) / num_views)

#######################
### UPSAMPLE IMAGES ###
#######################

else:
    print('Upsampling images before generating events.')

    num_views = args.num_views

    # extract last component of directory path
    dataset_tag = os.path.split(args.dataset)[1]
    print(f'Working with dataset tag {dataset_tag}')

    # make an input directory for upsampling inside the dataset directory
    upsamp_input_dir = os.path.join(args.dataset, args.upsamp_dir, 'upsamp_input')
    if not args.upsampling_done:

        # if upsampling was partially done, start from a given viewpoint index
        if args.upsample_starting_traj > 0:
            upsamp_input_dir = os.path.join(args.dataset, args.upsamp_dir, 'upsamp_input_partial')
            print(f'Upsampling starting from trajectory {args.upsample_starting_traj}')

        # remove directory if it already exists
        if os.path.exists(upsamp_input_dir):
            shutil.rmtree(upsamp_input_dir)
        os.makedirs(upsamp_input_dir)

    # imfiles = sorted(glob.glob(os.path.join(args.dataset, "*rgb*.png")))
    # num_timesteps = len(imfiles) / args.num_views
    # if num_timesteps % 1 == 0:
    #     num_timesteps = int(num_timesteps)
    # else:
    #     print('num_timesteps=len(imfiles)/args.num_views is not an integer! Exiting.')
    #     exit()

    print(f'In upsampling prep: found {num_trajs} trajs.')

    if args.upsample_starting_traj == 0:
        # set an output directory for upsampling inside the dataset directory
        upsamp_output_dir = os.path.join(args.dataset, args.upsamp_dir, 'upsamp_output')
    else:
        upsamp_output_dir = os.path.join(args.dataset, args.upsamp_dir, 'upsamp_output_partial')

    if not args.upsampling_done:

        start_traj = args.upsample_starting_traj

        # set up each viewpoint's own seq___ directory of inputs
        for traj_i in range(start_traj, num_trajs):

            folder_name = f'seq{str(traj_i).zfill(5)}'

            # make seq{traj_i}/imgs directory for each view and fill it
            os.makedirs(os.path.join(upsamp_input_dir, f'{folder_name}/imgs'))
            # for f in imfiles[traj_i*num_timesteps:(traj_i+1)*num_timesteps]:
            #     shutil.copy(f, os.path.join(upsamp_input_dir, f'{folder_name}/imgs/.'))
            for i in range(train_trajlength[traj_i]):
                im = train_ims[i+np.cumsum(train_trajlength)[traj_i]-train_trajlength[traj_i]]
                cv2.imwrite(os.path.join(upsamp_input_dir, f'{folder_name}/imgs/{str(i).zfill(5)}.png'), im*255)

            # make fps.txt file
            with open(os.path.join(upsamp_input_dir, f'{folder_name}/fps.txt'), 'w') as f:
                fps = train_trajlength[traj_i] / (train_meta[np.cumsum(train_trajlength)[traj_i]-1, 1] - train_meta[np.cumsum(train_trajlength)[traj_i]-train_trajlength[traj_i], 1])
                f.write(f'{args.scale_up_fps * fps}') #NOTE not sure how this affects output (is the model trained to achieve a certain fps?)
        
        # run upsampling for all viewpoint seq directories at once
        print('Upsampling images. Running upsampler via subprocess.call().')
        subprocess.call(f"python /home/anish1/rpg_vid2e/upsampling/upsample.py --input_dir {upsamp_input_dir} --output_dir {upsamp_output_dir}", shell=True)

print(f'Found {num_trajs} trajectories with length mean {np.mean(train_trajlength):.2f} and std {np.std(train_trajlength):.2f}')

#######################
### GENERATE EVENTS ###
#######################

# generating events must be done per cohesive, single-view trajectory
# i need to run esim torch once per accumulated trajectory
# frames are organized as:
# times = 0.0, 0.0, 0.0, ..., x.x, x.x, x.x, ..., 1.0, 1.0, 1.0

# useful to have
train_trajstarts = np.cumsum(train_trajlength)-train_trajlength

# preserve log images for later difflog-based events approximation calculation
log_images_all = []

if args.path_evs_cont == '':

    ### run esim_torch ###
    if not args.cpu:
        import esim_torch
    else:
        import esim_py

    events = []

    for traj_idx in range(num_trajs):

        if not args.cpu:

            esim = esim_torch.ESIM(contrast_threshold_neg=neg_thresh,
                                contrast_threshold_pos=pos_thresh,
                                refractory_period_ns=.5e6) # note this is in ns

            if not args.upsample:

                traj_ids = np.arange(train_trajstarts[traj_idx], np.cumsum(train_trajlength)[traj_idx])
                images = train_ims[traj_ids]
                timestamps = train_meta[traj_ids, 1] - train_meta[traj_ids, 1][0]

            else:

                folder_name = f'seq{str(traj_idx).zfill(5)}'

                image_files = sorted(glob.glob(os.path.join(upsamp_output_dir, f'{folder_name}/imgs', "*.png")))
                images = np.stack([load_im_as_gray(f) for f in image_files])
                timestamps = np.genfromtxt(os.path.join(upsamp_output_dir, f'{folder_name}/timestamps.txt'), dtype="float64")
                print(f'In upsampled {folder_name}, found {len(timestamps)} timestamps and {len(image_files)} images.')

            timestamps_ns = (timestamps * 1e9).astype("int64")

            num_frames = len(timestamps_ns) - 1
            log_images = np.log(images.astype("float32") + 1e-10)
            log_images_all.append(log_images)

            if not args.only_difflog:

                st_genevents = time.time()

                # generate torch tensors
                print(f"Trajectory {traj_idx+1}/{num_trajs} loading data to GPU")
                device = torch.device("cuda")
                log_images = torch.from_numpy(log_images).to(device)
                timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

                # generate events with GPU support
                print("Generating events")
                # separate forward pass into batches to reduce GPU memory requirement
                # with torch.no_grad():
                #     events.append(esim.forward(log_images, timestamps_ns))
                traj_events = []
                batch_size = 100
                for i in range(0, num_frames, batch_size):
                    with torch.no_grad():
                        traj_events.append(esim.forward(log_images[i:i+batch_size], timestamps_ns[i:i+batch_size]))
                    for k in traj_events[-1].keys():
                        traj_events[-1][k] = traj_events[-1][k].cpu()


                # compile batches into a single dict
                full_batch = {'x': torch.Tensor([]), 'y': torch.Tensor([]), 't': torch.Tensor([]), 'p': torch.Tensor([])}
                for ev_batch in traj_events:
                    full_batch['x'] = torch.cat((full_batch['x'], ev_batch['x']))
                    full_batch['y'] = torch.cat((full_batch['y'], ev_batch['y']))
                    full_batch['t'] = torch.cat((full_batch['t'], ev_batch['t']))
                    full_batch['p'] = torch.cat((full_batch['p'], ev_batch['p']))
                
                events.append(full_batch)
                for k in events[-1].keys():
                    events[-1][k] = events[-1][k].cpu()

                print(f"Trajectory {traj_idx+1}/{num_trajs} done in {time.time()-st_genevents:.2f} s")

        else:


            """

            # esim_py (cpu) uses cpp code that needs a list of absolute image paths and a list of timestamps

            esim = esim_py.EventSimulator(pos_thresh, neg_thresh, 0.0, 1e-5, True)

            esim.setParameters(pos_thresh, neg_thresh, 0.0, 1e-5, True)

            if is_png:
                traj_image_files = sorted(glob.glob(opj(train_dirs[traj_idx], '*.png')))
                # # delete indices of traj_image_files according to train_unmatched_ids_ims
                # traj_image_files = [traj_image_files[i] for i in range(len(traj_image_files)) if i not in train_unmatched_ids_ims[traj_idx]]

            else:
                exit('Not is_png not implemented yet for cpu-based events generation.')

            timestamps = train_meta[train_trajstarts[traj_idx]:np.cumsum(train_trajlength)[traj_idx], 1] - train_meta[train_trajstarts[traj_idx]:np.cumsum(train_trajlength)[traj_idx], 1][0]
            timestamps_ns = (timestamps * 1e9).astype("int64")

            # returns N x 4 ndarray
            traj_events = esim.generateFromStampedImageSequence(traj_image_files, timestamps_ns)

            # convert into dict of {t, x, y, p} to be consistent in this script
            traj_events = {'t': torch.from_numpy(traj_events[:, 0]), 'x': torch.from_numpy(traj_events[:, 1]), 'y': torch.from_numpy(traj_events[:, 2]), 'p': torch.from_numpy(traj_events[:, 3])}

            # list of dicts
            events.append(traj_events)

            """

            print('esim_py not implemented yet for cpu-based events generation.')
            exit()

else:

    events_objs = np.load(args.path_evs_cont, allow_pickle=True).tolist()

    # Create an empty list to store the dictionary objects
    events = []
    # Iterate over each element in the NumPy array
    for event_obj in events_objs:
        # Extract the individual components from the NumPy array
        t, x, y, p = event_obj.T
        # Create a dictionary object using the extracted components
        event_dict = {'t': torch.from_numpy(t), 'x': torch.from_numpy(x), 'y': torch.from_numpy(y), 'p': torch.from_numpy(p)}
        # Append the dictionary object to the 'events' list
        events.append(event_dict)

##################################
### SAVE CONTINUOUS EVENT DATA ###
##################################

if args.acc_scheme == 'continuous':

    if not args.only_difflog:

        st_t = time.time()

        # reformat into a single numpy array
        # view_idx * [t, x, y, p]
        # note different sizes means the data array contains dtype objects
        events_objs = np.asarray( [ np.vstack((event['t'].cpu().numpy(), event['x'].cpu().numpy(), event['y'].cpu().numpy(), event['p'].cpu().numpy())).T for event in events ] , dtype=object )
        
        for traj_idx in range(events_objs.shape[0]):
            print(f'{events_objs[traj_idx].shape[0]} events in traj idx {traj_idx}')

        save_filename = os.path.join(args.output_path, f"evs{'_upsamp' if args.upsample else ''}{'_tf' if args.do_transform else ''}+f'{'_'+{args.output_suffix} if args.output_suffix!='' else ''}'+.pkl")

        # print(f'Size of events is {sys.getsizeof(events_objs)/1e9} GB')
        print(f'Saving events to {save_filename}')
        st_saving_t = time.time()

        import pickle
        with open(save_filename, 'wb') as f:
            pickle.dump(events_objs, f, protocol=4)
        
        print(f'Saving pkl done in {time.time()-st_saving_t:.2f} s')

        print(f'Continuous done in {time.time()-st_t:.2f} s')

        # print(f'Size of events is {sys.getsizeof(events_objs)/1e9} GB')
        # np.save(os.path.join(args.output_path, f'evs_{dataset_tag}.npy'), events_objs, allow_pickle=True)

# NOTE these are forming color frames, i.e. RGB images that represent events
# but not even correctly, since they do not sum the positive and negative events to get a net color; they will be biased towards more recent events in the time window
elif args.acc_scheme == 'N':

    acc_interval = 5000
    num_frames = events['x'].shape[0] // acc_interval
    print(f"Saving {num_frames} event psuedoframes")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    background_color = 128
    for i in range(num_frames):
        frame = background_color*np.ones((images[0].shape[0], images[0].shape[1], 4), dtype=np.uint8) # 4 channels for rgba
        acc_events = {k: v[i*acc_interval:(i+1)*acc_interval].cpu().numpy() for k,v in events.items()}
        frame[acc_events['y'], acc_events['x'], acc_events['p']+1] = 255
        frame[acc_events['y'], acc_events['x'], 3] = 255
        Image.fromarray(frame).convert("RGBA").save(args.output_path+f"ev_{str(i).zfill(4)}.png")

###################################
### SAVE FRAME-BASED EVENT DATA ###
###################################

# For time-based accumulation when generating from flightmare images,
# since upsampling might not be great, let's flatten into per-time window frames
# based on the timestamp of each sample.
# There will be a varying number of frames(=num_timestamps) per traj, so store
# as a ndarray object array.

if args.acc_scheme == 'time' or args.also_time:

    alltrajs_frames = []
    alltrajs_frames_difflog = []
    
    # frames = np.zeros((num_views, des_num_frames, images[0].shape[0], images[0].shape[1]))
    # rgbframes = np.zeros((num_views, num_frames, images[0].shape[0], images[0].shape[1]))

    st_t = time.time()

    for traj_idx in range(num_trajs):

        if not args.only_difflog:

            # slicing of continuous Vid2E-generated event stream

            ts = events[traj_idx]['t']

            print(f'{len(ts)} events in traj idx {traj_idx} for {train_trajlength[traj_idx]-1} frames')

            # one fewer frames than the number of samples
            frames = np.zeros((train_trajlength[traj_idx]-1, train_ims.shape[1], train_ims.shape[2]))

            # while not confident in upsampling and therefore continuous event stream,
            # set time windows based on sampling periods in metadata
            st_slicing_t = time.time()
            for i in range(frames.shape[0]):

                t_start = 1e9 * (train_meta[train_trajstarts[traj_idx]+i, 1] - train_meta[train_trajstarts[traj_idx], 1])
                t_end = 1e9 * (train_meta[train_trajstarts[traj_idx]+i+1, 1] - train_meta[train_trajstarts[traj_idx], 1])

                ev_idxs_pos = torch.bitwise_and(torch.bitwise_and((ts >= t_start), (ts < t_end)), (events[traj_idx]['p'] > 0)).cpu()
                ev_idxs_neg = torch.bitwise_and(torch.bitwise_and((ts >= t_start), (ts < t_end)), (events[traj_idx]['p'] < 0)).cpu()

                # accumulate all events by summation and scale by positive/negative thresholds
                frame = pos_thresh*np.histogram2d(events[traj_idx]['x'][ev_idxs_pos].cpu().numpy(), events[traj_idx]['y'][ev_idxs_pos].cpu().numpy(), bins=(frames.shape[2], frames.shape[1]), range=[[0, frames.shape[2]], [0, frames.shape[1]]])[0] - neg_thresh*np.histogram2d(events[traj_idx]['x'][ev_idxs_neg].cpu().numpy(), events[traj_idx]['y'][ev_idxs_neg].cpu().numpy(), bins=(frames.shape[2], frames.shape[1]), range=[[0, frames.shape[2]], [0, frames.shape[1]]])[0]

                frames[i] = frame.T
        
            alltrajs_frames.append(frames)

            print(f'Done slicing in {time.time()-st_slicing_t:.2f} s')

        # computing approximation of events via difflog

        if len(log_images_all) > 0:
            frames_difflog = []
            for im_i in range(1, log_images_all[traj_idx].shape[0]):

                # approximation of events calculation
                difflog = log_images_all[traj_idx][im_i] - log_images_all[traj_idx][im_i-1]

                # thresholding
                events_frame_difflog = np.zeros_like(difflog)

                if not np.abs(difflog).max() < max(pos_thresh, neg_thresh):

                    # quantize difflog by thresholds
                    pos_idxs = np.where(difflog > 0.0)
                    neg_idxs = np.where(difflog < 0.0)
                    events_frame_difflog[pos_idxs] = (difflog[pos_idxs] // pos_thresh) * pos_thresh
                    events_frame_difflog[neg_idxs] = (difflog[neg_idxs] // -neg_thresh) * -neg_thresh

                frames_difflog.append(events_frame_difflog)

            alltrajs_frames_difflog.append(np.array(frames_difflog))

    if not args.only_difflog:

        # save Vid2E frames

        print(f'Saving {len(alltrajs_frames)} trajectories of evframes to {os.path.join(args.output_path, f"evs_frames.npy")}')

        # save as a numpy object array
        if len(alltrajs_frames) == 1:
            # in this single-trajectory case, save as float
            np.save(os.path.join(args.output_path, f'evs_frames'+f"{'_tf' if args.do_transform else ''}"+f"{'_'+{args.output_suffix} if args.output_suffix!='' else ''}"+'.npy'), np.asarray(alltrajs_frames))
        else:
            # assign each element of an object array the data from each trajectory
            object_array = np.empty(len(alltrajs_frames), dtype=object)
            for i in range(len(alltrajs_frames)):
                object_array[i] = alltrajs_frames[i]
            np.save(os.path.join(args.output_path, f'evs_frames'+f"{'_tf' if args.do_transform else ''}"+f"{'_'+{args.output_suffix} if args.output_suffix!='' else ''}"+'.npy'), object_array)

    # save difflog frames
    if len(log_images_all) > 0:

        print(f'Saving {len(alltrajs_frames_difflog)} trajectories of difflog-computed evframes to {os.path.join(args.output_path, f"evs_frames_difflog.npy")}')

        # save as a numpy object array
        if len(alltrajs_frames_difflog) == 1:
            # in this single-trajectory case, save as float
            np.save(os.path.join(args.output_path, f'evs_frames_difflog'+f"{'_tf' if args.do_transform else ''}"+f"{'_'+{args.output_suffix} if args.output_suffix!='' else ''}"+'.npy'), np.asarray(alltrajs_frames_difflog))
        else:
            np.save(os.path.join(args.output_path, f'evs_frames_difflog'+f"{'_tf' if args.do_transform else ''}"+f"{'_'+{args.output_suffix} if args.output_suffix!='' else ''}"+'.npy'), np.asarray(alltrajs_frames_difflog, dtype=object))

    print(f'Frames done in {time.time()-st_t:.2f} s')
