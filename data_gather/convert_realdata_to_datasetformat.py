# description: processes data generated from the bash command:
# bash data_from_rosbags.sh /home/anish/evfly_ws/data/datasets/2-27_bags /home/anish/evfly_ws/data/datasets/2-27_align
# which reads from a slow-played rosbag and aligns depth and event frames.
# it compiles all event frames npy files into a single one for the dataset, renames directories to timestamps (.2f),
# and makes fake data.csv files and images.
# this is to be used with the learner_lstm directly or can be converted to an H5 daatset via the to_h5.py script.

import os, sys, glob, csv
import numpy as np
import cv2

def process_directory(directory_path, des_h, des_w):
    print(f"Processing directory: {directory_path}")

    # List all subdirectories in the given directory
    subdirectories = sorted([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])

    if not subdirectories:
        print("No subdirectories found.")

    all_evframes = []

    for subdir in subdirectories:

        print(f"Processing subdirectory: {subdir}")

        # add in _im.png files
        depth_filenames = sorted(glob.glob(os.path.join(directory_path, subdir, '*_depth.png')))
        timestamps = [f.split('/')[-1].split('_')[0] for f in depth_filenames]

        evframes = np.load(os.path.join(directory_path, subdir, 'evframes.npy'), allow_pickle=True)
        print(f'Found {len(depth_filenames)} depth files; {len(evframes)} evframes')

        # occasionally there might be a break in the recording from the rosbag, with mismatched depth ims and evframes. Delete respective mismatched ones, with some buffer
        if len(evframes) != len(depth_filenames):
            print(f'Number of evframes {len(evframes)} does not match number of depth files {len(depth_filenames)}')

            # choose how much data to keep
            N = len(depth_filenames) - 10
            while N > len(evframes):
                N -= 10
            
            print(f'Keeping {N} depth files and evframes')
            print(f'Removing depth files {[os.path.basename(fn) for fn in depth_filenames[N:]]}')

            # remove extra depth files
            for f in depth_filenames[N:]:
                os.remove(f)

            depth_filenames = depth_filenames[:N]
            timestamps = timestamps[:N]
            evframes = evframes[:N]
        
        # load, resize, and save depth files if needed
        resize_needed = False
        for i, depth_filename in enumerate(depth_filenames):
            depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
            if depth.shape[0] != des_h or depth.shape[1] != des_w:
                if not resize_needed:
                    print(f'Resizing depth files from {depth.shape[:2]} to {(des_h, des_w)}')
                    resize_needed = True
                depth = cv2.resize(depth, (des_w, des_h))
                cv2.imwrite(depth_filename, depth)

        # spoofed image filenames
        im_filenames = [f.replace('_depth.png', '_im.png') for f in depth_filenames]
        # make files of blank images, each a PNG of size 260x346
        for im_filename in im_filenames:
            # os.system(f'convert -size 260x346 xc:white {im_filename}')
            cv2.imwrite(im_filename, np.ones((des_h, des_w), dtype=np.uint8)*255)

        # find first timestamp to rename the folder with, but with .2f
        first_timestamp = depth_filenames[0].split('/')[-1].split('_')[0][:-1]
        # eliminate . in the first_timestamp
        first_timestamp = first_timestamp.replace('.', '')
        print(f'First timestamp is {first_timestamp}')

        # spoof a data.csv of numrows len(depth_filenames) and numcolumns 21
        meta = np.zeros((len(depth_filenames), 21))
        for i, timestamp in enumerate(timestamps):
            # # make header line
            # if i == 0:
            #     meta[i, 0] = 'Fake-Header'
            meta[i, 1] = float(timestamp)
            meta[i, 2] = 4.0 # made-up desVel
            meta[:, 13] = 4.0 # made-up velocity command of forward
        # save as data.csv
        np.savetxt(os.path.join(directory_path, subdir, 'data.csv'), meta, delimiter=',')

        # clumsily add a header row of text
        with open(os.path.join(directory_path, subdir, 'data.csv'), 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        new_row = ['Fake-Header']*21
        rows.insert(0, new_row)
        with open(os.path.join(directory_path, subdir, 'data.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

        # save evframes into larger array and remove file
        traj_evframes = evframes[1:]
        # resize if needed
        if traj_evframes.shape[1] != des_h or traj_evframes.shape[2] != des_w:
            print(f'Resizing evframes from {traj_evframes.shape[1:]} to {(des_h, des_w)}')
            traj_evframes = np.array([cv2.resize(im, (des_w, des_h)) for im in traj_evframes])
        all_evframes.append(traj_evframes) # skip the first evframe to keep with standard of simulated data
        os.remove(os.path.join(directory_path, subdir, 'evframes.npy'))

        # rename the folder to the first timestamp
        os.rename(os.path.join(directory_path, subdir), os.path.join(directory_path, first_timestamp))

        print()

    # save all evframes into one file at higher level dir
    print(f'Saving all evframes into one file; {len(all_evframes)} trajectories')
    all_evframes = np.array(all_evframes, dtype=object)
    np.save(os.path.join(directory_path, 'evs_frames.npy'), all_evframes)
    print('Done')

if __name__ == "__main__":
    # Check if the command-line argument is provided

    if len(sys.argv) < 4:
        print("Usage: python script.py <directory_path> <desired_height> <desired_width>")
        sys.exit(1)

    if len(sys.argv) == 4:
        des_h = int(sys.argv[2])
        des_w = int(sys.argv[3])

    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <directory_path>")
    #     sys.exit(1)

    # Get the directory path from command-line arguments
    input_directory = sys.argv[1]

    # Check if the given path is a valid directory
    if not os.path.isdir(input_directory):
        print(f"Error: {input_directory} is not a valid directory.")
        sys.exit(1)

    # Call the function to process the directory
    process_directory(input_directory, des_h, des_w)

