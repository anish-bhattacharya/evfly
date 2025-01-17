# convert traditional dataset structure to h5 dataset file

import glob, os, sys, time
from os.path import join as opj
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import cv2

import getpass
uname = getpass.getuser()
sys.path.append(f'/home/{uname}/evfly_ws/src/evfly/learner')
from learner import Learner

def convert_to_h5(learner, h5_path):
    print(f'Converting dataset to h5 file {h5_path}')
    # if exists, delete it
    if os.path.exists(h5_path):
        print(f'File {h5_path} already exists, deleting it')
        os.remove(h5_path)
    dataset = h5py.File(h5_path, 'w')

    train_traj_starts = np.cumsum(learner.train_trajlength) - learner.train_trajlength

    for traj_i, traj_name in enumerate(learner.train_dirs):

        # if traj_name ends with a / then remove it
        if traj_name[-1] == '/':
            traj_name = traj_name[:-1]

        # make group with last part of trajectory name
        traj_group = dataset.create_group(traj_name.split('/')[-1])

        traj_group.create_dataset('data', data=learner.train_meta[train_traj_starts[traj_i]:train_traj_starts[traj_i]+learner.train_trajlength[traj_i]])
        traj_group.create_dataset('ims', data=learner.train_ims[train_traj_starts[traj_i]:train_traj_starts[traj_i]+learner.train_trajlength[traj_i]])
        traj_group.create_dataset('depths', data=learner.train_depths[train_traj_starts[traj_i]:train_traj_starts[traj_i]+learner.train_trajlength[traj_i]])
        traj_group.create_dataset('trajlength', data=learner.train_trajlength[traj_i])
        traj_group.create_dataset('desvel', data=learner.train_desvel[train_traj_starts[traj_i]:train_traj_starts[traj_i]+learner.train_trajlength[traj_i]])
        if learner.train_evs is not None:
            traj_group.create_dataset('evs', data=learner.train_evs[traj_i])
        traj_group.create_dataset('dirs', data=learner.train_dirs[traj_i])
        traj_group.create_dataset('dirs_ids', data=learner.train_dirs_ids[traj_i])

    dataset.close()

    print(f'Saved dataset to {h5_path}')

def list_hdf5_groups(hdf5_filename):
    group_list = []
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        hdf5_file.visititems(lambda name, obj: group_list.append(name) if isinstance(obj, h5py.Group) else None)
    return group_list

def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        for key in obj.keys():
            print(f"{sep},'-',{key},':',{obj[key]}")
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj)==h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print(f"{sep}+'\t','-',{key},':',{obj.attrs[key]}")

def h5dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
         descend_obj(f[group])

# main function
if __name__ == '__main__':

    if len(sys.argv) < 4:
        print(f'Usage: python to_h5.py <dataset> <task:convert/view> <do_transform:True/False>')
        sys.exit(1)
    dataset = sys.argv[1] # absolute path
    task = sys.argv[2]
    do_transform = sys.argv[3] == 'True'
    if len(sys.argv) > 4:
        if sys.argv[4] == 'difflog':
            events = 'evs_frames_difflog'
        else:
            events = 'evs_frames'
    else:
        events = 'evs_frames'
    print(f'Using events {events} (difflog 4th argument triggers evs_frames_difflog, else uses evs_frames)')

    h5_path = dataset+'.h5'

    if task == 'convert':
        ### convert a dataset
        learner = Learner(dataset_name=dataset, short=0, no_model=True, val_split=0.0, events=events, do_transform=do_transform, use_h5=False)
        print(f'Loaded dataset {dataset}')
        convert_to_h5(learner, h5_path)

    elif task == 'view':
        ### view a dataset
        # groups = list_hdf5_groups(h5_path)
        h5dump(h5_path)
        # print("List of groups:")
        # for group in groups:
        #     print(group)

    else:
        print(f'Unknown task {task}')
        sys.exit(1)



