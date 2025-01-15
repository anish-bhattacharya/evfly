# a learner that learns a perception -> action mapping
# Anish Bhattacharya, 2024

# NOTE you might need to unset PYTHONPATH

DEPLOYMENT = False

import glob, os, sys
from os.path import join as opj
import numpy as np
import torch
import torchvision.transforms.functional as TF
from datetime import datetime
import time
if not DEPLOYMENT:
    from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from math import radians, sin, cos

from dataloading import *
from learner_models import *
if not DEPLOYMENT:
    from evaluation_tools import eval_plotter

# NOTE this suppresses tensorflow warnings and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import getpass
uname = getpass.getuser()

# a class that trains an LSTM to predict actions from depth images
# Learner can be loaded in two ways:
# 1. just for dataloading, in which case dataset_name is provided and usually no_model=True, or
# 2. for model training, in which case just args is provided
class Learner:
    def __init__(self, args=None, dataset_name=None, short=0, no_model=False, val_split=0.2, events='', do_transform=False, use_h5=True):
        
        ################
        ## Parameters ##
        ################

        self.args = args

        if self.args is not None:

            self.device = args.device
            self.basedir = args.basedir
            self.logdir = args.logdir
            self.datadir = args.datadir
            self.ws_suffix = args.ws_suffix
            self.dataset_name = args.dataset
            self.data_augmentation = args.data_augmentation
            self.evs_min_cutoff = args.evs_min_cutoff
            self.rescale_depth = args.rescale_depth
            self.rescale_evs = args.rescale_evs
            self.domain_randomization = args.domain_randomization
            self.bev = args.bev
            self.short = args.short
            self.use_h5 = args.use_h5

            self.model_type = args.model_type
            self.skip_type = args.skip_type
            self.velpred = args.velpred
            self.num_recurrent = args.num_recurrent
            self.num_in_channels = args.num_in_channels
            self.num_out_channels = args.num_out_channels
            self.val_split = args.val_split
            self.seed = args.seed
            self.batch_size = args.batch_size
            self.load_trainval = args.load_trainval
            # self.load_checkpoint = args.load_checkpoint
            self.checkpoint_path = args.checkpoint_path
            self.combine_checkpoints = args.combine_checkpoints
            self.lr = args.lr
            self.N_eps = args.N_eps
            self.lr_warmup_epochs = args.lr_warmup_epochs
            self.lr_decay = args.lr_decay
            self.save_model_freq = args.save_model_freq
            self.val_freq = args.val_freq
            self.optional_loss_param = args.optional_loss_param
            self.events = args.events
            self.keep_collisions = args.keep_collisions
            self.do_transform = args.do_transform
            self.resize_input = args.resize_input
            self.eval_tools_freq = args.eval_tools_freq
            self.eval_tools_on_best = args.eval_tools_on_best
            self.print_trainprogress_freq = args.print_trainprogress_freq
            self.loss_weights = args.loss_weights
            self.split_method = args.split_method
            self.num_outputs = args.num_outputs

            # model-specific args

            # encoder
            self.enc_num_layers = args.enc_num_layers
            self.enc_kernel_sizes = args.enc_kernel_sizes
            self.enc_kernel_strides = args.enc_kernel_strides
            self.enc_out_channels = args.enc_out_channels
            self.enc_activations = args.enc_activations
            self.enc_pool_type = args.enc_pool_type
            self.enc_invert_pool_inputs = args.enc_invert_pool_inputs
            self.enc_pool_kernels = args.enc_pool_kernels
            self.enc_pool_strides = args.enc_pool_strides
            self.enc_conv_function = args.enc_conv_function

            # decoder
            self.dec_num_layers = args.dec_num_layers
            self.dec_kernel_sizes = args.dec_kernel_sizes
            self.dec_kernel_strides = args.dec_kernel_strides
            self.dec_out_channels = args.dec_out_channels
            self.dec_activations = args.dec_activations
            self.dec_pool_type = args.dec_pool_type
            self.dec_pool_kernels = args.dec_pool_kernels
            self.dec_pool_strides = args.dec_pool_strides
            self.dec_conv_function = args.dec_conv_function

            # fc
            self.fc_num_layers = args.fc_num_layers
            self.fc_layer_sizes = args.fc_layer_sizes
            self.fc_activations = args.fc_activations
            self.fc_dropout_p = args.fc_dropout_p

        else:

            self.device = 'cuda' if not no_model else 'cpu'
            self.basedir = f'/home/{uname}/evfly_ws/src/evfly'
            self.logdir = 'learner/logs'
            self.datadir = '../../data/datasets'
            self.ws_suffix = ''
            self.dataset_name = dataset_name
            self.data_augmentation = 0.0
            self.evs_min_cutoff = 0.0
            self.rescale_depth = 0.0
            self.rescale_evs = 0.0
            self.domain_randomization = 0.0
            self.bev = 0
            self.short = short
            self.use_h5 = use_h5

            self.model_type = 'LSTMNet'
            self.skip_type = 'crop'
            self.velpred = 0
            self.num_recurrent = [0]
            self.num_in_channels = 2
            self.num_out_channels = 1
            self.val_split = val_split
            self.seed = -2 # no randomization
            self.batch_size = 0
            self.load_trainval = True
            # self.load_checkpoint = False
            self.checkpoint_path = None
            self.combine_checkpoints = False
            self.lr = 1e-5
            self.N_eps = 500
            self.lr_warmup_epochs = 5
            self.lr_decay = False
            self.save_model_freq = 25
            self.val_freq = 10
            self.optional_loss_param = [0.0, 0.0]
            self.events = events
            self.keep_collisions = True
            self.do_transform = do_transform
            self.resize_input = None
            self.eval_tools_freq = 0
            self.eval_tools_on_best = False
            self.print_trainprogress_freq = 1
            self.loss_weights = None
            self.split_method = 'train-val'
            self.num_outputs = 2

            self.enc_num_layers = 2
            self.enc_kernel_sizes = [5, 5]
            self.enc_kernel_strides = [2, 2]
            self.enc_out_channels = [16, 64]
            self.enc_activations = ['relu', 'relu']
            self.enc_pool_type = 'max'
            self.enc_invert_pool_inputs = False
            self.enc_pool_kernels = [2, 2]
            self.enc_pool_strides = [2, 2]
            self.enc_conv_function = 'conv2d'
            self.dec_num_layers = 2
            self.dec_kernel_sizes = [5, 5]
            self.dec_kernel_strides = [2, 2]
            self.dec_out_channels = [64, 16]
            self.dec_activations = ['relu', 'relu']
            self.dec_pool_type = 'none'
            self.dec_pool_kernels = [2, 2]
            self.dec_pool_strides = [2, 2]
            self.dec_conv_function = 'upconv2d'
            self.fc_num_layers = 2
            self.fc_layer_sizes = [128, 64]
            self.fc_activations = ['relu', 'relu']
            self.fc_dropout_p = 0.5

        # if dataset_name is not a list, then it is a single name so make it a length-1 list
        if type(self.dataset_name) != list:
            self.dataset_name = [self.dataset_name]

        # make dictionaries enc_params and dec_params with the above args
        self.enc_params = {
            'num_layers': self.enc_num_layers,
            'kernel_sizes': self.enc_kernel_sizes,
            'kernel_strides': self.enc_kernel_strides,
            'out_channels': self.enc_out_channels,
            'activations': self.enc_activations,
            'pool_type': self.enc_pool_type,
            'invert_pool_inputs': self.enc_invert_pool_inputs,
            'pool_kernels': self.enc_pool_kernels,
            'pool_strides': self.enc_pool_strides,
            'conv_function': self.enc_conv_function,
        }
        self.dec_params = {
            'num_layers': self.dec_num_layers,
            'kernel_sizes': self.dec_kernel_sizes,
            'kernel_strides': self.dec_kernel_strides,
            'out_channels': self.dec_out_channels,
            'activations': self.dec_activations,
            'pool_type': self.dec_pool_type,
            'pool_kernels': self.dec_pool_kernels,
            'pool_strides': self.dec_pool_strides,
            'conv_function': self.dec_conv_function,
        }
        self.fc_params = {
            'num_layers': self.fc_num_layers,
            'layer_sizes': self.fc_layer_sizes,
            'activations': self.fc_activations,
            'dropout_p': self.fc_dropout_p,
        }

        # if self.checkpoint_path is a list of length 1, then just take the first element
        if type(self.checkpoint_path) == list and len(self.checkpoint_path) == 1:
            self.checkpoint_path = self.checkpoint_path[0]

        if self.events != '':
            if self.do_transform: # legacy
                self.events += '_tf.npy'
            else:
                self.events += '.npy'

        if self.device != 'cpu':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.last_eval_plot_ep = 0
        self.previous_tag = None

        # set seed for numpy and torch if nonnegative
        if self.seed >= 0:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        ###############
        ## Workspace ##
        ###############

        expname = datetime.now().strftime('d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname) + self.ws_suffix
        temp_workspace = self.workspace
        wkspc_ctr = 2
        while os.path.exists(temp_workspace):
            temp_workspace = self.workspace + f'_{wkspc_ctr}'
            wkspc_ctr += 1
        self.workspace = temp_workspace
        os.makedirs(self.workspace)
        self.writer = SummaryWriter(self.workspace)

        # save ordered args, config, and a logfile to write stdout to
        if self.args is not None:
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            f = opj(self.workspace, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')

        # also save learner_lstm.py and learner_models.py
        # by copying them from the evfly/learner directory
        f = opj(self.workspace, 'learner.py')
        os.system(f'cp /home/{uname}/evfly_ws/src/evfly/learner/learner.py {f}')
        f = opj(self.workspace, 'learner_models.py')
        os.system(f'cp /home/{uname}/evfly_ws/src/evfly/learner/learner_models.py {f}')
        f = opj(self.workspace, 'dataloading.py')
        os.system(f'cp /home/{uname}/evfly_ws/src/evfly/learner/dataloading.py {f}')

        self.mylogger(f'[Learner init] Making workspace {self.workspace}')

        # handle combine_checkpoints flag that is often set to True by mistake
        if self.combine_checkpoints and type(self.checkpoint_path) != list:
            self.combine_checkpoints = False
            self.mylogger(f'[Learner init] combine_checkpoints is True, but checkpoint_path is not a list of checkpoints (or len=1), so setting it to False')

        # assert self.dataset_name is not None, 'Dataset name not provided, neither through args nor through dataset_name kwarg'
        if self.dataset_name == None or self.dataset_name[0] is None or self.dataset_name[0] == '' or self.dataset_name[0] == 'None':
            self.dataset_name = [None]
            self.mylogger(f'[Learner init] No dataset name provided, not loading a dataset!')

        self.dataset_dir = []
        for dn in self.dataset_name:
            self.dataset_dir.append(opj(self.datadir, dn))

        #################
        ## Dataloading ##
        #################

        if not (self.checkpoint_path == '' or self.checkpoint_path == ['']) and self.load_trainval:
            self.mylogger('[Learner init] Trying to load train_val_dirs from checkpoint...')
            try:
                train_val_dirs = tuple(np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True))
                self.mylogger('[Learner init] Loaded train_val_dirs from checkpoint')
            except:
                self.mylogger('[Learner init] Could not load train_val_dirs from checkpoint, dataloading from scratch')
                train_val_dirs = None
        else:
            train_val_dirs = None

        self.learner_dataloading(val_split=self.val_split, short=self.short, seed=self.seed, train_val_dirs=train_val_dirs, events=self.events, keep_collisions=self.keep_collisions)

        self.num_training_steps = self.train_trajlength.shape[0]
        self.num_val_steps = self.val_trajlength.shape[0]
        self.lowest_val_loss = torch.inf
        self.lr_warmup_iters = self.lr_warmup_epochs * self.num_training_steps

        ##################################
        ## Define network and optimizer ##
        ##################################

        if not no_model:

            # if model_type is a length one list, make it a string
            if isinstance(self.model_type, list) and len(self.model_type) == 1:
                self.model_type = self.model_type[0]
            
            # if model_type is not a list (not multiple models) then progress with single model initialization
            if not isinstance(self.model_type, list):

                self.mylogger('[SETUP] Establishing model and optimizer.')

                if self.model_type == 'ConvNet_w_VelPred':

                    self.model = ConvNet_w_VelPred(num_in_channels=self.num_in_channels, num_recurrent=self.num_recurrent[1], num_outputs=self.num_outputs, enc_params=self.enc_params, fc_params=self.fc_params, input_shape=[1, 1, self.resize_input[0], self.resize_input[1]], logger=self.mylogger).to(self.device).float()

                elif self.model_type == 'OrigUNet':

                    self.mylogger(f'[SETUP] Setting evs_min_cutoff to {self.evs_min_cutoff}')

                    self.model = OrigUNet(num_in_channels=self.num_in_channels, num_out_channels=self.num_out_channels, num_recurrent=self.num_recurrent, input_shape=self.train_ims.shape, logger=self.mylogger, velpred=self.velpred, enc_params=self.enc_params, fc_params=self.fc_params, form_BEV=self.bev, evs_min_cutoff=self.evs_min_cutoff, skip_type=self.skip_type).to(self.device).float()

                else:

                    self.mylogger(f'[SETUP] Invalid model_type {self.model_type}. Exiting.')
                    exit()

            # if multiple model types need initialization, then write custom code snippets here to do so
            else:

                self.mylogger(f'[SETUP] Multiple model types {self.model_type} provided.')
                
                if self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'VITFLY_ViTLSTM':
                    self.mylogger(f'[SETUP] Creating model of type {self.model_type[0]} and {self.model_type[1]}')
                    self.model = OrigUNet_w_VITFLY_ViTLSTM(
                        num_in_channels=self.num_in_channels, 
                        num_out_channels=self.num_out_channels,
                        num_recurrent=self.num_recurrent, 
                        input_shape=[1, 1, self.resize_input[0], self.resize_input[1]], 
                        logger=self.mylogger,
                        velpred=self.velpred, 
                        enc_params=self.enc_params, 
                        dec_params=self.dec_params, 
                        fc_params=self.fc_params, 
                        form_BEV=self.bev,
                        evs_min_cutoff=self.evs_min_cutoff,
                        skip_type=self.skip_type, 
                        is_deployment=False,
                    ).to(self.device).float()

                elif self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'ConvNet_w_VelPred':
                    self.mylogger(f'[SETUP] Creating model of type {self.model_type[0]} and {self.model_type[1]}')
                    self.model = OrigUNet_w_ConvNet_w_VelPred(
                        num_in_channels=self.num_in_channels, 
                        num_out_channels=self.num_out_channels,
                        num_recurrent=self.num_recurrent, 
                        input_shape=[1, 1, self.resize_input[0], self.resize_input[1]], 
                        logger=self.mylogger,
                        velpred=self.velpred, 
                        enc_params=self.enc_params, 
                        dec_params=self.dec_params, 
                        fc_params=self.fc_params, 
                        form_BEV=self.bev,
                        evs_min_cutoff=self.evs_min_cutoff,
                        skip_type=self.skip_type, 
                        is_deployment=False,
                    ).to(self.device).float()

                else:
                    self.mylogger(f'[SETUP] Multi-model_type {self.model_type} not implemented yet. Exiting.')
                    exit()

            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            self.num_eps_trained = 0
            self.load_from_checkpoint(self.checkpoint_path)

            # print number of params and trainable params
            self.mylogger(f'[SETUP] Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}')
            self.mylogger(f'[SETUP] Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

            self.total_its = self.num_eps_trained * self.num_training_steps

    # a useful logger that prints messages to stdout and writes them to a logfile
    # it chunks messages based on their tags at the beginning of each message
    def mylogger(self, msg):
        # Extract the tag from the message using square brackets
        tag = msg.split('[')[1].split(']')[0] if '[' in msg and ']' in msg else None

        # Check if there's a tag and if it's different from the previous one print a newline
        if tag is not None and tag != self.previous_tag:
            print('\n', end='') # Print a newline
            self.logfile.write('\n')

        print(msg)
        self.logfile.write(msg+'\n')

        self.previous_tag = tag

    def combine_state_dicts(self, state_dicts, model_names=None):
        """Combines multiple state dicts into a single state dict.
        Args:
            state_dicts: A list of state dicts.
        Returns:
            A single state dict that combines all of the state dicts in the input list.
        """
        combined_state_dict = {}
        for sd_i, state_dict in enumerate(state_dicts):
            for key, value in state_dict.items():

                if model_names is not None:
                    prefix = model_names[sd_i]
                    key = f"{prefix}.{key}"

                if key in combined_state_dict:
                    pass # do not overwrite it; first state_dict takes precedence
                else:
                    combined_state_dict[key] = value
        return combined_state_dict

    def load_from_checkpoint(self, checkpoint_path):

        # if checkpoint_path is empty then just return
        # this happens when we evaluate at the beginning of training
        if checkpoint_path == '' or checkpoint_path == [''] or checkpoint_path == None or checkpoint_path == [None] or checkpoint_path == [] or checkpoint_path == ['None']:
            print(f'[SETUP] In load_from_checkpoint, but checkpoint_path is empty, so not loading from checkpoint')
            return

        try:
            self.num_eps_trained = int(checkpoint_path[-10:-4])
        except:
            self.num_eps_trained = 0
            self.mylogger(f'[SETUP] Could not parse number of epochs trained from checkpoint path {checkpoint_path}, using 0')
        self.mylogger(f'[SETUP] Loading checkpoint from {checkpoint_path}, already trained for {self.num_eps_trained} epochs')
        
        # if a model is trained partially via multiple training runs / checkpoints, call this to combine them
        if self.combine_checkpoints:
            self.mylogger(f'[SETUP] Combining checkpoints {checkpoint_path} into a single state dict')
            self.model.load_state_dict(self.combine_state_dicts([torch.load(cp, map_location=self.device) for cp in checkpoint_path], model_names=[self.model_type[0].lower(), self.model_type[1].lower()]))

        else:

            # if model contains multiple model components, load each individually
            if not isinstance(self.model_type, list) or (isinstance(self.model_type, list) and len(self.model_type) == 1):

                if self.model_type == 'OrigUNet_w_ConvUNet_w_VelPred':
                    self.model.origunet.load_state_dict(torch.load(checkpoint_path[0], map_location=self.device))
                    self.model.convunet_w_velpred.load_state_dict(torch.load(checkpoint_path[1], map_location=self.device))
                else:
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
                    
            elif isinstance(self.model_type, list) and self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'VITFLY_ViTLSTM':

                self.model.origunet.load_state_dict(torch.load(checkpoint_path[0], map_location=self.device))
                self.model.vitfly_vitlstm.load_state_dict(torch.load(checkpoint_path[1], map_location=self.device))

            elif self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'ConvNet_w_VelPred':
                self.model.origunet.load_state_dict(torch.load(checkpoint_path[0], map_location=self.device))
                self.model.convnet_w_velpred.load_state_dict(torch.load(checkpoint_path[1], map_location=self.device))


    def learner_dataloading(self, val_split, short=0, seed=None, train_val_dirs=None, events='', keep_collisions=False):

        # initialize lists storing all data
        self.train_meta = []
        self.train_velcmd = []
        self.train_ims = []
        self.train_depths = []
        self.train_trajlength = []
        self.train_desvel = []
        self.train_evs = []
        self.train_dirs = []
        self.train_dirs_ids = []
        self.val_meta = []
        self.val_velcmd = []
        self.val_ims = []
        self.val_depths = []
        self.val_trajlength = []
        self.val_desvel = []
        self.val_evs = []
        self.val_dirs = []
        self.val_dirs_ids = []

        self.dataset_numtrajs = []

        for data_dir in self.dataset_dir:

            data_dir_fullpath = opj(self.basedir, data_dir)

            ## Dataloading

            self.mylogger(f'[DATALOADER] Loading from {data_dir} from set {self.dataset_dir}')
            
            train_data, val_data, is_png = dataloader(data_dir_fullpath, val_split=val_split, short=short, seed=seed, train_val_dirs=train_val_dirs, events=events, keep_collisions=keep_collisions, logger=self.mylogger, do_clean_dataset=False, do_transform=self.do_transform, use_h5=self.use_h5, resize_input=self.resize_input, split_method=self.split_method, rescale_depth=self.rescale_depth, rescale_evs=self.rescale_evs, evs_min_cutoff=self.evs_min_cutoff)
            
            # temporarily spoof train data as val data
            # val_data = train_data

            train_meta, (train_ims, train_depths), train_trajlength, train_desvel, train_evs, train_dirs, train_dirs_ids = train_data
            
            val_meta, (val_ims, val_depths), val_trajlength, val_desvel, val_evs, val_dirs, val_dirs_ids = val_data
        
            self.mylogger(f'[DATALOADER] Dataloading done | train images {train_ims.shape}, val images {val_ims.shape}')

            ## Preloading

            # convert data to torch tensors but keep on cpu for later gpu-loading via a custom dataloader
            train_meta, train_ims, train_depths, train_desvel, train_evs = preload((train_meta, train_ims, train_depths, train_desvel, train_evs), 'cpu')

            val_meta, val_ims, val_depths, val_desvel, val_evs = preload((val_meta, val_ims, val_depths, val_desvel, val_evs), 'cpu')

            self.mylogger(f'[DATALOADER] Loaded data onto cpu memory as torch tensors')

            ## Checks

            if train_meta.shape[0] > 0:
                assert train_ims.max() <= 1.0 and train_ims.min() >= 0.0, 'Images not normalized (values outside [0.0, 1.0])'
                assert train_ims.max() > 0.50, "Images not normalized (values only below 0.10, possibly due to not normalizing images from 'old' dataset)"
            else:
                self.mylogger('[DATALOADER] NOTE NO TRAINING DATA SELECTED!')

            ## Miscellaneous

            # extract velocity commands from metadata
            train_velcmd = train_meta[:, range(13, 16) if is_png else range(12, 15)]
            val_velcmd = val_meta[:, range(13, 16) if is_png else range(12, 15)]

            ## Save data per dataset

            # note that while train_ims is a N_alltrajs x H x W tensor,
            # train_evs is a list of len num_trajs, where each element is N_traj_i-1 x H x W

            self.train_meta.append(train_meta)
            self.train_velcmd.append(train_velcmd)
            self.train_ims.append(train_ims)
            self.train_depths.append(train_depths)
            self.train_trajlength.append(train_trajlength)
            self.train_desvel.append(train_desvel)
            self.train_evs.append(train_evs)
            self.train_dirs.append(train_dirs)
            self.train_dirs_ids.append(train_dirs_ids)

            self.val_meta.append(val_meta)
            self.val_velcmd.append(val_velcmd)
            self.val_ims.append(val_ims)
            self.val_depths.append(val_depths)
            self.val_trajlength.append(val_trajlength)
            self.val_desvel.append(val_desvel)
            self.val_evs.append(val_evs)
            self.val_dirs.append(val_dirs)
            self.val_dirs_ids.append(val_dirs_ids)

            self.dataset_numtrajs.append((len(train_trajlength), len(val_trajlength)))

        # concatenate all data
        # trajstarts, not computed here, must now add up each previous dataset's trajlengths
        # dirs and dirs_ids must be specially handled
        self.train_meta = torch.cat(self.train_meta, axis=0)
        self.train_velcmd = torch.cat(self.train_velcmd, axis=0)
        self.train_ims = torch.cat(self.train_ims, axis=0)
        self.train_depths = torch.cat(self.train_depths, axis=0)
        self.train_desvel = torch.cat(self.train_desvel, axis=0)
        self.train_evs = [traj_evs for dataset_evs in self.train_evs for traj_evs in dataset_evs]
        self.val_meta = torch.cat(self.val_meta, axis=0)
        self.val_velcmd = torch.cat(self.val_velcmd, axis=0)
        self.val_ims = torch.cat(self.val_ims, axis=0)
        self.val_depths = torch.cat(self.val_depths, axis=0)
        self.val_desvel = torch.cat(self.val_desvel, axis=0)
        self.val_evs = [traj_evs for dataset_evs in self.val_evs for traj_evs in dataset_evs]

        self.train_trajlength = np.concatenate(self.train_trajlength, axis=0)
        self.val_trajlength = np.concatenate(self.val_trajlength, axis=0)

        self.train_dirs = np.concatenate(self.train_dirs, axis=0).tolist()
        self.val_dirs = np.concatenate(self.val_dirs, axis=0).tolist()

        self.train_dirs_ids = np.concatenate(self.train_dirs_ids, axis=0).tolist()
        self.val_dirs_ids = np.concatenate(self.val_dirs_ids, axis=0).tolist()

        # save train and val dirs in workspace for later use
        np.save(opj(self.workspace, 'train_val_dirs.npy'), np.array((self.train_dirs, self.val_dirs, self.train_dirs_ids, self.val_dirs_ids), dtype=object))

    def lr_scheduler(self, it):
        if it < self.lr_warmup_iters:
            lr = (0.9*self.lr)/self.lr_warmup_iters * it + 0.1*self.lr
        else:
            if self.lr_decay:
                lr = self.lr * (0.1 ** ((it-self.lr_warmup_iters) / (self.N_eps*self.num_training_steps)))
            else:
                lr = self.lr
        return lr

    def save_model(self, ep, best=-2):
        ep_str = str(ep).zfill(6)
        if best == -2: # not a best model
            self.mylogger(f'[SAVE] Saving model at epoch {ep}')
            path = self.workspace
            model_path = opj(path, f'model_ep{ep_str}.pth')
            torch.save(self.model.state_dict(), model_path)
            self.mylogger(f'[SAVE] Model saved at {path}')
        else: # -1 indicates best overall loss, 0, 1, etc indicate best loss term 0, 1, etc
            suffix = f'_best_' if best < 0 else f'_best{best}_'
            ep_suffix = f'ep{ep_str}'
            self.mylogger(f'[SAVE] Saving best (type {best}) model at epoch {ep}')
            model_path = opj(self.workspace, f'model{suffix}{ep_suffix}.pth')
            # if previous best model file of this type exists, delete it
            files_to_remove = glob.glob(opj(self.workspace, f'model{suffix}*'))
            for f in files_to_remove:
                os.remove(f)
            torch.save(self.model.state_dict(), model_path)
            self.mylogger(f'[SAVE] Best model saved at {model_path}')

    def eval_tools(self, ep, load_ckpt=False):
        self.last_eval_plot_ep = ep
        st_evalplot = time.time()
        self.mylogger(f'[SAVE] Generating evaluation plot at ep {ep}')
        model_path = opj(self.workspace, f'model_{str(ep).zfill(6)}.pth')
        if not os.path.exists(model_path):
            # find any model path with any string between 'model' and ep{}.pth
            model_path = glob.glob(opj(self.workspace, f'model*{str(ep).zfill(6)}.pth'))[0]
        if not os.path.exists(model_path):
            self.mylogger(f'[SAVE] Model checkpoint not found at {model_path}, skipping eval plot generation')
            return
        self.evalPlot_fig, evalPlot_title = eval_plotter(self, model_path, load_ckpt=load_ckpt)
        self.mylogger(f'[SAVE] Evaluation plot generated at ep {ep} in {time.time() - st_evalplot:.2f}s')
        self.writer.add_figure('val/plot', self.evalPlot_fig, global_step=ep)
        self.writer.flush()
        # self.evalPlot_fig.clf()
        plt.close(self.evalPlot_fig)

    def train(self):

        self.mylogger(f'[TRAIN] Training for {self.N_eps} epochs')
        if self.events != '':
            self.mylogger(f'[TRAIN] Events path is specified as path={self.events}')
        train_start = time.time()

        # starting indices of trajectories in dataset
        self.train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength

        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):

            if self.eval_tools_freq > 0:
                # periodically generate eval plots
                if (ep - self.num_eps_trained) % self.eval_tools_freq == 0:
                    self.eval_tools(ep)

            # periodically save model checkpoint (incl. at start of training)
            if (ep - self.num_eps_trained) % self.save_model_freq == 0:
                self.save_model(ep, best=-2)

            # periodically evaluate on validation set
            if (ep - self.num_eps_trained) % self.val_freq == 0:
                self.validation(ep)

            ep_loss = 0
            ep_loss_terms = []
            gradnorm = 0

            # shuffling order of training data trajectories here
            # since we index data using train_traj_starts, we can just shuffle that!
            shuffled_traj_indices = np.random.permutation(len(self.train_traj_starts))
            train_traj_starts = self.train_traj_starts[shuffled_traj_indices]
            train_traj_lengths = self.train_trajlength[shuffled_traj_indices]

            ### Training loop ###
            self.model.train()

            for it in range(self.num_training_steps):

                loss_terms = []
                (loss, loss_terms), (pred, extras) = self.run_model(it, train_traj_starts, train_traj_lengths, shuffled_traj_indices, 'train', batch_size=self.batch_size)

                ep_loss += loss
                ep_loss_terms.append(loss_terms)
                
                gradnorm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=torch.inf)

                new_lr = self.lr_scheduler(self.total_its-self.num_eps_trained*self.num_training_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                self.total_its += 1

            ep_loss /= self.num_training_steps
            gradnorm /= self.num_training_steps
            ep_loss_terms = np.mean(ep_loss_terms, axis=0)

            if ep % self.print_trainprogress_freq == 0:
                terms_string = ', '.join([f'{loss_term:.3f}' for loss_term in ep_loss_terms])
                self.mylogger(f'[TRAIN] Completed epoch {ep + 1}/{self.num_eps_trained + self.N_eps}, ep_loss = {ep_loss:.3f}, terms = {terms_string}, time = {time.time() - train_start:.2f}s, time/epoch = {(time.time() - train_start)/(ep + 1 - self.num_eps_trained):.2f}s')

            self.writer.add_scalar('train/loss', ep_loss, ep)
            self.writer.add_scalar('train/gradnorm', gradnorm, ep)
            self.writer.add_scalar('train/lr', new_lr, ep)
            for i, loss_term in enumerate(ep_loss_terms):
                self.writer.add_scalar(f'train/loss_term_{i}', loss_term, ep)
            self.writer.flush()

        self.mylogger(f'[TRAIN] Training complete, total time = {time.time() - train_start:.2f}s')
        self.save_model(ep, best=-2)

        if self.eval_tools_on_best:
            # read epoch values from saved model_best*_ep{epoch}.pth checkpoint files
            best_epochs = []
            for f in glob.glob(opj(self.workspace, 'model_best*.pth')):
                best_epochs.append(int(f.split('_')[-1][2:-4]))
            best_epochs.sort()
            for ep in best_epochs:
                self.eval_tools(ep, load_ckpt=True)

    def validation(self, ep):

        val_start = time.time()

        with torch.no_grad():

            ep_loss = 0
            ep_loss_terms = []

            # starting index of trajectories in dataset
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength
            val_traj_starts = np.hstack((val_traj_starts, -1)) # -1 as the end of 

            ### Validation loop ###
            self.model.eval()

            for it in range(self.num_val_steps):

                (loss, loss_terms), pred = self.run_model(it, val_traj_starts, self.val_trajlength, np.arange(len(val_traj_starts)), 'val')

                ep_loss += loss
                ep_loss_terms.append(loss_terms)

            ep_loss /= self.num_val_steps
            ep_loss_terms = np.mean(ep_loss_terms, axis=0)

            # if in first epoch, initialize a list of lowest: [total val loss, val loss term 0, val loss term 1, etc]
            if ep == self.num_eps_trained or ep == self.num_eps_trained + 1:
                self.lowest_val_loss = []
                for i in range(len(ep_loss_terms)+1):
                    self.lowest_val_loss.append(torch.inf)

            # write to tensorboard
            if ep % self.print_trainprogress_freq == 0:
                terms_string = ', '.join([f'{loss_term:.3f}' for loss_term in ep_loss_terms])
                self.mylogger(f'[VAL] Validated epoch {ep + 1}/{self.num_eps_trained + self.N_eps} over {self.val_ims.shape[0]} images, val_loss = {ep_loss:.6f}, terms = {terms_string}, time taken = {time.time() - val_start:.2f} s')
            self.writer.add_scalar('val/loss', ep_loss, ep)
            for i, loss_term in enumerate(ep_loss_terms):
                self.writer.add_scalar(f'val/loss_term_{i}', loss_term, ep)
                # check if new lowest val loss term
                if loss_term < self.lowest_val_loss[i+1]:
                    self.lowest_val_loss[i+1] = loss_term
                    self.mylogger(f'[VAL] New lowest val_loss term {i} = {self.lowest_val_loss[i+1]:.6f} at ep {ep + 1}/{self.num_eps_trained + self.N_eps}, saving model')
                    self.save_model(ep, best=i)
            self.writer.flush()

            # check if new lowest val loss
            if ep_loss < self.lowest_val_loss[0]:
                self.lowest_val_loss[0] = ep_loss
                self.mylogger(f'[VAL] New lowest val_loss = {self.lowest_val_loss[0]:.6f} at ep {ep + 1}/{self.num_eps_trained + self.N_eps}, saving model')
                self.save_model(ep, best=-1)

    def calculate_valid_crop_size(self, angle_radians, width, height):
        # Calculate the absolute values of the trigonometric functions
        cos_angle = abs(cos(angle_radians))
        sin_angle = abs(sin(angle_radians))

        # Calculate the bounding box dimensions after rotation
        rotated_width = width * cos_angle + height * sin_angle
        rotated_height = width * sin_angle + height * cos_angle

        # Calculate the valid crop size
        if rotated_width > 0 and rotated_height > 0:
            valid_crop_width = width * height / rotated_height
            valid_crop_height = width * height / rotated_width
        else:
            valid_crop_width = 0
            valid_crop_height = 0

        return np.floor(valid_crop_width).astype(int), np.floor(valid_crop_height).astype(int)
    
    def apply_vertical_motion_blur(self, image, kernel_size):
        # # Create a vertical motion blur kernel
        # kernel = np.zeros((kernel_size, kernel_size))
        # kernel[:, kernel_size // 2] = 1 / kernel_size
        # blurred = cv2.filter2D(image, -1, kernel)
        # return blurred
    
        # Ensure the image tensor is in the format (B, C, H, W)
        if image.dim() == 3:  # If (C, H, W)
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Create a vertical motion blur kernel in Torch
        kernel = torch.zeros((kernel_size, kernel_size))
        kernel[:, kernel_size // 2] = 1.0 / kernel_size  # Vertical line of 1's in the middle column
        kernel = kernel.view(1, 1, kernel_size, kernel_size)  # Shape for convolution (out_channels, in_channels, H, W)
        
        # Apply the blur kernel with 2D convolution (assuming grayscale, for RGB, apply separately to each channel)
        if image.shape[1] == 3:  # For RGB images, apply separately to each channel
            blurred = torch.cat([torch.nn.functional.conv2d(image[:, i:i+1], kernel, padding='same') 
                                for i in range(3)], dim=1)
        else:
            blurred = torch.nn.functional.conv2d(image, kernel, padding='same')
        
        return blurred.squeeze(0)  # Remove batch dimension if needed

    def blur_images_with_sinusoidal_amplitude(self, image_sequence, max_kernel_size=15, frequency=0.1):
        blurred_sequence = []
        
        for idx, image in enumerate(image_sequence):
            # Calculate sinusoidal blur amplitude
            amplitude = (np.sin(2 * np.pi * frequency * idx) + 1) / 2  # Normalize between 0 and 1
            kernel_size = int(1 + amplitude * (max_kernel_size - 1))  # Map to a suitable range of kernel sizes
            kernel_size = max(1, kernel_size)  # Ensure kernel size is at least 1
            
            # Apply the vertical motion blur with the calculated kernel size
            blurred_image = self.apply_vertical_motion_blur(image, kernel_size)
            blurred_sequence.append(blurred_image)
        
        return torch.stack(blurred_sequence, dim=0)

    def augment(self, inputs, gts_pair):

        gts_vels, gts = gts_pair

        # geometric transformations
        # roll rotation
        if np.random.rand() < 0.1:
            roll_angle = np.random.uniform(-20.0, 20.0)
            inputs = TF.rotate(inputs, roll_angle)
            gts = TF.rotate(gts, roll_angle)
        
            # calculate the largest valid image size under rotation center-crop, then resize back up to original size
            new_w, new_h = self.calculate_valid_crop_size(roll_angle*3.14/180, inputs.shape[3], inputs.shape[2])
            inputs = TF.resized_crop(inputs, inputs.shape[2]//2-new_h//2, inputs.shape[3]//2-new_w//2, new_h, new_w, (inputs.shape[2], inputs.shape[3]))
            gts = TF.resized_crop(gts, gts.shape[2]//2-new_h//2, gts.shape[3]//2-new_w//2, new_h, new_w, (gts.shape[2], gts.shape[3]))

        # (optional) adding periodic vertical blurring to mimic pitching
        if False: # np.random.rand() < 0.1:
            max_kernel_size = np.random.randint(15, 30)
            frequency = np.random.uniform(0.03, 0.1)
            inputs = self.blur_images_with_sinusoidal_amplitude(inputs, max_kernel_size=max_kernel_size, frequency=frequency)
            gts = self.blur_images_with_sinusoidal_amplitude(gts, max_kernel_size=max_kernel_size, frequency=frequency)

        # slight perspective warping
        # TODO

        # left-right flipping
        # note, this needs to flip the y-velocity direction
        if np.random.rand() < 0.1:
            inputs = TF.hflip(inputs)
            gts = TF.hflip(gts)
            gts_vels[:, 1] = -gts_vels[:, 1]
        
        # event-specific transformations
        # each image should get different adjustments here
        # slight scaling
        if np.random.rand() < 0.2:
            scale_factors = np.random.uniform(0.25, 4.0, size=(1))
            inputs = inputs * torch.Tensor(scale_factors, device=self.device)
            inputs = torch.clamp(inputs, -1.0, 1.0)
            if self.num_out_channels == 2:
                gts = gts * torch.Tensor(scale_factors, device=self.device)
                gts = torch.clamp(gts, -1.0, 1.0)
        
        # add noise
        if np.random.rand() < 0.1:
            noise = torch.randn_like(inputs) * 0.00001
            inputs = inputs + noise

        # flip polarity (this does not impact bem representations)
        if np.random.rand() < 0.1:
            scale_factors = np.random.choice([-1.0, 1.0], size=(inputs.shape[0], 1, 1, 1))
            inputs = inputs * torch.Tensor(scale_factors, device=self.device)
            if self.num_out_channels == 2:
                gts = gts * torch.Tensor(scale_factors, device=self.device)

        return inputs, [gts_vels, gts]

    def run_model(self, it, traj_starts, traj_lengths, traj_ids, mode, return_inputs=False, seq_input=False, batch_size=0, do_step=True):

        if mode == 'train':
            meta = self.train_meta
            ims = self.train_ims
            depths = self.train_depths
            desvel = self.train_desvel
            velcmd = self.train_velcmd
            evs = self.train_evs
        elif mode == 'val':
            meta = self.val_meta
            ims = self.val_ims
            depths = self.val_depths
            desvel = self.val_desvel
            velcmd = self.val_velcmd
            evs = self.val_evs
        else:
            self.mylogger(f'[RUN_MODEL] Invalid run_model mode {mode}. Exiting.')
            exit()

        self.optimizer.zero_grad()

        # define weights for loss terms
        if self.loss_weights is not None:
            weights = torch.Tensor(self.loss_weights)
        else:
            weights = torch.ones((len(gt_norms)))

        # compute loss
        loss = 0.0
        loss_terms = torch.zeros_like(weights)

        # init pred and gt tensors
        # preds_full_vel is of size (trajlength, 3)
        # preds_full_vision is of size (trajlength, 1, H, W) for depth ims or (trajlength, 2, H, W) for evs
        preds_full_vel = torch.zeros((traj_lengths[it]-1, 3)).cpu()
        preds_full_vision = torch.zeros((traj_lengths[it]-1, 1, self.train_ims.shape[-2], self.train_ims.shape[-1])).cpu() # 2nd channel used to be self.num_out_channels, but model.forward should make a single-channel output if num outputs = 2 (evframe prediction)
        preds_full = (preds_full_vel, preds_full_vision)
        gts_full = (torch.zeros_like(preds_full_vel).cpu(), torch.zeros_like(preds_full_vision).cpu())

        # compute batchified indices
        ids = np.arange(traj_starts[it]+1, traj_starts[it]+traj_lengths[it]) # trajectory indices
        batch_size = len(ids) if batch_size <= 0 else batch_size
        batched_id_sets = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

        if return_inputs:
            traj_input_ims_full = ims[ids, ...].unsqueeze(1)
            traj_input_evs_full = evs[traj_ids[it]].unsqueeze(1)
            traj_input_desvels_full = desvel[ids].unsqueeze(1)

        ### BATCH LOOP ###

        # batchify input to smaller-than-trajectory-length chunks
        # batch size 0 runs this loop once, over the whole trajectory
        for i_batch, batch_ids in enumerate(batched_id_sets):

            ## GET INPUT ##

            # get batch of traj_input vision data
            if self.num_in_channels == 1:
                if depths is not None:
                    traj_input = depths[batch_ids, ...].unsqueeze(1)
                else:
                    self.mylogger(f'[RUN_MODEL] num_in_channels = 1 but no depths available. Exiting.')
                    exit()

            elif self.num_in_channels == 2:

                # traditional method of loading pre-computed event frames
                if mode != 'train':

                    if evs is not None:
                        traj_input = evs[traj_ids[it]][batch_ids-1-traj_starts[it], ...].unsqueeze(1)
                    else:
                        self.mylogger(f'[RUN_MODEL] num_in_channels = 2 but no evs available. Exiting.')
                        exit()

            else:
                self.mylogger(f'[RUN_MODEL] Invalid num_in_channels {self.num_in_channels}. Only num_in_channels of 1 (depth) or 2 (evs) is implemented. Exiting.')
                exit()

            ## GET GROUND TRUTH ##

            # get batch of gt vision data
            if self.num_out_channels == 1:
                if depths is not None:
                    gt_frames = depths[batch_ids, ...].unsqueeze(1)
                else:
                    self.mylogger(f'[RUN_MODEL] num_out_channels = 1 but no depths available. Exiting.')
                    exit()
            elif self.num_out_channels == 2:
                if evs is not None:
                    gt_frames = evs[traj_ids[it]][batch_ids-1-traj_starts[it], ...].unsqueeze(1)
                else:
                    self.mylogger(f'[RUN_MODEL] num_out_channels = 2 but no evs available. Exiting.')
                    exit()
            else:
                self.mylogger(f'[RUN_MODEL] Invalid num_out_channels {self.num_out_channels}. Only num_out_channels of 1 (depth) or 2 (evs) is implemented. Exiting.')
                exit()

            # get batch of desvels and gt velcmds
            traj_input_desvels = desvel[batch_ids].view(-1, 1)
            traj_input_velcmds = velcmd[batch_ids, ...]

            # calculate gt norms list
            gt = (traj_input_velcmds, gt_frames)
            gt_norms = (gt[0]/traj_input_desvels, gt[1])

            def return_pred(pred):
                return pred

            # move inputs to device
            traj_input = traj_input.to(self.device).float()
            traj_input_desvels = traj_input_desvels.to(self.device).float()
            gt_norms = [gt_norm.to(self.device).float() for gt_norm in gt_norms]

            # run data augmentation
            if self.data_augmentation != 0.0 and mode == 'train':
                traj_input, gt_norms = self.augment(traj_input, gt_norms)

            ## RUN MODEL ##

            # run model
            extras = ()
            # batch (trajectory) query
            if not seq_input:

                if self.model_type == 'OrigUNet':

                    preds, extras = self.model([traj_input, traj_input_desvels, None])
                    preds = (preds, extras[0])
                
                elif self.model_type == 'ConvNet_w_VelPred':

                    preds, extras = self.model([traj_input, traj_input_desvels, None])
                    preds = (preds, torch.zeros_like(gt_norms[1]))

                # for VITFLY models
                elif 'VITFLY_' in self.model_type:
                        
                    preds, extras = self.model([traj_input, traj_input_desvels, None, None])
                    preds = (preds, torch.zeros_like(gt_norms[1]))
                    # if extras is None:
                    extras = [torch.zeros_like(preds[1])]

                    preds[0][:, 2] = 0.0

                elif isinstance(self.model_type, list):

                    if self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'VITFLY_ViTLSTM':

                        preds, extras = self.model([traj_input, traj_input_desvels, [None, None], None])
                        preds = (preds, extras[0])

                        preds[0][:, 2] = 0.0

                    elif self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'ConvNet_w_VelPred':

                        preds, extras = self.model([traj_input, traj_input_desvels, [None, None], None])
                        preds = (preds, extras[0])

                else:
                
                    preds, extras = self.model([traj_input, traj_input_desvels]) #, (init_hidden_state, init_cell_state)])

            # if inputs are sequentially, individually queried (legacy)
            else:
                preds = []
                for single_input, single_vel in zip(traj_input, traj_input_desvels):
                    single_out = self.model([single_input.unsqueeze(0), single_vel.unsqueeze(0)])[0]
                    preds.append(single_out)
                preds = torch.stack(preds).squeeze()

            # progressively fill preds and gt with batchified preds and gt
            preds_full[0][batch_ids-(traj_starts[it]+1), ...] = preds[0].cpu()
            preds_full[1][batch_ids-(traj_starts[it]+1), ...] = preds[1].cpu()
            gts_full[0][batch_ids-(traj_starts[it]+1), ...] = gt[0].cpu()
            gts_full[1][batch_ids-(traj_starts[it]+1), ...] = gt[1].cpu()

            ## COMPUTE LOSS ##

            batch_loss = 0.0
            for i, (weight, gt_norm, pred) in enumerate(zip(weights, gt_norms, preds)):
                
                # pred vel loss
                if i == 0 and self.args.optional_loss_param[0] != 0.0:

                    loss_term = F.mse_loss(gt_norm, pred, reduction='none')
                    loss_term_value = loss_term.mean().item()
                    scaler_mask = torch.logical_or(gt_norm[:,1].abs() > 0.00, gt_norm[:,2].abs() > 0.00) # find where y or z commands are nonzero
                    opposite_mask = ~scaler_mask
                    scaler = self.args.optional_loss_param[0]*scaler_mask.float() + 1.0*opposite_mask.float() # scale up where y or z commands are nonzero
                    loss_term *= scaler.unsqueeze(1).repeat(1, 3)
                    loss_term = loss_term.mean()

                # pred vision loss
                elif i == 1 and self.args.optional_loss_param[1] != 0.0:

                    loss_term = F.mse_loss(gt_norm, pred, reduction='none')
                    loss_term_value = loss_term.mean().item()
                    
                    if self.args.optional_loss_param[1] < 0: # do inverse loss scaling w.r.t depth values

                        # since gt norm is normalized and may get close to 0, we add a small value to prevent term blowing up
                        loss_term *= 1.0 / (gt_norm + 0.1) #  5.0 * (1.5 - gt_norm)

                    if self.args.optional_loss_param[1] == -2.0: # only train on depth pixels below some threshold

                        loss_term *= (gt_norm < 0.99).float()

                    loss_term = loss_term.mean()


                # no optional loss param
                else:

                    loss_term = F.mse_loss(gt_norm, pred)
                    loss_term_value = loss_term.mean().item()
                    loss_term = loss_term.mean()

                # backprop-able loss
                batch_loss += weight * loss_term
                
                loss_terms[i] += loss_term_value
                loss += weight * loss_term

            ### END LOSS TERM LOOP

            if mode == 'train' and do_step:
                # compute gradients and make training step
                batch_loss.backward()
                self.optimizer.step()

        ### END BATCH LOOP

        # assert that loss is not NaN
        assert not torch.isnan(loss), f'[RUN_MODEL] Loss is NaN at iteration {it}'

        # convert loss_terms to numpy
        loss_terms = loss_terms.detach().cpu().numpy()

        # return results of run_model
        if not return_inputs:
            return (loss, loss_terms), (return_pred(preds_full), extras)
        else:
            return (loss, loss_terms), (return_pred(preds_full), extras), (traj_input_ims_full, traj_input_evs_full, traj_input_desvels_full, gts_full)

def argparsing(filename=None):

    if filename is not None:
        default_config_files = [filename]
    else:
        default_config_files = [f'/home/{uname}/evfly_ws/src/evfly/learner/configs/config.txt']

    import configargparse
    parser = configargparse.ArgumentParser(default_config_files=default_config_files)

    ## general parameters ##
    parser.add_argument('--config', is_config_file=True, help='config file relative path')
    parser.add_argument('--basedir', type=str, default=f'/home/{uname}/evfly_ws/src/evfly', help='path to repo')
    parser.add_argument('--logdir', type=str, default='learner/logs', help='path to relative logging directory')
    parser.add_argument('--datadir', type=str, default=f'/home/{uname}/evfly_ws/src/evfly', help='path to relative dataset directory')
    
    ## experiment-level and learner params ##
    # some are legacy and may be unused
    parser.add_argument('--ws_suffix', type=str, default='', help='suffix if any to workspace name')
    parser.add_argument('--model_type', nargs='+', type=str, default='LSTMNet', help='list of strings matching model names in learner_models.py')
    parser.add_argument('--velpred', type=int, default=0, help='whether to add a velpred NN head to the model (0: none, 1: convnet+velpredfc to upsampled unet output, 11: convnet+velpredfc to unet output before upsampling, 2: convnet+velpredfc to unet middle layer output)')
    parser.add_argument('--dataset', nargs='+', type=str, default=None, help='name of dataset; is formatted as a list since we may want to select data from multiple datasets')
    parser.add_argument('--use_h5', action='store_true', help='whether to load dataset from a .h5 file')
    parser.add_argument('--short', type=int, default=0, help='if nonzero, how many trajectory folders to load')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of dataset to use for validation')
    parser.add_argument('--seed', type=int, default=None, help='random seed to use for python random, numpy, and torch; -2 does not shuffle data or set np/torch seed, -1 sets seed to current time for data and does not set np/torch seed, >=0 sets the seed to that value')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size to extract chunks from each trajectory for train/val')
    parser.add_argument('--device', type=str, default='cuda', help='generic cuda device; specific GPU should be specified in CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load_trainval', action='store_true', help='whether to load the train/val split from the given checkpoint path')
    parser.add_argument('--checkpoint_path', action='append', help='absolute path to model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--N_eps', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate for')
    parser.add_argument('--lr_decay', action='store_true', help='whether to use lr_decay, hardcoded to exponentially decay to 0.01 * lr by end of training')
    parser.add_argument('--save_model_freq', type=int, default=25, help='frequency with which to save model checkpoints')
    parser.add_argument('--val_freq', type=int, default=10, help='frequency with which to evaluate on validation set')
    parser.add_argument('--optional_loss_param', nargs='+', type=float, default=None, help='list of optional loss params that dictate different behavior per loss term')
    parser.add_argument('--num_recurrent', nargs='+', type=int, default=0, help='number of recurrent layers to use in an architecture, usually via LSTM; list to apply to different parts of an architecture')
    parser.add_argument('--events', type=str, default='', help='if non empty then train from this path to an events file; a file name including frames will load event frames')
    parser.add_argument('--keep_collisions', action='store_true', help='keep trajectories with collisions when dataloading')
    parser.add_argument('--do_transform', action='store_true', help='(legacy) transform images/depths/evs to auto-gimbal according to rotation state estimate (approximate)')
    parser.add_argument('--eval_tools_freq', type=int, default=0, help='frequency with which to generate evaluation plots')
    parser.add_argument('--eval_tools_on_best', action='store_true', help='whether to run eval_tools at best val loss epochs')
    parser.add_argument('--print_trainprogress_freq', type=int, default=1, help='frequency with which to print training progress statistics')
    parser.add_argument('--num_out_channels', type=int, default=1, help='number of output channels for prediction image (used for ConvUNet model; 1 for depth as default, 2 for evframes)')
    parser.add_argument('--num_in_channels', type=int, default=2, help='number of input channels for input to a model (used for ConvUNet model; 2 for evframes as default, or 1 for depth/images)')
    parser.add_argument('--resize_input', nargs='+', type=int, default=None, help='2-length list of ints specifying [H, W] to downsample images, depths, and event frames to')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=None, help='list of weights for loss terms, must be same length as number of loss terms')
    parser.add_argument('--split_method', type=str, default='train-val', help='whether split should be done in train-val or val-train order, post-shuffle. due to legacy code in which split was val-train.')
    parser.add_argument('--num_outputs', type=int, default=2, help='(legacy) number of outputs for a model; typical use-case is the length of desired velocity vector for prediction, where ConvUNet_w_VelPred model may output 2-vector and calculate third component to complete a unit norm vector')
    parser.add_argument('--rescale_depth', type=float, default=0.0, help='rescale depth values by this factor to be in range [0, 1] (default 0.0 does not rescale)')
    parser.add_argument('--rescale_evs', type=float, default=0.0, help='rescale evs values by this factor to be in range [-1, 1] (default 0.0 does not rescale; -1.0 rescales by maximum vlaue per frame)')
    parser.add_argument('--domain_randomization', type=float, default=0.0, help='(legacy)')
    parser.add_argument('--bev', type=int, default=0, help='whether to use a binary event image (0: No, 1: abs(evframe), 2: bev (actual BEV)). referred to as BEM in the paper.')
    parser.add_argument('--skip_type', type=str, default='crop', help='what kind of processing should be made over skip connections (default: crop, other options: interp). Only relevant for OrigUNet type models.')
    parser.add_argument('--combine_checkpoints', action='store_true', help='whether to combine torch state dicts from the inputted checkpoints into a single model')
    parser.add_argument('--data_augmentation', type=float, default=0.0, help='whether to use data augmentation methods in dataloader (0.0: no augmentation, 1.0: augmentation, other values to be implemented)')
    parser.add_argument('--evs_min_cutoff', type=float, default=0.0, help='bottom percent of absolute event frame values to cut')

    ## model-specific arguments ##

    # encoder
    parser.add_argument('--enc_num_layers', type=int, default=2, help='number of layers in encoder')
    parser.add_argument('--enc_kernel_sizes', nargs='+', type=int, default=[5, 5], help='list of kernel sizes for encoder')
    parser.add_argument('--enc_kernel_strides', nargs='+', type=int, default=[2, 2], help='list of kernel strides for encoder')
    parser.add_argument('--enc_out_channels', nargs='+', type=int, default=[16, 64], help='list of output channels for encoder')
    parser.add_argument('--enc_activations', nargs='+', type=str, default=['relu', 'relu'], help='list of activations for encoder')
    parser.add_argument('--enc_pool_type', type=str, default='max', help='pooling type for encoder')
    parser.add_argument('--enc_invert_pool_inputs', action='store_true', help='multiply input to pool function by -1, then undo inversion after pool')
    parser.add_argument('--enc_pool_kernels', nargs='+', type=int, default=[2, 2], help='list of pool strides for encoder')
    parser.add_argument('--enc_pool_strides', nargs='+', type=int, default=[2, 2], help='list of pool strides for encoder')
    parser.add_argument('--enc_conv_function', type=str, default='conv2d', help='convolution function for encoder')
    
    # decoder
    parser.add_argument('--dec_num_layers', type=int, default=2, help='number of layers in decoder')
    parser.add_argument('--dec_kernel_sizes', nargs='+', type=int, default=[5, 5], help='list of kernel sizes for decoder')
    parser.add_argument('--dec_kernel_strides', nargs='+', type=int, default=[2, 2], help='list of kernel strides for decoder')
    parser.add_argument('--dec_out_channels', nargs='+', type=int, default=[64, 16], help='list of output channels for decoder')
    parser.add_argument('--dec_activations', nargs='+', type=str, default=['relu', 'sigmoid'], help='list of activations for decoder')
    parser.add_argument('--dec_pool_type', type=str, default='max', help='pooling type for decoder')
    parser.add_argument('--dec_pool_kernels', nargs='+', type=int, default=[2, 2], help='list of pool strides for decoder')
    parser.add_argument('--dec_pool_strides', nargs='+', type=int, default=[2, 2], help='list of pool strides for decoder')
    parser.add_argument('--dec_conv_function', type=str, default='upconv2d', help='convolution function for decoder')

    # fc
    parser.add_argument('--fc_num_layers', type=int, default=3, help='number of layers in fully connected')
    parser.add_argument('--fc_layer_sizes', nargs='+', type=int, default=[128, 32, 1], help='list of layer sizes for fully connected')
    parser.add_argument('--fc_activations', nargs='+', type=str, default=['leaky_relu', 'leaky_relu', 'tanh'], help='list of activations for fully connected')
    parser.add_argument('--fc_dropout_p', type=float, default=0.1, help='dropout probability for fully connected')

    ## miscellaneous ##

    # added from run_competition for completeness in arg list (only included for deployment compatibility)
    parser.add_argument("--align_evframe", help="Rectify event frames", required=False, action="store_true")
    parser.add_argument("--vision_based", help="Fly vision-based", required=False, dest="vision_based", action="store_true")
    parser.add_argument("--ppo_path", help="PPO neural network policy", required=False, default=None)
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to model checkpoint')
    parser.add_argument("--keyboard", help="Fly state-based mode but take velocity commands from keyboard WASD", required=False, dest="keyboard", action="store_true")
    parser.add_argument("--planner", help="Fly state-based mode but calculate use a path planner and follow it with a custom controller", required=False, dest="planner", action="store_true")

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    print(f'[CONFIGARGPARSE] Parsing args from config file {args.config}')

    return args

if __name__ == '__main__':

    args = argparsing()
    print(args)

    learner = Learner(args)
    try:
        learner.train()
    except KeyboardInterrupt:
        print('[MAIN] Keyboard interrupt detected, exiting.')
        learner.logfile.close()
        exit()
