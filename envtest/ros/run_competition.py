#!/usr/bin/python3
import rospy
from std_msgs.msg import Empty, String, Header
from geometry_msgs.msg import Vector3, TwistStamped
from sensor_msgs.msg import Image
from dodgeros_msgs.msg import Command, QuadState
from envsim_msgs.msg import ObstacleArray, Obstacle
from cv_bridge import CvBridge, CvBridgeError
import message_filters

# from rl_example import load_rl_policy
from user_code import compute_command_state_based
from utils import AgileCommandMode, AgileQuadState, AgileCommand

import argparse
import time
import numpy as np
import pandas as pd
import os, sys
from os.path import join as opj
import cv2
import torch
import yaml

import getpass
uname = getpass.getuser()

AF_PATH = f'/home/{uname}/evfly_ws/src/evfly'
SMALL_EPS = 1e-5

sys.path.append(AF_PATH+'/learner')
from learner import argparsing
from learner_models import *
sys.path.append(AF_PATH+'/utils')
from ev_utils import *

class AgilePilotNode:
    def __init__(self, vision_based=False, ppo_path=None, model_type=None, model_path=None, num_recurrent=None, keyboard=False, use_planner=False, exp_name=None, total_num_exps=1):
        print("[RUN_COMPETITION] Initializing agile_pilot_node...")
        rospy.init_node("agile_pilot_node", anonymous=False)

        self.vision_based = vision_based
        self.run_expert_in_parallel = True
        self.num_recurrent = num_recurrent
        self.keyboard = keyboard

        self.exp_name = exp_name
        self.total_num_exps = total_num_exps

        print(f"[RUN_COMPETITION] vision_based = {self.vision_based}, run_expert_in_parallel = {self.run_expert_in_parallel}, num_recurrent = {self.num_recurrent}, keyboard = {self.keyboard}")

        #######################
        ### USER PARAMETERS ###
        #######################

        # used for repeated trials with steadily increasing desired velocity
        # if self.exp_name is not None and self.exp_name != '':
        #     try:
        #         print('[RUN_COMPETITION] Extracting desired velocity from exp_name...')
        #         exp_num = int(self.exp_name[-3:])
        #         progress = exp_num / self.total_num_exps
        #         self.desiredVel = 3.0 + 4.0 * progress
        #     except:
        #         self.desiredVel = np.random.uniform(low=3.0, high=7.0)
        
        # else:
        self.desiredVel = 5.0
        
        print()
        print(f"[RUN_COMPETITION] Desired velocity = {self.desiredVel:.2f}")
        print()

        # if in state mode, set these to false to save on computation
        self.do_events = True and self.vision_based
        self.use_gimbal = False and self.vision_based # legacy
        
        self.save_events = False
        self.saved_events = False
        self.save_im_dbg2 = False
        self.saved_im_dbg2 = False
        self.plot_cmd = False

        #######################

        ########################################
        ## Set up NN and other configurations ##
        ########################################

        # read arguments from learner/configs/lstm.txt
        self.args = argparsing(filename=AF_PATH+'/learner/configs/lstm.txt')
        # define enc and dec params
        # make dictionaries enc_params and dec_params with the above args
        self.enc_params = {
            'num_layers': self.args.enc_num_layers,
            'kernel_sizes': self.args.enc_kernel_sizes,
            'kernel_strides': self.args.enc_kernel_strides,
            'out_channels': self.args.enc_out_channels,
            'activations': self.args.enc_activations,
            'pool_type': self.args.enc_pool_type,
            'invert_pool_inputs': self.args.enc_invert_pool_inputs,
            'pool_kernels': self.args.enc_pool_kernels,
            'pool_strides': self.args.enc_pool_strides,
            'conv_function': self.args.enc_conv_function,
        }
        self.dec_params = {
            'num_layers': self.args.dec_num_layers,
            'kernel_sizes': self.args.dec_kernel_sizes,
            'kernel_strides': self.args.dec_kernel_strides,
            'out_channels': self.args.dec_out_channels,
            'activations': self.args.dec_activations,
            'pool_type': self.args.dec_pool_type,
            'pool_kernels': self.args.dec_pool_kernels,
            'pool_strides': self.args.dec_pool_strides,
            'conv_function': self.args.dec_conv_function,
        }
        self.fc_params = {
            'num_layers': self.args.fc_num_layers,
            'layer_sizes': self.args.fc_layer_sizes,
            'activations': self.args.fc_activations,
            'dropout_p': self.args.fc_dropout_p,
        }

        self.num_recurrent = self.args.num_recurrent

        # load yaml of parameters
        with open('../../envsim/parameters/simple_sim_pilot.yaml') as file:
            pilot_params = yaml.load(file, Loader=yaml.FullLoader)
        self.takeoff_height = pilot_params['takeoff_height']

        self.eval_config_file_path = '../../envtest/ros/evaluation_config.yaml'
        with open(self.eval_config_file_path) as file:
            eval_config_params = yaml.load(file, Loader=yaml.FullLoader)
        self.goal_distance = eval_config_params['target']

        self.config_file_path = '../../flightmare/flightpy/configs/vision/config.yaml'
        with open(self.config_file_path) as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader)

        camera_params = config_params['rgb_camera']
        self.image_h, self.image_w = (camera_params['height'], camera_params['width'])
        self.gimbal_h, self.gimbal_w = (60, 90)
        self.gimbal_fov = camera_params['fov']

        self.rl_policy = None
        if ppo_path is not None:
            self.rl_policy = load_rl_policy(ppo_path)
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        
        quad_name = "kingfisher"

        # logging
        self.init = 0
        self.col = None
        self.t1 = 0 #Time flag
        self.timestamp = 0 #Time stamp initial
        # self.last_valid_im = None #Image that will be logged
        self.data_format = {'timestamp':[],
                            'desired_vel':[],
                            'quat_1':[],
                            'quat_2':[],
                            'quat_3':[],
                            'quat_4':[],
                            'pos_x':[],
                            'pos_y':[],
                            'pos_z':[],
                            'vel_x':[],
                            'vel_y':[],
                            'vel_z':[],
                            'velcmd_x':[],
                            'velcmd_y':[],
                            'velcmd_z':[],
                            'ct_cmd':[],
                            'br_cmd_x':[],
                            'br_cmd_y':[],
                            'br_cmd_z':[],
                            'is_collide': [],
                            }
        self.data_buffer = pd.DataFrame(self.data_format) # store in the data frame
        self.data_buffer_maxlength = 10
        
        self.log_ctr = 0 # counter for the csv, unused for now
        
        # if goal distance is 60, end of data collection xrange is 50
        self.data_collection_xrange = [0+5, self.goal_distance-.17*self.goal_distance]

        # make the folder for the epoch
        self.folder = AF_PATH+f"/utils/rollouts/{int(time.time()*1000) if (self.exp_name is None or self.exp_name == '') else self.exp_name}"
        os.mkdir(self.folder)

        # if this is a named experiment, save the config file to maintain information of run, including scene/env/etc
        if self.exp_name is not None and self.exp_name != '':
            os.system(f'cp {self.config_file_path} {self.folder}/config.yaml')

        self.events = np.zeros((self.gimbal_h, self.gimbal_w)) if self.use_gimbal else np.zeros((self.image_h, self.image_w))

        # if save_events, save each event frame via the log function and then save as a npy
        self.evims = []
        self.im_dbg2s = []

        self.state_poss = []
        self.state_vels = []
        self.expert_command = None
        self.expert_commands = []
        self.vision_commands = []
        self.spline_poss = []
        self.spline_vels = []
        self.plotted_commands = False

        # load trained model
        if self.args.checkpoint_path is not None and self.vision_based:
            print(f"[RUN_COMPETITION] Model loading from {self.args.checkpoint_path} ...")
            self.device = torch.device("cpu")

            if self.args.model_type == 'ConvNet_w_VelPred' or (isinstance(self.args.model_type, list) and len(self.args.model_type) == 1 and self.args.model_type[0] == 'ConvNet_w_VelPred'):

                self.model = ConvNet_w_VelPred(num_in_channels=self.args.num_in_channels, 
                                               num_recurrent=self.args.num_recurrent[1], num_outputs=self.args.num_outputs,
                                               enc_params=self.enc_params,
                                               fc_params=self.fc_params,
                                               input_shape=[1, 1, 68, 148]).to(self.device).float()
                # NOTE hardcoded resize input here for velpred11 part of model

            elif self.args.model_type == 'OrigUNet' or (isinstance(self.args.model_type, list) and len(self.args.model_type) == 1 and self.args.model_type[0] == 'OrigUNet'):

                self.model = OrigUNet(num_in_channels=self.args.num_in_channels, 
                                      num_out_channels=self.args.num_out_channels, 
                                      num_recurrent=self.args.num_recurrent, 
                                      input_shape=[1, 1, self.args.resize_input[0], self.args.resize_input[1]], 
                                      velpred=self.args.velpred, 
                                      enc_params=self.enc_params, 
                                      fc_params=self.fc_params,
                                      form_BEV=self.args.bev,
                                      evs_min_cutoff=0.0,
                                      skip_type=self.args.skip_type).to(self.device).float()
                
            elif self.args.model_type == 'OrigUNet_w_VITFLY_VitLSTM' or isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'VITFLY_ViTLSTM':

                self.model = OrigUNet_w_VITFLY_ViTLSTM(
                                num_in_channels=self.args.num_in_channels,
                                num_out_channels=self.args.num_out_channels,
                                num_recurrent=self.args.num_recurrent,
                                input_shape=[1, 1, self.args.resize_input[0], self.args.resize_input[1]],
                                velpred=self.args.velpred,
                                enc_params=self.enc_params,
                                dec_params=self.dec_params,
                                fc_params=self.fc_params,
                                form_BEV=self.args.bev,
                                evs_min_cutoff=0.0,
                                skip_type=self.args.skip_type,
                                is_deployment=False,
                            ).to(self.device).float()

            elif self.args.model_type == 'OrigUNet_w_ConvNet_w_VelPred' or isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'ConvNet_w_VelPred':

                self.model = OrigUNet_w_ConvNet_w_VelPred(
                                num_in_channels=self.args.num_in_channels,
                                num_out_channels=self.args.num_out_channels,
                                num_recurrent=self.args.num_recurrent,
                                input_shape=[1, 1, self.args.resize_input[0], self.args.resize_input[1]],
                                velpred=self.args.velpred,
                                enc_params=self.enc_params,
                                dec_params=self.dec_params,
                                fc_params=self.fc_params,
                                num_outputs=self.args.num_outputs,
                                form_BEV=self.args.bev,
                                evs_min_cutoff=0.0,
                                skip_type=self.args.skip_type,
                                is_deployment=False,
                            ).to(self.device).float()

            else:
                print(f'[RUN_COMPETITION] Invalid self.args.model_type {self.args.model_type}. Exiting.')
                exit()

            # print number of params and trainable params
            print(f'[SETUP] Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}')
            print(f'[SETUP] Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

            if len(self.args.checkpoint_path) > 1:
                
                if self.args.combine_checkpoints:

                    print(f'[SETUP] Combining checkpoints {self.args.checkpoint_path} into a single state dict')
                    self.model.load_state_dict(self.combine_state_dicts([torch.load(cp, map_location=self.device) for cp in self.args.checkpoint_path], model_names=[self.args.model_type[0].lower(), self.args.model_type[1].lower()]))

                else:
                    
                    if self.args.model_type == 'OrigUNet_w_ConvUNet_w_VelPred':

                        self.model.origunet.load_state_dict(torch.load(self.args.checkpoint_path[0], map_location=self.device))
                        self.model.convunet_w_velpred.load_state_dict(torch.load(self.args.checkpoint_path[1], map_location=self.device))

                    elif isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'VITFLY_ViTLSTM':

                        self.model.origunet.load_state_dict(torch.load(self.args.checkpoint_path[0], map_location=self.device), strict=False)
                        self.model.nd_vitlstm.load_state_dict(torch.load(self.args.checkpoint_path[1], map_location=self.device))

                    elif self.args.model_type == 'OrigUNet_w_ConvNet_w_VelPred' or isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'ConvNet_w_VelPred':

                        self.model.origunet.load_state_dict(torch.load(self.args.checkpoint_path[0], map_location=self.device))
                        self.model.convnet_w_velpred.load_state_dict(torch.load(self.args.checkpoint_path[1], map_location=self.device))

                    else:
                        raise ValueError(f"[RUN_COMPETITION] model_type {self.args.model_type} not recognized for self.args.checkpoint_path list of length > 1. Exiting.")

            else:
                print(f'[RUN_COMPETITION] self.args.checkpoint_path of length 1, loading from {self.args.checkpoint_path[0]}')
                self.model.load_state_dict(torch.load(self.args.checkpoint_path[0], map_location=self.device))

            self.model.eval()

            # Initialize hidden state
            self.model_hidden_state = [None]

            print(f"[RUN_COMPETITION] Model loaded")
            time.sleep(2)

        self.start_time = 0
        self.logged_time_flag = 0

        self.first_data_write = False

        self.current_cmd_controller = None
        self.current_cmd = None

        self.keyboard_input = ''
        self.got_keypress = 0.0

        # initialize to bogus obstacle array with 10 obstacles at 1000, 1000, 1000
        self.obs_msg = self.create_obstacle_array()

        # vision member variables
        self.depth = np.zeros((self.image_h, self.image_w))
        self.depth_t = None
        self.depth_im_threshold = 0.9 # increased from 0.1 (max depth seems to be ~ 0.885)
        self.im = np.zeros((self.image_h, self.image_w))
        self.im_t = None
        self.im_ctr = 0
        self.prev_im = np.zeros((self.image_h, self.image_w))

        self.depth_gimbal = np.zeros((self.gimbal_h, self.gimbal_w))
        self.im_gimbal = np.zeros((self.gimbal_h, self.gimbal_w))
        self.prev_im_gimbal = np.zeros((self.gimbal_h, self.gimbal_w))
        self.gimbal = None
        self.pts = [[0,0], [0,0], [0,0], [0,0]]

        self.im_dbg1 = None
        self.im_dbg2 = None

        # place to store extras from learned model
        self.extras = None

        # manual synchronization variables
        self.accepted_delta_t_im_depth = 0.01

        self.csv_file = AF_PATH+'/flightmare/flightpy/configs/vision/'+config_params['environment']['level']+'/'+config_params['environment']['env_folder']+'/static_obstacles.csv'
        self.is_trees = 'trees' in config_params['environment']['level'] or 'forest' in config_params['environment']['level']
        self.quad_radius = config_params['quad_radius']

        #####################
        ## ROS subscribers ##
        #####################

        # Logic subscribers
        self.start_sub = rospy.Subscriber(
            "/" + quad_name + "/start_navigation",
            Empty,
            self.start_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Observation subscribers
        # we are making odom, image, and depth approximately time synchronized for logging purposes
        self.odom_sub = message_filters.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/state",
            QuadState,
        )
        self.im_sub = message_filters.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/image",
            Image,
        )
        self.depth_sub = message_filters.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/depth",
            Image,
        )
        timesync = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.im_sub, self.depth_sub], queue_size=10, slop=self.accepted_delta_t_im_depth)
        timesync.registerCallback(self.observation_callback)

        self.obstacle_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles",
            ObstacleArray,
            self.obstacle_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.cmd_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/command",
            Command,
            self.cmd_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.keyboard_sub = rospy.Subscriber(
            "/keyboard_input",
            String,
            self.keyboard_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        ####################
        ## ROS publishers ##
        ####################

        # Command publishers
        self.cmd_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/feedthrough_command",
            Command,
            queue_size=1,
        )
        self.linvel_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/velocity_command",
            TwistStamped,
            queue_size=1,
        )
        self.im_dbg1_pub = rospy.Publisher(
            "/debug_img1",
            Image,
            queue_size=1,
        )
        self.im_dbg2_pub = rospy.Publisher(
            "/debug_img2",
            Image,
            queue_size=1,
        )
        print("[RUN_COMPETITION] Initialization completed!")

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

    #############################
    ## Vision-based controller ##
    #############################

    def compute_command_vision_based(self):

        # Example of LINVEL command (velocity is expressed in world frame)
        command_mode = 2
        command = AgileCommand(command_mode)
        command.t = self.state.t
        command.yawrate = 0.0
        command.mode = 2
        
        ###############
        ## Load data ##
        ###############
        
        # determine model input image
        if self.do_events:
            im = self.events
        else:
            im = self.depth_gimbal if self.use_gimbal else self.depth

        im = torch.Tensor(im)

        # if im is not of size resize_input (see default config file configs/lstm.txt), resize
        if im.shape[0] != self.args.resize_input[0] or im.shape[1] != self.args.resize_input[1]:
            im = torch.nn.functional.interpolate(im.unsqueeze(0).unsqueeze(0), size=(self.args.resize_input[0], self.args.resize_input[1]), mode='bilinear', align_corners=False).squeeze()

        if self.do_events:
            # set this by the percentile
            im_scaledown_factor = torch.quantile(torch.abs(im), 0.97)
        else:
            im_scaledown_factor = 1.0

        # print(f'[RUN_COMPETITION] self.args.model_type = {self.args.model_type}')

        # set hidden state
        if sum(self.num_recurrent) > 0 and ( self.state.pos[0] < 0.5 or self.model_hidden_state is None ):
            
            # set hidden state to be zeros instead
            if (isinstance(self.args.model_type, list) and len(self.args.model_type) == 1 and self.args.model_type[0] == 'OrigUNet') or self.args.model_type == 'OrigUNet':

                print(f'[RUN_COMPETITION] Resetting hidden state for model_type OrigUNet')

                self.model_hidden_state = [[None, None]]

            elif isinstance(self.args.model_type, list) and len(self.args.model_type) > 1 and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'VITFLY_ViTLSTM':

                print(f'[RUN_COMPETITION] Resetting hidden state for model_type OrigUNet, VITFLY_ViTLSTM')
                self.model_hidden_state = ((None, None), None) # for origunet+X, this is ((origunet_unet_hidden, origunet_velpred_hidden), X_hidden)

            elif isinstance(self.args.model_type, list) and len(self.args.model_type) > 1 and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'ConvNet_w_VelPred':

                print(f'[RUN_COMPETITION] Resetting hidden state for model_type OrigUNet, ConvNet_w_VelPred')
                self.model_hidden_state = ((None, None), None)

            else:
                self.model_hidden_state = (torch.zeros(self.model.lstm.num_layers, self.model.lstm.hidden_size).float(), torch.zeros(self.model.lstm.num_layers, self.model.lstm.hidden_size).float())

        # else:

        #     print(len(self.model_hidden_state))

        ###############
        ## Run model ##
        ###############

        # print('===============================================================')
        # print(len(self.model_hidden_state))
        # print('===============================================================')
        # print(self.model_hidden_state)

        with torch.no_grad():
            out = self.model([
                torch.clamp(im.view(1, 1, im.shape[-2], im.shape[-1]) / im_scaledown_factor, -1. if self.do_events else 0.0, 1.), # input_frame
                torch.tensor(self.desiredVel).view(1, 1).float(), # desvel
                *self.model_hidden_state # hidden state
                ])
                # hidden state that for combo origunet+X model should be an unraveled iterable of ((origunet_unet_hidden, origunet_velpred_hidden), X_hidden)

        x, self.extras = out

        if isinstance(self.args.model_type, list) and len(self.args.model_type) > 1 and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'VITFLY_ViTLSTM':

            self.model_hidden_state = self.extras[2]

        elif isinstance(self.args.model_type, list) and len(self.args.model_type) > 1 and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'ConvNet_w_VelPred':

            self.model_hidden_state = self.extras[2]

        elif self.model.__class__.__name__ == 'ConvUNet_w_VelPred' or self.model.__class__.__name__ == 'OrigUNet':

            self.model_hidden_state = [self.extras[2]]
        
        elif self.model.__class__.__name__ == 'ConvNet_w_VelPred':
        
            self.model_hidden_state = self.extras

        elif 'LSTM' in self.model.__class__.__name__:
        
            self.model_hidden_state = self.extras
        
        elif self.num_recurrent > 0:
        
            print(f'[RUN_COMPETITION] model.__class__.__name__ = {self.model.__class__.__name__} with num_recurrent > 0 but hidden state handling has not been implemented. Exiting.')
            exit()

        # print(f'[RUN_COMPETITION VISION_BASED] model output {x}')

        x = x.squeeze().detach().numpy()

        command.velocity = x * self.desiredVel
        # possibly necessary scalers if using a pretrained V(phi) from another environment
        # command.velocity[1] *= 2.0

        # manual drone acceleration phase
        min_xvel_cmd = 1.0
        hardcoded_ctl_threshold = 2.0
        if self.state.pos[0] < hardcoded_ctl_threshold:
            command.velocity[0] = max(min_xvel_cmd, (self.state.pos[0]/hardcoded_ctl_threshold)*self.desiredVel)

        return command

    def cmd_callback(self, msg):
        self.current_cmd_controller = msg

    def keyboard_callback(self, msg):
        self.got_keypress = rospy.Time().now().to_sec()
        self.keyboard_input = msg.data

    # legacy
    def readVel(self,file):
        with open(file,"r") as f:
            x = f.readlines()
            for i in range(len(x)):
                if i == 0:
                    return float(x[i].split("\n")[0])

    # compute estimated events from two stored images, with thresholds inputted
    # network was trained from evims of floats binned by 0.2, so estimate that here
    def compute_events(self, neg_thresh=0.2, pos_thresh=0.2, gimbal=False):

        if gimbal:
            im = self.im_gimbal
            prev_im = self.prev_im_gimbal
            h = self.gimbal_h
            w = self.gimbal_w
        else:
            im = self.im
            prev_im = self.prev_im
            h = self.image_h
            w = self.image_w

        if im is None or prev_im is None:
            self.events = np.zeros((h, w))
            return
        
        # approximation of events calculation
        difflog = np.log(im + SMALL_EPS) - np.log(prev_im + SMALL_EPS)

        # thresholding
        self.events = np.zeros_like(difflog)

        if np.abs(difflog).max() < max(pos_thresh, neg_thresh):
            return

        # quantize difflog by thresholds
        pos_idxs = np.where(difflog > 0.0)
        neg_idxs = np.where(difflog < 0.0)
        self.events[pos_idxs] = (difflog[pos_idxs] // pos_thresh) * pos_thresh
        self.events[neg_idxs] = (difflog[neg_idxs] // -neg_thresh) * -neg_thresh

        return

    # approximate time-synced callback with three sensor measurements: odom state, rgb image, depth image
    def observation_callback(self, odom_msg, im_msg, depth_msg):

        # # debug prints
        # print(f'in observation callback!')
        # print(f'odom_msg timestamp {odom_msg.header.stamp.to_nsec()/1e9}')
        # print(f'im_msg timestamp {im_msg.header.stamp.to_nsec()/1e9}')
        # print(f'depth_msg timestamp {depth_msg.header.stamp.to_nsec()/1e9}')
        # print()

        ###################
        ### SUBSCRIBERS ###
        ###################

        # handle odom
        self.state_callback(odom_msg)

        # handle image
        if self.im_callback(im_msg) < 0:
            return

        # handle depth
        if self.depth_callback(depth_msg) < 0:
            return

        ###################

        # legacy
        # usable keypress?
        if rospy.Time().now().to_sec() - self.got_keypress > 0.1:
            self.keyboard_input = ''
        
        # run expert regardless of method
        if not self.vision_based or (self.vision_based and self.run_expert_in_parallel):
        
            self.expert_command, extras = compute_command_state_based(
                state=self.state,
                obstacles=self.obs_msg,
                desiredVel=self.desiredVel,
                rl_policy=self.rl_policy,
                keyboard=self.keyboard,
                is_trees=self.is_trees
            )
            collisions = extras['collisions']
            wpt_idx = extras['wpt_idx']
            spline_poss = extras['spline_poss']
            spline_vels = extras['spline_vels']

        else:

            self.expert_command = None
            collisions = None
            wpt_idx = None
            spline_poss = None
            spline_vels = None

        # debug image 2; changeable debug image
        self.im_dbg2 = self.im.copy() # copying full image
        
        # if im_dgb2 is single-channel, make 3-channel
        if len(self.im_dbg2.shape) == 2:
            self.im_dbg2 = cv2.cvtColor((self.im_dbg2*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            # self.im_dbg2 = np.stack((self.im_dbg2,)*3, axis=-1)

        # if in vision command mode, compute vision command and publish
        vision_command = None
        if self.vision_based:

            start_compute_time = time.time()
            vision_command = self.compute_command_vision_based()

            # useful occasional prints for debugging
            if self.im_ctr % 20 == 0:
                print(f'[RUN_COMPETITION] compute_command_vision_based took {time.time() - start_compute_time:.3f} seconds')
                print(f'[RUN_COMPETITION] events min = {self.events.min():.2f}, events max = {self.events.max():.2f}, events 0.97 quantile = {torch.quantile(torch.abs(torch.Tensor(self.events)), 0.97):.2f}')

                print(f'[RUN_COMPETITION] depth min = {self.extras[0].min():.2f}, depth max = {self.extras[0].max():.2f}, depth 0.97 quantile = {torch.quantile(torch.abs(self.extras[0]), 0.97):.2f}')

            # if UNet type model, visualize the first element of extras which is fully interpolated up-to-size depth prediction from evframe
            if self.model.__class__.__name__ == 'OrigUNet' or \
               (isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'VITFLY_ViTLSTM') or \
               (isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'ConvNet_w_VelPred'):
                
                self.im_dbg2 = (np.stack((self.extras[0].squeeze().detach().numpy(),)*3, axis=-1) * 255).astype(np.uint8)
            
            self.im_dbg2_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.im_dbg2, encoding="passthrough"))

            self.command = vision_command

        # if in state mode, compute state command and publish
        else:

            # user_code expert
            self.command = self.expert_command

            # debug image 2 will overlay the collision array of points as white dots,
            # where if collision[i, j] == 1 it is red,
            # and the wpt_idx as a green dot
            if collisions is not None:
                x_px_offset = self.im_dbg2.shape[1] / (collisions.shape[1]+1) # float
                y_px_offset = self.im_dbg2.shape[0] / (collisions.shape[0]+1) # float
                # collisions array goes from physical top left (body frame y=15, z=15) to bottom left
                # coordinates in waypoint frame
                for yi in range(collisions.shape[0]):

                    for xi in range(collisions.shape[1]):

                        if collisions[yi, xi] == 1:
                            color = (0, 0, 255) # red
                        else:
                            color = (255, 0, 0) # blue
                        pt_in = (int((xi+1)*x_px_offset), int((yi+1)*y_px_offset))
                        self.im_dbg2 = cv2.circle(self.im_dbg2, pt_in, 2, color, -1)

                # mark chosen waypoint with green circle
                if wpt_idx is not None:
                    pt_in_chosen = (int((wpt_idx[1]+1)*x_px_offset), int((wpt_idx[0]+1)*y_px_offset))
                    cv2.circle(self.im_dbg2, pt_in_chosen, 6, (0, 255, 0), -1)

        self.publish_command(self.command)

        # publish debug images
        # debug image 1; image and events overlayed + velocity command arrow

        if self.use_gimbal:

            im_dbg1 = self.depth_gimbal.copy() if not self.do_events else self.im_gimbal.copy()
            h, w = self.gimbal_h, self.gimbal_w

        else:

            im_dbg1 = self.depth.copy() if not self.do_events else self.im.copy()
            h, w = self.image_h, self.image_w

        if self.do_events:

            im_dbg1_evs = visualize_evim(self.events) # copying cropped and horizon-aligned image
            # add in image for better visualization
            im_dbg1 = np.stack(((im_dbg1*255.0).astype(np.uint8),)*3, axis=-1)
            if self.events is not None:
                im_dbg1[np.where(self.events != 0.0)] = im_dbg1_evs[np.where(self.events != 0.0)]

        arrow_start = (w//2, h//2)
        if not self.vision_based and wpt_idx is not None:
            arrow_end = pt_in_chosen
        else:
            arrow_end = (int(w/2-self.command.velocity[1]*(w/3)), int(h/2-self.command.velocity[2]*(h/3)))
        
        self.im_dbg1 = im_dbg1
        # self.im_dbg1 = cv2.arrowedLine( im_dbg1, arrow_start, arrow_end, (0, 0, 0), h//60, tipLength=0.2)
        self.im_dbg1_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.im_dbg1, encoding="passthrough"))

        self.im_dbg2 = cv2.arrowedLine( self.im_dbg2, arrow_start, arrow_end, (0, 0, 0), h//80, tipLength=0.15)
        self.im_dbg2_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.im_dbg2, encoding="bgr8"))

        # under some conditions, log sensor data
        # state, image, and depth image
        if self.state.pos[0] > self.data_collection_xrange[0] and self.state.pos[0] < self.data_collection_xrange[1]:
            self.log_data(log_expert=self.run_expert_in_parallel)
            self.plotted_commands = False
            self.expert_commands.append(self.expert_command)
            self.vision_commands.append(vision_command)
            self.spline_poss.append(spline_poss)
            self.spline_vels.append(spline_vels)
            self.state_vels.append(self.state.vel)
            self.state_poss.append(self.state.pos)

        # once the drone is beyond the collection range, save a plot of expert and vision commands
        if self.state.pos[0] > self.data_collection_xrange[1] and not self.plotted_commands and self.plot_cmd:

            print(f'[RUN_COMPETITION] Plotting commands...')

            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(3, 2, figsize=(8, 8))

            axs[0, 0].plot([pos[0] for pos in self.spline_poss], label='spline pos') if spline_poss is not None else None
            axs[0, 0].plot([pos[0] for pos in self.state_poss], label='state pos')
            axs[0, 0].set_ylabel(f"x pos")
            axs[0, 0].legend()
            axs[0, 0].grid()

            axs[1, 0].plot([pos[1] for pos in self.spline_poss], label='spline pos') if spline_poss is not None else None
            axs[1, 0].plot([pos[1] for pos in self.state_poss], label='state pos')
            axs[1, 0].set_ylabel(f"y pos")
            axs[1, 0].legend()
            axs[1, 0].grid()

            axs[2, 0].plot([pos[2] for pos in self.spline_poss], label='spline pos') if spline_poss is not None else None
            axs[2, 0].plot([pos[2] for pos in self.state_poss], label='state pos')
            axs[2, 0].set_ylabel(f"z pos")
            axs[2, 0].legend()
            axs[2, 0].grid()

            axs[0, 1].plot([cmd.velocity[0] for cmd in self.vision_commands], label='pred', marker='.') if self.vision_based else None
            axs[0, 1].plot([cmd.velocity[0] for cmd in self.expert_commands], label='cmd') if expert_command is not None else None
            axs[0, 1].plot([vel[0] for vel in self.spline_vels], label='spline vel') if spline_vels is not None else None
            axs[0, 1].plot([vel[0] for vel in self.state_vels], label='state vel')
            axs[0, 1].set_ylabel(f"x vel")
            axs[0, 1].legend()
            axs[0, 1].grid()

            axs[1, 1].plot([cmd.velocity[1] for cmd in self.vision_commands], label='pred', marker='.') if self.vision_based else None
            axs[1, 1].plot([cmd.velocity[1] for cmd in self.expert_commands], label='cmd') if expert_command is not None else None
            axs[1, 1].plot([vel[1] for vel in self.spline_vels], label='spline vel') if spline_vels is not None else None
            axs[1, 1].plot([vel[1] for vel in self.state_vels], label='state vel')
            axs[1, 1].set_ylabel(f"y vel")
            axs[1, 1].legend()
            axs[1, 1].grid()

            axs[2, 1].plot([cmd.velocity[2] for cmd in self.vision_commands], label='pred', marker='.') if self.vision_based else None
            axs[2, 1].plot([cmd.velocity[2] for cmd in self.expert_commands], label='cmd') if expert_command is not None else None
            axs[2, 1].plot([vel[2] for vel in self.spline_vels], label='spline vel') if spline_vels is not None else None
            axs[2, 1].plot([vel[2] for vel in self.state_vels], label='state vel')
            axs[2, 1].set_ylabel(f"z vel")
            axs[2, 1].legend()
            axs[2, 1].grid()

            fig.savefig(f"{self.folder}/cmd_plot.png")

            print(f'[RUN_COMPETITION] Saving plotted commands figure')

            # clear and delete fig
            plt.clf()
            plt.close(fig)

            print(f'[RUN_COMPETITION] Closed figure')

            self.plotted_commands = True

        # save collected evims
        if self.state.pos[0] > self.data_collection_xrange[1] and self.save_events and not self.saved_events:
            print(f'[RUN_COMPETITION] Saving evims as npy file')
            np.save(f"{self.folder}/evims.npy", self.evims)
            print(f'Saving evims to {self.folder}/evims.npy done')
            self.saved_events = True

        # save collected im_dbg2s
        if self.state.pos[0] > self.data_collection_xrange[1] and self.save_im_dbg2 and not self.saved_im_dbg2:
            print(f'[RUN_COMPETITION] Saving im_dbg2s as npy file')
            np.save(f"{self.folder}/im_dbg2s.npy", self.im_dbg2s)
            print(f'Saving im_dbg2s to {self.folder}/im_dbg2s.npy done')
            self.saved_im_dbg2 = True

    #### END OBSERVATION_CALLBACK

    def log_data(self, log_expert=False):

        # get the current time stamp
        # NOTE use image timestamp since this is important for calculating events
        # and we are using approximate time syncing
        timestamp = np.round(self.im_t, 3)

        data_entry = [
                        timestamp,
                        self.desiredVel,
                        self.state.att[0],
                        self.state.att[1],
                        self.state.att[2],
                        self.state.att[3],
                        self.state.pos[0],
                        self.state.pos[1],
                        self.state.pos[2],
                        self.state.vel[0],
                        self.state.vel[1],
                        self.state.vel[2],
                        self.command.velocity[0] if not log_expert else self.expert_command.velocity[0],
                        self.command.velocity[1] if not log_expert else self.expert_command.velocity[1],
                        self.command.velocity[2] if not log_expert else self.expert_command.velocity[2],
                        self.current_cmd_controller.collective_thrust,
                        self.current_cmd_controller.bodyrates.x,
                        self.current_cmd_controller.bodyrates.y,
                        self.current_cmd_controller.bodyrates.z,
                        self.col,
                        ]

        self.data_buffer = self.data_buffer.append(pd.Series(data_entry, index=self.data_buffer.columns), ignore_index=True)

        # append data to csv file every data_buffer_maxlength entries
        if len(self.data_buffer) >= self.data_buffer_maxlength:
            self.data_buffer.to_csv(opj(self.folder, 'data.csv'), mode='a', header=not self.first_data_write, index=True)
            self.data_buffer = pd.DataFrame(self.data_format)
            self.first_data_write = True

        # write images every log call
        cv2.imwrite(f"{self.folder}/{timestamp:.3f}_im.png", (self.im*255).astype(np.uint8))
        cv2.imwrite(f"{self.folder}/{timestamp:.3f}_depth.png", (self.depth*255).astype(np.uint8))

        if self.save_events and not self.saved_events:
            self.evims.append(self.events)

        if self.save_im_dbg2 and not self.saved_im_dbg2:
            self.im_dbg2s.append(self.im_dbg2)

    def fix_corrupted_depth(self, depth_image, neighbors=5):
        corrupted_indices = np.where(depth_image == 0.0)
        if len(corrupted_indices) == 0:
            return depth_image
        
        # Iterate through each corrupted pixel
        for i in range(len(corrupted_indices[0])):
            row, col = corrupted_indices[0][i], corrupted_indices[1][i]

            # Extract the neighborhood around the corrupted pixel
            neighborhood = depth_image[max(0, row - neighbors):min(depth_image.shape[0], row + neighbors + 1),
                                       max(0, col - neighbors):min(depth_image.shape[1], col + neighbors + 1)]

            # Exclude the corrupted pixel itself (center of the neighborhood)
            neighborhood = neighborhood[neighborhood != 0.0]

            # Interpolate the corrupted pixel value as the mean of its neighbors
            interpolated_value = np.mean(neighborhood)

            # Assign the interpolated value to the corrupted pixel
            depth_image[row, col] = interpolated_value

        return depth_image

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)
        try:
            self.col = self.if_collide(self.obs_msg.obstacles[0])
        except:
            self.col = False

    def im_callback(self, im_msg):

        # legacy
        # if self.image_w is None or self.image_h is None:
            # take these values from the config file instead
            # self.image_w = im_msg.width
            # self.image_h = im_msg.height
        if self.gimbal is None and self.use_gimbal:
            self.gimbal = Gimbal(self.gimbal_fov, self.image_w, self.image_h)

        try:
            im = self.cv_bridge.imgmsg_to_cv2(im_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("[IM_CALLBACK] CvBridge Error: {0}".format(e))
            return -1
        
        self.im_ctr += 1
        self.im_t = im_msg.header.stamp.to_nsec() / 1e9 # float with 9 digits past decimal

        # for rgb images, convert to normalized single channel,
        # preferably in the same way as Vid2E
        if len(im.shape) == 3 or im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = im.astype(np.float32)/255.0

        # save image
        self.prev_im = self.im
        self.im = im

        # legacy
        # # compute gimbaled images and save
        # self.prev_im_gimbal = self.im_gimbal
        # q = np.array([self.state.att[0], self.state.att[1], self.state.att[2], self.state.att[3]])
        # self.im_gimbal, self.pts = self.gimbal.do_gimbal(self.im, q, self.gimbal_w, self.gimbal_h, do_clip=True)

        if self.do_events:
            # compute event batch
            self.compute_events(gimbal=self.use_gimbal)

        return 0

    def depth_callback(self, depth_msg):

        if self.image_w is None or self.image_h is None:
            self.image_w = depth_msg.width
            self.image_h = depth_msg.height
            if self.gimbal is None and self.use_gimbal:
                self.gimbal = Gimbal(self.gimbal_fov, self.image_w, self.image_h)

        try:
            im = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except CvBridgeError as e:
            rospy.logerr("[DEPTH_CALLBACK] CvBridge Error: {0}".format(e))
            return -1
        
        self.depth_t = depth_msg.header.stamp.to_nsec()/1e9
        im = np.clip(im / self.depth_im_threshold, 0, 1)

        self.depth = self.fix_corrupted_depth(im)
        
        # legacy
        # # compute gimbaled images and save
        # q = np.array([self.state.att[0], self.state.att[1], self.state.att[2], self.state.att[3]])
        # self.depth_gimbal, self.pts = self.gimbal.do_gimbal(self.depth, q, self.gimbal_w, self.gimbal_h, do_clip=True)

        return 0

    def obstacle_callback(self, obs_data):
        self.obs_msg = obs_data

    def if_collide(self, obs):
        """
        Borrowed and modified from evaluation_node
        """

        if self.is_trees:
            dist = np.linalg.norm(np.array([obs.position.x, obs.position.y]))
        else:
            dist = np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z]))
        # margin is distance to object center minus object radius minus drone radius (estimated)
        margin = dist - obs.scale - self.quad_radius
        # Ground hit condition
        if margin < 0 or self.state.pos[2] <= 0.1:
            return True
        else:
            return False

    def publish_command(self, command):
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
                return
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        print("[RUN_COMPETITION] Start publishing commands!")
        self.publish_commands = True

    def create_obstacle(self):
        # Create an obstacle with specified position and scale
        obs = Obstacle()
        obs.position = Vector3(1000, 1000, 1000)
        obs.scale = 0.5
        return obs

    def create_obstacle_array(self):
        # Create an ObstacleArray message
        obs_array = ObstacleArray()
        obs_array.header = Header()
        obs_array.header.stamp = rospy.Time.now()
        obs_array.t = rospy.get_time()  # Current time as float64
        obs_array.num = 10  # Number of obstacles

        # Create 10 obstacles and add to the obstacle array
        for _ in range(10):
            obs = self.create_obstacle()
            obs_array.obstacles.append(obs)

        return obs_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot.")
    parser.add_argument("--vision_based", help="Fly vision-based", required=False, dest="vision_based", action="store_true")
    parser.add_argument("--ppo_path", help="PPO neural network policy", required=False, default=None)
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='string matching model name in lstmArch.py')
    parser.add_argument('--model_path', nargs='+', type=str, default=None, help='list of absolute paths to model checkpoints (multiple entries means multi-part model)')
    parser.add_argument('--num_recurrent', type=int, default=None, help='number of lstm layers, needs to be passed in for some models like LSTMNetwFC')
    parser.add_argument("--keyboard", help="Fly state-based mode but take velocity commands from keyboard WASD", required=False, dest="keyboard", action="store_true")
    parser.add_argument("--planner", help="Fly state-based mode but calculate use a path planner and follow it with a custom controller", required=False, dest="planner", action="store_true")
    parser.add_argument('--exp_name', type=str, default='', help='string to call current experiment')
    parser.add_argument('--total_num_exps', type=int, default=1, help='total number of experiments')

    args = parser.parse_args()
    print(f'[RUN_COMPETITION] args: {args}')

    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, ppo_path=args.ppo_path, model_type=args.model_type, model_path=args.model_path, num_recurrent=args.num_recurrent, keyboard=args.keyboard, use_planner=False, exp_name=args.exp_name, total_num_exps=args.total_num_exps)

    rospy.spin()
