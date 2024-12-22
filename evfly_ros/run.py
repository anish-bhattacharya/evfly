#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
# from dvs_msgs.msg import Event, EventArray
from cv_bridge import CvBridge
import cv2
import sys, os
import time
from std_msgs.msg import UInt8MultiArray
import torch

import getpass
uname = getpass.getuser()

EVFLY_PATH = f'/home/{uname}/evfly_ws/src/evfly'
CALIB_FILE = '/utils/calib_7-28/K.yaml'

sys.path.append(EVFLY_PATH+'/utils')
from ev_utils import simple_evim
from calibration_tools.rectify_bag import Aligner

sys.path.append(EVFLY_PATH+'/learner')
from learner import argparsing
from learner_models import *
import vitfly_models

class ImageSubscriberNode:
    def __init__(self):
        rospy.init_node('evfly_ros') #, anonymous=True)

        self.des_fwd_vel = 1.0 # forward velocity [m/s]
        self.dodge_scaler = 2.0 # scaling factor for lateral dodging
        self.ramp_duration = 3.0 # duration of forward speed ramp-up [s]
        self.des_z = 0.8 # desired z height [m]

        self.evcam_height, self.evcam_width = 480, 640
        self.evs_min_cutoff = 0.15 # lower percentage of events to remove from event frame, helps with noisy textures in the real world
        self.rate_hz = 15 # rate at which to run node

        # positional safety guard; drone will issue velocity commands 0 if it goes out of these bounds
        self.x_range = [-2, 50]
        self.y_range = [-20, 20]
        self.z_range = [-2, 3]

        self.print_debug = False

        self.images = []
        self.evframe = None
        self.cv_bridge = CvBridge()

        self.safety_guard_triggered = False

        ### INITIALIZE MODEL

        # parse args
        self.args = argparsing(filename=EVFLY_PATH+'/learner/configs/config.txt')

        print()
        print(self.args)
        print()

        # optional aligning of evframe
        self.align_evframe = self.args.align_evframe
        if self.align_evframe:
            print("[EVFLY_ROS RUN INIT] Aligning evframes!")
            self.aligner = Aligner(calib_file=EVFLY_PATH+CALIB_FILE)

        # define enc and dec params
        # make dictionaries enc_params and dec_params with the above args
        self.device = self.args.device
        self.enc_params = {
            'num_layers': self.args.enc_num_layers,
            'kernel_sizes': self.args.enc_kernel_sizes,
            'kernel_strides': self.args.enc_kernel_strides,
            'out_channels': self.args.enc_out_channels,
            'activations': self.args.enc_activations,
            'pool_type': self.args.enc_pool_type,
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

        # define our model
        if self.args.model_type == 'OrigUNet' or (isinstance(self.args.model_type, list) and len(self.args.model_type) == 1 and self.args.model_type[0] == 'OrigUNet'):
            self.model = OrigUNet(
                num_in_channels=self.args.num_in_channels, 
                num_out_channels=self.args.num_out_channels, 
                num_recurrent=self.args.num_recurrent, 
                input_shape=[1, 1, self.args.resize_input[0], self.args.resize_input[1]], 
                velpred=self.args.velpred, 
                enc_params=self.enc_params, 
                fc_params=self.fc_params,
                device=self.device,
                is_deployment=False,
                form_BEV=self.args.bev, 
                evs_min_cutoff=self.evs_min_cutoff,
                skip_type=self.args.skip_type
                ).to(self.device).float()

        elif isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'VITFLY_ViTLSTM':
            print(f'[EVFLY_ROS RUN INIT] Creating model of type {self.args.model_type[0]} and {self.args.model_type[1]}')
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
                evs_min_cutoff=self.evs_min_cutoff, 
                skip_type=self.args.skip_type, 
                is_deployment=False,
                ).to(self.device).float()

        else:
            print(f"Model type {self.args.model_type} not recognized!")
            exit()

        # print number of parameters in model
        print(f'[evfly_ros INIT] Model {self.args.model_type} loaded with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.')

        ### LOAD FROM CHECKPOINT PTH

        print("[EVFLY_ROS RUN INIT] Loading from checkpoints.")

        if len(self.args.checkpoint_path) > 1:

            if self.args.combine_checkpoints:

                print(f'[SETUP] Combining checkpoints {self.args.checkpoint_path} into a single state dict')
                self.model.load_state_dict(self.combine_state_dicts([torch.load(cp, map_location=self.device) for cp in self.args.checkpoint_path], model_names=['origunet', 'vitfly_vitlstm']))

            else:
                
                if isinstance(self.args.model_type, list) and self.args.model_type[0] == 'OrigUNet' and self.args.model_type[1] == 'VITFLY_ViTLSTM':

                    self.model.origunet.load_state_dict(torch.load(self.args.checkpoint_path[0], map_location=self.device))
                    self.model.vitfly_vitlstm.load_state_dict(torch.load(self.args.checkpoint_path[1], map_location=self.device))

                else:
                    raise ValueError(f"[RUN_COMPETITION] model_type {self.args.model_type} not recognized for self.args.checkpoint_path list of length > 1. Exiting.")
        else:
            self.model.load_state_dict(torch.load(self.args.checkpoint_path[0], map_location=self.device))

        print("[EVFLY_ROS RUN INIT] Done loading from checkpoints.")

        self.model.eval()

        self.velpred_hidden_state = None
        self.origunet_hidden_state = None

        ### SUBSCRIBERS

        # events in the form of a uint8 array that is reshaped in the callback
        self.proc_evs = None
        self.image_subscriber = rospy.Subscriber('/output/proc_evs', UInt8MultiArray, self.image_callback, queue_size=1)

        # odometry
        self.odom = None
        self.odom_subscriber = rospy.Subscriber('/input_odom', Odometry, self.odom_callback, queue_size=1)

        # trigger signal
        self.last_trigger_t = 0.0
        self.first_trigger_t = None
        self.trigger_subscriber = rospy.Subscriber('/trigger', Empty, self.trigger_callback, queue_size=1)

        ### PUBLISHERS

        self.pred_depth = None
        self.pred_depth_publisher = rospy.Publisher('/output/pred_depth', Image, queue_size=1)
        self.pred_vel = None
        self.pred_vel_publisher = rospy.Publisher('/output/pred_vel', TwistStamped, queue_size=1)

        self.evim = None
        self.evim_publisher = rospy.Publisher('/output/evim', Image, queue_size=1)
        self.dbg_im = None
        self.dbg_im_publisher = rospy.Publisher('/output/dbg_im', Image, queue_size=1)

        self.vel_msg = TwistStamped()
        self.vel_msg.header.stamp = rospy.Time.now()
        self.vel_msg.twist.linear.x = 0.0
        self.vel_msg.twist.linear.y = 0.0
        self.vel_msg.twist.linear.z = 0.0

        self.vel_cmd_publisher = rospy.Publisher('/robot/cmd_vel', TwistStamped, queue_size=1)

        self.rate = rospy.Rate(self.rate_hz)  # 5Hz or 10Hz
 
        self.ct = 0

    def combine_state_dicts(self, state_dicts, model_names=None):
        """Combines multiple state dicts into a single state dict.
        Args:
            state_dicts: A list of state dicts.
            (optional) model_names: specifies prefix to prepend to each state dict's keys
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

    def trigger_callback(self, msg):
        if self.first_trigger_t is None:
            self.first_trigger_t = rospy.Time().now().to_sec()
        self.last_trigger_t = rospy.Time().now().to_sec()

    def odom_callback(self, msg):
        self.odom = msg

    def run_model(self):

        input_frame = torch.from_numpy(self.evframe).view(1, 1, self.args.resize_input[0], self.args.resize_input[1]).to(self.device).float()

        # percentile scaling
        evim_scaledown_factor = torch.quantile(input_frame.abs(), .97)
        if self.print_debug:
            print(f'evframe max/min = {input_frame.max():.3f}/{input_frame.min():.3f}, 97th percentile = {evim_scaledown_factor:.3f}')
        input_frame = torch.clip(input_frame/evim_scaledown_factor, -1.0, 1.0)

        desvel = torch.Tensor([[4.0]]).to(self.device)

        st_modelfwd_time = time.time()

        full_input = [input_frame, desvel, [self.origunet_hidden_state, None], self.velpred_hidden_state]

        with torch.no_grad():
            x_vel, (x_depth, _, ((self.origunet_hidden_state, _), self.velpred_hidden_state)) = self.model(full_input)

        if self.print_debug:
            print(f"model fwd took {time.time()-st_modelfwd_time:.3f}")

        self.pred_vel = x_vel.cpu().detach().numpy().squeeze() # keeping this 0-1
        self.pred_depth = x_depth.cpu().detach().numpy().squeeze() if x_depth is not None else None

        if self.print_debug:
            print(f'pred depth min/max = {self.pred_depth.min():.3f}/{self.pred_depth.max():.3f}, 90th percent = {np.percentile(self.pred_depth, .90):.3f}')

        self.dbg_im = self.pred_depth.copy()
        self.dbg_im = np.clip(self.dbg_im, 0.0, 1.0)
        self.dbg_im = (np.stack([self.dbg_im]*3, axis=2)*255).astype(np.uint8)
        # write arrow
        im_w, im_h = 346, 260
        arrow_start = (im_w//2, int(.8*im_h))
        arrow_end = (int(arrow_start[0]-self.pred_vel[1]*(im_h/2)), int(arrow_start[1]))
        arrow_color = (0, 0, 255)

        self.dbg_im = cv2.arrowedLine( self.dbg_im, arrow_start, arrow_end, arrow_color, im_h//100, tipLength=0.3)

    def publish_pred_depth(self):

        # convert to uint8
        pred_depth_clipped = np.clip(self.pred_depth, 0.0, 1.0)
        pred_depth_uint8 = (pred_depth_clipped * 255).astype(np.uint8)

        # convert to rosmsg
        pred_depth_rosmsg = self.cv_bridge.cv2_to_imgmsg(pred_depth_uint8)

        # publish
        pred_depth_rosmsg.header.stamp = rospy.Time.now()
        self.pred_depth_publisher.publish(pred_depth_rosmsg)

    def publish_pred_vel(self):

        self.pred_vel *= self.des_fwd_vel # scale up from 1m/s

        self.vel_msg = TwistStamped()
        self.vel_msg.header.stamp = rospy.Time.now()
        self.vel_msg.twist.linear.x = self.pred_vel[0]
        self.vel_msg.twist.linear.y = self.pred_vel[1] * self.dodge_scaler
        if self.odom is not None:
            self.vel_msg.twist.linear.z = 1.5 * (self.des_z - self.odom.pose.pose.position.z)
        else:
            self.vel_msg.twist.linear.z = 0.0
        self.pred_vel_publisher.publish(self.vel_msg)

    def publish_dbg_im(self):

        # convert to rosmsg
        dbg_im_rosmsg = self.cv_bridge.cv2_to_imgmsg(self.dbg_im, encoding="bgr8")

        # publish
        self.dbg_im_publisher.publish(dbg_im_rosmsg)

    def publish_evim(self):

        self.evim, _ = simple_evim(self.evframe, scaledown_percentile=97, style='redblue-on-white')
        self.evim = self.cv_bridge.cv2_to_imgmsg(self.evim, encoding="rgb8")
        self.evim_publisher.publish(self.evim)

    def image_callback(self, msg):

        self.proc_evs = np.frombuffer(msg.data, dtype=np.uint8)
        self.proc_evs = self.proc_evs.reshape(self.evcam_height, self.evcam_width)

    def evs_process(self):

        if self.proc_evs is not None:
            
            self.evframe = self.proc_evs.copy().astype(np.float32)
            self.evframe -= 128
            self.evframe *= 0.2

            # optionally align the event frame before running model on it
            if self.align_evframe:
                self.evframe = self.aligner.align(davis=self.evframe, depth=None)['davis']

            # model takes 260x346 input, so if the evframe is not of this shape then 
            # either center-crop or resize. if using a short focal length lens, center-cropping
            # is recommended.
            if self.evframe.shape[0] != 260 or self.evframe.shape[1] != 346:
                # we center-crop instead of resize
                # 480//2 - 260//2 : 480//2 + 260//2
                # 640//2 - 346//2 : 640//2 + 346//2
                self.evframe = self.evframe[480//2 - 260//2 : 480//2 + 260//2,
                                            640//2 - 346//2 : 640//2 + 346//2]
                # self.evframe = cv2.resize(self.evframe, dsize=(346, 260), interpolation=cv2.INTER_CUBIC)

            self.run_model()

            if self.pred_depth is not None:
                self.publish_pred_depth()
            if self.pred_vel is not None:
                self.publish_pred_vel()
            if self.evframe is not None:
                self.publish_evim()
            if self.dbg_im is not None:
                self.publish_dbg_im()
            if self.print_debug:
                print()

    def run(self):
        while not rospy.is_shutdown():
            
            self.evs_process()

            # if in safe range
            if self.odom is None or \
               (not self.safety_guard_triggered and \
               self.x_range[0] < self.odom.pose.pose.position.x < self.x_range[1] and \
               self.y_range[0] < self.odom.pose.pose.position.y < self.y_range[1] and \
               self.z_range[0] < self.odom.pose.pose.position.z < self.z_range[1]):

                if rospy.Time().now().to_sec() - self.last_trigger_t < 0.1:
                    print("commanding!")

                    # manual, discretized ramp-up; in first second of commands, reduce fwd/dodging vel by /2.0
                    ramp_duration = self.ramp_duration
                    if rospy.Time().now().to_sec() - self.first_trigger_t < ramp_duration:

                        ramp_time = rospy.Time().now().to_sec() - self.first_trigger_t
                        ramp_scaler = ramp_time / ramp_duration

                        self.vel_msg.twist.linear.x *= ramp_scaler
                        self.vel_msg.twist.linear.y *= ramp_scaler

                        self.vel_msg.twist.linear.x = max(min(1.0 + self.vel_msg.twist.linear.x, self.des_fwd_vel), 0.0)

                    self.vel_cmd_publisher.publish(self.vel_msg)

                else:

                    if self.last_trigger_t > 0.0:
                        
                        self.vel_msg.twist.linear.x = 0.0
                        self.vel_msg.twist.linear.y = 0.0
                        self.vel_msg.twist.linear.z = 0.0
                        self.vel_cmd_publisher.publish(self.vel_msg)

            # safety guard
            else:

                print("out of bounds! stopping.")
                self.safety_guard_triggered = True
                self.vel_msg.twist.linear.x = 0.0
                self.vel_msg.twist.linear.y = 0.0
                self.vel_msg.twist.linear.z = 0.0
                self.vel_cmd_publisher.publish(self.vel_msg)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        evfly_ros = ImageSubscriberNode()
        evfly_ros.run()
    except rospy.ROSInterruptException:
        pass
