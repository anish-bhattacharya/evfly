#!/usr/bin/python3

from utils import AgileCommandMode, AgileCommand
# from rl_example import rl_example

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

import glob, os, sys, time
from os.path import join as opj

from dataModify import Gimbal

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../learner'))
from learner_models import *

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
from ev_utils import *

# 3D line determined by two points (x1, y1, z1) and (x2, y2, z2)
# sphere determined by a center point (x3, y3, z3) and radius r
# quantity b^2 - 4ac < 0 then there is no intersection, where:
# b = 2*( (x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3) )
# a = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
# c = x3^2 + y3^2 + z3^2 + x1^2 + y1^2 + z1^2 - 2*(x3*x1 + y3*y1 + z3*z1) - r^2
# line is a 2-tuple of 3-tuples, obstacle is a 2-tuple of the center 3-tuple and the radius float
def check_collision(line, obstacle, is_tree=False):

    (x1, y1, z1), (x2, y2, z2) = line
    (x3, y3, z3), r = obstacle

    # this doesn't work
    # if is_tree:

    #     # 2D circle case
    #     a = y1 - y2
    #     b = x2 - x1
    #     c = (x1 - x2) * y1 + x1 * (y2 - y1)
    #     return np.abs(b * x3 + a * y3 + c) / np.sqrt(a**2 + (b)**2) <= r

    # else:

    # spoof tree occupancy by making a sphere's z value the same as the ego z value
    if is_tree:
        z3 = 0

    # 3D sphere case
    b = 2 * ((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    c = (
        x3**2
        + y3**2
        + z3**2
        + x1**2
        + y1**2
        + z1**2
        - 2 * (x3 * x1 + y3 * y1 + z3 * z1)
        - r**2
    )
    return b**2 - 4 * a * c >= 0

# helper function for vectorized expert policy (method_id = 1)
def find_closest_zero_index(arr):
    center = np.array(arr.shape) // 2  # find the center point of the array
    dist_to_center = np.abs(np.indices(arr.shape) - center.reshape(-1, 1, 1)).sum(0)  # calculate distance to center for each element
    zero_indices = np.argwhere(arr == 0)  # find indices of all zero elements
    if len(zero_indices) == 0:
        return None  # if no zero elements, return None
    dist_to_zeros = dist_to_center[tuple(zero_indices.T)]  # get distances to center for zero elements
    min_dist_indices = np.argwhere(dist_to_zeros == dist_to_zeros.min()).flatten()  # find indices of zero elements with minimum distance to center
    chosen_index = np.random.choice(min_dist_indices)  # randomly choose one of the zero elements with minimum distance to center
    return tuple(zero_indices[chosen_index])  # return index tuple

def compute_command_state_based(state, obstacles, desiredVel, rl_policy=None, keyboard=False, keyboard_input='', splines=None, planner_start_time=None, is_trees=False):
    # print("Computing command based on obstacle information!")
    # print("Obstacles: ", obstacles)

    """
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 10.0
    command.bodyrates = [0.0, 0.0, 0.0]
    """

    # LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.yawrate = 0.0

    obst_dist_threshold = 10
    obst_inflate_factor = 1.0
    method_id = 1 # 0 = old spiral method, 1 = new re-factored, 2 = constant, 3 = keyboard, 4 = path planning
    if keyboard:
        import select
        method_id = 3

    # calculate an obstacle-free waypoint
    x_displacement = 8
    grid_center_offset = 8
    grid_displacement = .5 # 2.5
    num_wpts = np.arange(-grid_center_offset, grid_center_offset + grid_displacement, grid_displacement).size

    start = time.time()

    # useful debug information
    collisions = None
    wpt_idx = None
    spline_poss = None
    spline_vels = None

    # old expert
    if method_id == 0:

        wpts_2d = np.zeros((num_wpts, num_wpts, 2))
        for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
                wpts_2d[yi, xi] = [x, y]

        # the first layer of wpts_2d is actually the world y axis, the second is z axis
        # the third, the x axis, should all be +5m forward
        x_slice = x_displacement * np.ones((num_wpts, num_wpts))
        wpts_2d = np.concatenate((x_slice[:, :, None], wpts_2d), axis=2)

        # try spiraling outward again but just using bounds instead, and selecting blocks
        idx_midpt = num_wpts // 2
        curr_x = idx_midpt
        curr_y = idx_midpt
        x_bound = 1
        y_bound = -1
        wpt_idxs_2d = []
        count = 0
        while curr_x < num_wpts:
            if count % 4 == 0:
                x_bound = count / 4 + 1
            if (count - 1) % 4 == 0:
                y_bound = -((count - 1) / 4 + 1)

            if not count % 2:  # x-dir vector
                xvals = np.arange(
                    curr_x, idx_midpt + x_bound, -1 if x_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(np.repeat(int(curr_y), xvals.size), xvals)
                ]
                curr_x = idx_midpt + x_bound
                x_bound *= -1
            else:  # y-dir vector
                yvals = np.arange(
                    curr_y, idx_midpt + y_bound, -1 if y_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(yvals, np.repeat(int(curr_x), yvals.size))
                ]
                curr_y = idx_midpt + y_bound
                y_bound *= -1

            count += 1

        # iterate through waypoints, spiraling outwards from center
        for wpt_idx in wpt_idxs_2d:
            found_valid_pt = True
            # check if the current wpt is valid for all obstacles ahead of our current position
            for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.scale > 0.01 and obst.position.x < obst_dist_threshold]:
                if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                    found_valid_pt = False
                    break
            if found_valid_pt:
                break

        # CHECK AGAIN WITH OBSTACLE SCALE REDUCED TO .17
        if not found_valid_pt:
            print("[EXPERT] Didn't find a feasible path, Searching again with less inflation!")
            for wpt_idx in wpt_idxs_2d:
                found_valid_pt = True
                # check if the current wpt is valid for all obstacles ahead of our current position
                for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                    if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+0.17)):
                        found_valid_pt = False
                        break
                if found_valid_pt:
                    break
        
        # simplest controller: waypoint --PID--> linear velocity command
        yvel = 1.25 * (wpts_2d[wpt_idx][1])
        # x_scale_down_factor = (grid_center_offset - np.abs(yvel))/grid_center_offset
        xvel = max(desiredVel, 1 * (wpts_2d[wpt_idx][0]))
        zvel = 1.25 * wpts_2d[wpt_idx][2]

    # new expert
    elif method_id == 1:

        num_wpts_x = num_wpts
        x_grid = np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)
        if is_trees:
            num_wpts_y = 1
            y_grid = np.array([0])
        else:
            num_wpts_y = num_wpts
            y_grid = np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)

        wpts_2d = np.zeros((num_wpts_y, num_wpts_x, 3))
        collisions = np.zeros((num_wpts_y, num_wpts_x))

        for xi, x in enumerate(x_grid):

            for yi, y in enumerate(y_grid):

                # print(f'[USER CODE] xi: {xi}, yi: {yi}, x: {x}, y: {y}')

                wpts_2d[yi, xi] = [x_displacement, x, y]
                for obst in [obst for obst in obstacles.obstacles if obst.position.x + (obst.scale + obst_inflate_factor) > 0 and obst.position.x - (obst.scale + obst_inflate_factor) < obst_dist_threshold]:

                    # print(f'wpt: {wpts_2d[yi, xi]} \t obst: {obst.position.x, obst.position.y, obst.position.z, obst.scale+obst_inflate_factor}')

                    if check_collision(((0, 0, 0), (wpts_2d[yi, xi])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor), is_tree=is_trees):
                        collisions[yi, xi] = 1
                        # print(f'COLLISION wpt = {wpts_2d[yi, xi]} obst x distance = {obst.position.x - (obst.scale + obst_inflate_factor)}')
                        break

        if collisions.sum() == collisions.size:

            # print(f'[EXPERT] No collision-free path found')
            xvel, yvel, zvel = (desiredVel, 0., 0.)

        else:

            wpt_idx = find_closest_zero_index(collisions)
            wpt = wpts_2d[wpt_idx[0], wpt_idx[1]]

            # make the desired velocity vector of magnitude desiredVel
            wpt = wpt / np.linalg.norm(wpt) * desiredVel
            xvel = wpt[0]
            yvel = wpt[1]
            zvel = wpt[2]

    # just fly forward
    elif method_id == 2:

        xvel, yvel, zvel = (4.0, 0., 0.)

    elif method_id == 3:

        xvel, yvel, zvel = (2., 0., 0.)

        # print(f'[EXPERT] Keyboard input: {keyboard_input}')

        # Check if there is any keypress
        if keyboard_input == 'w':
            zvel = 1.0
        elif keyboard_input == 's':
            zvel = -1.0
        elif keyboard_input == 'a':
            yvel = 1.0
        elif keyboard_input == 'd':
            yvel = -1.0

        # norm the command vector up to desiredVel
        scaler = desiredVel/np.linalg.norm([xvel, yvel, zvel])
        xvel, yvel, zvel = (xvel*scaler, yvel*scaler, zvel*scaler)

    elif method_id == 4:

        if splines is None:
            
            print('[EXPERT] No spline provided for path planning')
            xvel, yvel, zvel = (desiredVel, 0., 0.)
        
        else:

            # print('[EXPERT] Using spline for path planning')

            # don't start trajectory until we pass some x position threshold
            # reset start_time until we are ready
            if state.pos[0] < splines[0](0.0) + 0.2:

                xvel, yvel, zvel = (desiredVel, 0., 0.)

                # # get current time
                # start_time = state.t

                # print(f'[EXPERT] Waiting to start trajectory until altitude is close to {splines[2](0.0)} (start_time = {start_time:.2f}))')

            else:

                curr_time = state.t - planner_start_time

                # # directly get first derivative of position splines which is velocity
                # xvel = splines[0](curr_time, 1)
                # yvel = splines[1](curr_time, 1)
                # zvel = splines[2](curr_time, 1)

                # p gain on position error
                xvel = 1.0 * (splines[0](curr_time) - state.pos[0])
                yvel = 1.0 * (splines[1](curr_time) - state.pos[1])
                zvel = 1.0 * (splines[2](curr_time) - state.pos[2])

                spline_poss = np.array([splines[0](curr_time), splines[1](curr_time), splines[2](curr_time)])
                spline_vels = np.array([splines[0](curr_time, 1), splines[1](curr_time, 1), splines[2](curr_time, 1)])

                # print(f'[EXPERT] xvel: {xvel}, yvel: {yvel}, zvel: {zvel} at curr_time: {curr_time:.2f}')

    if time.time() - int(time.time()) < 0.1: # print this as infrequently as possible
        print(f'[EXPERT] Expert method {method_id} took {time.time() - start:.3f} seconds')

    command.velocity = [xvel, yvel, zvel]

    # recover altitude if too low
    if state.pos[2] < 1:
        command.velocity[2] = (2 - state.pos[2]) * 2

    # # manual speedup
    # min_xvel_cmd = 1.0
    # hardcoded_ctl_threshold = 2.0
    # if state.pos[0] < hardcoded_ctl_threshold:
    #     command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)

    ################################################
    # !!! End !!!
    ################################################

    # If you want to test your RL policy
    if rl_policy is not None:
        command = rl_example(state, obstacles, rl_policy)

    extras = {'collisions': collisions,
              'wpt_idx': wpt_idx,
              'spline_poss': spline_poss,
              'spline_vels': spline_vels}

    return command, extras
