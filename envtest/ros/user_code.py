#!/usr/bin/python3

from utils import AgileCommand

import numpy as np

import glob, os, sys, time
from os.path import join as opj

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
    method_id = 1 # 1 = new re-factored, 2 = constant
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

    # refactored waypoint search expert
    if method_id == 1:

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

                wpts_2d[yi, xi] = [x_displacement, x, y]
                for obst in [obst for obst in obstacles.obstacles if obst.position.x + (obst.scale + obst_inflate_factor) > 0 and obst.position.x - (obst.scale + obst_inflate_factor) < obst_dist_threshold]:

                    if check_collision(((0, 0, 0), (wpts_2d[yi, xi])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor), is_tree=is_trees):
                        collisions[yi, xi] = 1
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

    extras = {'collisions': collisions,
              'wpt_idx': wpt_idx,
              'spline_poss': spline_poss,
              'spline_vels': spline_vels}

    return command, extras
