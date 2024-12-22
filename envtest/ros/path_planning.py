# path planner for trajectory generation through a known obstacle field
# use graph traversal to find a path through the obstacle field,
# not necessarily optimal

import numpy as np
import matplotlib.pyplot as plt
from read_obst_info import extract_objects_from_csv
from scipy.interpolate import CubicSpline

class Planner:

    # initialize planner
    def __init__(self, ranges=None, discretization=None, obst_inflation_factor=None):

        # parameters
        self.x_range = [0, 60] if ranges is None else ranges[0]
        self.y_range = [-20, 20] if ranges is None else ranges[1]
        self.z_range = [0, 20] if ranges is None else ranges[2]
        self.discretization = 1.0 if discretization is None else discretization
        self.obst_inflation_factor = 0.3 if obst_inflation_factor is None else obst_inflation_factor

        # define obstacle field
        self.map = np.zeros( (int((self.x_range[1]-self.x_range[0])/self.discretization) + 1,
                              int((self.y_range[1]-self.y_range[0])/self.discretization) + 1, 
                              int((self.z_range[1]-self.z_range[0])/self.discretization) + 1) )
        self.map_positions = np.zeros((*self.map.shape, 3))
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                for k in range(self.map.shape[2]):
                    self.map_positions[i,j,k] = np.array([i,j,k]) * self.discretization + np.array([self.x_range[0], self.y_range[0], self.z_range[0]])

        print(f'[Planner] Map defined of shape {self.map.shape} with discretization {self.discretization} and ranges {self.x_range}, {self.y_range}, {self.z_range}')
        return

    # check if point is in collision with given obstacle
    def is_in_collision(self, p, obstacle):
        obstacle_center = np.array([*obstacle[:3]])
        obstacle_radius = np.array(obstacle[3]) + self.obst_inflation_factor
        collision = np.logical_and( np.logical_and( np.abs(p[0] - obstacle_center[0]) <= obstacle_radius[0] , np.abs(p[1] - obstacle_center[1]) <= obstacle_radius[1] ) , np.abs(p[2] - obstacle_center[2]) <= obstacle_radius[2] )
        return collision

    # assuming obstacles are in the form of a list of tuples (x, y, z, radius)
    # fill map with inflated obstacles by setting corresponding map locations to 1
    def fill_map(self, obstacles):
        for obstacle in obstacles:
            obstacle_center = np.array([*obstacle[:3]])
            obstacle_radius = np.array(obstacle[3]) + self.obst_inflation_factor
            # use map_positions to find all idxs of points inside obstacle
            # use equation of ellipsoid given 3-element radius
            idxs = np.where( np.logical_and( np.logical_and( np.abs(self.map_positions[:, :, :, 0] - obstacle_center[None, None, None, 0]) <= obstacle_radius[0] , np.abs(self.map_positions[:, :, :, 1] - obstacle_center[None, None, None, 1]) <= obstacle_radius[1] ) , np.abs(self.map_positions[:, :, :, 2] - obstacle_center[None, None, None, 2]) <= obstacle_radius[2] ) )
            self.map[idxs] = 1

        print(f'[Planner] Number of obstacles: {len(obstacles)}')
        print(f'[Planner] Number of occupied points in map: {int(np.sum(self.map))}/{np.prod(self.map.shape)} ({100*np.sum(self.map)/np.prod(self.map.shape):.2f}%)')
        return

    # initialize matplotlib fig, ax for visualization
    def init_visualization(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # set axis limits
        self.ax.set_xlim3d(self.x_range[0], self.x_range[1])
        self.ax.set_ylim3d(self.y_range[0], self.y_range[1])
        self.ax.set_zlim3d(self.z_range[0], self.z_range[1])
        # set axes aspect ratio equal
        self.ax.set_aspect('auto')
        # flip y axis
        # self.ax.invert_yaxis()
        # set viewing angle
        self.ax.view_init(elev=87, azim=-180)
        return

    # visualize 3D map with scatterplot of points,
    # free space are small blue points,
    # obstacles are large red points
    def plot_map(self):
        self.ax.scatter(*self.map_positions[self.map == 0].T, s=1, color='blue', label='Free Space')
        self.ax.scatter(*self.map_positions[self.map == 1].T, s=1, color='red', label='Obstacles')
        return

    # visualize obstacles
    def plot_obstacles(self, obstacles, resolution=10):
        for obstacle in obstacles:
            center = obstacle[:3]
            radius = obstacle[3]
            # Create a mesh grid for the sphere's surface
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            x = center[0] + radius[0] * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius[1] * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius[2] * np.outer(np.ones(np.size(u)), np.cos(v))
            # Plot the sphere
            self.ax.plot_surface(x, y, z, color='b', alpha=0.6)
        return

    # visualize path
    def plot_path(self, path):
        self.ax.plot(*np.array(path).T, color='green', label='Path')
        self.ax.legend()

    # return index of map position closest to p
    def idx_map(self, p):
        # if p is not a ndarray make it one
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        idx = np.unravel_index(np.argmin(np.linalg.norm(self.map_positions - p[None, None, None, :], axis=-1)), self.map_positions.shape[:3])
        return idx

    # query information from a point in the map, typically whether a point is in collision
    def query_map(self, p):
        idx = self.idx_map(p)
        # check map position at idx matches p
        # does not consider floats right now
        # assert np.allclose(self.map_positions[idx], p), f'p={p} does not match map_positions[idx]={self.map_positions[idx]}'
        return self.map[idx]

    # check if point is within map bounds
    def is_valid_point(self, p):
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        return np.all(p >= np.array([self.x_range[0], self.y_range[0], self.z_range[0]])) and np.all(p <= np.array([self.x_range[1], self.y_range[1], self.z_range[1]]))

    # return index of obstacle that point is in collision with
    def query_obstacles(self, p, obstacles):
        if self.query_map(p) == 0:
            print(f'[Planner] Querying obstacle list, but point {p} is not in collision! Returning -1.')
        found_obstacles = []
        for i, obstacle in enumerate(obstacles):
            obstacle_center = np.array([*obstacle[:3]])
            obstacle_radius = np.array(obstacle[3]) + self.obst_inflation_factor
            if self.is_in_collision(p, obstacle):
                found_obstacles.append(i)
        return found_obstacles

    # find path from start to end
    # for now this is not the optimal path
    # also for now, operate in 2D and consider any x >= end[0] a success
    def find_path(self, start, end):
        path = []

        if self.query_map(start) == 1:
            print('Start or end position is in collision! Returning -1.')
            return -1, None

        # get discretized start and end positions
        start = self.map_positions[self.idx_map(start)]
        end = self.map_positions[self.idx_map(end)]

        path.append(start)

        relevant_obstacles = []

        # move in +x direction until we reach desired x distance
        while path[-1][0] < end[0]:
            
            # get next point
            next_point = path[-1] + np.array([self.discretization, 0, 0])

            # check if next point is in collision
            if self.query_map(next_point) == 1:

                relevant_obstacles.extend(self.query_obstacles(next_point, self.obstacles))

                # pop last point from path
                path.pop()

                # search in +/- y direction for a free point
                left_point = next_point + np.array([0, self.discretization, 0])
                while self.is_valid_point(left_point) and self.query_map(left_point) == 1:
                    left_point[1] += self.discretization
                right_point = next_point + np.array([0, -self.discretization, 0])
                while self.is_valid_point(right_point) and self.query_map(right_point) == 1:
                    right_point[1] -= self.discretization
                
                # check if we found a free point
                if not self.is_valid_point(left_point) and not self.is_valid_point(right_point):
                    print('No path found (left/right search out of bounds)! Returning -1.')
                    return -1, None
                
                # check if left or right point is closer to next_point
                if not self.is_valid_point(left_point):
                    next_point = right_point
                elif not self.is_valid_point(right_point):
                    next_point = left_point
                elif np.linalg.norm(next_point - left_point) < np.linalg.norm(next_point - right_point):
                    next_point = left_point
                else:
                    next_point = right_point

            # add next point to path
            path.append(next_point)

        return path, relevant_obstacles
    
    # fit a cubic spline to a set of points in a path
    def fit_spline(self, points, velocity=1.0):

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # find timesteps
        timesteps = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1) / velocity)
        timesteps = np.insert(timesteps, 0, 0.0)

        print(f'[Planner] With given velocity {velocity}, timesteps ranges from {timesteps[0]:.2f} to {timesteps[-1]:.2f}.')

        # fit spline to each dimension
        bcs = [((1, 0.0), (1, velocity)), ((1, 0.0), (1, 0.0)), ((1, 0.0), (1, 0.0))]
        splines = [CubicSpline(timesteps, points[:,i], bc_type=bcs[i]) for i in range(points.shape[1])]

        print(f'[Planner] Fitted splines to {len(splines)} dimensions.')

        return splines, timesteps

    # plot spline
    def plot_splines(self, splines, timesteps):
        t_values = np.linspace(timesteps[0], timesteps[-1], 100)
        spline_points = np.array([[spl(t) for spl in splines] for t in t_values])
        self.ax.plot(*spline_points.T, color='red', label='Spline')
        self.ax.legend()
        return

    # given filled map calculate path and fit spline
    def calculate_path_and_spline(self, start, end, velocity=1.0):

        if not isinstance(start, np.ndarray):
            start = np.array(start)
        if not isinstance(end, np.ndarray):
            end = np.array(end)

        self.path, self.relevant_obstacles = self.find_path(start, end)
        if self.path == -1:
            print('No valid path found! Setting to reference trajectory.')
            self.path = [start.tolist(), end.tolist()]
        print(f'[Planner] Avoided collisions with {len(self.relevant_obstacles)} obstacles.')
        self.splines, self.ts = self.fit_spline(self.path, velocity)
        return

    # fill map with obstacles from a csv file
    def input_obstacles_from_csv(self, csv_file):
        self.obstacles = extract_objects_from_csv(csv_file)
        self.fill_map(self.obstacles)
        return

    # given multidimensional spline, plot derivate of spline over time in individual subplots against time
    def plot_spline_derivatives(self, splines, timesteps):
        fig, axs = plt.subplots(len(splines), 1)
        fig.suptitle('Spline Derivatives')
        for i in range(len(splines)):
            axs[i].plot(timesteps, splines[i](timesteps, 1), label='1st Derivative')
            # axs[i].plot(timesteps, splines[i](timesteps, 2), label='2nd Derivative')
            axs[i].legend()
        return

if __name__ == '__main__':

    # define planner
    planner = Planner(ranges=[[0, 60], [-20, 20], [0, 20]], discretization=1.0, obst_inflation_factor=0.3)

    # define start and end points
    velocity = 4.0
    start = np.array([0, 0, 3])
    end = np.array([60, 0, 3])

    # define obstacles
    # obstacles = [(10, 1, 3, 2), (20, -3, 3, 2)] #, (30, 0, 5, 2), (40, 0, 5, 2), (50, 0, 5, 2)]
    csv_file = '/home/anish/evfly_ws/src/evfly/flightmare/flightpy/configs/vision/medium_trees/environment_0/static_obstacles.csv'
    planner.obstacles = extract_objects_from_csv(csv_file)

    planner.fill_map(planner.obstacles)

    # visualize map
    # planner.plot_map()

    # find path
    path, relevant_obstacles = planner.find_path(start, end)
    if path == -1:
        print('No path found! Exiting.')
        exit()
    print(f'[Planner] Avoided collisions with {len(relevant_obstacles)} obstacles.')
    # print(path)

    # visualization
    planner.init_visualization()

    # # plot obstacles that have center points between certain z values
    # obstacles_to_plot = [obstacle for obstacle in obstacles if obstacle[2] >= start[2]-2 and obstacle[2] <= start[2]+2]

    planner.plot_obstacles([planner.obstacles[obst_i] for obst_i in relevant_obstacles])
    # planner.plot_obstacles(planner.obstacles)

    # visualize path
    planner.plot_path(path)
    # plt.show()

    # fit spline to path
    splines, ts = planner.fit_spline(path, velocity)
    planner.plot_splines(splines, ts)
    plt.show()

    # # plot spline derivatives
    # planner.plot_spline_derivatives(splines, np.linspace(ts[0], ts[-1], 100))
    # plt.show()

    # # min snap trajectory
    # from mistgen.mist import mist_generator
    # mg = mist_generator()
    # mst_xs, mst_ys, mst_ts = mg.mist_2d_gen(np.array(path).T, np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 10)
    # mg_vaj_xy = mg.mist_2d_vaj_gen(mst_xs, mst_ys, mst_ts)
    # mg.mist_2d_vis(np.array(path).T, mst_xs, mst_ys, mst_ts, mg_vaj_xy, True, True, True)
    # plt.show()

    # print(f'xs.shape = {mst_xs.shape}')
    # print(f'ys.shape = {mst_ys.shape}')
    # print(f'ts.shape = {mst_ts.shape}')
    # print('====================')
    # print(f'xs = {mst_xs}')
    # print(f'ys = {mst_ys}')
    # print(f'ts = {mst_ts}')



