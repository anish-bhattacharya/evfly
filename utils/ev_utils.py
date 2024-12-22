# import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

def bin_evim(evim, target_maxabs_value, pos_thresh=0.2, neg_thresh=0.2):
    binned_evim = evim * target_maxabs_value
    pos_ids = evim > 0
    neg_ids = evim < 0
    binned_evim[pos_ids] = evim[pos_ids] // pos_thresh
    binned_evim[neg_ids] = evim[neg_ids] // neg_thresh
    return binned_evim

def simple_evim(evframe, scaledown_percentile=100, style='gray'):
    import torch

    if isinstance(evframe, torch.Tensor):
        evframe = evframe.cpu().detach().numpy()

    # if evframe is np array
    if isinstance(evframe, np.ndarray):

        if scaledown_percentile is not None:
            if scaledown_percentile <= 1:
                scaledown_percentile *= 100.0
            scaledown_factor = np.percentile(np.abs(evframe), scaledown_percentile)
            evframe_sc = np.clip(evframe/scaledown_factor, -1.0, 1.0)
        else:
            evframe_sc = evframe
        
        if style == 'gray':
            normalized_array = 255 * (evframe_sc - np.min(evframe_sc)) / (np.max(evframe_sc) - np.min(evframe_sc))
        
            encoding = '8UC1'

        elif style == 'redblue-on-black':
            normalized_array = np.zeros((*evframe_sc.shape, 3)) * 255
            pos_ids = evframe_sc > 0
            neg_ids = evframe_sc < 0

            normalized_array[pos_ids, 0] = 255 * evframe_sc[pos_ids]
            normalized_array[pos_ids, 1] = 0
            normalized_array[pos_ids, 2] = 0

            normalized_array[neg_ids, 2] = 255 * -evframe_sc[neg_ids]
            normalized_array[neg_ids, 0] = 0
            normalized_array[neg_ids, 1] = 0

            encoding = 'rgb8'

        elif style == 'redblue-on-white':
            # normalized_vals = (evframe_sc - np.min(evframe_sc)) / (np.max(evframe_sc) - np.min(evframe_sc))-.5 # -.5,+.5
            normalized_array = np.ones((*evframe_sc.shape, 3)) * 255
            pos_ids = evframe_sc > 0
            neg_ids = evframe_sc < 0
            zero_ids = evframe_sc == 0

            # print counts
            # print(f'pos: {pos_ids.sum()}, neg: {neg_ids.sum()}, zero: {zero_ids.sum()}')

            normalized_array[pos_ids, 0] = 255 # * evframe_sc[pos_ids]
            normalized_array[pos_ids, 1] = 255 - 255 * evframe_sc[pos_ids]
            normalized_array[pos_ids, 2] = 255 - 255 * evframe_sc[pos_ids]

            normalized_array[neg_ids, 0] = 255 - 255 * -evframe_sc[neg_ids]
            normalized_array[neg_ids, 1] = 255 - 255 * -evframe_sc[neg_ids]
            normalized_array[neg_ids, 2] = 255 # * evframe_sc[neg_ids]

            encoding = 'rgb8'

        else:
            raise ValueError("[simple_evim] style not recognized")

        evim = normalized_array.astype(np.uint8)

    else:
        raise ValueError("[simple_evim] evframe must be a numpy array or a torch tensor")

    return evim, encoding

def visualize_evim(evim, pos_thresh=0.2, neg_thresh=0.2, darken_factor=0.7):
    import torch

    # darken_factor=1.0 means no extra darkening

    # print(evim.shape)

    frame = np.zeros((*evim.shape, 3))
    binned = bin_evim(evim, target_maxabs_value=1.0, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
    if isinstance(binned, np.ndarray):
        binned = torch.from_numpy(binned)
    try:
        binned = binned.cpu()
    except:
        pass

    # print(binned.shape)

    # R or B pixel based on net of all events at pixel during timeframe
    neg_ids = (binned<0).nonzero()
    pos_ids = (binned>0).nonzero()
    # note in below, binned[neg_ids] < 0 and binned[pos_ids] > 0
    # if len(neg_ids) > 1:
        # print(neg_ids.shape)
    frame[neg_ids[:,0], neg_ids[:,1], 0] = darken_factor + binned[neg_ids[:,0], neg_ids[:,1]]/binned.abs().max()/(1/darken_factor)
    frame[neg_ids[:,0], neg_ids[:,1], 1] = darken_factor + binned[neg_ids[:,0], neg_ids[:,1]]/binned.abs().max()/(1/darken_factor)
    # if len(pos_ids) > 1:
    frame[pos_ids[:,0], pos_ids[:,1], 1] = darken_factor - binned[pos_ids[:,0], pos_ids[:,1]]/binned.abs().max()/(1/darken_factor)
    frame[pos_ids[:,0], pos_ids[:,1], 2] = darken_factor - binned[pos_ids[:,0], pos_ids[:,1]]/binned.abs().max()/(1/darken_factor)

    return (frame*255.0).astype(np.uint8)

def form_eventframe(view_events, H, W, times0=None, times1=None, N=None, device='cpu', is_half_res=False, pos_thresh=0.2, neg_thresh=0.2, all_events=False):

    # slice events
    if not all_events:

        if len(view_events) == 0:
            return np.zeros((H, W)), times0

        if times0 is None:
            print('times0 argument is None but it must be given to establilsh a starting point for the events slicing!')
            exit()

        if times1 is not None:
            # extract the time-sliced events
            # it's likely that view_events is on cpu
            valid_ev_idxs_timed = np.bitwise_and(view_events[:,0] >= times0*1e9, view_events[:,0] < times1[0]*1e9)
            view_events_timed = view_events[valid_ev_idxs_timed]
        elif N is not None:
            print(f'times0: {times0}')
            view_events_timed = view_events[view_events[:,0] >= times0*1e9][:N]
            times1 = (view_events_timed[-1,0]+1) / 1e9
        else:
            raise ValueError("form_eventframe() requires either times1 or N to be not None")

        view_events_timed_pos = view_events_timed[view_events_timed[:,-1] > 0]
        view_events_timed_neg = view_events_timed[view_events_timed[:,-1] < 0]
        frame = pos_thresh*np.histogram2d(view_events_timed_pos[:, 1], view_events_timed_pos[:, 2], bins=(W, H), range=[[0, W], [0, H]])[0] - neg_thresh*np.histogram2d(view_events_timed_neg[:, 1], view_events_timed_neg[:, 2], bins=(W, H), range=[[0, W], [0, H]])[0]
        frame = frame.T

        # # if continuous eventstream was made half-res, then 4x events have coalesced into a single pixel.
        # # make up for it by dividing the final sum per pixel by 4.
        # if is_half_res:
        #     frame = (frame / 4.0) // pos_thresh * pos_thresh

        return frame, times1

    # convert all input events
    else:

        if len(view_events) == 0:
            return np.zeros((H, W))

        view_events_pos = view_events[view_events[:,-1] > 0]
        view_events_neg = view_events[view_events[:,-1] == 0]

        frame = pos_thresh*np.histogram2d(view_events_pos[:, 1], view_events_pos[:, 2], bins=(W, H), range=[[0, W], [0, H]])[0] - neg_thresh*np.histogram2d(view_events_neg[:, 1], view_events_neg[:, 2], bins=(W, H), range=[[0, W], [0, H]])[0]
        frame = frame.T

        return frame

def plot_events(x, y, t, p, num_events=None, last_perc_larger=None):
    if num_events is not None:
        idx = np.linspace(0, x.shape[0]-1, num_events).astype(np.int64)
        x = x[idx]
        y = y[idx]
        t = t[idx]
        p = p[idx]
    
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis labels
    ax.set_xlabel('time')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # Remove labels and ticks from the X and Y axes
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Remove axis labels for X and Y
    ax.set_zlabel('')
    ax.set_xlabel('')
    ax.set_xticklabels('')
    
    # Set the viewpoint to adjust axis orientation
    ax.view_init(elev=0, azim=-1.)  # Adjust azimuth and elevation as needed
    # elev 20.0, azim -25.0

    if last_perc_larger is None:
        # Plot with very small points
        colors = ['b' if val == 1 else 'r' for val in p]
        ax.scatter(t, x, y, c=colors, marker='.', s=1, linewidths=0)  # 's' parameter for point size
    else:
        last_mask = np.zeros(t.shape[0], dtype=bool)
        last_mask[int((1-last_perc_larger) * t.shape[0]):] = True

        colors = ['b' if val == 1 else 'r' for val in p[~last_mask]]
        ax.scatter(t[~last_mask], x[~last_mask], y[~last_mask], c=colors, marker='.', s=1, linewidths=0)  # 's' parameter for point size
        colors = ['b' if val == 1 else 'r' for val in p[last_mask]]
        ax.scatter(t[last_mask], x[last_mask], y[last_mask], c=colors, marker='.', s=4, linewidths=0)  # 's' parameter for point size
    
    # Remove the background color
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Set equal limits for Y and Z axes to make their aspect ratios equal
    y_limits = ax.get_ylim()  # Get current limits of Y axis
    z_limits = ax.get_zlim()  # Get current limits of Z axis
    max_range = max(y_limits[1] - y_limits[0], z_limits[1] - z_limits[0])
    ax.set_ylim(y_limits[0], y_limits[0] + max_range)
    ax.set_zlim(z_limits[0], z_limits[0] + max_range)

    x_limits = ax.get_xlim()
    stretch_factor = 2  # Stretch the X axis by a factor of 2
    new_x_range = (x_limits[1] - x_limits[0]) * stretch_factor

    # Adjust X limits to stretch it
    ax.set_xlim(x_limits[0], x_limits[0] + new_x_range)

    return fig

def plot_events_open3d(x, y, t, p, num_events=None, last_perc_larger=None, pcd=None):
    import open3d as o3d
    # subsample events to num_events
    if num_events is not None:
        idx = np.linspace(0, x.shape[0]-1, num_events).astype(np.int64)
        x = x[idx]
        y = y[idx]
        t = t[idx]
        p = p[idx]

    # print(f'p values = {np.unique(p)}')

    # if True:
    #     idx = p == 0
    #     x = x[idx]
    #     y = y[idx]
    #     t = t[idx]
    #     p = p[idx]

    # data = np.random.rand(1000, 3) * 10
    data = np.column_stack((x, y, t))
    # data[:, 2] = data[:, 2] - data[:, 2].min()
    # data[:, 2] = data[:, 2] / 5e5

    # Create a point cloud
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    # # verify data
    # print(data.shape)
    # print(data)
    # print(pcd)

    # Create a color map
    colors = np.zeros((x.shape[0], 3))
    colors[p == 1, 0] = 1
    colors[p == 0, 2] = 1
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

    # # Visualize the point cloud
    # # o3d.visualization.draw_geometries([pcd])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin = pcd.get_center())
    # # vis.add_geometry(axis)

    # opt = vis.get_render_option()
    # opt.point_size = 1  # Larger points to visually approximate circles

    # vis.run()
    # vis.destroy_window()


