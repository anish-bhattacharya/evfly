# create completely new environments

import glob, os, sys
import csv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

new_dir = sys.argv[1]
num_environments = int(sys.argv[2])
num_obstacles = int(sys.argv[3])
obst_scheme = sys.argv[4]

# obstacle parameters
xrange = [5, 60]
yrange = [-20, 20]
zrange = [0, 0]
rollrange = [0, 0]
pitchrange = [-np.pi, np.pi] # this is apparent yaw in unity
yawrange = [0, 0] # this is apparent pitch in unity
scxrange = [0.5, 0.5]
scyrange = [0.5, 0.5]
sczrange = [0.5, 0.5]

basedir = '/home/anish/evfly_ws/src/evfly/flightmare/flightpy/configs/vision'
new_dir = os.path.join(basedir, new_dir)

# make it if it doesn't exist; if it does, erase contents after prompting user
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
else:
    print(f'Directory {new_dir} already exists. \nErase contents? (y/n)')
    response = input()
    if response == 'y':
        os.system('rm -r ' + new_dir + '/*')
    else:
        print('Not erasing. Exiting.')
        exit()

for env_i in range(num_environments):

    env_name = f'environment_{env_i}'
    new_env_dir = os.path.join(new_dir, env_name)

    # make new env dir; if it exists, erase contents
    if not os.path.exists(new_env_dir):
        os.makedirs(new_env_dir)
    else:
        os.system('rm -r ' + new_env_dir + '/*')
    
    # spoof a dynamic obstacle
    # copy contents of /home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle directory to each environment directory
    os.system('cp -r /home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle/* ' + new_env_dir)

    # make static_obstacles.csv
    if obst_scheme == 'random':

        name = 'rpg_box01'

        x = np.random.uniform(xrange[0], xrange[1], num_obstacles)
        y = np.random.uniform(yrange[0], yrange[1], num_obstacles)
        z = np.random.uniform(zrange[0], zrange[1], num_obstacles)

        roll = np.random.uniform(rollrange[0], rollrange[1], num_obstacles)
        pitch = np.random.uniform(pitchrange[0], pitchrange[1], num_obstacles)
        yaw = np.random.uniform(yawrange[0], yawrange[1], num_obstacles)
        qx = np.zeros(num_obstacles)
        qy = np.zeros(num_obstacles)
        qz = np.zeros(num_obstacles)
        qw = np.zeros(num_obstacles)
        for q_i in range(num_obstacles):
            rot = Rotation.from_euler('xyz', [roll[q_i], pitch[q_i], yaw[q_i]], degrees=False)
            q = rot.as_quat()
            qx[q_i] = q[0]
            qy[q_i] = q[1]
            qz[q_i] = q[2]
            qw[q_i] = q[3]

        scx = np.random.uniform(scxrange[0], scxrange[1], num_obstacles)
        scy = np.random.uniform(scyrange[0], scyrange[1], num_obstacles)
        scz = np.random.uniform(sczrange[0], sczrange[1], num_obstacles)

    else:

        NotImplementedError('Only random obstacle scheme implemented')
        exit()

    # fill data array
    data = np.zeros((num_obstacles, 11), dtype=object)
    for i in range(num_obstacles):
        data[i][0] = name
        data[i][1] = x[i]
        data[i][2] = y[i]
        data[i][3] = z[i]
        data[i][4] = qw[i]
        data[i][5] = qx[i]
        data[i][6] = qy[i]
        data[i][7] = qz[i]
        data[i][8] = scx[i]
        data[i][9] = scy[i]
        data[i][10] = scz[i]

    # save to csv
    output_file = os.path.join(new_env_dir, 'static_obstacles.csv')
    np.savetxt(output_file, data, delimiter=',', fmt='%s')

print(f'Created {new_dir} with {num_environments} environments and {num_obstacles} obstacles each, placed with scheme {obst_scheme}.')
