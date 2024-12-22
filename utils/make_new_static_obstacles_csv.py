# create new static obstacles environments from given ones

import glob, os, sys
import csv
import numpy as np
import pandas as pd

env_dir = sys.argv[1]
new_dir = sys.argv[2]
percent_obsts_to_keep = float(sys.argv[3])
randomize_selection = sys.argv[4] == 'random'

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

dirs = sorted(glob.glob(os.path.join(env_dir, 'environment_*')))

for dir in dirs:
    basename = os.path.basename(dir)
    new_dirname = os.path.join(new_dir, basename)

    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname)

    # read csv file
    input_file = os.path.join(dir, 'static_obstacles.csv')
    output_file = os.path.join(new_dirname, 'static_obstacles.csv')
    
    data = []
    # data = pd.read_csv(input_file).values
    with open(input_file, 'r') as file:
        csv_reader = csv.reader(file)
        # Iterate through each row in the CSV file
        for row in csv_reader:
            data.append(row)
    # Convert the list of lists to a NumPy array
    data = np.array(data)

    data_new = data.copy()
    for i in range(data.shape[0]):

        # orientation
        data_new[i][4] = 0
        data_new[i][5] = 0
        data_new[i][6] = 0
        data_new[i][7] = 1

        # scale
        width = 0.5 # np.random.uniform(0.4, 0.5)
        data_new[i][8] = width
        data_new[i][9] = 100.0
        data_new[i][10] = width

    # index of obstacles to keep
    if randomize_selection:
        indices_to_keep = np.random.choice(data_new.shape[0], int(data_new.shape[0]*percent_obsts_to_keep), replace=False)
    else:
        indices_to_keep = np.arange(int(data_new.shape[0]*percent_obsts_to_keep))

    data_new = data_new[indices_to_keep]

    # data_new = data_new[:int(data_new.shape[0]*percent_obsts_to_keep)]

    # with open(output_file, 'w') as f:
    np.savetxt(output_file, data_new, delimiter=',', fmt='%s')

    # spoof a dynamic obstacle
    # copy contents of /home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle directory to each environment directory
    os.system('cp -r /home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle/* ' + new_dirname)

    # exit()
