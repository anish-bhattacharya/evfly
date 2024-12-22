# take in directory as sys arg and load dynamic_obstacles.yaml file

import sys
import os
import yaml
import random
import csv

dataset = sys.argv[1]

# load dynamic_obstacles.yaml file
yaml_file = os.path.join(dataset, "dynamic_obstacles.yaml")

with open(yaml_file, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# get element called 'N' and read as int
N = data['N']

# make adjustments to data
# TODO: randomize the prefabs
prefabs = ['rpg_box01', 'rpg_box02', 'rpg_box03', 'Barrel_v1_LD3']

print(f'Randomizing {N} prefabs from selection {prefabs}')

# make list length N of random prefabs
random_prefab = [random.choice(prefabs) for i in range(N)]

# edit dynamic_obstacles.yaml
for i in range(N):
    data['Object'+str(i+1)]['prefab'] = random_prefab[i]

# save data back to dynamic_obstacles.yaml file
# with open(yaml_file, 'w') as stream:
with open(yaml_file, 'w') as stream:
    try:
        yaml.dump(data, stream, default_flow_style=False, sort_keys=False)
    except yaml.YAMLError as exc:
        print(exc)

print(f'Updated {yaml_file} with random prefabs')

# now load in each static_kr_{i}.csv file individually and edit the scale
# scale is in last three columns of each row
        
def edit_scale(row, prefab_id):
    row[0] = prefab_id
    if prefab_id == 'rpg_box01':
        row[-3] = '0.5'
        row[-2] = '0.5'
        row[-1] = '0.5'
        row[3] = '0.0'
    elif prefab_id == 'rpg_box02':
        row[-3] = '0.5'
        row[-2] = '10'
        row[-1] = '0.5'
        row[3] = f'{str(float(row[-2])/2.0-1.0)}'
    elif prefab_id == 'rpg_box03':
        row[-3] = '0.25'
        row[-2] = '10'
        row[-1] = '1'
        row[3] = '-1.0' #f'{str(float(row[-2])/2.0)}'
    elif prefab_id == 'Barrel_v1_LD3':
        row[-3] = '1'
        row[-2] = '10'
        row[-1] = '1'
        row[3] = f'{str(float(row[-2])/2.0-1.0)}'
    else:
        print('Invalid prefab name')
        exit(1)
    return row

for i in range(N):

    csv_file = os.path.join(dataset, f"static_kr_{i}.csv")

    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

        # Edit the last three elements of each row
        modified_rows = [edit_scale(row, random_prefab[i]) for row in rows]

    # Write the modified content back to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modified_rows)

print(f'Updated {N} static_kr_(i).csv files with appropriate scales for prefabs')

print('Done')
