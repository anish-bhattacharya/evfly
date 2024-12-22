# we need to convert the given static_obstacles.csv files from each environment folder into the static_kr_{i}.csv files to be used by our dynamic obstacle moving code.

import os, sys, glob
import csv

env_in_path = sys.argv[1]

environment_dirs = sorted(glob.glob(os.path.join(env_in_path, 'environment_*')))

query_csv_file = os.path.join(environment_dirs[0], 'static_obstacles.csv')
# get number of rows in static obstacles
with open(query_csv_file, 'r') as file:
  # Create a csv reader
  csv_reader = csv.reader(file)
  N = 0
  for row in csv_reader:
    N += 1

# set base directory to parent directory of env_in_path
base_directory = os.path.dirname(env_in_path)

# set output directory to basename of env_in_path
output_directory = os.path.basename(env_in_path)
custom_output_directory = os.path.join(base_directory, 'custom_'+output_directory)
# if exists, delete and remake
if os.path.exists(custom_output_directory):
    os.system('rm -r ' + custom_output_directory)
os.makedirs(custom_output_directory)

print(f"Converting {N} obstacles in {len(environment_dirs)} environments from {env_in_path} to {custom_output_directory}")
# exit()

# Loop through each row
for row in range(N):
    # Create a new CSV file for each row
    output_file = os.path.join(base_directory, "custom_"+output_directory, f"static_kr_{row}.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Loop through each environment folder
        for environment in range(len(environment_dirs)):
            # Read the original CSV file
            input_file = os.path.join(base_directory, output_directory, f"environment_{environment}/static_obstacles.csv")
            with open(input_file, mode='r') as input_csv:
                reader = csv.reader(input_csv)
                
                # Extract the desired elements from the row
                extracted_row = list(reader)[row]
                desired_info = [
                                extracted_row[0], # name
                                extracted_row[1], # pos x
                                extracted_row[2], # pos y
                                extracted_row[3], # pos z
                                extracted_row[4], # q w
                                extracted_row[5], # q x
                                extracted_row[6], # q y
                                extracted_row[7], # q z
                                extracted_row[8], # scale x
                                extracted_row[9], # scale y
                                extracted_row[10] # scale z
                                ]
                
                # Write the extracted row to the new CSV file
                writer.writerow(desired_info)

# # write yaml file
# basetext = \
# '''
#   csvtraj: traj_501743091401155677
#   loop: false
#   position:
#   - 48.057093511629645
#   - 3.4678879517038
#   - 7.923471482431094
#   prefab: rpg_box01
#   rotation:
#   - 0.4846908786270495
#   - -0.35626152747714557
#   - -0.02281735819161093
#   - 0.7985185310188774
#   scale:
#   - 1.5386464779409343
#   - 1.5386464779409343
#   - 1.5386464779409343
# '''

# copy basetext from file /home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle/dynamic_obstacles.yaml
basetext = '\n'
# Open the file
with open('/home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle/dynamic_obstacles.yaml', 'r') as file:
    # Read lines 3 to 18
    for i, line in enumerate(file, start=1):
        if 3 <= i <= 18:
            basetext += line
        elif i > 18:
            break

# print(f'basetext: \n{basetext}')
# exit()

# copy the directory /home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle/csvtrajs into the custom_output_directory
os.system('cp -r /home/anish/evfly_ws/src/evfly/fake_dynamic_obstacle/csvtrajs ' + os.path.join(base_directory, "custom_"+output_directory))

# copy a bogus static obstacles file from /home/anish/evfly_ws/src/evfly/fake_static_obstacle/static_obstacles.csv into the custom_output_directory
os.system('cp /home/anish/evfly_ws/src/evfly/fake_static_obstacle/static_obstacles.csv ' + os.path.join(base_directory, "custom_"+output_directory))

filename = os.path.join(base_directory, "custom_"+output_directory, "dynamic_obstacles.yaml")
with open(filename, mode='a') as file:
    file.write(f"N: {N}\n")
for o in range(N):
    with open(filename, mode='a') as file:
        file.write(f"Object{o+1}:{basetext}")
