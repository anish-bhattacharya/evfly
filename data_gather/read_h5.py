import h5py
import sys

# Open the HDF5 file
file_path = sys.argv[1]
with h5py.File(file_path, 'r') as h5_file:
    # Access the dataset 't'
    t_dataset = h5_file['t']
    
    # Print the first 10 values
    first_10_t = t_dataset[:10]
    print("First 10 values of 't':", first_10_t)
    
    # Print the last 10 values
    last_10_t = t_dataset[-10:]
    print("Last 10 values of 't':", last_10_t)
