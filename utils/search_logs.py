import os
import sys

# # Define the search strings
# search_strings = ["2-12"]

# Set the root directory containing the log directories
root_directory = sys.argv[1]
search_strings = sys.argv[2:]

# Function to check if both search strings are in a file
def file_contains_strings(file_path, strings):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return all(s in content for s in strings)
    except FileNotFoundError:
        return False

# Function to search through log directories
def search_log_dirs(root_dir):
    for log_dir in sorted(os.listdir(root_dir)):
        log_dir_path = os.path.join(root_dir, log_dir)
        config_file_path = os.path.join(log_dir_path, 'config.txt')
        if os.path.isdir(log_dir_path) and file_contains_strings(config_file_path, search_strings):
            print(log_dir)

# Search the log directories
search_log_dirs(root_directory)
