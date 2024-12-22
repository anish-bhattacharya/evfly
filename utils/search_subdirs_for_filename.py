import os, sys
import fnmatch

def search_files(base_dir, patterns):
    for root, dirs, files in sorted(os.walk(base_dir)):
        matched = False
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                matched = True
                break
            if matched:
                break
        print(f"{os.path.basename(root)}: {'yes' if matched else 'no'}")

if __name__ == "__main__":
    directory = sys.argv[1]
    patterns = [sys.argv[2]]
    search_files(directory, patterns)
