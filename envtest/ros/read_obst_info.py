import csv

def extract_objects_from_csv(csv_file_path):
    objects = []

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        # # Assuming the first row contains headers, skip it
        # next(csv_reader, None)

        for row in csv_reader:
            try:
                x = float(row[1])  # Assuming x is in column 1 (0-indexed)
                y = float(row[2])  # Assuming y is in column 2 (0-indexed)
                z = float(row[3])  # Assuming z is in column 3 (0-indexed)
                # cvs radius is in format y, z, x ???
                radius = (float(row[10]), float(row[8]), float(row[9]))  # Assuming radius is in column 8 (0-indexed)
                
                objects.append((x, y, z, radius))
            except (ValueError, IndexError):
                print(f"Skipping invalid row: {row}")

    return objects

# main function
if __name__ == '__main__':

    # Example usage:
    csv_file_path = '/home/anish/evfly_ws/src/evfly/flightmare/flightpy/configs/vision/easy/environment_0/static_obstacles.csv'  # Replace with your actual CSV file path
    object_list = extract_objects_from_csv(csv_file_path)

    # Display the extracted objects
    for obj in object_list:
        print(obj)
