import os

folder_path = "/home/ziyan/02_research/pytorch-superpoint/logs/superpoint_mvsec/checkpoints"

# List all files in the folder
files = os.listdir(folder_path)

# Iterate over each file
for file in files:
    # Extract the numerical part of the file name
    try:
        num_part = int(''.join(filter(str.isdigit, file)))
    except ValueError:
        # If the file name doesn't contain numerical part, skip
        continue
    
    # Check if the numerical part is greater than 1000
    if num_part > 1000:
        # If so, delete the file
        os.remove(os.path.join(folder_path, file))
        # print(os.path.join(folder_path, file))
