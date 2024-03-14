import os
import shutil

# Set the paths
source_folder = '/home/wengweihao/Downloads/kvasir-instrument/masks'  # Replace with the path to the folder containing the images
train_folder = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/data/kvasir/train/mask'    # Replace with the path to the 'train' folder
txt_file = '/home/wengweihao/Downloads/kvasir-instrument/train.txt'           # Replace with the path to the 'train.txt' file

# Create the 'train' folder if it doesn't exist
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

# Read the names from the txt file
with open(txt_file, 'r') as file:
    names = file.read().splitlines()

# Move the corresponding .jpg files
for name in names:
    source_file = os.path.join(source_folder, name + '.png')
    destination_file = os.path.join(train_folder, name + '.png')
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)
