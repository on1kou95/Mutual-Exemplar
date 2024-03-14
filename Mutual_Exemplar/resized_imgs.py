from PIL import Image
import os

# Set the directory path where your images are stored
source_folder = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r'
# Set the directory path where you want to save the resized images
target_folder = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r_e'

# Create target folder if it doesn't exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Set the target size (width, height)
TARGET_SIZE = (720,576)

# Loop through all files in the directory
for filename in os.listdir(source_folder):
    # Construct the full file path of the source image
    source_file_path = os.path.join(source_folder, filename)

    # Construct the full file path where the resized image will be saved
    target_file_path = os.path.join(target_folder, filename)

    # Check if the file is an image based on its extension
    # You might need to add or remove extensions depending on your needs
    if source_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        try:
            # Open the image
            with Image.open(source_file_path) as img:
                # Resize the image
                img_resized = img.resize(TARGET_SIZE, Image.ANTIALIAS)

                # Save the resized image in the target folder
                img_resized.save(target_file_path)

                print(f'Resized and saved {filename} to {target_folder} successfully.')
        except Exception as e:
            print(f'Error resizing {filename}: {e}')
