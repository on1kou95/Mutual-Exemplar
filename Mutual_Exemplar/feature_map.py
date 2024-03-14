from PIL import Image
import numpy as np

# Set the size of the feature map (e.g., 10x10 cells)
grid_size = 10

# Set the size of each cell in pixels (e.g., 50x50 pixels)
cell_size = 50

# Initialize an empty image
image_size = grid_size * cell_size
feature_map = np.zeros((image_size, image_size, 3), dtype=np.uint8)

# Fill each cell with a different shade of yellow
for i in range(grid_size):
    for j in range(grid_size):
        # Generate a random shade of yellow for each cell
        # Yellow is made by mixing red and green in equal intensity
        intensity = np.random.randint(10, 256)
        feature_map[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size, 0] = intensity  # Red channel
        feature_map[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size, 1] = intensity  # Green channel
        # Blue channel is kept low or 0

# Convert the NumPy array to a PIL image
feature_map_image = Image.fromarray(feature_map)

# Save the image
save_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r_e/feature_map_image3.png'
feature_map_image.save(save_path)

print("Feature map saved at:", save_path)
