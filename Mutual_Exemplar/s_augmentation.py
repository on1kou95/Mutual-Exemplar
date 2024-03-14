import cv2
import numpy as np
from skimage.util import random_noise
from skimage import exposure

def random_apply_transformations_and_save(image_path, save_path):
    image = cv2.imread(image_path)
    rows, cols = image.shape[:2]

    # # Apply random rotation
    # angle = np.random.randint(-180, 180)
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    # image = cv2.warpAffine(image, M, (cols, rows))

    # Apply random flip
    flip_code = np.random.choice([-1, 0, 1]) # -1: flip both axes, 0: vertical flip, 1: horizontal flip
    image = cv2.flip(image, flip_code)

    # Apply random affine transformation
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = pts1 + np.random.uniform(-5, 5, pts1.shape).astype(np.float32)
    M_affine = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M_affine, (cols, rows))

    # Apply random grayscale noise
    mode = np.random.choice(['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'])
    image = (random_noise(image, mode=mode) * 255).astype(np.uint8)

    # Apply random Gaussian blur
    blur_value = np.random.choice([3, 5, 7]) # Kernel size
    image = cv2.GaussianBlur(image, (blur_value, blur_value), 0)

    # Apply random color jitter (using gamma correction for simplicity)
    gamma = np.random.uniform(0.5, 2.0)
    image = exposure.adjust_gamma(image, gamma=gamma)

    # # Apply random GridMask
    # def gridmask(image, d_range=(1, 50), ratio=0.5):
    #     d = np.random.randint(d_range[0], d_range[1])
    #     h, w = image.shape[:2]
    #     mask = np.zeros((h, w), dtype=np.uint8)
    #     for i in range(0, h, d):
    #         for j in range(0, w, d):
    #             mask[i:int(i+d*ratio), j:int(j+d*ratio)] = 1
    #     return cv2.bitwise_and(image, image, mask=mask)
    #
    # image = gridmask(image)

    # Save the transformed image
    cv2.imwrite(save_path, image)

# Example usage:
image_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r_e/2/1val_04_input.png'
save_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r_e/2/s2.png'
random_apply_transformations_and_save(image_path, save_path)
