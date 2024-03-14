import numpy as np
from skimage import io
from skimage.metrics import mean_squared_error


def calculate_metrics(image1_path, image2_path):
    # 加载图像并转换为二值图像
    image1 = io.imread(image1_path, as_gray=True)
    image2 = io.imread(image2_path, as_gray=True)
    image1 = (image1 > 0.5).astype(np.uint8)
    image2 = (image2 > 0.5).astype(np.uint8)

    # 计算DSC
    intersection = np.logical_and(image1, image2).sum()
    dsc = (2. * intersection) / (image1.sum() + image2.sum())

    # 计算IoU
    union = np.logical_or(image1, image2).sum()
    iou = intersection / union

    # 计算MAE
    mae = np.abs(image1 - image2).mean()

    # 计算F-measure
    precision = intersection / image2.sum()
    recall = intersection / image1.sum()
    f_measure = 2 * (precision * recall) / (precision + recall)

    return dsc, iou, mae, f_measure


# 替换'image1_path'和'image2_path'为你的图像路径
image1_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test/saved_images_2/val_03_gt.png'
image2_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test/saved_images_2/val_04_gt.png'

dsc, iou, mae, f_measure = calculate_metrics(image1_path, image2_path)
print(f"DSC: {dsc}, IoU: {iou}, MAE: {mae}, F-measure: {f_measure}")
