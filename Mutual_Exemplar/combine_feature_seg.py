from PIL import Image

# 载入分割图
segmentation_image_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r_e/1val_04_pred_1.png'
segmentation_image = Image.open(segmentation_image_path).convert('RGB')

# 获取分割图的尺寸
segmentation_size = segmentation_image.size

# 载入特征图
feature_map_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r_e/feature_map_image1.png'
feature_map = Image.open(feature_map_path).convert('RGB')

# 将特征图调整大小以匹配分割图的尺寸
resized_feature_map = feature_map.resize(segmentation_size, Image.ANTIALIAS)

# 合并图像，这里简单地通过将两个图像的像素值相加进行合并
# 注意：这种方法可能导致一些像素值超过255，需要裁剪
merged_image = Image.blend(resized_feature_map, segmentation_image, alpha=0.5)

# 保存合并后的图像
merged_image_path = '/home/wengweihao/Documents/EndoNeRF/Min_Max_Similarity/logs/kvasir/test_r_e/pred_2a.png'
merged_image.save(merged_image_path)

merged_image_path
