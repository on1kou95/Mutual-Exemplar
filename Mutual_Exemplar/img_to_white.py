from PIL import Image
import os

def convert_nonzero_pixels_to_white(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)  # 创建目标文件夹（如果不存在）

    for filename in os.listdir(source_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_folder, filename)
            with Image.open(img_path) as img:
                img = img.convert("RGBA")
                data = img.getdata()

                new_data = []
                for item in data:
                    # 将非零像素转换为白色，保持透明度不变
                    if item[:3] != (0, 0, 0):
                        new_data.append((255, 255, 255, item[3]))
                    else:
                        new_data.append(item)

                img.putdata(new_data)
                # 构建目标文件路径并保存
                target_path = os.path.join(target_folder, filename)
                img.save(target_path)

source_folder = '/home/wengweihao/Downloads/instrument_1_4_training/instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels'  # 替换为源文件夹路径
target_folder = '/home/wengweihao/Downloads/instrument_1_4_training/instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels2'  # 替换为目标文件夹路径
convert_nonzero_pixels_to_white(source_folder, target_folder)
