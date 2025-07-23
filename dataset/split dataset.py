import os
import shutil
import random


def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # 获取所有类别目录
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in classes:
        # 创建相应的目录结构
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # 获取该类别下的所有图片文件
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        # 计算每个子集的大小
        total_images = len(images)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        test_size = total_images - train_size - val_size

        # 分配图片到各个子集
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # 复制图片到相应的目录
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

    print("数据集划分完成！")


# 源数据集路径
source_dataset_path = "Rice Leaf Disease1"

# 目标路径
train_path = "../dataset1/Rice Leaf Disease/train"
val_path = "../dataset1/Rice Leaf Disease/val"
test_path = "../dataset1/Rice Leaf Disease/test"

# 创建目标路径
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 划分数据集
split_dataset(source_dataset_path, train_path, val_path, test_path)
