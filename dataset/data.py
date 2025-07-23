from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 将数据集图片放缩成224*224大小
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#
#     "test": transforms.Compose([transforms.Resize((224, 224)),  # 调整图片大小为224*224
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# # (2)、加载数据集
#
# train_dataset = torchvision.datasets.ImageFolder(root="./dataset/en_Wheat_split/train",
#                                                  transform=data_transform["train"])
# print(train_dataset.class_to_idx)  # 获取训练集中不同类别对应的索引序号
# val_dataset = torchvision.datasets.ImageFolder(root="./dataset/en_Wheat_split/val",
#                                                transform=data_transform["test"])
# print(val_dataset.class_to_idx)
# test_dataset = torchvision.datasets.ImageFolder(root="./dataset/en_Wheat_split/test",
#                                                 transform=data_transform["test"])
# print(test_dataset.class_to_idx)  # 获取测试集中不同类别对应的索引序号

import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_PestDisease_dataloaders(data_dir="", batch_size=32, num_workers=4):
    """
    返回病虫害数据集的 DataLoader

    参数:
        data_dir (str): 数据集目录路径，默认为 "./dataset/en_Wheat_split"
        batch_size (int): 批量大小，默认为 32
        num_workers (int): 加载数据时使用的子进程数，默认为 4

    返回:
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
        test_loader (DataLoader): 测试集数据加载器
    """
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=data_transform["train"]
    )
    print("Train dataset classes:", train_dataset.class_to_idx)

    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=data_transform["test"]
    )
    print("Validation dataset classes:", val_dataset.class_to_idx)

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=data_transform["test"]
    )
    print("Test dataset classes:", test_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader , test_loader