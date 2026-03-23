import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 加载训练集（不进行归一化）
train_set = datasets.CIFAR10(root='./data', train=True, download=False,
                             transform=transforms.ToTensor())

train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=2)


def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # 当前batch的样本数
        images = images.view(batch_samples, images.size(1), -1)  # 重塑为 [batch, channels, height*width]
        mean += images.mean(2).sum(0)   # 计算每个通道的空间均值，然后求和
        std += images.std(2).sum(0)     # 计算每个通道的空间标准差，然后求和
        total_samples += batch_samples   # 累计总样本数

    mean /= total_samples  # 计算平均均值
    std /= total_samples   # 计算平均标准差
    return mean, std

if __name__ == '__main__':
    mean, std = compute_mean_std(train_loader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")