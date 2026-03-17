from torchvision import datasets, transforms

# 指定保存目录（可任意改）
datasets.CIFAR10(root='./data',           # 会生成 ./data/cifar-10-python.tar.gz
                 train=True,              # 下训练集
                 download=True,           # 关键：True 就自动下载
                 transform=transforms.ToTensor())

datasets.CIFAR10(root='./data',
                 train=False,             # 下测试集
                 download=True,
                 transform=transforms.ToTensor())

print("CIFAR-10 下载并解压完成！")