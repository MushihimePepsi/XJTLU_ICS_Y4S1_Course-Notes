import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# 设置超参数
number_of_workers = 0
batch_size = 128
valid_size = 0.2
n_epochs = 35

# 数据转换和增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
])

# 下载数据集
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# 划分训练集和验证集
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# 定义采样器
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=number_of_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=number_of_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=number_of_workers)

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# 可视化辅助函数
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    plt.imshow(np.transpose(img, (1, 2, 0)))


# 定义简化的Inception模块
class SimplifiedInception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 分支1: 1×1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 分支2: 1×1降维 → 3×3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 分支3: 3×3池化 → 1×1卷积
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        # 在通道维度上拼接
        return torch.cat([branch1, branch2, branch3], dim=1)
    # 输出通道数 = 32 + 64 + 32 = 128


# 定义完整的CNN网络
class InceptionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 第一层：普通卷积层 (3×3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 第二层：第一个Inception模块
        self.inception1 = SimplifiedInception(in_channels=64)  # 输出128通道
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        # 第三层：第二个Inception模块
        self.inception2 = SimplifiedInception(in_channels=128)  # 输出128通道
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # 第四层：第三个Inception模块
        self.inception3 = SimplifiedInception(in_channels=128)  # 输出128通道
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4

        # 全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 4x4 -> 1x1
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入: [B, 3, 32, 32]
        x = self.conv1(x)  # -> [B, 64, 32, 32]
        x = self.inception1(x)  # -> [B, 128, 32, 32]
        x = self.pool1(x)  # -> [B, 128, 16, 16]
        x = self.inception2(x)  # -> [B, 128, 16, 16]
        x = self.pool2(x)  # -> [B, 128, 8, 8]
        x = self.inception3(x)  # -> [B, 128, 8, 8]
        x = self.pool3(x)  # -> [B, 128, 4, 4]
        x = self.avgpool(x)  # -> [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # -> [B, 128]
        x = self.fc(x)  # -> [B, num_classes]
        return x


# 创建模型
model = InceptionCNN(num_classes=10)

# 打印模型结构
print(model)

# 如果有GPU，将模型移到GPU
if train_on_gpu:
    model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=0.001,
                       betas=(0.9, 0.999),
                       weight_decay=5e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# 训练模型
valid_loss_min = np.inf
train_losses = []
valid_losses = []

for epoch in range(1, n_epochs + 1):
    # 训练阶段
    model.train()
    train_loss = 0.0

    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    # 验证阶段
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

    # 计算平均损失
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # 打印统计信息
    print(f'Epoch: {epoch:2d} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

    # 更新学习率
    scheduler.step()

    # 保存最佳模型
    if valid_loss <= valid_loss_min:
        print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'inception_model_cifar.pt')
        valid_loss_min = valid_loss

print('Training completed!')

# 测试模型
model.load_state_dict(torch.load('inception_model_cifar.pt'))

test_loss = 0.0
class_correct = [0] * 10
class_total = [0] * 10

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))

        if train_on_gpu:
            correct = correct_tensor.cpu().numpy()
        else:
            correct = correct_tensor.numpy()

        # 统计每个类别的准确率
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i]
            class_total[label] += 1

# 计算整体测试准确率
test_loss = test_loss / len(test_loader.dataset)
print(f'\nTest Loss: {test_loss:.6f}')

# 打印每个类别的准确率
for i in range(10):
    if class_total[i] > 0:
        print(f'Test Accuracy of {class_names[i]:>10s}: {100 * class_correct[i] / class_total[i]:2.0f}% '
              f'({int(class_correct[i])}/{int(class_total[i])})')

# 计算整体准确率
total_correct = sum(class_correct)
total = sum(class_total)
print(f'\nOverall Test Accuracy: {100 * total_correct / total:2.0f}% ({int(total_correct)}/{int(total)})')


# 可视化部分测试结果
def visualize_predictions(model, test_loader, num_images=20):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    if train_on_gpu:
        images = images.cuda()

    with torch.no_grad():
        output = model(images)
        _, preds = torch.max(output, 1)

    if train_on_gpu:
        images = images.cpu()
        preds = preds.cpu()
        labels = labels.cpu()

    images = images.numpy()
    preds = preds.numpy()
    labels = labels.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in range(num_images):
        ax = fig.add_subplot(2, num_images // 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(f"{class_names[preds[idx]]} ({class_names[labels[idx]]})",
                     color=("green" if preds[idx] == labels[idx] else "red"))
    plt.show()


# 可视化预测结果
visualize_predictions(model, test_loader)

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, n_epochs + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


# 打印模型参数统计
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f'\nTotal Trainable Parameters: {total_params:,}')
print(f'Model Size: {total_params * 4 / (1024 ** 2):.2f} MB (assuming float32)')

# 保存完整模型（包含结构）
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'valid_losses': valid_losses,
    'test_accuracy': 100 * total_correct / total
}, 'complete_inception_model.pth')

print('Model saved successfully!')