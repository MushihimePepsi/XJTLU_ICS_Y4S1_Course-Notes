import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据转换
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 加载数据集

train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

train_ld = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_ld = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),   # 256×1×1
            nn.Flatten(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# 训练和测试函数
def train_epoch(model, device, train_ld, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_ld:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

def test_epoch(model, device, test_ld, criterion):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_ld:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# 主程序
if __name__ == '__main__':
    # device = torch.device("cpu")  # 有 GPU 可改成 "cuda:0"
    device = torch.device("cuda:0")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    EPOCHS = 30
    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, device, train_ld, criterion, optimizer)
        test_acc = test_epoch(model, device, test_ld, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_cifar10_cnn.pt')
        print(f'Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%')
    print('Best test accuracy: {:.2f}%'.format(best_acc))