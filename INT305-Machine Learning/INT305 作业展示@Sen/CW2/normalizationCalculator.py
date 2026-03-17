import torch
from torchvision import datasets, transforms

data_path = './data'
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

mean = 0.0
std = 0.0
num_samples = 0

for images, _ in train_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    num_samples += batch_samples

mean = mean / num_samples

for images, _ in train_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    std += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])

std = torch.sqrt(std / (num_samples * images.size(2)))

print("Mean:", mean)
print("Std:", std)