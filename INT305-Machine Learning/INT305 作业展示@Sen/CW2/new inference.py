import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter

# 检查CUDA是否可用
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Testing on CPU ...')
else:
    print('CUDA is available! Testing on GPU ...')

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# 可视化辅助函数（与训练代码保持一致）
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    plt.imshow(np.transpose(img, (1, 2, 0)))


# 定义简化的Inception模块（与训练代码完全一致）
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
        return torch.cat([branch1, branch2, branch3], dim=1)


# 定义完整的CNN网络（与训练代码完全一致）
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
        self.inception1 = SimplifiedInception(in_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层：第二个Inception模块
        self.inception2 = SimplifiedInception(in_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四层：第三个Inception模块
        self.inception3 = SimplifiedInception(in_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.inception3(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# 主程序开始
# ============================================================================

print("=" * 60)
print("Inception CNN Model Testing")
print("=" * 60)

# 创建模型（结构与训练时完全相同）
model = InceptionCNN(num_classes=10)
print("模型创建完成")

# 将模型移到GPU（如果可用）
if train_on_gpu:
    model.cuda()
    print("模型已移至GPU")
else:
    print("在CPU上运行")

# 加载训练好的模型
print("\n加载模型权重...")
model_name = 'inception_model_cifar.pt'  # 与训练代码中保存的文件名一致

try:
    # 加载模型权重
    if train_on_gpu:
        model.load_state_dict(torch.load(model_name))
    else:
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    print(f"成功加载模型权重: {model_name}")
except Exception as e:
    print(f"加载模型时出错: {e}")
    print("请确保模型文件存在且路径正确")
    exit(1)


# 计算参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f"\n可训练参数总数: {total_params:,}")
print(f"模型大小: {total_params * 4 / (1024 ** 2):.2f} MB (float32)")

# 损失函数
criterion = nn.CrossEntropyLoss()

# 测试数据预处理（与训练时相同，但不包含数据增强）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
])

# 加载测试数据
print("\n" + "=" * 60)
print("加载测试数据...")
print("=" * 60)

test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform_test)

batch_size = 128  # 与训练时相同
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=0, shuffle=False)

print(f"测试集大小: {len(test_data)}")
print(f"批次大小: {batch_size}")
print(f"总批次数量: {len(test_loader)}")

# ============================================================================
# 基本测试：计算总体准确率
# ============================================================================

print("\n" + "=" * 60)
print("开始测试...")
print("=" * 60)

model.eval()
test_loss = 0.0
class_correct = [0] * 10
class_total = [0] * 10

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

        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i]
            class_total[label] += 1

# 平均测试损失
test_loss = test_loss / len(test_loader.dataset)
print(f'\n测试损失: {test_loss:.6f}\n')

# 打印每个类别的准确率
print("各类别准确率:")
print("-" * 60)
for i in range(10):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'{class_names[i]:12s}: {accuracy:5.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})')

# 总体准确率
overall_accuracy = 100. * sum(class_correct) / sum(class_total)
print("-" * 60)
print(f'总体准确率: {overall_accuracy:.2f}% ({sum(class_correct):5d}/{sum(class_total):5d})')

# ============================================================================
# 详细分析
# ============================================================================

print("\n" + "=" * 60)
print("详细分析")
print("=" * 60)

# 收集所有预测结果用于详细分析
all_predictions = []
all_labels = []
all_confidences = []
all_correct_conf = []
all_incorrect_conf = []
correct_samples = []
incorrect_samples = []

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        probabilities = F.softmax(output, dim=1)
        confidence, preds = torch.max(probabilities, 1)

        # 将数据转移到CPU
        if train_on_gpu:
            preds_cpu = preds.cpu().numpy()
            target_cpu = target.cpu().numpy()
            confidence_cpu = confidence.cpu().numpy()
            data_cpu = data.cpu().numpy()
        else:
            preds_cpu = preds.numpy()
            target_cpu = target.numpy()
            confidence_cpu = confidence.numpy()
            data_cpu = data.numpy()

        # 收集数据
        all_predictions.extend(preds_cpu)
        all_labels.extend(target_cpu)
        all_confidences.extend(confidence_cpu)

        # 分析每个样本
        for i in range(len(target)):
            true_label = target_cpu[i]
            pred_label = preds_cpu[i]
            conf = confidence_cpu[i]

            if pred_label == true_label:
                all_correct_conf.append(conf)
                # 收集少量正确样本用于可视化
                if len(correct_samples) < 10 and batch_idx < 3:
                    correct_samples.append({
                        'image': data_cpu[i],
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': conf
                    })
            else:
                all_incorrect_conf.append(conf)
                # 收集少量错误样本用于可视化
                if len(incorrect_samples) < 10 and batch_idx < 3:
                    incorrect_samples.append({
                        'image': data_cpu[i],
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': conf
                    })

# 置信度分析
print(f"测试样本总数: {len(all_labels)}")
print(f"正确分类样本数: {len(all_correct_conf)}")
print(f"错误分类样本数: {len(all_incorrect_conf)}")
print(f"正确分类平均置信度: {np.mean(all_correct_conf):.3f}")
print(f"错误分类平均置信度: {np.mean(all_incorrect_conf):.3f}")
print(f"置信度差异: {np.mean(all_correct_conf) - np.mean(all_incorrect_conf):.3f}")

# ============================================================================
# 混淆矩阵
# ============================================================================

print("\n" + "=" * 60)
print("生成混淆矩阵...")
print("=" * 60)

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Samples'})
plt.title('Inception CNN - CIFAR-10 Confusion matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_inception.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 分类报告
# ============================================================================

print("\n详细分类报告:")
print("-" * 60)
report = classification_report(all_labels, all_predictions,
                               target_names=class_names,
                               digits=3)
print(report)

# ============================================================================
# 可视化正确分类的图片
# ============================================================================

print("\n可视化正确分类示例...")

if len(correct_samples) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('Inception CNN - Correct Classification Examples', fontsize=16, y=1.02)

    # 显示前10个正确样本
    for idx in range(min(10, len(correct_samples))):
        sample = correct_samples[idx]
        img = sample['image']

        ax = axes[idx // 5, idx % 5]
        imshow(img)
        ax.set_title(f"{class_names[sample['true_label']]}\n置信度: {sample['confidence']:.3f}",
                     color='green', fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(min(10, len(correct_samples)), 10):
        ax = axes[idx // 5, idx % 5]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('correctly_classified_inception.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("没有正确分类的样本")

# ============================================================================
# 可视化错误分类的图片
# ============================================================================

print("\n可视化错误分类示例...")

if len(incorrect_samples) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('Inception CNN - 错误分类示例', fontsize=16, y=1.02)

    # 显示前10个错误样本
    for idx in range(min(10, len(incorrect_samples))):
        sample = incorrect_samples[idx]
        img = sample['image']

        ax = axes[idx // 5, idx % 5]
        imshow(img)
        ax.set_title(
            f"真实: {class_names[sample['true_label']]}\n预测: {class_names[sample['pred_label']]}\n置信度: {sample['confidence']:.3f}",
            color='red', fontsize=9)
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(min(10, len(incorrect_samples)), 10):
        ax = axes[idx // 5, idx % 5]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('misclassified_inception.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("没有错误分类的样本")

# ============================================================================
# 置信度分布图
# ============================================================================

print("\n生成置信度分布图...")

if all_correct_conf and all_incorrect_conf:
    plt.figure(figsize=(10, 5))
    plt.hist([all_correct_conf, all_incorrect_conf],
             bins=20,
             label=['Correct class', 'Incorrect class'],
             alpha=0.7,
             color=['green', 'red'])
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Inception CNN - Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('confidence_distribution_inception.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 最常见错误分类模式
# ============================================================================

print("\n最常见错误分类模式:")
print("-" * 60)

if incorrect_samples:
    # 从所有错误中统计（不仅仅是可视化样本）
    error_pairs = []
    for i in range(len(all_labels)):
        if all_predictions[i] != all_labels[i]:
            error_pairs.append((all_labels[i], all_predictions[i]))

    if error_pairs:
        error_counter = Counter(error_pairs)
        most_common_errors = error_counter.most_common(5)

        print("前5个最常见错误模式:")
        for (true_idx, pred_idx), count in most_common_errors:
            true_name = class_names[true_idx]
            pred_name = class_names[pred_idx]
            print(f"{true_name:12s} → {pred_name:12s}: {count:3d} 次")

# ============================================================================
# 高置信度错误分析
# ============================================================================

print("\n高置信度错误分析:")
print("-" * 60)

if all_incorrect_conf:
    # 定义高置信度阈值
    high_conf_threshold = 0.7

    # 统计高置信度错误
    high_conf_errors = [c for c in all_incorrect_conf if c > high_conf_threshold]
    high_conf_error_ratio = len(high_conf_errors) / len(all_incorrect_conf) * 100

    print(f"高置信度阈值: >{high_conf_threshold}")
    print(f"高置信度错误数量: {len(high_conf_errors)}")
    print(f"错误中高置信度的比例: {high_conf_error_ratio:.1f}%")

    if high_conf_errors:
        print(f"最高置信度错误: {max(all_incorrect_conf):.3f}")
        print(f"高置信度错误平均置信度: {np.mean(high_conf_errors):.3f}")

# ============================================================================
# 按类别分析
# ============================================================================

print("\n按类别分析:")
print("-" * 60)

print(f"{'类别':<12} {'准确率':<8} {'正确数':<8} {'总数':<8} {'正确平均置信度':<15} {'错误平均置信度':<15}")
print("-" * 80)

for i in range(10):
    # 收集该类别所有样本的置信度
    class_correct_confs = []
    class_incorrect_confs = []

    for j in range(len(all_labels)):
        if all_labels[j] == i:
            if all_predictions[j] == i:
                class_correct_confs.append(all_confidences[j])
            else:
                class_incorrect_confs.append(all_confidences[j])

    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    correct_avg_conf = np.mean(class_correct_confs) if class_correct_confs else 0
    incorrect_avg_conf = np.mean(class_incorrect_confs) if class_incorrect_confs else 0

    print(f"{class_names[i]:<12} {accuracy:<8.2f} {class_correct[i]:<8} {class_total[i]:<8} "
          f"{correct_avg_conf:<15.3f} {incorrect_avg_conf:<15.3f}")

# ============================================================================
# 最终总结
# ============================================================================

print("\n" + "=" * 60)
print("测试分析完成！")
print("=" * 60)
print(f"模型名称: Inception CNN")
print(f"模型文件: {model_name}")
print(f"总体准确率: {overall_accuracy:.2f}%")
print(f"测试损失: {test_loss:.6f}")
print(f"可训练参数: {total_params:,}")
print("\n已生成以下分析文件:")
print("1. confusion_matrix_inception.png - 混淆矩阵")
print("2. correctly_classified_inception.png - 正确分类示例")
print("3. misclassified_inception.png - 错误分类示例")
print("4. confidence_distribution_inception.png - 置信度分布图")
print("=" * 60)

# 保存结果到文本文件
with open('test_results_inception.txt', 'w') as f:
    f.write(f"Inception CNN 测试结果\n")
    f.write("=" * 50 + "\n")
    f.write(f"模型文件: {model_name}\n")
    f.write(f"总体准确率: {overall_accuracy:.2f}%\n")
    f.write(f"测试损失: {test_loss:.6f}\n")
    f.write(f"可训练参数: {total_params:,}\n\n")

    f.write("各类别准确率:\n")
    f.write("-" * 40 + "\n")
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        f.write(f"{class_names[i]:12s}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})\n")

    f.write("\n置信度分析:\n")
    f.write("-" * 40 + "\n")
    f.write(f"测试样本总数: {len(all_labels)}\n")
    f.write(f"正确分类样本数: {len(all_correct_conf)}\n")
    f.write(f"错误分类样本数: {len(all_incorrect_conf)}\n")
    f.write(f"正确分类平均置信度: {np.mean(all_correct_conf):.3f}\n")
    f.write(f"错误分类平均置信度: {np.mean(all_incorrect_conf):.3f}\n")

print(f"\n详细结果已保存到: test_results_inception.txt")