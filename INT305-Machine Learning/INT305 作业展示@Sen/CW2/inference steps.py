import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter

# 检查CUDA是否可用
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Testing on CPU ...')
else:
    print('CUDA is available! Testing on GPU ...')

# 加载类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# 图像显示函数
# 删除或修改原来的imshow函数
def imshow(img, mean=None, std=None):
    """
    支持自定义归一化参数的imshow
    """
    if mean is None:
        mean = [0.4914, 0.4822, 0.4465]  # 使用你的mean
    if std is None:
        std = [0.2470, 0.2435, 0.2616]  # 使用你的std

    # 转换为numpy（如果是tensor）
    if torch.is_tensor(img):
        img = img.cpu().numpy()

    # 反归一化
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = img * std + mean

    # 裁剪到有效范围
    img = np.clip(img, 0, 1)

    # 调整维度并显示
    plt.imshow(np.transpose(img, (1, 2, 0)))


# 定义CNN模型（必须与训练时相同）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            # first block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # fully connected layer
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# 创建模型
model = Net()
if train_on_gpu:
    model.cuda()

# 加载训练好的模型
print("Model loading...")
model.load_state_dict(torch.load('model_cifar.pt'))
print(f"Successfully loaded model")

# 损失函数
criterion = nn.CrossEntropyLoss()

# 测试数据预处理（与训练时相同，但不包含数据增强）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std= (0.2470, 0.2435, 0.2616))
])

# 加载测试数据
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform_test)

batch_size = 128
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=0)

print("\n" + "=" * 60)
print("Start Testing")
print("=" * 60)

# ============================================================================
# 基本测试：计算总体准确率
# ============================================================================

model.eval()
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data, target in test_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# 平均测试损失
test_loss = test_loss / len(test_loader.dataset)
print(f'测试损失: {test_loss:.6f}\n')

# print accuracy of every class
for i in range(10):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(
            f'Type {class_names[i]:12s} Accuracy: {accuracy:5.2f}% ({int(class_correct[i]):4d}/{int(class_total[i]):4d})')

# print overall accuracy
overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
print(
    f'\nTest Accuracy (Overall): {overall_accuracy:.2f}% ({int(np.sum(class_correct)):5d}/{int(np.sum(class_total)):5d})')

# ============================================================================
# 详细分析：收集所有预测用于进一步分析
# ============================================================================

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)

# 收集所有预测结果
all_predictions = []
all_confidences = []
all_labels = []
all_correct_confidences = []  # 新增：所有正确样本的置信度
all_incorrect_confidences = []  # 新增：所有错误样本的置信度
correct_samples = []  # 用于可视化的少量样本
incorrect_samples = []  # 用于可视化的少量样本

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        probabilities = F.softmax(output, dim=1)
        confidence, preds = torch.max(probabilities, 1)

        # 1. 收集所有数据用于分析
        all_confidences.extend(confidence.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        # 2. 按正确/错误分类收集所有置信度
        for i in range(len(target)):
            if preds[i].item() == target[i].item():
                all_correct_confidences.append(confidence[i].item())
            else:
                all_incorrect_confidences.append(confidence[i].item())

        # 3. 收集少量样本用于可视化（只取前几个批次）
        if batch_idx < 5:
            for i in range(len(target)):
                image_data = data[i].cpu().numpy()
                true_label = target[i].item()
                pred_label = preds[i].item()
                conf = confidence[i].item()

                sample_info = {
                    'image': image_data,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': conf
                }

                if true_label == pred_label:
                    correct_samples.append(sample_info)
                else:
                    incorrect_samples.append(sample_info)

# 诊断输出
print(f"总测试样本数: {len(all_labels)}")
print(f"所有正确样本的置信度数量: {len(all_correct_confidences)}")
print(f"所有错误样本的置信度数量: {len(all_incorrect_confidences)}")
print(f"用于可视化的正确样本数: {len(correct_samples)}")
print(f"用于可视化的错误样本数: {len(incorrect_samples)}")
print(f"从置信度计算出的准确率: {100 * len(all_correct_confidences) / len(all_labels):.2f}%")

# ============================================================================
# 混淆矩阵
# ============================================================================

print("\nConfusion matrix:...")
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Samples'})
plt.title('CIFAR-10 Confusion matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 分类报告
# ============================================================================

print("\nDetailed report:")
print("-" * 60)
report = classification_report(all_labels, all_predictions,
                               target_names=class_names,
                               digits=3)
print(report)

# ============================================================================
# 可视化正确分类的图片
# ============================================================================

print("\nVisualization Correct Examples...")
if len(correct_samples) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('Correct Classification Examples', fontsize=16, y=1.02)

    # 定义归一化参数
    MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    STD = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)


    # 辅助函数：正确的反归一化
    def denormalize(img):
        img = img * STD + MEAN
        img = np.clip(img, 0, 1)
        return np.transpose(img, (1, 2, 0))


    # 选择前10个正确分类的样本
    samples_to_show = min(10, len(correct_samples))
    for idx in range(samples_to_show):
        sample = correct_samples[idx]
        img = sample['image']

        # ✅ 使用正确的反归一化
        img_display = denormalize(img)

        ax = axes[idx // 5, idx % 5]
        ax.imshow(img_display)
        ax.set_title(f"{class_names[sample['true_label']]}\nConfidence value: {sample['confidence']:.3f}",
                     color='green', fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(samples_to_show, 10):
        ax = axes[idx // 5, idx % 5]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('correctly_classified.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No correct classifications")
# ============================================================================
# 可视化错误分类的图片
# ============================================================================

print("\nVisualization Incorrect Examples...")
if len(incorrect_samples) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('Incorrect Classification Examples', fontsize=16, y=1.02)

    # 定义归一化参数（与上面相同）
    MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    STD = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)


    def denormalize(img):
        img = img * STD + MEAN
        img = np.clip(img, 0, 1)
        return np.transpose(img, (1, 2, 0))


    # 选择前10个错误分类的样本
    samples_to_show = min(10, len(incorrect_samples))
    for idx in range(samples_to_show):
        sample = incorrect_samples[idx]
        img = sample['image']

        # ✅ 使用正确的反归一化
        img_display = denormalize(img)

        ax = axes[idx // 5, idx % 5]
        ax.imshow(img_display)
        ax.set_title(
            f"Real: {class_names[sample['true_label']]}\nPredicted: {class_names[sample['pred_label']]}\nConfidence value: {sample['confidence']:.3f}",
            color='red', fontsize=9)
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(samples_to_show, 10):
        ax = axes[idx // 5, idx % 5]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('misclassified.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No incorrect classifications")

# ============================================================================
# 置信度分析 - 使用所有数据
# ============================================================================

print("\nConfidence value analysis (使用所有测试数据):")
print("-" * 60)

if all_correct_confidences and all_incorrect_confidences:
    print(f"Total test samples: {len(all_labels)}")
    print(f"Correct classification samples: {len(all_correct_confidences)}")
    print(f"Incorrect classification samples: {len(all_incorrect_confidences)}")
    print(f"Accuracy from confidence analysis: {100 * len(all_correct_confidences) / len(all_labels):.2f}%")
    print(f"Correct classification average confidence value: {np.mean(all_correct_confidences):.3f}")
    print(f"Incorrect classification average confidence value: {np.mean(all_incorrect_confidences):.3f}")

    # 置信度分布图 - 使用所有数据
    plt.figure(figsize=(10, 5))
    plt.hist([all_correct_confidences, all_incorrect_confidences],
             bins=20,
             label=['Correct', 'Incorrect'],
             alpha=0.7,
             color=['green', 'red'])
    plt.xlabel('Confidence value')
    plt.ylabel('Frequency (Number of samples)')
    plt.title(f'Correct vs Incorrect Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No data for confidence analysis")

# ============================================================================
# 新增：高置信度错误分析
# ============================================================================

print("\nHigh-confidence error analysis:")
print("-" * 60)

if all_incorrect_confidences:
    # 定义高置信度阈值
    high_conf_threshold = 0.7

    # 统计高置信度错误
    high_conf_errors = [c for c in all_incorrect_confidences if c > high_conf_threshold]
    high_conf_error_ratio = len(high_conf_errors) / len(all_incorrect_confidences) * 100

    print(f"High-confidence threshold: >{high_conf_threshold}")
    print(f"High-confidence errors: {len(high_conf_errors)}")
    print(f"Percentage of errors with high confidence: {high_conf_error_ratio:.1f}%")

    if high_conf_errors:
        print(f"Highest confidence among errors: {max(all_incorrect_confidences):.3f}")
        print(f"Average confidence of high-confidence errors: {np.mean(high_conf_errors):.3f}")

        # 高置信度错误分布
        plt.figure(figsize=(8, 4))
        plt.hist(high_conf_errors, bins=10, color='red', alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of High-Confidence Errors (>{high_conf_threshold})\n{len(high_conf_errors)} samples')
        plt.axvline(x=high_conf_threshold, color='r', linestyle='--', alpha=0.5,
                    label=f'Threshold ({high_conf_threshold})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('high_confidence_errors.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# 新增：按类别分析置信度
# ============================================================================

print("\n" + "=" * 60)
print("各类别置信度分析")
print("=" * 60)

# 按类别收集置信度
class_correct_conf = [[] for _ in range(10)]
class_incorrect_conf = [[] for _ in range(10)]

# 需要重新处理数据或使用已收集的数据
# 这里我们使用已收集的correct_samples和incorrect_samples作为示例
# 注意：这只是部分数据，如果需要完整数据需要重新处理

print(f"{'类别':<12} {'正确平均':<8} {'错误平均':<8} {'差距':<8} {'正确样本数':<12} {'错误样本数':<12}")
print("-" * 80)

for i, class_name in enumerate(class_names):
    # 收集该类别的所有置信度（从所有数据中）
    class_correct = []
    class_incorrect = []

    # 遍历所有批次的数据（这里简化处理，实际应该从所有数据中提取）
    # 为了简化，我们先使用可视化样本中的数据
    for sample in correct_samples:
        if sample['true_label'] == i:
            class_correct.append(sample['confidence'])

    for sample in incorrect_samples:
        if sample['true_label'] == i:
            class_incorrect.append(sample['confidence'])

    if class_correct:
        correct_avg = np.mean(class_correct)
    else:
        correct_avg = 0

    if class_incorrect:
        incorrect_avg = np.mean(class_incorrect)
    else:
        incorrect_avg = 0

    print(f"{class_name:<12} {correct_avg:<8.3f} {incorrect_avg:<8.3f} "
          f"{correct_avg - incorrect_avg:<8.3f} {len(class_correct):<12} {len(class_incorrect):<12}")

# ============================================================================
# 新增：高风险混淆对分析
# ============================================================================

print("\n" + "=" * 60)
print("高风险混淆对分析（置信度>0.7的错误）")
print("=" * 60)

# 从incorrect_samples中找出高置信度错误
high_conf_errors = []
for sample in incorrect_samples:
    if sample['confidence'] > 0.7:  # 高置信度阈值
        true_name = class_names[sample['true_label']]
        pred_name = class_names[sample['pred_label']]
        high_conf_errors.append((true_name, pred_name, sample['confidence']))

if high_conf_errors:
    # 按置信度排序
    high_conf_errors.sort(key=lambda x: x[2], reverse=True)

    print(f"发现 {len(high_conf_errors)} 个高置信度(>0.7)错误（来自可视化样本）：")
    print("\n前10个最自信的错误：")
    for i, (true_name, pred_name, conf) in enumerate(high_conf_errors[:10], 1):
        print(f"{i:2d}. {true_name:10s} → {pred_name:10s} : {conf:.3f}")

    # 统计最常见的混淆对
    pair_counter = Counter([(true, pred) for true, pred, _ in high_conf_errors])

    print("\n最常见的混淆模式：")
    for (true_name, pred_name), count in pair_counter.most_common(5):
        print(f"{true_name:10s} → {pred_name:10s} : {count}次")
else:
    print("在可视化样本中没有发现高置信度错误")

# ============================================================================
# 最常见错误分类模式
# ============================================================================

print("\nNormal incorrect classification pattern:")
print("-" * 60)

if incorrect_samples:
    error_pairs = []
    for sample in incorrect_samples:
        error_pairs.append((sample['true_label'], sample['pred_label']))

    error_counter = Counter(error_pairs)
    most_common_errors = error_counter.most_common(5)

    for (true_idx, pred_idx), count in most_common_errors:
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        print(f"{true_name:12s} → {pred_name:12s}: {count:3d} times")

# ============================================================================
# 最终总结
# ============================================================================

print("\n" + "=" * 60)
print("测试分析完成！")
print("=" * 60)
print("已生成以下文件:")
print("1. confusion_matrix.png - 混淆矩阵")
print("2. correctly_classified.png - 正确分类示例")
print("3. misclassified.png - 错误分类示例")
print("4. confidence_distribution.png - 置信度分布图")
print("5. high_confidence_errors.png - 高置信度错误分布")
print("\n模型性能总结:")
print(f"- 总体准确率: {overall_accuracy:.2f}%")
print(f"- 测试损失: {test_loss:.6f}")
print(f"- 正确分类平均置信度: {np.mean(all_correct_confidences):.3f}")
print(f"- 错误分类平均置信度: {np.mean(all_incorrect_confidences):.3f}")
print("=" * 60)

# 可选：显示一批测试图像及其预测
if __name__ == '__main__':
    print("\nShowing some predicted examples...")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    if train_on_gpu:
        images = images.cuda()

    model.eval()
    with torch.no_grad():
        output = model(images)
        _, preds_tensor = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)
        confidences, _ = torch.max(probabilities, 1)

    if train_on_gpu:
        preds = preds_tensor.cpu().numpy()
        images_np = images.cpu().numpy()
        confs = confidences.cpu().numpy()
    else:
        preds = preds_tensor.numpy()
        images_np = images.numpy()
        confs = confidences.numpy()

    # 定义反归一化参数
    MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    STD = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)

    fig = plt.figure(figsize=(25, 6))
    for idx in range(min(20, len(images_np))):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])

        # ✅ 正确的反归一化显示
        img = images_np[idx]
        img = img * STD + MEAN
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))

        ax.imshow(img)
        true_label = labels[idx].item()
        pred_label = preds[idx]
        conf = confs[idx]

        color = "green" if pred_label == true_label else "red"
        ax.set_title(
            f"Predicted: {class_names[pred_label]}\nActual: {class_names[true_label]}\nConfidence value: {conf:.3f}",
            color=color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()