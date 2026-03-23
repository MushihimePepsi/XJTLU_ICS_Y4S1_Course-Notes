import numpy as np

# 假设的类别名称
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 假设的每个类别的正确预测数和总预测数
class_correct = np.array([789, 901, 667, 599, 768, 751, 844, 823, 905, 901])
class_total = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

# 打印每个类别的准确率
for i in range(len(class_names)):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(
            f'Type {class_names[i]:12s} Accuracy: {accuracy:5.2f}% ({int(class_correct[i]):4d}/{int(class_total[i]):4d})')

# 总体准确率
overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
print(
    f'\nTest Accuracy (Overall): {overall_accuracy:.2f}% ({int(np.sum(class_correct)):5d}/{int(np.sum(class_total)):5d})')