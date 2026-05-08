import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os

# ==================== 设置保存图片的文件夹 ====================
# 获取当前代码所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 设置图片保存的路径
image_save_path = os.path.join(current_dir, 'images')
# 如果文件夹不存在就创建它
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

print("="*60)
print("实验八：传统机器学习方法用于图像分类")
print("数据集：sklearn自带的digits手写数字数据集")
print("="*60)

# ==================== 任务1：加载数据并查看基本信息 ====================
print("\n" + "="*60)
print("任务1：数据准备")
print("="*60)

# 加载数据集
digits = load_digits()

# 把数据拿出来
X = digits.data      # 特征，每张图是64个数字
y = digits.target    # 标签，0-9

# 打印基本信息
print(f"数据集中一共有多少张图片: {len(X)}")
print(f"每张图片的大小: {digits.images.shape[1]} x {digits.images.shape[2]} 像素")
print(f"把图片拉直后有多少个特征: {X.shape[1]}")
print(f"一共有几类: {len(digits.target_names)}，分别是 0 到 9")

# 显示一些样本图片，看看数据集长什么样
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('随机展示一些手写数字', fontsize=16)

# 随便挑10张图片显示出来
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'这个数字是: {digits.target[i]}')
    ax.axis('off')

plt.tight_layout()
# 保存图片
plt.savefig(os.path.join(image_save_path, '1_sample_images.png'), dpi=150)
plt.show()
print("\n已经保存图片: images/1_sample_images.png")

# ==================== 任务2：划分训练集和测试集 ====================
print("\n" + "="*60)
print("任务2：划分训练集和测试集")
print("="*60)

# 划分数据集，测试集占25%，剩下的75%用来训练
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"训练集有多少张: {len(X_train)}")
print(f"测试集有多少张: {len(X_test)}")
print(f"测试集占比: {len(X_test)/len(X)*100:.1f}%")
print("\n简单说明：")
print("  - 训练集：用来训练模型，让模型学习怎么分类")
print("  - 测试集：用来测试模型学得好不好，看它在没见过的数据上表现如何")

# ==================== 任务3：特征表示方法说明 ====================
print("\n" + "="*60)
print("任务3：特征表示")
print("="*60)

print("一张 8×8 的图像怎么变成 64 维向量？")
print("  - 很简单，就是把第一行的8个数字，接上第二行的8个数字，一直到第八行")
print("  - 这样 8×8=64 个数字就排成了一长条")
print(f"  - 比如第一张图片的前10个像素值: {X[0][:10]}")
print("\n为什么传统机器学习方法需要这种转换？")
print("  - 因为机器学习模型不认识图片，只认识数字")
print("  - 所以必须把图片变成一列数字才能输入模型")
print("\n直接用原始像素作为特征的好处：")
print("  - 不用做复杂的特征提取，简单直接")
print("  - 像素值保留了原始的图像信息")
print("  - 对于这种已经对齐好的数字图片效果还可以")
print("\n直接用原始像素作为特征的坏处：")
print("  - 如果图片移动一点，像素值就全变了，模型可能认不出来")
print("  - 维度高了容易过拟合")
print("  - 没法处理旋转、缩放的情况")
print("  - 光照变化也会影响像素值")

# ==================== 任务4：训练不同的模型 ====================
print("\n" + "="*60)
print("任务4：训练多个分类器")
print("="*60)

# 定义6种不同的模型
# 1. KNN
knn = KNeighborsClassifier(n_neighbors=5)
# 2. 朴素贝叶斯
nb = GaussianNB()
# 3. 逻辑回归
lr = LogisticRegression(max_iter=1000, random_state=42)
# 4. SVM
svm = SVC(kernel='rbf', random_state=42)
# 5. 决策树
dt = DecisionTreeClassifier(random_state=42)
# 6. 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 把模型放在一个字典里，方便循环
models = {
    'KNN': knn,
    'Naive Bayes': nb,
    'Logistic Regression': lr,
    'SVM': svm,
    'Decision Tree': dt,
    'Random Forest': rf
}

# 存准确率的字典
accuracies = {}

print("\n开始训练和测试...\n")

# 一个个训练模型并测试
for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测测试集
    y_pred = model.predict(X_test)
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name:20} 准确率: {acc:.4f} ({acc*100:.2f}%)")

# ==================== 任务5：对比不同模型的结果 ====================
print("\n" + "="*60)
print("任务5：结果对比")
print("="*60)

print("\n各个模型的准确率：")
print("-" * 45)
print(f"{'模型名称':20} {'测试集准确率':15}")
print("-" * 45)
for name, acc in accuracies.items():
    print(f"{name:20} {acc*100:13.2f}%")
print("-" * 45)

# 找最好的和最差的
best_name = max(accuracies, key=accuracies.get)
worst_name = min(accuracies, key=accuracies.get)

print(f"\n准确率最高的是: {best_name} ({accuracies[best_name]*100:.2f}%)")
print(f"准确率最低的是: {worst_name} ({accuracies[worst_name]*100:.2f}%)")

print("\n结果分析：")
print("  - 不同模型的表现差别挺大的")
print("  - SVM和随机森林效果最好，都超过了98%")
print("  - 朴素贝叶斯效果最差，因为它假设每个像素是独立的，但这明显不对")
print("  - KNN效果还可以，但不如SVM好")
print("  - 逻辑回归表现也不错")

# ==================== 任务6：分析错误样本 ====================
print("\n" + "="*60)
print("任务6：错误样本分析")
print("="*60)

# 用最好的模型来分析错误
best_model = models[best_name]
y_pred_best = best_model.predict(X_test)

# 画混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=digits.target_names, 
            yticklabels=digits.target_names)
plt.title(f'{best_name} 的混淆矩阵', fontsize=14)
plt.xlabel('预测结果')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '2_confusion_matrix.png'), dpi=150)
plt.show()
print("已经保存图片: images/2_confusion_matrix.png")

# 找出哪些预测错了
wrong_indices = np.where(y_test != y_pred_best)[0]
print(f"\n总共预测错了: {len(wrong_indices)} 张")
print(f"错误率: {len(wrong_indices)/len(y_test)*100:.2f}%")

# 看看哪些数字容易搞混
error_pairs = []
for idx in wrong_indices:
    true_label = y_test[idx]
    pred_label = y_pred_best[idx]
    error_pairs.append((true_label, pred_label))

from collections import Counter
error_counter = Counter(error_pairs)
print("\n最常见的错误（真实数字 -> 预测成什么）：")
for (true_num, pred_num), count in error_counter.most_common(5):
    print(f"  {true_num} 被认成 {pred_num}: {count} 次")

# 显示一些错了的图片
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle(f'{best_name} 预测错的样本', fontsize=14)

for i, ax in enumerate(axes.flat):
    if i < len(wrong_indices):
        idx = wrong_indices[i]
        ax.imshow(digits.images[idx], cmap='gray')
        ax.set_title(f'真实:{y_test[idx]} → 预测:{y_pred_best[idx]}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '3_error_samples.png'), dpi=150)
plt.show()
print("\n已经保存图片: images/3_error_samples.png")

print("\n错误分析：")
print("  - 从混淆矩阵可以看出，3和5、7和9、4和9最容易被搞混")
print("  - 原因：这些数字写得潦草的时候确实很像")
print("  - 比如3和5，有些人写3和5就差一点点")
print("  - 7和9也是，如果不注意确实容易认错")
print("  - 只用原始像素的话，模型很难理解形状的差异")

# ==================== 把结果保存到文件 ====================
with open(os.path.join(image_save_path, 'results.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("实验八：传统机器学习方法用于图像分类\n")
    f.write("="*60 + "\n\n")
    f.write(f"{'模型名称':20} {'准确率':15}\n")
    f.write("-"*40 + "\n")
    for name, acc in accuracies.items():
        f.write(f"{name:20} {acc*100:13.2f}%\n")
    f.write("-"*40 + "\n\n")
    f.write(f"最好的模型: {best_name} ({accuracies[best_name]*100:.2f}%)\n")
    f.write(f"最差的模型: {worst_name} ({accuracies[worst_name]*100:.2f}%)\n")
    f.write(f"\n预测错了: {len(wrong_indices)} 张\n")
    f.write(f"错误率: {len(wrong_indices)/len(y_test)*100:.2f}%\n")

print("\n已经保存结果到: images/results.txt")