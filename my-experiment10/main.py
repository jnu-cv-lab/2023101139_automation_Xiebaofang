import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置保存路径
current_dir = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(current_dir, 'images')
os.makedirs(image_save_path, exist_ok=True)

print("="*60)
print("实验十：CNN训练过程分析、优化器对比、特征可视化")
print("="*60)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 数据加载 ====================
print("\n加载MNIST数据集...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 划分训练集和验证集
train_size = 50000
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集: {len(train_subset)}")
print(f"验证集: {len(val_subset)}")
print(f"测试集: {len(test_dataset)}")


# ==================== 定义CNN模型 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==================== 训练函数 ====================
def train_one_model(optimizer_name, lr, num_epochs=5):
    """训练模型并返回结果"""
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
    
    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    
    return {
        'model': model,
        'test_acc': test_acc,
        'all_preds': all_preds,
        'all_labels': all_labels_list,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


# ==================== 任务1：基准模型 ====================
print("\n" + "="*60)
print("任务1：基准模型（Adam, lr=0.001）")
print("="*60)

base_result = train_one_model('Adam', 0.001, num_epochs=5)
print(f"\n基准模型测试准确率: {base_result['test_acc']:.2f}%")


# ==================== 任务2：优化器对比 ====================
print("\n" + "="*60)
print("任务2：优化器对比")
print("="*60)

opt_configs = [
    ('SGD', 0.01),
    ('SGD+Momentum', 0.01),
    ('Adam', 0.001)
]

opt_results = {}

for opt_name, lr in opt_configs:
    print(f"\n--- 训练 {opt_name} (lr={lr}) ---")
    result = train_one_model(opt_name, lr, num_epochs=5)
    opt_results[opt_name] = result
    print(f"{opt_name} 测试准确率: {result['test_acc']:.2f}%")


# 绘制优化器对比曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, res in opt_results.items():
    axes[0].plot(range(1, 6), res['val_losses'], label=name, marker='o')
    axes[1].plot(range(1, 6), res['val_accs'], label=name, marker='s')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('优化器对比 - 验证Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy (%)')
axes[1].set_title('优化器对比 - 验证Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '1_optimizer_comparison.png'), dpi=150)
plt.show()
print("\n已保存: images/1_optimizer_comparison.png")


# ==================== 任务3：学习率对比 ====================
print("\n" + "="*60)
print("任务3：学习率对比（Adam）")
print("="*60)

lr_values = [0.1, 0.01, 0.001]
lr_results = {}

for lr in lr_values:
    print(f"\n--- 训练 lr={lr} ---")
    result = train_one_model('Adam', lr, num_epochs=5)
    lr_results[lr] = result
    print(f"lr={lr} 测试准确率: {result['test_acc']:.2f}%")


# 绘制学习率对比曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr, res in lr_results.items():
    axes[0].plot(range(1, 6), res['val_losses'], label=f'lr={lr}', marker='o')
    axes[1].plot(range(1, 6), res['val_accs'], label=f'lr={lr}', marker='s')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('学习率对比 - 验证Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy (%)')
axes[1].set_title('学习率对比 - 验证Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '2_lr_comparison.png'), dpi=150)
plt.show()
print("\n已保存: images/2_lr_comparison.png")


# ==================== 任务4：卷积核可视化 ====================
print("\n" + "="*60)
print("任务4：卷积核可视化")
print("="*60)

conv1_weights = base_result['model'].conv1.weight.data.cpu()

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('第一层卷积核（32个，每个3x3）', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < 32:
        kernel = conv1_weights[i, 0]
        kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min())
        ax.imshow(kernel_norm, cmap='gray')
        ax.set_title(f'{i+1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '3_conv1_kernels.png'), dpi=150)
plt.show()
print("已保存: images/3_conv1_kernels.png")


# ==================== 任务5：Feature map可视化 ====================
print("\n" + "="*60)
print("任务5：Feature map可视化")
print("="*60)

# 取一张测试图片
test_iter = iter(test_loader)
sample_images, sample_labels = next(test_iter)
sample_image = sample_images[0:1].to(device)

# 获取第一层卷积输出
base_result['model'].eval()
with torch.no_grad():
    conv1_output = base_result['model'].conv1(sample_image)

# 显示原图
plt.figure(figsize=(4, 4))
plt.imshow(sample_image[0, 0].cpu(), cmap='gray')
plt.title(f'输入图片（标签: {sample_labels[0].item()}）')
plt.axis('off')
plt.savefig(os.path.join(image_save_path, '4_input_image.png'), dpi=150)
plt.show()

# 显示feature maps
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('第一层卷积输出的Feature Maps（32个通道）', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < 32:
        fm = conv1_output[0, i].cpu()
        ax.imshow(fm, cmap='gray')
        ax.set_title(f'Ch{i+1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '5_feature_maps.png'), dpi=150)
plt.show()
print("已保存: images/5_feature_maps.png")


# ==================== 任务6：错误样本分析 ====================
print("\n" + "="*60)
print("任务6：错误样本分析")
print("="*60)

all_preds = base_result['all_preds']
all_labels_list = base_result['all_labels']

# 找出错误
error_indices = []
for i in range(len(all_labels_list)):
    if all_preds[i] != all_labels_list[i]:
        error_indices.append(i)

print(f"总测试样本: {len(all_labels_list)}")
print(f"错误样本数: {len(error_indices)}")
print(f"错误率: {len(error_indices)/len(all_labels_list)*100:.2f}%")

# 统计混淆
error_pairs = []
for i in error_indices:
    error_pairs.append((all_labels_list[i], all_preds[i]))
error_counter = Counter(error_pairs)

print("\n最常见错误：")
for (true, pred), count in error_counter.most_common(5):
    print(f"  {true} -> {pred}: {count}次")

# 收集测试集所有图像
all_test_images = []
for images, labels in test_loader:
    for img in images:
        all_test_images.append(img)

# 显示错误样本图片
num_to_show = min(16, len(error_indices))
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('错误分类样本（真实->预测）', fontsize=16)

for i in range(num_to_show):
    row = i // 4
    col = i % 4
    idx = error_indices[i]
    img = all_test_images[idx].squeeze().numpy()
    true_l = all_labels_list[idx]
    pred_l = all_preds[idx]
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f'{true_l}->{pred_l}', fontsize=10, color='red')
    axes[row, col].axis('off')

# 隐藏多余的子图
for i in range(num_to_show, 16):
    row = i // 4
    col = i % 4
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '6_error_samples.png'), dpi=150)
plt.show()
print("已保存: images/6_error_samples.png")


# ==================== 任务7：混淆矩阵 ====================
print("\n" + "="*60)
print("任务7：混淆矩阵")
print("="*60)

cm = confusion_matrix(all_labels_list, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('测试集混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '7_confusion_matrix.png'), dpi=150)
plt.show()
print("已保存: images/7_confusion_matrix.png")


# ==================== 保存结果 ====================
with open(os.path.join(image_save_path, 'results.txt'), 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("实验十：CNN训练过程分析、优化器对比、特征可视化\n")
    f.write("="*60 + "\n\n")
    
    f.write("【基准模型】\n")
    f.write(f"测试准确率: {base_result['test_acc']:.2f}%\n\n")
    
    f.write("【优化器对比】\n")
    for name, res in opt_results.items():
        f.write(f"{name}: {res['test_acc']:.2f}%\n")
    f.write("\n")
    
    f.write("【学习率对比】\n")
    for lr, res in lr_results.items():
        f.write(f"lr={lr}: {res['test_acc']:.2f}%\n")
    f.write("\n")
    
    f.write("【错误分析】\n")
    f.write(f"总测试样本: {len(all_labels_list)}\n")
    f.write(f"错误样本数: {len(error_indices)}\n")
    f.write(f"错误率: {len(error_indices)/len(all_labels_list)*100:.2f}%\n")

print("\n实验完成！所有结果保存在 images 文件夹")