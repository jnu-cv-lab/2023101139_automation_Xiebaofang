import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 设置保存路径 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(current_dir, 'images')
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

print("="*60)
print("实验九：PyTorch 入门与图像分类")
print("数据集：MNIST 手写数字")
print("="*60)

# ==================== 任务1：环境准备 ====================
print("\n" + "="*60)
print("任务1：环境准备")
print("="*60)

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"使用设备: {device}")

# 测试张量操作
x = torch.tensor([1, 2, 3])
print(f"张量测试: {x}")

# ==================== 任务2：加载数据集 ====================
print("\n" + "="*60)
print("任务2：加载数据集")
print("="*60)

# 数据预处理：转成张量并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载 MNIST 数据集
full_train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 划分训练集和验证集（训练集50000，验证集10000）
train_size = 50000
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset, [train_size, val_size]
)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 显示8张样本图片
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('MNIST 手写数字样本', fontsize=16)

for i, ax in enumerate(axes.flat):
    img, label = full_train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'标签: {label}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '1_sample_images.png'), dpi=150)
plt.show()
print("已保存: images/1_sample_images.png")

# ==================== 任务3：定义 CNN 模型 ====================
print("\n" + "="*60)
print("任务3：定义 CNN 模型")
print("="*60)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1：输入1通道，输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层2：输入32通道，输出64通道，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：输入64*7*7，输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：输入128，输出10
        self.fc2 = nn.Linear(128, 10)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 28 -> 14
        x = self.pool(self.relu(self.conv1(x)))
        # 14 -> 7
        x = self.pool(self.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params}")
print(f"可训练参数量: {trainable_params}")

# ==================== 任务4：训练模型 ====================
print("\n" + "="*60)
print("任务4：训练模型")
print("="*60)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录训练过程
num_epochs = 5
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# ==================== 任务6：测试模型 ====================
print("\n" + "="*60)
print("任务6：测试模型")
print("="*60)

model.eval()
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_loader)
test_acc = 100 * correct / total

print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集 Accuracy: {test_acc:.2f}%")

# 显示8张测试图像的预测结果
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle(f'测试集预测结果 (准确率: {test_acc:.2f}%)', fontsize=16)

test_images = []
test_labels = []
for images, labels in test_loader:
    test_images = images
    test_labels = labels
    break

model.eval()
with torch.no_grad():
    outputs = model(test_images.to(device))
    _, predictions = torch.max(outputs, 1)

for i, ax in enumerate(axes.flat):
    img = test_images[i].squeeze().numpy()
    true_label = test_labels[i].item()
    pred_label = predictions[i].item()
    color = 'green' if true_label == pred_label else 'red'
    ax.imshow(img, cmap='gray')
    ax.set_title(f'真实:{true_label} → 预测:{pred_label}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '2_test_predictions.png'), dpi=150)
plt.show()
print("已保存: images/2_test_predictions.png")

# ==================== 任务7：绘制训练曲线 ====================
print("\n" + "="*60)
print("任务7：绘制训练曲线")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲线
ax1.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
ax1.plot(range(1, num_epochs+1), val_losses, label='Val Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练和验证 Loss 曲线')
ax1.legend()
ax1.grid(True)

# Accuracy 曲线
ax2.plot(range(1, num_epochs+1), train_accs, label='Train Acc', marker='o')
ax2.plot(range(1, num_epochs+1), val_accs, label='Val Acc', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('训练和验证 Accuracy 曲线')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '3_training_curves.png'), dpi=150)
plt.show()
print("已保存: images/3_training_curves.png")

# ==================== 保存结果到文件 ====================
with open(os.path.join(image_save_path, 'results.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("实验九：PyTorch 入门与图像分类\n")
    f.write("="*60 + "\n\n")
    f.write(f"PyTorch 版本: {torch.__version__}\n")
    f.write(f"使用设备: {device}\n\n")
    f.write(f"训练集大小: {len(train_dataset)}\n")
    f.write(f"验证集大小: {len(val_dataset)}\n")
    f.write(f"测试集大小: {len(test_dataset)}\n\n")
    f.write(f"模型总参数量: {total_params}\n\n")
    f.write("各 Epoch 结果:\n")
    f.write("-"*60 + "\n")
    for i in range(num_epochs):
        f.write(f"Epoch {i+1}: Train Loss={train_losses[i]:.4f}, Train Acc={train_accs[i]:.2f}%, ")
        f.write(f"Val Loss={val_losses[i]:.4f}, Val Acc={val_accs[i]:.2f}%\n")
    f.write("-"*60 + "\n\n")
    f.write(f"测试集最终结果:\n")
    f.write(f"  Loss: {test_loss:.4f}\n")
    f.write(f"  Accuracy: {test_acc:.2f}%\n")

print("\n已保存结果到: images/results.txt")

# ==================== 结果分析 ====================
print("\n" + "="*60)
print("任务8：结果分析")
print("="*60)

print("\n1. 训练 loss 是否随着 epoch 增加而下降？")
if train_losses[-1] < train_losses[0]:
    print("   是，训练 loss 从 {:.4f} 下降到 {:.4f}".format(train_losses[0], train_losses[-1]))
else:
    print("   否")

print("\n2. 验证 accuracy 是否随着训练逐渐提升？")
if val_accs[-1] > val_accs[0]:
    print("   是，验证 accuracy 从 {:.2f}% 提升到 {:.2f}%".format(val_accs[0], val_accs[-1]))
else:
    print("   否")

print("\n3. 训练 accuracy 和验证 accuracy 是否存在明显差距？")
gap = train_accs[-1] - val_accs[-1]
print(f"   最后 epoch 差值: {gap:.2f}%")
if gap > 2:
    print("   存在明显差距，可能原因是训练集过拟合或验证集数据分布略有不同")
else:
    print("   差距不大，模型泛化能力较好")

print("\n4. 哪些数字更容易被分错？")
# 分析混淆情况
from collections import Counter
errors = []
for true, pred in zip(all_labels, all_preds):
    if true != pred:
        errors.append((true, pred))
error_counter = Counter(errors)
print("   最常见的错误：")
for (true, pred), count in error_counter.most_common(5):
    print(f"      {true} 被认成 {pred}: {count} 次")

print("\n5. MNIST 和 CIFAR-10 哪个更难？")
print("   CIFAR-10 更难。原因是：")
print("   - MNIST 是灰度图，CIFAR-10 是彩色图")
print("   - MNIST 数字在中心，CIFAR-10 物体位置变化大")
print("   - CIFAR-10 有背景干扰，物体形状更复杂")

print("\n" + "="*60)
print("实验完成！")
print(f"结果图片保存在: {image_save_path}")
print("="*60)