"""
训练：Skeleton Transformer 羽毛球击球动作识别
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 设置保存路径 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(current_dir, 'images')
os.makedirs(image_save_path, exist_ok=True)

print("="*60)
print("羽毛球击球动作识别 - Skeleton Transformer 训练")
print("="*60)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 加载数据 ====================
print("\n加载预处理数据...")
processed_dir = os.path.join(current_dir, 'processed')

X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

with open(os.path.join(processed_dir, 'label_map.json'), 'r') as f:
    label_map = json.load(f)

# 反向映射
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")
print(f"类别数: {num_classes}")
print(f"类别映射: {label_map}")

# ==================== 数据集类 ====================
class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== Transformer 模型 ====================
class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, num_classes=6, dropout=0.1, max_len=30):
        super(SkeletonTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.classifier(x)
        return x

# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(loader, desc="训练", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="验证", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels

# ==================== 创建数据加载器 ====================
batch_size = 32
train_dataset = SkeletonDataset(X_train, y_train)
test_dataset = SkeletonDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# ==================== 模型初始化 ====================
model = SkeletonTransformer(
    input_dim=132,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    num_classes=num_classes,
    dropout=0.1,
    max_len=30
).to(device)

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 训练配置 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

num_epochs = 30
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0

print("\n开始训练...")
print("="*60)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(image_save_path, 'best_model.pth'))
        print(f"  -> 保存最佳模型 (准确率: {val_acc:.4f})")

# ==================== 最终评估 ====================
print("\n" + "="*60)
print("最终评估")
print("="*60)

model.load_state_dict(torch.load(os.path.join(image_save_path, 'best_model.pth')))
_, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

# 分类报告
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=[idx_to_label[i] for i in range(num_classes)]))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[idx_to_label[i] for i in range(num_classes)],
            yticklabels=[idx_to_label[i] for i in range(num_classes)])
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig(os.path.join(image_save_path, 'confusion_matrix.png'), dpi=150)
plt.show()
print("已保存: images/confusion_matrix.png")

# ==================== 绘制训练曲线 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
axes[0].plot(range(1, num_epochs+1), val_losses, label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练和验证 Loss 曲线')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, num_epochs+1), train_accs, label='Train Acc', marker='o')
axes[1].plot(range(1, num_epochs+1), val_accs, label='Val Acc', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('训练和验证 Accuracy 曲线')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, 'training_curves.png'), dpi=150)
plt.show()
print("已保存: images/training_curves.png")

# ==================== 保存结果 ====================
with open(os.path.join(image_save_path, 'training_results.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("羽毛球击球动作识别 - 训练结果\n")
    f.write("="*60 + "\n\n")
    f.write(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
    f.write("分类报告:\n")
    f.write(classification_report(all_labels, all_preds, target_names=[idx_to_label[i] for i in range(num_classes)]))
    f.write("\n最佳模型保存路径: images/best_model.pth\n")

print("\n训练完成！")
print(f"最佳模型保存: images/best_model.pth")
print(f"训练曲线: images/training_curves.png")
print(f"混淆矩阵: images/confusion_matrix.png")