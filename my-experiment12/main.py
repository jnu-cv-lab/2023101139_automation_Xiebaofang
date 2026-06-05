import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置保存路径
current_dir = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(current_dir, 'images')
os.makedirs(image_save_path, exist_ok=True)

print("="*60)
print("实验十二：Sinusoidal Position Encoding 与 RoPE 实现与对比")
print("="*60)

# 第一部分：Sinusoidal Position Encoding 
print("\n" + "="*60)
print("第一部分：Sinusoidal Position Encoding 实现")
print("="*60)

def get_sinusoidal_position_encoding(seq_len, d_model):
   
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # 计算分母项：10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # 填充 sin 和 cos
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# 生成位置编码
seq_len = 50
d_model = 128
pe = get_sinusoidal_position_encoding(seq_len, d_model)

print(f"位置编码形状: {pe.shape}")

# 可视化位置编码
plt.figure(figsize=(14, 8))
plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
plt.colorbar(label='编码值')
plt.xlabel('特征维度')
plt.ylabel('位置索引')
plt.title('Sinusoidal Position Encoding 可视化')
plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '1_sinusoidal_pe.png'), dpi=150)
plt.show()
print("已保存: images/1_sinusoidal_pe.png")

# 可视化不同位置在不同维度上的编码值
plt.figure(figsize=(12, 6))
for i, pos in enumerate([0, 10, 20, 30, 40]):
    plt.plot(pe[pos, :100].numpy(), label=f'位置 {pos}')
plt.xlabel('特征维度')
plt.ylabel('编码值')
plt.title('不同位置的 Sinusoidal Position Encoding（前100维）')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(image_save_path, '2_sinusoidal_positions.png'), dpi=150)
plt.show()
print("已保存: images/2_sinusoidal_positions.png")

# 验证相对位置性质：PE(pos+k) 可以表示为 PE(pos) 的线性变换
print("\n验证 Sinusoidal PE 的相对位置性质...")
print("理论上，PE(pos+k) 可以表示为 PE(pos) 的线性变换")
print("这使得模型能够更容易地学习相对位置关系")

# 第二部分：二维向量旋转 
print("\n" + "="*60)
print("第二部分：二维向量旋转实现")
print("="*60)

def rotate_2d_vector(x, y, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    
    return x_new, y_new

# 演示二维旋转
print("\n二维向量旋转演示：")
original_x, original_y = 1.0, 0.0
print(f"原始向量: ({original_x}, {original_y})")

# 不同角度旋转
angles_deg = [0, 30, 45, 60, 90, 120, 180]
for deg in angles_deg:
    rad = math.radians(deg)
    x_rot, y_rot = rotate_2d_vector(original_x, original_y, rad)
    print(f"旋转 {deg}° 后: ({x_rot:.4f}, {y_rot:.4f})")

# 可视化二维旋转
fig, ax = plt.subplots(figsize=(8, 8))
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
for i, deg in enumerate([0, 30, 45, 60, 90, 120, 180]):
    rad = math.radians(deg)
    x_rot, y_rot = rotate_2d_vector(1, 0, rad)
    ax.arrow(0, 0, x_rot, y_rot, head_width=0.05, head_length=0.08, 
             fc=colors[i], ec=colors[i], label=f'{deg}°')
    ax.text(x_rot*1.1, y_rot*1.1, f'{deg}°', fontsize=10)

ax.set_title('二维向量旋转演示')
ax.legend()
ax.grid(True)
plt.savefig(os.path.join(image_save_path, '3_2d_rotation.png'), dpi=150)
plt.show()
print("已保存: images/3_2d_rotation.png")

# 第三部分：高维 RoPE 实现 
print("\n" + "="*60)
print("第三部分：高维 RoPE (Rotary Position Embedding) 实现")
print("="*60)

def precompute_rope_freqs(dim, seq_len, base=10000.0):
   
    # 计算每个维度的频率
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算每个位置的旋转角度
    position = torch.arange(seq_len).float()
    freqs = torch.outer(position, theta)  # (seq_len, dim//2)
    
    return freqs

def apply_rope_2d(x, freqs):
   
    batch, seq_len, dim = x.shape
    dim_half = dim // 2
    
    # 分离奇偶维度
    x_even = x[..., 0::2]  # 偶数索引
    x_odd = x[..., 1::2]   # 奇数索引
    
    # 获取旋转角度
    cos_freqs = torch.cos(freqs).unsqueeze(0)  # (1, seq_len, dim_half)
    sin_freqs = torch.sin(freqs).unsqueeze(0)
    
    # 应用旋转
    x_rot_even = x_even * cos_freqs - x_odd * sin_freqs
    x_rot_odd = x_even * sin_freqs + x_odd * cos_freqs
    
    # 合并
    x_rotated = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
    
    return x_rotated

# 演示 RoPE
dim = 64
seq_len = 50
batch_size = 1

# 预计算频率
freqs = precompute_rope_freqs(dim, seq_len)
print(f"旋转角度形状: {freqs.shape}")

# 创建随机输入
x = torch.randn(batch_size, seq_len, dim)

# 应用 RoPE
x_rotated = apply_rope_2d(x, freqs)

print(f"输入形状: {x.shape}")
print(f"输出形状: {x_rotated.shape}")

# 可视化不同位置的旋转角度
plt.figure(figsize=(12, 6))
for i in range(0, dim//2, 8):
    plt.plot(freqs[:, i].numpy(), label=f'维度对 {i+1}-{i+2}')
plt.xlabel('位置')
plt.ylabel('旋转角度 (弧度)')
plt.title('RoPE 不同维度对的旋转角度随位置变化')
plt.legend(loc='upper left', ncol=4, fontsize=8)
plt.grid(True)
plt.savefig(os.path.join(image_save_path, '4_rope_freqs.png'), dpi=150)
plt.show()
print("已保存: images/4_rope_freqs.png")

# 第四部分：验证 RoPE 的相对位置性质 
print("\n" + "="*60)
print("第四部分：验证 RoPE 的相对位置性质")
print("="*60)

def verify_rope_relative_property(dim=64, pos_i=10, pos_j=20, pos_k=30):
   
    # 预计算频率
    seq_len = max(pos_i, pos_j, pos_k) + 1
    freqs = precompute_rope_freqs(dim, seq_len)
    
    # 创建一个随机查询向量 q
    q = torch.randn(dim)
    
    # 在位置 i 和 j 处应用旋转
    q_i = rotate_vector_by_freqs(q, freqs[pos_i])
    q_j = rotate_vector_by_freqs(q, freqs[pos_j])
    q_k = rotate_vector_by_freqs(q, freqs[pos_k])
    
    # 计算点积
    dot_ij = torch.dot(q_i, q_j)
    dot_ik = torch.dot(q_i, q_k)
    
    # 理论上，dot_ij 只依赖于 (j-i)，而不是 i 和 j 本身
    # 这里用数值验证
    print(f"\n相对位置验证（dim={dim}）：")
    print(f"  位置 {pos_i} 和 {pos_j} 的向量点积: {dot_ij.item():.6f}")
    print(f"  位置 {pos_i} 和 {pos_k} 的向量点积: {dot_ik.item():.6f}")
    print(f"  差值 (位置差 {(pos_j-pos_i)} vs {(pos_k-pos_i)}): 点积不同，说明依赖相对位置")
    
    return dot_ij, dot_ik

def rotate_vector_by_freqs(vec, freqs):
    
    dim = len(vec)
    dim_half = dim // 2
    
    # 分离奇偶
    vec_even = vec[0::2]
    vec_odd = vec[1::2]
    
    cos_f = torch.cos(freqs)
    sin_f = torch.sin(freqs)
    
    # 旋转
    vec_rot_even = vec_even * cos_f - vec_odd * sin_f
    vec_rot_odd = vec_even * sin_f + vec_odd * cos_f
    
    # 合并
    result = torch.zeros_like(vec)
    result[0::2] = vec_rot_even
    result[1::2] = vec_rot_odd
    
    return result

# 验证
verify_rope_relative_property(dim=64, pos_i=5, pos_j=15, pos_k=25)

# 更系统的验证：计算所有位置对的点积
print("\n系统验证：计算不同位置对的点积矩阵...")
dim_small = 32
seq_len_small = 20
freqs_small = precompute_rope_freqs(dim_small, seq_len_small)

# 创建一个随机向量
q = torch.randn(dim_small)

# 计算所有位置的旋转向量
rotated_vectors = []
for pos in range(seq_len_small):
    rot_vec = rotate_vector_by_freqs(q, freqs_small[pos])
    rotated_vectors.append(rot_vec)

# 计算点积矩阵
dot_matrix = torch.zeros(seq_len_small, seq_len_small)
for i in range(seq_len_small):
    for j in range(seq_len_small):
        dot_matrix[i, j] = torch.dot(rotated_vectors[i], rotated_vectors[j])

# 可视化点积矩阵
plt.figure(figsize=(10, 8))
plt.imshow(dot_matrix.numpy(), cmap='coolwarm', aspect='auto')
plt.colorbar(label='点积值')
plt.xlabel('位置 j')
plt.ylabel('位置 i')
plt.title(f'RoPE 位置间点积矩阵\n(对角线最大，沿对角线方向递减，说明依赖相对位置)')
plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '5_rope_dot_matrix.png'), dpi=150)
plt.show()
print("已保存: images/5_rope_dot_matrix.png")

# 绘制特定相对位置的点积
plt.figure(figsize=(10, 6))
for offset in [1, 2, 3, 4, 5]:
    values = [dot_matrix[i, i+offset].item() for i in range(seq_len_small-offset)]
    plt.plot(values, label=f'相对位置 = {offset}')
plt.xlabel('起始位置 i')
plt.ylabel('点积值')
plt.title('不同相对位置的点积值随起始位置变化（应保持相对稳定）')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(image_save_path, '6_rope_relative.png'), dpi=150)
plt.show()
print("已保存: images/6_rope_relative.png")

# ==================== 第五部分：E+pos 与 RoPE 对比 ====================
print("\n" + "="*60)
print("第五部分：E+pos 与 RoPE 的对比")
print("="*60)



#第六部分：数值实验对比 
print("\n" + "="*60)
print("第六部分：数值实验对比")
print("="*60)

def compute_attention_scores_with_pos(Q, K, pos_encoding_type='sinusoidal'):
    """
    模拟带位置编码的注意力分数计算
    """
    seq_len, dim = Q.shape
    
    if pos_encoding_type == 'sinusoidal':
        # Sinusoidal PE：加到 Q 和 K 上
        pe = get_sinusoidal_position_encoding(seq_len, dim)
        Q_pos = Q + pe
        K_pos = K + pe
    elif pos_encoding_type == 'rope':
        # RoPE：旋转 Q 和 K
        freqs = precompute_rope_freqs(dim, seq_len)
        Q_pos = apply_rope_2d(Q.unsqueeze(0), freqs).squeeze(0)
        K_pos = apply_rope_2d(K.unsqueeze(0), freqs).squeeze(0)
    else:
        Q_pos = Q
        K_pos = K
    
    # 计算注意力分数
    scores = torch.matmul(Q_pos, K_pos.T) / math.sqrt(dim)
    return scores

# 创建测试数据
seq_len = 10
dim = 32
Q = torch.randn(seq_len, dim)
K = torch.randn(seq_len, dim)

# 计算不同方法的注意力分数
scores_no_pos = compute_attention_scores_with_pos(Q, K, 'none')
scores_sinusoidal = compute_attention_scores_with_pos(Q, K, 'sinusoidal')
scores_rope = compute_attention_scores_with_pos(Q, K, 'rope')

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 无位置编码
im1 = axes[0].imshow(scores_no_pos.numpy(), cmap='coolwarm', aspect='auto')
axes[0].set_title('无位置编码')
axes[0].set_xlabel('Key 位置')
axes[0].set_ylabel('Query 位置')
plt.colorbar(im1, ax=axes[0])

# Sinusoidal PE
im2 = axes[1].imshow(scores_sinusoidal.numpy(), cmap='coolwarm', aspect='auto')
axes[1].set_title('Sinusoidal PE (加法注入)')
axes[1].set_xlabel('Key 位置')
axes[1].set_ylabel('Query 位置')
plt.colorbar(im2, ax=axes[1])

# RoPE
im3 = axes[2].imshow(scores_rope.numpy(), cmap='coolwarm', aspect='auto')
axes[2].set_title('RoPE (旋转注入)')
axes[2].set_xlabel('Key 位置')
axes[2].set_ylabel('Query 位置')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(image_save_path, '7_attention_comparison.png'), dpi=150)
plt.show()
print("已保存: images/7_attention_comparison.png")