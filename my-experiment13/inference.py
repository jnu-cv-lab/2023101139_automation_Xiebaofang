"""
推理：用训练好的模型预测单个视频的动作类别
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import json

# ==================== 模型定义（必须和训练时一致）====================
class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, num_classes=6, dropout=0.1, max_len=30):
        super(SkeletonTransformer, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# ==================== 配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, 'processed')
model_path = os.path.join(current_dir, 'images', 'best_model.pth')
label_map_path = os.path.join(processed_dir, 'label_map.json')

# 加载标签映射
with open(label_map_path, 'r') as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkeletonTransformer(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("="*50)
print("模型加载成功！")
print(f"设备: {device}")
print(f"类别: {label_map}")
print("="*50)

# ==================== MediaPipe 初始化 ====================
pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_pose_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)

def normalize_pose(keypoints):
    if keypoints is None or len(keypoints) < 33:
        return keypoints
    left_hip = keypoints[23][:2]
    right_hip = keypoints[24][:2]
    hip_center = (left_hip + right_hip) / 2
    left_shoulder = keypoints[11][:2]
    right_shoulder = keypoints[12][:2]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width < 1e-6:
        shoulder_width = 1.0
    normalized = keypoints.copy()
    for i in range(len(normalized)):
        normalized[i, 0] = (normalized[i, 0] - hip_center[0]) / shoulder_width
        normalized[i, 1] = (normalized[i, 1] - hip_center[1]) / shoulder_width
        normalized[i, 2] = normalized[i, 2] / shoulder_width
    return normalized

def resize_sequence(seq, target_len=30):
    current_len = seq.shape[0]
    if current_len == target_len:
        return seq
    indices = np.linspace(0, current_len - 1, target_len)
    resized = []
    for idx in indices:
        floor_idx = int(np.floor(idx))
        ceil_idx = min(floor_idx + 1, current_len - 1)
        if floor_idx == ceil_idx:
            resized.append(seq[floor_idx])
        else:
            alpha = idx - floor_idx
            resized.append((1 - alpha) * seq[floor_idx] + alpha * seq[ceil_idx])
    return np.array(resized)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extract_pose_from_frame(frame)
        if kp is not None:
            kp = normalize_pose(kp)
            frames.append(kp.flatten())
    cap.release()
    if len(frames) == 0:
        return None
    seq = np.array(frames)
    seq = resize_sequence(seq, 30)
    return seq

def predict_video(video_path):
    """预测单个视频的动作类别"""
    print(f"\n处理视频: {video_path}")
    
    # 提取骨架序列
    seq = process_video(video_path)
    if seq is None:
        print("错误：无法提取骨架序列")
        return None
    
    # 转换为 tensor 并预测
    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(seq_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    # 输出结果
    pred_label = idx_to_label[pred_class]
    print(f"\n{'='*40}")
    print(f"预测结果: {pred_label}")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"{'='*40}")
    
    # 显示所有类别概率
    print("\n各类别概率:")
    for i, (name, idx) in enumerate(label_map.items()):
        prob = probs[0][idx].item()
        bar = "█" * int(prob * 50)
        print(f"  {name:20}: {prob:.4f} {bar}")
    
    return pred_label, confidence

# ==================== 示例：预测一个测试视频 ====================
if __name__ == "__main__":
    import sys
    
    # 方式1：命令行传入视频路径
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        predict_video(video_path)
    else:
        # 方式2：指定一个测试视频路径
        # 这里需要你自己设置一个测试视频的路径
        print("使用方法: python inference.py <视频路径>")
        print("示例: python inference.py data/forehand_drive/001.mp4")
        
        # 也可以手动指定一个测试视频
        # 取消下面的注释，把路径改成你的测试视频
        # test_video = "data/forehand_drive/001.mp4"
        # predict_video(test_video)