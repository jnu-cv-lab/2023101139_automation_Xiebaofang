"""
预处理：使用 MediaPipe 提取视频骨架序列
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
TARGET_FRAMES = 30

# 类别映射（根据你的文件夹名称）
CATEGORY_MAP = {
    'forehand_drive': 0,
    'forehand_lift': 1,
    'forehand_net_shot': 2,
    'forehand_clear': 3,
    'backhand_drive': 4,
    'backhand_net_shot': 5,
}

# ==================== MediaPipe 初始化 ====================
pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_pose_from_frame(frame):
    """提取一帧的关键点"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)

def normalize_pose(keypoints):
    """归一化"""
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
    """重采样到目标长度"""
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
    """处理单个视频"""
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
    seq = resize_sequence(seq, TARGET_FRAMES)
    return seq

def process_all(data_dir, out_dir):
    """处理所有视频"""
    videos = []
    for cat, label in CATEGORY_MAP.items():
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            print(f"跳过: {cat_dir} 不存在")
            continue
        for f in os.listdir(cat_dir):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({'path': os.path.join(cat_dir, f), 'label': label})
    
    print(f"找到 {len(videos)} 个视频")
    X, y = [], []
    for v in tqdm(videos, desc="处理中"):
        seq = process_video(v['path'])
        if seq is not None:
            X.append(seq)
            y.append(v['label'])
    
    if len(X) == 0:
        print("错误：没有成功处理任何视频")
        return
    
    X = np.array(X)
    y = np.array(y)
    print(f"成功处理 {len(X)} 个视频，序列形状: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    with open(os.path.join(out_dir, 'label_map.json'), 'w') as f:
        json.dump(CATEGORY_MAP, f, indent=2)
    
    print(f"完成！结果保存在 {out_dir}")

if __name__ == "__main__":
    cur = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur, 'data')
    out_dir = os.path.join(cur, 'processed')
    print("="*50)
    print("羽毛球动作识别 - 数据预处理")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {out_dir}")
    print("="*50)
    if not os.path.exists(data_dir):
        print(f"错误: {data_dir} 不存在")
        exit(1)
    process_all(data_dir, out_dir)
    print("预处理完成！")