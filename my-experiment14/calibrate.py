import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

PATTERN_SIZE = (9, 6)      
SQUARE_SIZE = 25           

# 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, 'images')
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

print("="*60)
print("相机标定 - 棋盘格")
print("="*60)
print(f"棋盘格内角点: {PATTERN_SIZE[0]} x {PATTERN_SIZE[1]}")
print(f"方格边长: {SQUARE_SIZE} mm")
print(f"图片目录: {images_dir}")

# 获取图片 
# 方式1：使用 images 文件夹中的多张图片
image_paths = glob.glob(os.path.join(images_dir, '*.jpg')) + \
              glob.glob(os.path.join(images_dir, '*.png')) + \
              glob.glob(os.path.join(images_dir, '*.jpeg'))

# 方式2：使用单张图片（如果图片数量不足）
if len(image_paths) == 0:
    print("\n没有找到图片，使用单张图片演示...")
    # 如果你只有一张图片，放在 images 文件夹并命名为 chessboard.png
    single_image = os.path.join(images_dir, 'chessboard.png')
    if os.path.exists(single_image):
        image_paths = [single_image]
    else:
        print(f"请将棋盘格图片放到 {images_dir} 文件夹")
        print("或者拍至少15张不同角度的棋盘格照片")
        exit(1)

print(f"找到 {len(image_paths)} 张图片")

#生成棋盘格三维坐标
pattern_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
pattern_points[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
pattern_points *= SQUARE_SIZE

print(f"棋盘格三维坐标点数: {len(pattern_points)}")

# 第三步：检测角点 
imgpoints = []   # 存储检测到的角点像素坐标
imgpoints_good = []  # 存储检测到角点的图片路径

print("\n正在检测角点...")

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"  无法读取: {img_path}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    
    if ret:
        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        imgpoints.append(corners_sub)
        imgpoints_good.append(img_path)
        
        # 绘制角点并保存
        img_copy = img.copy()
        cv2.drawChessboardCorners(img_copy, PATTERN_SIZE, corners_sub, ret)
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(results_dir, f'corners_{base_name}'), img_copy)
        print(f"  ✅ 成功: {base_name} ({len(corners_sub)} 个角点)")
    else:
        print(f"  ❌ 失败: {os.path.basename(img_path)}")

print(f"\n成功检测角点的图片: {len(imgpoints)} 张")

if len(imgpoints) == 0:
    print("错误：没有检测到任何角点")
    print("请检查棋盘格图片和 PATTERN_SIZE 是否正确")
    exit(1)

#相机标定 
print("\n正在标定相机...")

# 生成所有图片的三维坐标
object_points = [pattern_points] * len(imgpoints)

# 标定
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, imgpoints, gray.shape[::-1], None, None
)

print("\n" + "="*60)
print("标定结果")
print("="*60)

print(f"\n重投影误差: {ret:.6f} 像素")

print(f"\n相机内参矩阵 K:")
print(K)

print(f"\n畸变参数 D = [k1, k2, p1, p2, k3]:")
print(dist.flatten())

print(f"\n每张图片的外参:")
for i, img_path in enumerate(imgpoints_good):
    rvec = rvecs[i]
    tvec = tvecs[i]
    print(f"  图片 {i+1}: 旋转向量 {rvec.flatten()[:3]}, 平移向量 {tvec.flatten()[:3]}")

# 去畸变处理 
print("\n" + "="*60)
print("去畸变处理")
print("="*60)

# 对第一张检测到角点的图片进行去畸变
if len(imgpoints_good) > 0:
    first_img_path = imgpoints_good[0]
    img_orig = cv2.imread(first_img_path)
    h, w = img_orig.shape[:2]
    
    # 方法1：使用 cv2.undistort
    img_undistorted = cv2.undistort(img_orig, K, dist)
    
    # 方法2：使用 cv2.getOptimalNewCameraMatrix 裁剪黑边
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    img_undistorted2 = cv2.undistort(img_orig, K, dist, None, new_camera_matrix)
    
    # 保存结果
    base_name = os.path.basename(first_img_path)
    cv2.imwrite(os.path.join(results_dir, f'original_{base_name}'), img_orig)
    cv2.imwrite(os.path.join(results_dir, f'undistorted_{base_name}'), img_undistorted)
    cv2.imwrite(os.path.join(results_dir, f'undistorted_cropped_{base_name}'), img_undistorted2)
    
    print(f"已保存原始图像: original_{base_name}")
    print(f"已保存去畸变图像: undistorted_{base_name}")
    print(f"已保存去畸变裁剪图像: undistorted_cropped_{base_name}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB))
    axes[1].set_title('去畸变后')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(img_undistorted2, cv2.COLOR_BGR2RGB))
    axes[2].set_title('去畸变 + 裁剪黑边')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'undistort_comparison.png'), dpi=150)
    plt.show()
    print("已保存对比图: undistort_comparison.png")
else:
    print("没有图片可用于去畸变演示")

#保存标定结果 
with open(os.path.join(results_dir, 'calibration_results.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("相机标定结果\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"棋盘格内角点: {PATTERN_SIZE[0]} x {PATTERN_SIZE[1]}\n")
    f.write(f"方格边长: {SQUARE_SIZE} mm\n")
    f.write(f"有效图片数: {len(imgpoints)}\n\n")
    
    f.write(f"重投影误差: {ret:.6f} 像素\n\n")
    
    f.write("相机内参矩阵 K:\n")
    f.write(str(K) + "\n\n")
    
    f.write("畸变参数 D = [k1, k2, p1, p2, k3]:\n")
    f.write(str(dist.flatten()) + "\n\n")
    
    f.write("图像分辨率:\n")
    f.write(f"  {w} x {h}\n\n")
    
    f.write("分析:\n")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    f.write(f"  fx = {fx:.2f}, fy = {fy:.2f}\n")
    f.write(f"  cx = {cx:.2f}, cy = {cy:.2f}\n")
    f.write(f"  图像中心: ({w/2:.2f}, {h/2:.2f})\n")
    f.write(f"  fx/fy 差值: {abs(fx - fy):.2f}\n")
    f.write(f"  cx与图像中心偏移: ({cx - w/2:.2f}, {cy - h/2:.2f})\n")

print(f"\n结果已保存到 {results_dir}")
print("="*60)
print("标定完成！")