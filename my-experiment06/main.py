import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 设置路径 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

box_path = os.path.join(images_dir, 'box.png')
scene_path = os.path.join(images_dir, 'box_in_scene.png')

print("="*60)
print("实验：基于 OpenCV 的局部特征检测、描述与图像匹配")
print("="*60)

# ==================== 读取图像 ====================
print("\n1. 读取图像...")
img_box = cv2.imread(box_path)
img_scene = cv2.imread(scene_path)

if img_box is None or img_scene is None:
    print("错误：请确保 box.png 和 box_in_scene.png 在 images 文件夹中")
    exit()

print(f"box.png 尺寸: {img_box.shape}")
print(f"box_in_scene.png 尺寸: {img_scene.shape}")

# ==================== 任务1：ORB 特征检测 ====================
print("\n" + "="*60)
print("任务1：ORB 特征检测")
print("="*60)

# 创建 ORB 检测器
orb = cv2.ORB_create(nfeatures=1000)

# 检测关键点和描述子
kp1, des1 = orb.detectAndCompute(img_box, None)
kp2, des2 = orb.detectAndCompute(img_scene, None)

print(f"box.png 关键点数量: {len(kp1)}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}")
print(f"描述子维度: {des1.shape[1]}")

# 可视化关键点
img_box_kp = cv2.drawKeypoints(img_box, kp1, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
img_scene_kp = cv2.drawKeypoints(img_scene, kp2, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)

cv2.imwrite(os.path.join(images_dir, '1_box_keypoints.png'), img_box_kp)
cv2.imwrite(os.path.join(images_dir, '2_scene_keypoints.png'), img_scene_kp)
print("\n已保存:")
print("  - images/1_box_keypoints.png (box.png 特征点)")
print("  - images/2_scene_keypoints.png (scene.png 特征点)")

# ==================== 任务2：ORB 特征匹配 ====================
print("\n" + "="*60)
print("任务2：ORB 特征匹配")
print("="*60)

# 创建暴力匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 进行匹配
matches = bf.match(des1, des2)

# 按照距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

print(f"总匹配数量: {len(matches)}")

# 显示前30个匹配
top_n = 30
matches_top = matches[:top_n]

# 绘制初始匹配图
img_matches_initial = cv2.drawMatches(img_box, kp1, img_scene, kp2, matches_top, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(images_dir, '3_initial_matches.png'), img_matches_initial)
print(f"\n已保存: images/3_initial_matches.png (前{top_n}个匹配)")

# ==================== 任务3：RANSAC 剔除错误匹配 ====================
print("\n" + "="*60)
print("任务3：RANSAC 剔除错误匹配")
print("="*60)

# 提取匹配点的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 使用 RANSAC 计算 Homography 矩阵
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 将 mask 转换为布尔列表
mask = mask.ravel().tolist()

# 统计内点数量
inliers = sum(mask)
total_matches = len(matches)
inlier_ratio = inliers / total_matches

print(f"总匹配数量: {total_matches}")
print(f"RANSAC 内点数量: {inliers}")
print(f"内点比例: {inlier_ratio:.4f} ({inlier_ratio*100:.2f}%)")
print(f"\nHomography 矩阵:")
print(H)

# 提取内点匹配
inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

# 绘制 RANSAC 后的匹配图
img_matches_ransac = cv2.drawMatches(img_box, kp1, img_scene, kp2, inlier_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(images_dir, '4_ransac_matches.png'), img_matches_ransac)
print("\n已保存: images/4_ransac_matches.png (RANSAC后匹配图)")

# ==================== 任务4：目标定位 ====================
print("\n" + "="*60)
print("任务4：目标定位")
print("="*60)

# 获取 box.png 的四个角点
h, w = img_box.shape[:2]
box_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 将角点投影到场景图中
projected_corners = cv2.perspectiveTransform(box_corners, H)

# 绘制四边形边框
img_result = img_scene.copy()
img_result = cv2.polylines(img_result, [np.int32(projected_corners)], True, (0, 255, 0), 3)

cv2.imwrite(os.path.join(images_dir, '5_target_location.png'), img_result)
print("已保存: images/5_target_location.png (目标定位结果)")

# 判断定位是否成功
print("\n定位是否成功: 是" if inlier_ratio > 0.3 else "定位是否成功: 否")
print("说明: 内点比例大于30%通常表示定位成功")

# ==================== 任务6：参数对比实验 ====================
print("\n" + "="*60)
print("任务6：参数对比实验 (nfeatures 不同取值)")
print("="*60)

def run_experiment(nfeatures):
    """运行不同 nfeatures 参数实验"""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img_box, None)
    kp2, des2 = orb.detectAndCompute(img_scene, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if mask is not None:
        mask = mask.ravel().tolist()
        inliers = sum(mask)
        inlier_ratio = inliers / len(matches) if len(matches) > 0 else 0
    else:
        inliers = 0
        inlier_ratio = 0
    
    success = inlier_ratio > 0.3
    
    return {
        'nfeatures': nfeatures,
        'kp1_num': len(kp1),
        'kp2_num': len(kp2),
        'matches_num': len(matches),
        'inliers': inliers,
        'inlier_ratio': inlier_ratio,
        'success': success
    }

# 测试不同的 nfeatures
nfeatures_list = [500, 1000, 2000]
results = []

print("\n{:<12} {:<18} {:<18} {:<12} {:<12} {:<12} {:<12}".format(
    "nfeatures", "模板图关键点数", "场景图关键点数", "匹配数量", "RANSAC内点数", "内点比例", "是否成功定位"))
print("-" * 100)

for n in nfeatures_list:
    r = run_experiment(n)
    results.append(r)
    print("{:<12} {:<18} {:<18} {:<12} {:<12} {:<12.4f} {:<12}".format(
        r['nfeatures'], r['kp1_num'], r['kp2_num'], 
        r['matches_num'], r['inliers'], r['inlier_ratio'], "是" if r['success'] else "否"))

# ==================== 结果汇总与可视化 ====================
print("\n" + "="*60)
print("结果汇总")
print("="*60)

# 创建对比图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('ORB 特征匹配实验对比', fontsize=16)

# 显示关键点图
axes[0, 0].imshow(cv2.cvtColor(img_box_kp, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'box.png 关键点 ({len(kp1)}个)')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_scene_kp, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'scene.png 关键点 ({len(kp2)}个)')
axes[0, 1].axis('off')

# 显示初始匹配图
axes[0, 2].imshow(cv2.cvtColor(img_matches_initial, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f'初始匹配 (前{top_n}个)')
axes[0, 2].axis('off')

# 显示 RANSAC 后匹配图
axes[1, 0].imshow(cv2.cvtColor(img_matches_ransac, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'RANSAC后匹配 (内点{inliers}个)')
axes[1, 0].axis('off')

# 显示目标定位结果
axes[1, 1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('目标定位结果')
axes[1, 1].axis('off')

# 显示参数对比表格
axes[1, 2].axis('tight')
axes[1, 2].axis('off')
table_data = [[r['nfeatures'], r['kp1_num'], r['kp2_num'], r['matches_num'], 
               r['inliers'], f"{r['inlier_ratio']:.3f}", "是" if r['success'] else "否"] for r in results]
table = axes[1, 2].table(cellText=table_data, 
                          colLabels=['nfeatures', '模板关键点', '场景关键点', '匹配数', '内点数', '内点比例', '定位成功'],
                          loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
axes[1, 2].set_title('参数对比实验')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '6_experiment_summary.png'), dpi=150)
plt.show()

print("\n已保存: images/6_experiment_summary.png (实验总结对比图)")

# ==================== 输出所有生成的文件 ====================
print("\n" + "="*60)
print("生成的文件列表")
print("="*60)
for f in os.listdir(images_dir):
    if f.endswith('.png') or f.endswith('.jpg'):
        print(f"  - images/{f}")

print("\n实验完成！")