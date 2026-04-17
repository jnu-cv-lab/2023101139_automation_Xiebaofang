import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 使用绝对路径 ====================
username = "xie_bao_fang"
base_path = f"/home/{username}/cv-course/my-experiment05"
images_dir = os.path.join(base_path, 'images')
test_jpg_path = os.path.join(images_dir, 'test.jpg')
os.makedirs(images_dir, exist_ok=True)

print("="*60)
print("几何变换实验 - 全能扫描王效果")
print(f"图片路径: {test_jpg_path}")
print(f"图片是否存在: {os.path.exists(test_jpg_path)}")

# ==================== 1. 生成测试图 ====================
print("\n1. 生成测试图...")

def create_test_image(size=600):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (150, 150), (450, 350), (0, 0, 255), 3)
    cv2.circle(img, (300, 250), 100, (0, 255, 0), 3)
    for y in range(100, 500, 80):
        cv2.line(img, (50, y), (550, y), (255, 0, 0), 2)
    for x in range(100, 550, 80):
        cv2.line(img, (x, 50), (x, 550), (255, 0, 0), 2)
    cv2.putText(img, "Rectangle", (180, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Circle", (270, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Parallel Lines", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

test_img = create_test_image(600)
cv2.imwrite(os.path.join(images_dir, '0_original.png'), test_img)

# ==================== 2. 相似变换 ====================
h, w = test_img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 30, 0.8)
similarity_img = cv2.warpAffine(test_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(os.path.join(images_dir, '1_similarity.png'), similarity_img)

# ==================== 3. 仿射变换 ====================
pts1 = np.float32([[50, 50], [550, 50], [50, 550]])
pts2 = np.float32([[50, 100], [550, 50], [150, 550]])
M = cv2.getAffineTransform(pts1, pts2)
affine_img = cv2.warpAffine(test_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(os.path.join(images_dir, '2_affine.png'), affine_img)

# ==================== 4. 透视变换 ====================
pts1 = np.float32([[50, 50], [550, 50], [50, 550], [550, 550]])
pts2 = np.float32([[80, 30], [520, 80], [30, 520], [570, 570]])
M = cv2.getPerspectiveTransform(pts1, pts2)
perspective_img = cv2.warpPerspective(test_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
cv2.imwrite(os.path.join(images_dir, '3_perspective.png'), perspective_img)
print("前4部分完成，图片已保存")

# ==================== 5. 几何性质分析 ====================
print("\n" + "="*60)
print("几何性质分析")
print("="*60)
print("""
相似变换: 保持直线、平行、垂直，圆可能变椭圆
仿射变换: 保持直线、平行，不保持垂直，圆变椭圆
透视变换: 只保持直线，平行和垂直都不保持
""")

# ==================== 6. 扫描王效果：A4纸校正并铺满 ====================
print("\n" + "="*60)
print("6. 全能扫描王效果：A4纸校正并铺满画面")

corrected_img = None
real_photo = None

if not os.path.exists(test_jpg_path):
    print(f"错误：找不到图片 {test_jpg_path}")
    print("请将A4纸照片命名为 test.jpg 放到 images 文件夹中")
else:
    print(f"成功找到图片: {test_jpg_path}")
    real_photo = cv2.imread(test_jpg_path)
    
    if real_photo is None:
        print("错误：无法读取图片，文件可能损坏")
    else:
        print(f"图片尺寸: {real_photo.shape}")
        h, w = real_photo.shape[:2]
        
        # 显示图片让用户选择角点
        points = []
        img_copy = real_photo.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append([x, y])
                    cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)
                    cv2.putText(img_copy, str(len(points)), (x+10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow('Select 4 corners of A4 paper', img_copy)
                    print(f"已选择点{len(points)}: ({x}, {y})")
                    
                    if len(points) == 4:
                        print("已选择4个角点，按 Enter 键继续...")
        
        cv2.imshow('Select 4 corners of A4 paper', img_copy)
        cv2.setMouseCallback('Select 4 corners of A4 paper', mouse_callback)
        print("\n请按顺序点击A4纸的四个角点：")
        print("  点1: 左上角")
        print("  点2: 右上角")
        print("  点3: 左下角")
        print("  点4: 右下角")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(points) == 4:
            print(f"\n选择的角点: {points}")
            pts_src = np.float32(points)
            
            # 关键：计算A4纸的宽度和高度（按比例，A4纸长宽比约为√2:1 ≈ 1.414）
            # 计算源图像中A4纸的实际宽度和高度
            width_top = np.linalg.norm(pts_src[1] - pts_src[0])   # 上边宽度
            width_bottom = np.linalg.norm(pts_src[3] - pts_src[2]) # 下边宽度
            height_left = np.linalg.norm(pts_src[2] - pts_src[0])   # 左边高度
            height_right = np.linalg.norm(pts_src[3] - pts_src[1])  # 右边高度
            
            # 取平均作为最终尺寸
            paper_width = int((width_top + width_bottom) / 2)
            paper_height = int((height_left + height_right) / 2)
            
            # 或者使用A4纸标准比例（长边:短边 = √2:1 ≈ 1.414）
            # 这里我们让输出图像的大小为 A4 比例（宽度:高度 = 1:1.414 或 1.414:1）
            # 为了铺满画面，我们让输出图像和原图大小一致，但只显示A4纸区域
            output_width = max(paper_width, paper_height)
            output_height = int(output_width * 1.414)  # A4比例
            
            # 也可以让输出图像铺满，使用固定的输出尺寸
            # 这里使用 A4 比例，输出尺寸为 800x1131 左右
            output_width = 800
            output_height = int(output_width * 1.414)  # 1131
            
            print(f"A4纸尺寸: 宽度={paper_width:.0f}, 高度={paper_height:.0f}")
            print(f"输出图像尺寸: {output_width} x {output_height}")
            
            # 目标四个角点（铺满整个输出图像，变成矩形电子版）
            pts_dst = np.float32([
                [0, 0],                    # 左上角
                [output_width, 0],         # 右上角
                [0, output_height],        # 左下角
                [output_width, output_height]  # 右下角
            ])
            
            try:
                # 计算透视变换矩阵
                M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                # 应用变换，输出指定大小的图像
                corrected_img = cv2.warpPerspective(real_photo, M, (output_width, output_height))
                
                # 保存结果
                cv2.imwrite(os.path.join(images_dir, '4_distorted_paper.png'), real_photo)
                cv2.imwrite(os.path.join(images_dir, '5_corrected_paper.png'), corrected_img)
                print("\n已保存: 4_distorted_paper.png（原始照片）")
                print("已保存: 5_corrected_paper.png（扫描王效果-电子版）")
                
                # 保存角点信息
                with open(os.path.join(images_dir, 'corners.txt'), 'w') as f:
                    f.write("选择的四个角点坐标（顺序：左上、右上、左下、右下）：\n")
                    for i, pt in enumerate(pts_src):
                        f.write(f"点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})\n")
                    f.write(f"\n输出图像尺寸: {output_width} x {output_height}\n")
                print("已保存: corners.txt")
                
            except Exception as e:
                print(f"透视变换失败: {e}")
        else:
            print(f"错误：只选择了 {len(points)} 个点，需要4个点")

# ==================== 7. 生成对比图 ====================
print("\n7. 生成对比图...")

# 准备显示的图像
img_original = test_img
img_similarity = similarity_img
img_affine = affine_img
img_perspective = perspective_img

if corrected_img is not None and real_photo is not None:
    img_distorted = real_photo
    img_corrected = corrected_img
else:
    h, w = 600, 800
    img_distorted = np.ones((h, w, 3), dtype=np.uint8) * 240
    img_corrected = np.ones((h, w, 3), dtype=np.uint8) * 240
    cv2.putText(img_distorted, "No test.jpg or correction failed", (50, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img_corrected, "Please select 4 corners correctly", (50, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('几何变换对比实验 - 全能扫描王效果', fontsize=16)

images_to_show = [
    (img_original, '原图（测试图）'),
    (img_similarity, '相似变换'),
    (img_affine, '仿射变换'),
    (img_perspective, '透视变换'),
    (img_distorted, '原始A4纸照片'),
    (img_corrected, '扫描王效果\n(电子版)')
]

for idx, (img, title) in enumerate(images_to_show):
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(title, fontsize=12)
    axes[row, col].axis('off')

plt.tight_layout()
result_path = os.path.join(images_dir, 'geometry_transform_result.png')
plt.savefig(result_path, dpi=150)
plt.show()

print(f"对比图已保存: {result_path}")

# ==================== 8. 输出结果 ====================
print("\n" + "="*60)
print("生成的文件列表")
print("="*60)
for filename in os.listdir(images_dir):
    print(f"  - images/{filename}")

print("\n实验完成！")
if corrected_img is not None:
    print("✅ 透视校正成功！")
    print("✅ 已生成扫描王效果：images/5_corrected_paper.png")
    print("   - A4纸区域已被单独裁剪出来")
    print("   - 已铺满整个画面")
    print("   - 变成矩形电子版效果")
else:
    print("❌ 透视校正失败，请检查角点选择是否正确")