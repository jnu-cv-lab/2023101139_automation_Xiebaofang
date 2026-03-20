import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 任务1：读取图片
print("="*50)
print("任务1：读取图片")
# 读取 images 文件夹下的 test.jpg
img = cv2.imread("my-experiment01/images/test.jpg")

# 检查图片是否读取成功
if img is None:
    print("错误：无法读取图片，请检查图片路径和文件名")
    exit()
print("图片读取成功！")
print("="*50)

# ==================== 任务2：输出图像基本信息 ====================
print("\n" + "="*50)
print("任务2：图像基本信息")
# 获取图像尺寸（高度、宽度、通道数）
height, width, channels = img.shape
print(f"图像尺寸：宽度 = {width} 像素，高度 = {height} 像素")
print(f"图像通道数：{channels}")
print(f"像素数据类型：{img.dtype}")
print("="*50)

# ==================== 任务3：显示原图 ====================
print("\n" + "="*50)
print("任务3：显示原图")
# OpenCV 读取的是 BGR 格式，而 Matplotlib 需要 RGB 格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用 Matplotlib 显示图片
plt.figure("原图显示", figsize=(10, 8))
plt.imshow(img_rgb)
plt.title("原始图像")
plt.axis('off')  # 关闭坐标轴
plt.show()
print("="*50)

# ==================== 任务4：转换为灰度图并显示 ====================
print("\n" + "="*50)
print("任务4：转换为灰度图")
# 将彩色图转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("灰度转换完成！")

# 显示灰度图
plt.figure("灰度图显示", figsize=(10, 8))
plt.imshow(gray_img, cmap='gray')  # cmap='gray' 表示用灰度色彩映射
plt.title("灰度图像")
plt.axis('off')
plt.show()
print("灰度图窗口已关闭")
print("="*50)

# ==================== 任务5：保存处理结果 ====================
print("\n" + "="*50)
print("任务5：保存灰度图")
# 将灰度图保存为新文件
cv2.imwrite("my-experiment01/images/gray_test.jpg", gray_img)
print("灰度图已保存为：my-experiment01/images/gray_test.jpg")

# 验证文件是否保存成功
import os
if os.path.exists("my-experiment01/images/gray_test.jpg"):
    file_size = os.path.getsize("my-experiment01/images/gray_test.jpg")
    print(f"文件保存成功！文件大小：{file_size} 字节")
else:
    print("文件保存失败，请检查路径")

print("="*50)

# ==================== 任务6：NumPy 简单操作 ====================
print("\n" + "="*50)
print("任务6：NumPy 操作示例")

# 示例1：输出某个像素点的值（取图像中心点）
center_y = height // 2
center_x = width // 2
pixel_value = img[center_y, center_x]
print(f"中心点 ({center_x}, {center_y}) 的 BGR 值为：{pixel_value}")

# 也可以输出灰度图的中心点像素值
gray_pixel = gray_img[center_y, center_x]
print(f"灰度图中心点像素值：{gray_pixel}")

# 示例2：裁剪图像左上角一块区域（100x100 像素）
crop_size = 100
cropped = img[0:crop_size, 0:crop_size]
cv2.imwrite("my-experiment01/images/cropped.jpg", cropped)
print(f"左上角 {crop_size}x{crop_size} 区域已裁剪保存为：my-experiment01/images/cropped.jpg")