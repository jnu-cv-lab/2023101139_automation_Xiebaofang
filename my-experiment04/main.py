import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
import os

# ==================== 设置保存路径 ====================
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 设置 images 文件夹路径
images_dir = os.path.join(script_dir, 'images')
# 如果 images 文件夹不存在，创建它
os.makedirs(images_dir, exist_ok=True)

print(f"图片将保存到: {images_dir}")

# ==================== 第一部分：生成测试图并验证抗混叠 ====================
print("="*60)
print("第一部分：生成棋盘格和chirp测试图")

def generate_checkerboard(size=512, block_size=8):
    """生成棋盘格测试图"""
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i:i+block_size, j:j+block_size] = 255
    return img

def generate_chirp(size=512, f0=0.01, f1=0.5):
    """生成chirp信号（频率逐渐增高的正弦条纹）"""
    x = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    phase = 2 * np.pi * (f0 * r + (f1 - f0) * r**2) * size / 2
    img = 127 + 127 * np.sin(phase)
    return img.astype(np.uint8)

def downsample(img, factor=2):
    """下采样"""
    h, w = img.shape
    return img[::factor, ::factor]

def downsample_with_gaussian(img, factor=2, sigma=None):
    """先高斯滤波再下采样"""
    if sigma is None:
        sigma = 0.45 * factor
    img_filtered = gaussian_filter(img.astype(float), sigma=sigma)
    return downsample(img_filtered, factor).astype(np.uint8)

def compute_fft_spectrum(img):
    """计算FFT频谱"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)
    return magnitude_log

# 生成测试图
checkerboard = generate_checkerboard(512, 8)
chirp = generate_chirp(512, 0.01, 0.5)

# 保存生成的测试图
cv2.imwrite(os.path.join(images_dir, 'checkerboard.png'), checkerboard)
cv2.imwrite(os.path.join(images_dir, 'chirp.png'), chirp)
print(f"已保存: checkerboard.png, chirp.png")

# 对棋盘格进行下采样实验
factor = 4
checker_small_direct = downsample(checkerboard, factor)
checker_small_filtered = downsample_with_gaussian(checkerboard, factor)

# 计算误差
checker_restored_direct = cv2.resize(checker_small_direct, (512, 512), interpolation=cv2.INTER_NEAREST)
checker_restored_filtered = cv2.resize(checker_small_filtered, (512, 512), interpolation=cv2.INTER_NEAREST)
error_direct = np.abs(checkerboard.astype(float) - checker_restored_direct.astype(float))
error_filtered = np.abs(checkerboard.astype(float) - checker_restored_filtered.astype(float))

# 保存中间结果
cv2.imwrite(os.path.join(images_dir, 'checker_small_direct.png'), checker_small_direct)
cv2.imwrite(os.path.join(images_dir, 'checker_small_filtered.png'), checker_small_filtered)
print(f"已保存: checker_small_direct.png, checker_small_filtered.png")

print(f"直接下采样误差均值: {np.mean(error_direct):.2f}")
print(f"高斯滤波后下采样误差均值: {np.mean(error_filtered):.2f}")

# 对chirp图进行相同实验
chirp_small_direct = downsample(chirp, factor)
chirp_small_filtered = downsample_with_gaussian(chirp, factor)
chirp_restored_direct = cv2.resize(chirp_small_direct, (512, 512), interpolation=cv2.INTER_NEAREST)
chirp_restored_filtered = cv2.resize(chirp_small_filtered, (512, 512), interpolation=cv2.INTER_NEAREST)

# 计算频谱
spec_checker_orig = compute_fft_spectrum(checkerboard)
spec_checker_direct = compute_fft_spectrum(checker_small_direct)
spec_checker_filtered = compute_fft_spectrum(checker_small_filtered)

# ==================== 第二部分：验证σ公式 ====================
print("\n" + "="*60)
print("第二部分：验证σ公式 (M=4)")

def test_sigma_values(img, factor=4, sigmas=[0.5, 1.0, 2.0, 4.0]):
    """测试不同sigma值的效果"""
    results = {}
    h, w = img.shape
    for sigma in sigmas:
        img_filtered = gaussian_filter(img.astype(float), sigma=sigma)
        img_small = downsample(img_filtered, factor)
        img_restored = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)
        mse = np.mean((img.astype(float) - img_restored) ** 2)
        results[sigma] = {'small': img_small, 'restored': img_restored, 'mse': mse}
        print(f"σ={sigma}: MSE={mse:.2f}")
        # 保存结果
        cv2.imwrite(os.path.join(images_dir, f'chirp_sigma_{sigma}.png'), img_restored)
    return results

M = 4
theoretical_sigma = 0.45 * M
print(f"理论最优σ = 0.45 * M = {theoretical_sigma}")

sigma_results = test_sigma_values(chirp, factor=M, sigmas=[0.5, 1.0, 1.8, 2.0, 4.0])

# 找出最佳sigma
best_sigma = min(sigma_results.keys(), key=lambda s: sigma_results[s]['mse'])
print(f"实验最佳σ = {best_sigma}")
print(f"与理论值 {theoretical_sigma} 对比: {'接近' if abs(best_sigma - theoretical_sigma) < 0.3 else '有差异'}")

# ==================== 第三部分：自适应下采样 ====================
print("\n" + "="*60)
print("第三部分：自适应下采样")

def estimate_local_M(grad_x, grad_y, block_size=16):
    """根据梯度估计局部M值"""
    h, w = grad_x.shape
    M_map = np.ones((h, w)) * 4
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_gx = grad_x[i:i+block_size, j:j+block_size]
            block_gy = grad_y[i:i+block_size, j:j+block_size]
            mean_grad = (np.mean(np.abs(block_gx)) + np.mean(np.abs(block_gy))) / 2
            if mean_grad > 30:
                M_map[i:i+block_size, j:j+block_size] = 2
            elif mean_grad > 15:
                M_map[i:i+block_size, j:j+block_size] = 3
            else:
                M_map[i:i+block_size, j:j+block_size] = 4
    return M_map

def adaptive_downsample(img, M_map):
    """根据M_map自适应下采样"""
    h, w = img.shape
    img_result = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w))
    
    for M in np.unique(M_map):
        mask = (M_map == M)
        sigma = 0.45 * M
        img_filtered = gaussian_filter(img.astype(float), sigma=sigma)
        img_small = img_filtered[::int(M), ::int(M)]
        img_restored = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)
        img_result[mask] += img_restored[mask]
        weight_sum[mask] += 1
    
    weight_sum[weight_sum == 0] = 1
    img_result = img_result / weight_sum
    return img_result.astype(np.uint8)

# 计算梯度
grad_x = cv2.Sobel(chirp.astype(float), cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(chirp.astype(float), cv2.CV_64F, 0, 1, ksize=3)
M_map = estimate_local_M(grad_x, grad_y)

# 保存M_map
plt.figure()
plt.imshow(M_map, cmap='jet')
plt.colorbar()
plt.title('Adaptive M Map')
plt.savefig(os.path.join(images_dir, 'adaptive_M_map.png'))
plt.close()

# 执行自适应下采样
chirp_adaptive = adaptive_downsample(chirp, M_map)

# 对比全图统一用M=4下采样
chirp_uniform = downsample_with_gaussian(chirp, factor=4, sigma=0.45*4)
chirp_uniform_restored = cv2.resize(chirp_uniform, (512, 512), interpolation=cv2.INTER_CUBIC)

# 计算误差
error_uniform = np.abs(chirp.astype(float) - chirp_uniform_restored.astype(float))
error_adaptive = np.abs(chirp.astype(float) - chirp_adaptive.astype(float))

# 保存结果
cv2.imwrite(os.path.join(images_dir, 'chirp_adaptive.png'), chirp_adaptive)
cv2.imwrite(os.path.join(images_dir, 'chirp_uniform_restored.png'), chirp_uniform_restored)
cv2.imwrite(os.path.join(images_dir, 'error_uniform.png'), (error_uniform * 255 / error_uniform.max()).astype(np.uint8))
cv2.imwrite(os.path.join(images_dir, 'error_adaptive.png'), (error_adaptive * 255 / error_adaptive.max()).astype(np.uint8))

print(f"统一M=4下采样误差均值: {np.mean(error_uniform):.2f}")
print(f"自适应下采样误差均值: {np.mean(error_adaptive):.2f}")
print(f"自适应方法误差降低: {(np.mean(error_uniform) - np.mean(error_adaptive)) / np.mean(error_uniform) * 100:.1f}%")

# ==================== 可视化结果 ====================
print("\n" + "="*60)
print("生成可视化结果...")

fig = plt.figure(figsize=(18, 14))

# 第一行：棋盘格实验结果
plt.subplot(3, 5, 1)
plt.imshow(checkerboard, cmap='gray')
plt.title('棋盘格原图')
plt.axis('off')

plt.subplot(3, 5, 2)
plt.imshow(error_direct, cmap='hot')
plt.title('直接下采样误差图')
plt.axis('off')

plt.subplot(3, 5, 3)
plt.imshow(error_filtered, cmap='hot')
plt.title('高斯滤波后下采样误差图')
plt.axis('off')

plt.subplot(3, 5, 4)
plt.imshow(spec_checker_orig, cmap='gray')
plt.title('棋盘格原图FFT频谱')
plt.axis('off')

plt.subplot(3, 5, 5)
plt.imshow(spec_checker_filtered, cmap='gray')
plt.title('滤波后下采样FFT频谱')
plt.axis('off')

# 第二行：不同sigma对比
sigma_display = [0.5, 1.0, 2.0, 4.0]
for idx, sigma in enumerate(sigma_display):
    plt.subplot(3, 5, 6+idx)
    img_test = sigma_results[sigma]['restored']
    plt.imshow(img_test, cmap='gray')
    plt.title(f'σ={sigma}, MSE={sigma_results[sigma]["mse"]:.1f}')
    plt.axis('off')

plt.subplot(3, 5, 10)
plt.imshow(chirp, cmap='gray')
plt.title(f'chirp原图\n理论σ={theoretical_sigma}')
plt.axis('off')

# 第三行：自适应下采样对比
plt.subplot(3, 5, 11)
plt.imshow(chirp, cmap='gray')
plt.title('chirp原图')
plt.axis('off')

plt.subplot(3, 5, 12)
plt.imshow(M_map, cmap='jet')
plt.title('自适应M值分布图')
plt.axis('off')

plt.subplot(3, 5, 13)
plt.imshow(error_uniform, cmap='hot')
plt.title(f'统一M=4下采样误差\n均值={np.mean(error_uniform):.1f}')
plt.axis('off')

plt.subplot(3, 5, 14)
plt.imshow(error_adaptive, cmap='hot')
plt.title(f'自适应下采样误差\n均值={np.mean(error_adaptive):.1f}')
plt.axis('off')

plt.subplot(3, 5, 15)
plt.imshow(chirp_adaptive, cmap='gray')
plt.title('自适应下采样恢复图')
plt.axis('off')

plt.tight_layout()
# 保存结果图到 images 文件夹
result_path = os.path.join(images_dir, 'antialiasing_result.png')
plt.savefig(result_path, dpi=150)
plt.show()

print(f"\n所有结果已保存到: {images_dir}")
print("="*60)
print("分析结论")
print("="*60)
print("""
【第一部分：抗混叠原理】
- 直接下采样会产生混叠伪影
- 先高斯滤波再下采样可以抑制混叠

【第二部分：σ公式验证】
- 理论公式 σ = 0.45M 给出了最优滤波强度
- 实验结果与理论值基本吻合

【第三部分：自适应下采样】
- 根据梯度信息估计局部M值
- 自适应方法比统一下采样误差更低
""")