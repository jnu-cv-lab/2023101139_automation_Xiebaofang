import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# ==================== 1. 图像读取与预处理 ====================
print("="*60)
print("1. 图像读取与预处理")
img_color = cv2.imread('/home/xie_bao_fang/cv-course/my-experiment03/images3/test.jpg')
if img_color is None:
    print("错误：请确保 test.jpg 存在")
    exit()
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"原图尺寸: {img_gray.shape}")

# ==================== 2. 下采样（缩小）====================
print("\n" + "="*60)
print("2. 下采样（缩小为原来的1/2）")

# 方法1：直接缩小（无预滤波）
img_small_direct = cv2.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

# 方法2：先高斯平滑再缩小
img_smooth = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
img_small_smooth = cv2.resize(img_smooth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

print(f"缩小后尺寸: {img_small_direct.shape}")

# ==================== 3. 图像恢复（放大回原尺寸）====================
print("\n" + "="*60)
print("3. 图像恢复（放大回原尺寸）")

# 最近邻插值
img_nn = cv2.resize(img_small_direct, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

# 双线性插值
img_linear = cv2.resize(img_small_direct, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

# 双三次插值
img_cubic = cv2.resize(img_small_direct, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_CUBIC)

print("三种方法恢复完成")

# ==================== 4. 空间域比较（MSE和PSNR）====================
print("\n" + "="*60)
print("4. 空间域比较")

def compute_psnr(img1, img2):
    """计算PSNR"""
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def compute_mse(img1, img2):
    """计算MSE"""
    return mean_squared_error(img1.flatten(), img2.flatten())

# 计算三种恢复方法的MSE和PSNR
mse_nn = compute_mse(img_gray, img_nn)
psnr_nn = compute_psnr(img_gray, img_nn)

mse_linear = compute_mse(img_gray, img_linear)
psnr_linear = compute_psnr(img_gray, img_linear)

mse_cubic = compute_mse(img_gray, img_cubic)
psnr_cubic = compute_psnr(img_gray, img_cubic)

print(f"最近邻插值: MSE={mse_nn:.4f}, PSNR={psnr_nn:.2f}dB")
print(f"双线性插值: MSE={mse_linear:.4f}, PSNR={psnr_linear:.2f}dB")
print(f"双三次插值: MSE={mse_cubic:.4f}, PSNR={psnr_cubic:.2f}dB")

# ==================== 5. 傅里叶变换分析 ====================
print("\n" + "="*60)
print("5. 傅里叶变换分析")

def compute_fft_spectrum(img):
    """计算二维FFT并返回频谱（对数、中心化）"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)
    return magnitude_log

# 计算三种图像的频谱
spectrum_original = compute_fft_spectrum(img_gray)
spectrum_small = compute_fft_spectrum(img_small_direct)
spectrum_restored = compute_fft_spectrum(img_linear)

# ==================== 6. DCT分析 ====================
print("\n" + "="*60)
print("6. DCT分析")

def compute_dct_energy_ratio(img, block_size=8, ratio=0.25):
    """
    计算DCT低频能量占比
    将图像分成block_size×block_size的块，计算每个块左上角(ratio比例)的能量占比
    """
    h, w = img.shape
    energy_ratios = []
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                # 对块进行DCT变换
                block_float = block.astype(np.float32)
                dct_block = cv2.dct(block_float)
                # 总能量
                total_energy = np.sum(dct_block ** 2)
                # 低频区域（左上角 quarter×quarter）
                low_size = int(block_size * ratio)
                low_energy = np.sum(dct_block[:low_size, :low_size] ** 2)
                if total_energy > 0:
                    energy_ratios.append(low_energy / total_energy)
    
    return np.mean(energy_ratios) if energy_ratios else 0

def compute_dct_log_spectrum(img):
    """计算整幅图像的DCT系数（取对数显示）"""
    img_float = img.astype(np.float32)
    dct_full = cv2.dct(img_float)
    dct_log = np.log(np.abs(dct_full) + 1)
    return dct_log

# 计算DCT能量比例
ratio_original = compute_dct_energy_ratio(img_gray)
ratio_nn = compute_dct_energy_ratio(img_nn)
ratio_linear = compute_dct_energy_ratio(img_linear)
ratio_cubic = compute_dct_energy_ratio(img_cubic)

print(f"原图低频能量占比: {ratio_original:.4f}")
print(f"最近邻恢复低频能量占比: {ratio_nn:.4f}")
print(f"双线性恢复低频能量占比: {ratio_linear:.4f}")
print(f"双三次恢复低频能量占比: {ratio_cubic:.4f}")

# 计算DCT对数频谱用于显示
dct_original = compute_dct_log_spectrum(img_gray)
dct_nn = compute_dct_log_spectrum(img_nn)
dct_linear = compute_dct_log_spectrum(img_linear)

# ==================== 7. 显示所有图像 ====================
print("\n" + "="*60)
print("7. 显示图像")

plt.figure(figsize=(15, 12))

# 第一行：原图、缩小图、恢复图
plt.subplot(3, 4, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('原图')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(img_small_direct, cmap='gray')
plt.title('缩小图 (1/2)')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(img_nn, cmap='gray')
plt.title(f'最近邻恢复\nMSE={mse_nn:.2f}, PSNR={psnr_nn:.1f}dB')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(img_linear, cmap='gray')
plt.title(f'双线性恢复\nMSE={mse_linear:.2f}, PSNR={psnr_linear:.1f}dB')
plt.axis('off')

# 第二行：FFT频谱
plt.subplot(3, 4, 5)
plt.imshow(spectrum_original, cmap='gray')
plt.title('原图 FFT 频谱')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(spectrum_small, cmap='gray')
plt.title('缩小图 FFT 频谱')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(spectrum_restored, cmap='gray')
plt.title('恢复图 FFT 频谱')
plt.axis('off')

plt.subplot(3, 4, 8)
# 空位用于布局
plt.axis('off')

# 第三行：DCT系数
plt.subplot(3, 4, 9)
plt.imshow(dct_original, cmap='gray')
plt.title(f'原图 DCT 系数\n低频能量占比={ratio_original:.3f}')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(dct_nn, cmap='gray')
plt.title(f'最近邻 DCT\n低频占比={ratio_nn:.3f}')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(dct_linear, cmap='gray')
plt.title(f'双线性 DCT\n低频占比={ratio_linear:.3f}')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(img_cubic, cmap='gray')
plt.title(f'双三次恢复\nMSE={mse_cubic:.2f}, PSNR={psnr_cubic:.1f}dB')
plt.axis('off')

plt.tight_layout()
plt.savefig('/home/xie_bao_fang/cv-course/my-experiment03/images3/result.png', dpi=150)
plt.show()

# ==================== 8. 分析结论 ====================
print("\n" + "="*60)
print("8. 分析结论")
print("="*60)
print("""
【空间域分析】
- PSNR值越高表示恢复图像质量越好，双三次插值通常获得最高的PSNR
- 双三次插值边缘更平滑，最近邻插值会产生锯齿效应

【频域分析（FFT）】
- 缩小后的图像频谱高频成分丢失（频谱中心周围较暗）
- 恢复图像无法完全恢复丢失的高频细节
- 频谱中心代表低频信息，四周代表高频信息

【DCT分析】
- DCT系数集中在左上角（低频区域），高频系数较小
- 图像越模糊，低频能量占比越高
- 恢复方法越好，能量分布越接近原图

【结论】
双三次插值恢复效果最佳，但计算复杂度最高；
双线性插值在质量和效率间取得平衡；
最近邻插值最快但质量最差。
""")