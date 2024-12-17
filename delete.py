import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def gaussian_filter(img, kernel_size, sigma):
    """应用高斯滤波器平滑图像"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def sobel_edge_detection(img):
    """提取图像边缘特征"""
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    return edges / edges.max()  # 归一化到[0, 1]

def gamma_correction(img, gamma):
    """伽马校正增强目标"""
    img_normalized = img / img.max()
    return np.power(img_normalized, gamma)

def laplacian_pyramid(img, levels=3):
    """构建拉普拉斯金字塔"""
    gaussian_pyramid = [img]
    for i in range(levels):
        img = gaussian_filter(gaussian_pyramid[-1], kernel_size=5, sigma=2)
        gaussian_pyramid.append(img)
    laplacian_pyramid = [gaussian_pyramid[i] - gaussian_filter(gaussian_pyramid[i], kernel_size=5, sigma=2)
                         for i in range(levels)]
    return laplacian_pyramid

def adaptive_filter(kernel_size, contrast):
    """自适应生成滤波核"""
    # 确定滤波核的中心位置
    h, w = contrast.shape
    center_x, center_y = h // 2, w // 2

    # 提取局部对比度，确保大小与滤波核一致
    half_k = kernel_size // 2
    local_contrast = contrast[center_x - half_k:center_x + half_k + 1,
                               center_y - half_k:center_y + half_k + 1]

    # 构造高斯核
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    x, y = np.meshgrid(x, y)
    sigma = kernel_size / 2
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # 将局部对比度应用于高斯核
    adaptive_kernel = gaussian_kernel * local_contrast
    return adaptive_kernel / adaptive_kernel.sum()


def compute_local_energy(img, neighborhood_size=5):
    """计算局部能量密度"""
    kernel = np.ones((neighborhood_size, neighborhood_size)) / (neighborhood_size**2)
    local_energy = cv2.filter2D(img**2, -1, kernel)
    return local_energy / local_energy.max()  # 归一化到[0, 1]

def detect_weak_targets(img, levels=3, gamma=2.0, neighborhood_size=5):
    """实现完整的改进 Top-Hat 算法"""
    # Step 1: 图像预处理
    smoothed = gaussian_filter(img, kernel_size=5, sigma=2)
    edges = sobel_edge_detection(smoothed)
    enhanced = gamma_correction(edges, gamma)

    # Step 2: 多尺度分解
    laplacian_layers = laplacian_pyramid(enhanced, levels=levels)

    # Step 3: 自适应滤波与目标增强
    filtered_layers = []
    for i, layer in enumerate(laplacian_layers):
        # 归一化对比度
        contrast = np.abs(layer) / (np.abs(layer).max() + 1e-8)

        # 提取每个局部区域的滤波核
        kernel = adaptive_filter(kernel_size=5, contrast=contrast)

        # 应用滤波
        filtered = cv2.filter2D(layer, -1, kernel)
        filtered_layers.append(filtered)

    # Step 4: 局部能量分析
    adjusted_layers = []
    for layer in filtered_layers:
        energy = compute_local_energy(layer, neighborhood_size=neighborhood_size)
        adjusted = layer * energy  # 动态调整显著性
        adjusted_layers.append(adjusted)

    # Step 5: 多尺度融合
    final_response = sum(adjusted_layers)  # 叠加所有层
    final_response = final_response / final_response.max()  # 归一化

    return final_response

# 测试代码
if __name__ == "__main__":
    # 加载灰度图像
    input_image = cv2.imread("./output_frames/Misc_394.jpg", cv2.IMREAD_GRAYSCALE)
    input_image = input_image.astype(np.float64) / 255.0  # 归一化到[0, 1]

    # 检测弱小目标
    response = detect_weak_targets(input_image)

    # # 可视化结果
    # cv2.imshow("Input Image", input_image)
    # cv2.imshow("Weak Target Detection", response)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.title("Original Image"), plt.imshow(input_image, cmap="gray"), plt.axis("off")
    plt.subplot(1, 2, 2), plt.title("detection_result"), plt.imshow(response, cmap="gray"), plt.axis("off")
    plt.tight_layout()
    plt.show()


