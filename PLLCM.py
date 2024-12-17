import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage.segmentation import random_walker
from skimage.util import img_as_float


# Step 1: 多尺度滑动窗口滤波
def multiscale_window_filter(image, scales=[3, 5, 7, 9, 11]):
    filtered_images = []
    for L in scales:
        if L // 4 == 0:  # 防止除以 0 错误
            continue
        # 构建窗口，中心为正数，边缘为负数
        M = np.zeros((L, L))
        M[L // 2, L // 2] = 4 * (L // 4)
        M[:, [0, -1]] = -1
        M[[0, -1], :] = -1

        # 滤波操作
        filtered = convolve(image, M)
        filtered_images.append(filtered / (4 * (L // 4)))

    return np.max(filtered_images, axis=0)


# Step 2: 提取候选目标像素
def extract_candidate_pixels(filtered_image, kTh=5.2):
    mean_val = np.mean(filtered_image)
    std_val = np.std(filtered_image)
    threshold = mean_val + kTh * std_val
    candidate_pixels = filtered_image > threshold
    return candidate_pixels


# Step 3: 构建基于RW的局部窗口并计算PLLCM
def calculate_pllcm(image, candidate_pixels):
    height, width = image.shape
    pllcm_map = np.zeros_like(image, dtype=float)

    # 定义11x11局部窗口的范围
    for y in range(5, height - 5):
        for x in range(5, width - 5):
            if candidate_pixels[y, x]:
                # 提取11x11的局部窗口
                local_window = image[y - 5:y + 6, x - 5:x + 6]

                # 构建标签图像用于RW
                labels = np.zeros_like(local_window)
                labels[0, :] = 1
                labels[-1, :] = 2
                labels[:, 0] = 3
                labels[:, -1] = 4
                labels[5, 5] = -1  # 中心候选目标

                # RW分割
                segmented_labels = random_walker(local_window, labels, beta=10)

                # 计算局部对比度
                target_area = segmented_labels == 5
                background_area = segmented_labels != 5

                # 确保区域不为空
                if np.any(target_area) and np.any(background_area):
                    pllcm_map[y, x] = local_window[target_area].mean() - local_window[background_area].mean()
                else:
                    pllcm_map[y, x] = 0  # 如果区域为空，设置为0

    return pllcm_map


# Step 4: 自适应阈值分割目标
def adaptive_threshold(pllcm_map, lTh=0.5):
    Emax = pllcm_map.max()
    Emean = pllcm_map.mean()
    threshold = lTh * Emax + (1 - lTh) * Emean
    return pllcm_map > threshold


# Main function to process the image
def process_image(image_path):
    # 加载并预处理图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = img_as_float(image)

    # Step 1: 滤波并生成初始对比度图
    filtered_image = multiscale_window_filter(image)

    # Step 2: 提取候选目标像素
    candidate_pixels = extract_candidate_pixels(filtered_image)

    # Step 3: 计算PLLCM增强图
    pllcm_map = calculate_pllcm(image, candidate_pixels)

    # Step 4: 分割目标
    target_mask = adaptive_threshold(pllcm_map)

    # 保存结果
    cv2.imwrite('pllcm_output.png', (target_mask * 255).astype(np.uint8))

    return target_mask


# 调用处理函数
result = process_image('./output_frames/first_frame.jpg')
cv2.imshow("Result", result.astype(np.uint8) * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
