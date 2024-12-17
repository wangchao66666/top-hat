import cv2
import numpy as np

def optimized_lcm(image, kernel_size=9):
    """
    使用卷积和归一化优化的 LCM 算法。
    """
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 定义卷积核，计算邻域的均值
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # 计算邻域的局部均值
    local_mean = cv2.filter2D(gray, -1, kernel)

    # 计算每个像素的平方（中心像素的 L_n * L_n）
    local_max_squared = gray.astype(np.float32) ** 2

    # 避免除零
    contrast = local_max_squared / (local_mean + 1e-6)

    # 对比度归一化到 0-255 的范围
    contrast = np.clip(contrast, 0, 255).astype(np.uint8)

    return contrast


def calculate_threshold_and_mask(contrast_map, k):
    """
    Calculate threshold and segment targets from contrast map.

    Parameters:
        contrast_map (numpy.ndarray): Input contrast map.
        k (float): Threshold multiplier.

    Returns:
        threshold (float): Calculated threshold value.
        target_mask (numpy.ndarray): Binary mask of detected targets.
    """
    # Compute Mean and Standard Deviation of the Contrast Map
    mean_contrast = np.mean(contrast_map)
    std_contrast = np.std(contrast_map)

    # Calculate Threshold (T = mean + k * std)
    threshold = mean_contrast + k * std_contrast

    # Segment Targets using Threshold
    _, target_mask = cv2.threshold(contrast_map, threshold, 255, cv2.THRESH_BINARY)

    return target_mask.astype(np.uint8)

def process_video_optimized(image_path):
    # 加载图像
    image = cv2.imread(image_path)

    # 检查图像是否成功加载
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # 计算优化后的 LCM 对比度图
    contrast_map = optimized_lcm(image)
    cv2.imshow('Contrast image', contrast_map)

    mask = calculate_threshold_and_mask(contrast_map, 3)

    cv2.imshow("Target Mask", mask)
    # # 找到对比度图中的最大亮点位置
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(contrast_map)
    # print(max_loc)
    #
    # # 在原始视频帧中标记最亮点的位置
    # cv2.circle(image, max_loc, 5, (0, 0, 255), -1)
    # cv2.putText(image, f"Brightest: {max_loc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (255, 0, 0), 2, cv2.LINE_AA)
    #
    # cv2.imshow('Optimized LCM Brightest Point Detection', image)

    # 释放资源并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图像路径
image_path = 'C:\\Users\\localhost\\Desktop\\open-sirst-v2-master\\images\\targets\\Misc_53.png'
process_video_optimized(image_path)
