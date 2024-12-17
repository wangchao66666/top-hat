'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def dynamic_top_hat(image, kernel_type="ellipse", kernel_size=(5, 10), filter_type="gaussian", sigma=1):
    """
    改进的动态 Top-Hat 算法，结合滤波预处理与动态形状调整。

    Parameters:
        image (numpy.ndarray): 输入灰度图像。
        kernel_type (str): 结构元素的类型，可选 "ellipse", "circle", "rectangle"。
        kernel_size (tuple): 结构元素的动态大小范围 (min_size, max_size)。
        filter_type (str): 滤波类型，可选 "gaussian", "median", "bilateral"。
        sigma (int): 高斯滤波的标准差（仅用于高斯滤波）。

    Returns:
        NWTH (numpy.ndarray): 改进的 White Top-Hat 结果。
        NBTH (numpy.ndarray): 改进的 Black Top-Hat 结果。
    """
    # -------------------
    # 1. 滤波预处理
    # -------------------
    if filter_type == "gaussian":
        filtered_image = cv2.GaussianBlur(image, (5, 5), sigma)
    elif filter_type == "median":
        filtered_image = cv2.medianBlur(image, 5)
    elif filter_type == "bilateral":
        filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    else:
        raise ValueError("Invalid filter type. Choose 'gaussian', 'median', or 'bilateral'.")

    # -------------------
    # 2. 动态生成结构元素
    # -------------------
    min_size, max_size = kernel_size
    if kernel_type == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_size, max_size))
    elif kernel_type == "circle":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_size, min_size))
    elif kernel_type == "rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_size, max_size))
    else:
        raise ValueError("Invalid kernel type. Choose 'ellipse', 'circle', or 'rectangle'.")

    # -------------------
    # 3. 改进的 White Top-Hat
    # -------------------
    morph_open = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
    NWTH = cv2.subtract(filtered_image, morph_open)  # NWTH = f - (f ○ kernel)

    # -------------------
    # 4. 改进的 Black Top-Hat
    # -------------------
    morph_close = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)
    NBTH = cv2.subtract(morph_close, filtered_image)  # NBTH = (f ● kernel) - f

    return NWTH, NBTH


# -------------------
# 示例运行
# -------------------
if __name__ == "__main__":
    # 加载图像
    image = cv2.imread("./output_video_png2/frame_0050.jpg", cv2.IMREAD_GRAYSCALE)

    # 调用动态 Top-Hat 算法
    NWTH, NBTH = dynamic_top_hat(
        image=image,
        kernel_type="circle",
        kernel_size=(5, 15),
        filter_type="gaussian",
        sigma=2
    )

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.title("Original Image"), plt.imshow(image, cmap="gray"), plt.axis("off")
    plt.subplot(1, 3, 2), plt.title("New White Top-Hat"), plt.imshow(NWTH, cmap="gray"), plt.axis("off")
    plt.subplot(1, 3, 3), plt.title("New Black Top-Hat"), plt.imshow(NBTH, cmap="gray"), plt.axis("off")
    plt.tight_layout()
    plt.show()
'''
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def dynamic_brightness_weight(image, T_high=0.8):
    """动态亮度权重调整，抑制高亮区域"""
    brightness_weight = np.exp(-image / T_high)
    return brightness_weight

def background_smoothing(image, sigma=5):
    """通过高斯平滑提取背景，同时获取细节信息"""
    smoothed_background = gaussian_filter(image, sigma=sigma)
    detail_layer = image - smoothed_background
    return smoothed_background, detail_layer

def enhanced_image_combination(image, detail_layer, lambda_detail=0.7):
    """综合背景与细节层，调整增强图像"""
    enhanced_image = image + lambda_detail * detail_layer
    enhanced_image = np.clip(enhanced_image, 0, 1)
    return enhanced_image

def gradient_magnitude(image):
    """计算梯度幅值"""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
    return gradient

# def dynamic_structural_element(image, gradient, sizes):
#     """多结构元素动态生成与融合"""
#     result = np.zeros_like(image, dtype=np.float64)
#     for size in sizes:
#         se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
#         # 白顶帽操作：提取比背景更亮的小目标
#         white_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)
#
#         # 黑顶帽操作：提取比背景更暗的小目标
#         black_tophat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, se)
#
#         weight = gradient  # 使用梯度作为权重
#         result += weight * (white_tophat+black_tophat)
#     result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
#     return result

# def generate_ellipse_matrix(width):
#     """
#     生成一个矩阵，其中背景为1，椭圆部分为0。
#
#     参数:
#     width (int): 椭圆的长轴（矩阵的宽）
#     height (int): 椭圆的短轴（矩阵的高）
#
#     返回:
#     numpy.ndarray: 包含椭圆的矩阵
#     """
#     # 确定矩阵大小，保证中心对齐
#     height =int(width*0.6)
#     matrix_size_x = width + 4  # 增加边距
#     matrix_size_y = height + 4  # 增加边距
#
#     # 创建矩阵并初始化为1
#     matrix = np.ones((matrix_size_y, matrix_size_x), dtype=int)
#
#     # 椭圆参数
#     center_x = matrix_size_x // 2
#     center_y = matrix_size_y // 2
#     a = width // 2  # 长轴半径
#     b = height // 2 + 1    # 短轴半径
#
#     # 绘制椭圆
#     for y in range(matrix_size_y):
#         for x in range(matrix_size_x):
#             # 椭圆公式: ((x - cx)/a)^2 + ((y - cy)/b)^2 <= 1
#             if ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1:
#                 matrix[y, x] = 0
#
#     # 绘制矩形：最大内接矩形
#     rect_half_width = a // 2 # 矩形宽为椭圆长轴
#     rect_half_height = b // 2 # 矩形高为椭圆短轴
#
#     start_x = center_x - rect_half_width
#     end_x = center_x + rect_half_width+1
#     start_y = center_y - rect_half_height
#     end_y = center_y + rect_half_height+1
#
#     # 设置矩形内部为1
#     for y in range(start_y, end_y):
#         for x in range(start_x, end_x):
#             matrix[y, x] = 1
#
#
#     return matrix.astype(np.uint8)

# def generate_structuring_elements(scale):
#     """
#     生成图1和图2的非对称结构元素。
#     参数:
#     - scale: 尺度，用于控制结构元素的大小。
#
#     返回:
#     - structuring_elements: 包含图1和图2的结构元素
#     """
#     # 图1结构元素：用于膨胀
#     ellipse_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))  # 椭圆
#     rect_b0 = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * scale, 2 * scale))  # 矩形
#
#     # 图2结构元素：用于腐蚀
#     rect_bi = cv2.getStructuringElement(cv2.MORPH_RECT, (scale // 2, scale // 2))  # 内部矩形
#     ellipse_b2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))  # 椭圆
#
#     return (ellipse_b, rect_b0, rect_bi, ellipse_b2)
#
# def remove_highlighted_regions(image, threshold_factor=1.5):
#     """
#     检测并减除高亮区域（如云彩、房屋等）。
#     参数:
#     - image: 输入灰度图像。
#     - threshold_factor: 动态阈值因子，高于此灰度值的区域会被减除。
#     返回:
#     - processed_image: 减除高亮区域后的图像。
#     """
#     # 动态阈值计算
#     mean_intensity = np.mean(image)
#     threshold = threshold_factor * mean_intensity
#
#     # 生成高亮掩膜
#     highlight_mask = (image > threshold).astype(np.uint8) * 255
#
#     # 对高亮区域进行形态学操作（扩展和平滑）
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#     highlight_mask = cv2.dilate(highlight_mask, kernel)
#     highlight_mask = cv2.erode(highlight_mask, kernel)
#
#     # 减除高亮区域
#     processed_image = cv2.inpaint(image, highlight_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
#     return processed_image
def newRingStrel(R_i, ratio=2.5):
    """
    构造矩形环状结构元素
    输入参数:
    R_i : 内半径，即目标区域的大小
    ratio : R_o 和 R_i 的比值，默认值为 2
    输出:
    B_0 : 小矩形结构元素，边长为 R_i
    B_1 : 环形矩形结构元素，外半径为 R_o，内半径为 R_i
    """
    # 计算外半径 R_o
    R_o = ratio * R_i

    # 1. 创建小矩形结构元素 B_0，大小为 R_i x R_i
    B_0 = np.ones((R_i, R_i), dtype=int)

    # 2. 创建环形矩形结构元素 B_1，外半径为 R_o，内半径为 R_i
    d = 2 * int(R_o) + 1  # 结构元素的大小
    B_1 = np.ones((d, d), dtype=int)  # 初始全为 1 的矩阵

    # 计算内半径区域的起始和结束索引
    start_index = int(R_o) + 1 - R_i
    end_index = int(R_o) + 1 + R_i

    # 将内半径区域置为 0，形成环形结构
    B_1[start_index:end_index, start_index:end_index] = 0

    return B_0.astype(np.uint8), B_1.astype(np.uint8)

def custom_morphology(image, gradient,scales=[3, 5, 7]):
    """
    使用自定义结构元素进行多尺度白顶帽和黑顶帽操作。

    参数:
    - image: 输入灰度图像。
    - scales: 多尺度结构元素的尺度列表。

    返回:
    - final_result: 白顶帽与黑顶帽操作结果的叠加。
    """

    white_tophat_sum = np.zeros_like(image, dtype=np.float64)
    black_tophat_sum = np.zeros_like(image, dtype=np.float64)

    for scale in scales:
        # 生成结构元素
        B0, B1= newRingStrel(scale)

        img_d = cv2.dilate(image, B1)

        img_e = cv2.erode(img_d, B0)

        white_tophat = image-img_e
        white_tophat[white_tophat < 0] = 0
        white_tophat = cv2.normalize( white_tophat, None, 0, 1, cv2.NORM_MINMAX)
        white_tophat_sum += white_tophat

        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 3, 1), plt.title("Original Image"), plt.imshow(image, cmap="gray"), plt.axis("off")
        # plt.subplot(1, 3, 2), plt.title("Improved Top-Hat Result"), plt.imshow(img_e, cmap="gray"), plt.axis("off")
        # plt.subplot(1, 3, 3), plt.title("Improved Top-Hat Result"), plt.imshow(white_tophat, cmap="gray"), plt.axis("off")
        # img_B_e=cv2.erode(img, B0)
        # img_b_d = cv2.dilate(img, B1)
        # black_tophat_sum

    # 结果归一化并叠加
    final_result =gradient *(white_tophat_sum)
    final_result = cv2.normalize(final_result, None, 0, 1, cv2.NORM_MINMAX)

    return final_result


def custom_morphology1(image, gradient, scales=[3, 5, 7]):
    """
    使用自定义结构元素进行多尺度白顶帽和黑顶帽操作，并进行叠加优化。

    参数:
    - image: 输入灰度图像。
    - gradient: 梯度图像，用于权重调整。
    - scales: 多尺度结构元素的尺度列表。

    返回:
    - final_result: 加强叠加优化后的结果图像。
    """
    white_tophat_stack = []

    for scale in scales:
        # 生成结构元素
        B0, B1 = newRingStrel(scale)

        # 进行膨胀和腐蚀操作
        img_d = cv2.dilate(image, B1)
        img_e = cv2.erode(img_d, B0)

        # 计算白顶帽
        white_tophat = image - img_e
        white_tophat[white_tophat < 0] = 0
        white_tophat = cv2.normalize(white_tophat, None, 0, 1, cv2.NORM_MINMAX)

        # 存储当前尺度的白顶帽结果
        white_tophat_stack.append(white_tophat)

    # 转换为多通道数组，便于操作
    white_tophat_stack = np.array(white_tophat_stack)

    # 根据出现的频率进行加权叠加
    weight_map = np.sum(white_tophat_stack > 0.5, axis=0) / len(scales)
    enhanced_sum = np.sum(white_tophat_stack * weight_map, axis=0)

    # 应用梯度增强并归一化
    final_result = gradient * enhanced_sum
    final_result = cv2.normalize(final_result, None, 0, 1, cv2.NORM_MINMAX)

    return final_result


def adaptive_contrast_enhancement(image, gradient, beta=1.0, brightness_weight=None):
    """自适应对比度增强"""
    mean_filter = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5))
    if brightness_weight is None:
        brightness_weight = np.ones_like(image)
    enhanced = image + beta * gradient * brightness_weight * (image - mean_filter)
    #std_filter = np.sqrt(cv2.boxFilter((image - mean_filter) ** 2, ddepth=-1, ksize=(5, 5)))  # 局部标准差
    # 自适应增强因子 Beta(x, y)
    #enhanced = image + beta * gradient * (image - mean_filter)
    enhanced = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX)
    return enhanced

def nonlinear_adjustment(image, gamma=2.0):
    """非线性调节（Sigmoid调整）"""
    sigmoid = 1 / (1 + np.exp(-gamma * (image - 0.5)))
    return cv2.normalize(sigmoid, None, 0, 1, cv2.NORM_MINMAX)



def low_pass_background_filter(image, cutoff_sigma=3):
    """利用低通滤波平滑背景，提取高频成分"""
    smoothed = gaussian_filter(image, sigma=cutoff_sigma)
    high_frequency_component = image - smoothed
    return smoothed, high_frequency_component

def improved_top_hat(image, sizes=[3, 5, 7], beta=1.0, gamma=2.0, T_high=1.0, sigma=5, lambda_detail=0.7):
    """
    改进 Top-Hat 算法
    :param image: 输入灰度图像
    :param sizes: 动态结构元素的大小列表
    :param beta: 对比度增强系数
    :param gamma: 非线性调整系数
    :return: 改进的弱小目标检测结果
    """

    # 1. 计算梯度幅值作为感知权重
    gradient = gradient_magnitude(image)

    # 2. 动态亮度权重，抑制高亮区域
    brightness_weight = dynamic_brightness_weight(image, T_high)

    # 3. 背景平滑与细节提取
    smoothed_background, detail_layer = background_smoothing(image, sigma=sigma)

    # 4. 综合背景与细节，调整增强图像
    image = enhanced_image_combination(smoothed_background, detail_layer, lambda_detail)

    # 2. 多结构元素动态融合
    morph_result = custom_morphology(image, gradient, sizes)

    # 3. 自适应对比度增强
    enhanced_image = adaptive_contrast_enhancement(morph_result, gradient, beta, brightness_weight)

    #smoothed, high_frequency = low_pass_background_filter(enhanced_image, cutoff_sigma=sigma)

    # 4. 非线性调整
    final_result = nonlinear_adjustment(enhanced_image, gamma)

    return final_result

# 测试改进算法
if __name__ == "__main__":
    # 读取输入图像（灰度）
    image_path = "C:\\Users\\localhost\\Desktop\\open-sirst-v2-master\\images\\targets\\Misc_50.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.normalize(image.astype("float64"), None, 0, 1, cv2.NORM_MINMAX)

    # 改进算法的检测
    result = improved_top_hat(image, sizes=[3, 5, 7, 9], beta=2.0, gamma=20.0, T_high=0.8, sigma=10, lambda_detail=0.7)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.title("Original Image"), plt.imshow(image, cmap="gray"), plt.axis("off")
    plt.subplot(1, 2, 2), plt.title("Improved Top-Hat Result"), plt.imshow(result, cmap="gray"), plt.axis("off")
    plt.tight_layout()
    plt.show()

    # # 转换图像到 0-255 并转换为 uint8 类型
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 保存结果
    output_dir = "top_hat_image"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_image1.jpg")
    cv2.imwrite(output_path, result)



# import numpy as np
# import cv2
# from scipy.ndimage import gaussian_filter
# from scipy.special import expit as sigmoid
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
# # Step 1: 图像预处理
# def preprocess_image(image, filter_type="gaussian", kernel_size=5):
#     """
#     对输入图像应用滤波，减少噪声。
#     参数:
#         image: 输入图像，numpy 数组格式。
#         filter_type: 滤波类型，可选 "gaussian" 或 "median"。
#         kernel_size: 滤波器的核大小。
#     返回:
#         预处理后的图像。
#     """
#     if filter_type == "gaussian":
#         # 使用高斯滤波器对图像进行平滑处理
#         # 高斯滤波公式：f_smooth(x, y) = f(x, y) * G(x, y)
#         return gaussian_filter(image, sigma=kernel_size / 6.0)
#     elif filter_type == "median":
#         # 使用中值滤波器去除椒盐噪声
#         return cv2.medianBlur(image.astype(np.uint8), kernel_size)
#     else:
#         # 如果传入的滤波类型不被支持，抛出错误
#         raise ValueError("不支持的滤波类型，请使用 'gaussian' 或 'median'。")
#
# # Step 2: 生成多结构元素
# def generate_structural_elements(base_size, scales):
#     """
#     生成不同尺度的结构元素。
#     参数:
#         base_size: 结构元素的基础尺寸。
#         scales: 尺度因子列表。
#     返回:
#         不同尺度的结构元素列表。
#     """
#     elements = []
#     for scale in scales:
#         size = int(base_size * scale)  # 按比例调整结构元素的大小
#         # 创建椭圆形结构元素
#         elements.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
#     return elements
#
# # Step 3: 动态权重计算
# def compute_dynamic_weight(image, neighborhood_size=15, epsilon=1e-6):
#     """
#     基于局部对比度计算动态权重。
#     参数:
#         image: 输入图像。
#         neighborhood_size: 计算局部区域的邻域大小。
#         epsilon: 防止除零的小值。
#     返回:
#         每个像素的动态权重。
#     """
#     # 计算局部均值（用平均滤波器实现）
#     mean_local = cv2.blur(image, (neighborhood_size, neighborhood_size))
#     # 计算局部标准差
#     std_local = cv2.GaussianBlur(image ** 2, (neighborhood_size, neighborhood_size), 0) - mean_local ** 2
#     std_local = np.sqrt(std_local + epsilon)  # 防止出现负值或零值
#     # 计算动态权重，基于局部对比度
#     weights = np.abs(image - mean_local) / (std_local + epsilon)
#     return weights
#
# # Step 4: 多结构元素响应计算
# def compute_multi_structure_response(image, structural_elements):
#     """
#     计算多结构元素的响应。
#     参数:
#         image: 输入图像。
#         structural_elements: 结构元素列表。
#     返回:
#         结构元素的综合响应。
#     """
#     response = np.zeros_like(image, dtype=np.float32)  # 初始化响应图像
#     for element in structural_elements:
#         # 使用每个结构元素进行形态学开运算
#         opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
#         response += opened  # 累加所有结构元素的响应
#     return response
#
# # Step 5: 自适应 Sigmoid 非线性调节
# def adaptive_sigmoid_adjustment(response, image, alpha=1.0, epsilon=1e-6):
#     """
#     应用自适应 Sigmoid 调节以增强对比度。
#     参数:
#         response: 多结构响应图像。
#         image: 原始图像，用于计算局部统计信息。
#         alpha: 控制曲线锐度的系数。
#         epsilon: 防止除零的小值。
#     返回:
#         调节后的响应图像。
#     """
#     # 计算邻域均值
#     neighborhood_mean = cv2.blur(image, (15, 15))
#     # 计算局部标准差，复用动态权重计算方法
#     std_local = compute_dynamic_weight(image, neighborhood_size=15, epsilon=epsilon)
#     # 根据局部对比度计算自适应参数 k
#     k = alpha * std_local
#     # 应用 Sigmoid 函数进行非线性调节
#     return sigmoid((response - neighborhood_mean) / (k + epsilon))
#
# # Step 6: 梯度显著性权重计算
# def compute_gradient_weight(image):
#     """
#     计算基于梯度的显著性权重。
#     参数:
#         image: 输入图像。
#     返回:
#         梯度权重图。
#     """
#     # 计算 x 和 y 方向的 Sobel 梯度
#     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     # 计算梯度幅值
#     gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
#     # 归一化梯度幅值，确保最大值为 1
#     return gradient_magnitude / np.max(gradient_magnitude)
#
# # Step 7: 显著性区域强化
# def enhance_saliency(adjusted_response, gradient_weights):
#     """
#     通过结合调节后的响应和梯度权重，强化显著性区域。
#     参数:
#         adjusted_response: 经 Sigmoid 调节后的响应图像。
#         gradient_weights: 梯度计算得到的显著性权重。
#     返回:
#         最终增强的图像。
#     """
#     # 显著性增强，通过逐像素相乘
#     return adjusted_response * gradient_weights
#
# # 主函数：改进的 Top-Hat 算法
# def improved_top_hat(image, base_size=15, scales=[0.8, 1.0, 1.2], alpha=1.0):
#     """
#     改进的 Top-Hat 算法，用于弱目标检测。
#     参数:
#         image: 输入图像，numpy 数组格式。
#         base_size: 结构元素的基础尺寸。
#         scales: 结构元素的尺度因子列表。
#         alpha: 自适应 Sigmoid 的控制系数。
#     返回:
#         增强后的图像，突出弱目标。
#     """
#     # 第一步：图像预处理
#     preprocessed = preprocess_image(image, filter_type="gaussian", kernel_size=5)
#
#     # 第二步：生成多尺度结构元素
#     structural_elements = generate_structural_elements(base_size, scales)
#
#     # 第三步：计算多结构元素的响应
#     response = compute_multi_structure_response(preprocessed, structural_elements)
#
#     # 第四步：自适应 Sigmoid 调节
#     adjusted_response = adaptive_sigmoid_adjustment(response, preprocessed, alpha=alpha)
#
#     # 第五步：计算梯度权重
#     gradient_weights = compute_gradient_weight(preprocessed)
#
#     # 第六步：显著性增强
#     final_result = enhance_saliency(adjusted_response, gradient_weights)
#
#     # 返回最终结果
#     return final_result
#
# # 示例用法
# if __name__ == "__main__":
#     # 加载示例图像
#     input_image = cv2.imread("./output_video_png/frame_0500.jpg", cv2.IMREAD_GRAYSCALE)
#
#     # 应用改进的 Top-Hat 算法
#     result = improved_top_hat(input_image, base_size=15, scales=[0.8, 1.0, 1.2], alpha=1.5)
#
#     # 显示结果
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1), plt.title("原始图像"), plt.imshow(input_image, cmap="gray"), plt.axis("off")
#     plt.subplot(1, 2, 2), plt.title("增强后的图像"), plt.imshow(result, cmap="gray"), plt.axis("off")
#     plt.tight_layout()
#     plt.show()
'''
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def gradient_magnitude(image):
    """计算梯度幅值"""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
    return gradient

def dynamic_brightness_weight(image, T_high=0.8):
    """动态亮度权重调整，抑制高亮区域"""
    brightness_weight = np.exp(-image / T_high)
    return brightness_weight

def background_smoothing(image, sigma=5):
    """通过高斯平滑提取背景，同时获取细节信息"""
    smoothed_background = gaussian_filter(image, sigma=sigma)
    detail_layer = image - smoothed_background
    return smoothed_background, detail_layer

def enhanced_image_combination(image, detail_layer, lambda_detail=0.7):
    """综合背景与细节层，调整增强图像"""
    enhanced_image = image + lambda_detail * detail_layer
    enhanced_image = np.clip(enhanced_image, 0, 1)
    return enhanced_image

def adaptive_contrast_enhancement(image, gradient, beta=1.0, brightness_weight=None):
    """自适应对比度增强"""
    mean_filter = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5))
    if brightness_weight is None:
        brightness_weight = np.ones_like(image)
    enhanced = image + beta * gradient * brightness_weight * (image - mean_filter)
    enhanced = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX)
    return enhanced

def low_pass_background_filter(image, cutoff_sigma=3):
    """利用低通滤波平滑背景，提取高频成分"""
    smoothed = gaussian_filter(image, sigma=cutoff_sigma)
    high_frequency_component = image - smoothed
    return smoothed, high_frequency_component

def improved_top_hat(image, beta=1.0, T_high=0.8, sigma=5, lambda_detail=0.7):
    """
    改进 Top-Hat 算法
    :param image: 输入灰度图像
    :param sizes: 动态结构元素的大小列表
    :param beta: 对比度增强系数
    :param gamma: 非线性调整系数
    :param T_high: 高亮区域的亮度阈值
    :param sigma: 背景平滑滤波强度
    :param lambda_detail: 细节层增强系数
    :return: 改进后的检测结果
    """
    # 1. 计算梯度幅值
    gradient = gradient_magnitude(image)

    # 2. 动态亮度权重，抑制高亮区域
    brightness_weight = dynamic_brightness_weight(image, T_high)

    # 3. 背景平滑与细节提取
    smoothed_background, detail_layer = background_smoothing(image, sigma=sigma)

    # 4. 综合背景与细节，调整增强图像
    enhanced_image = enhanced_image_combination(smoothed_background, detail_layer, lambda_detail)

    # 5. 自适应对比度增强
    enhanced_image = adaptive_contrast_enhancement(enhanced_image, gradient, beta, brightness_weight)

    # 6. 低通滤波处理背景，提取高频目标
    smoothed, high_frequency = low_pass_background_filter(enhanced_image, cutoff_sigma=sigma)

    # 7. 最终结果叠加高频成分
    final_result = high_frequency
    final_result = np.clip(final_result, 0, 1)

    return final_result

# 测试改进算法
if __name__ == "__main__":
    # 读取输入图像（灰度）
    image_path = "./output_video_png/frame_0000.jpg"  # 替换为实际路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.normalize(image.astype("float64"), None, 0, 1, cv2.NORM_MINMAX)

    # 改进算法检测
    result = improved_top_hat(image, beta=1.0, T_high=0.8, sigma=5, lambda_detail=0.7)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.title("Original Image"), plt.imshow(image, cmap="gray"), plt.axis("off")
    plt.subplot(1, 2, 2), plt.title("Improved Top-Hat Result"), plt.imshow(result, cmap="gray"), plt.axis("off")
    plt.tight_layout()
    plt.show()
'''

# import cv2
# import numpy as np
# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
# def gradient_magnitude(image):
#     """计算梯度幅值"""
#     sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
#     gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
#     return gradient
#
#
# def dynamic_structural_element(image, gradient, sizes, high_brightness_threshold=0.8):
#     """多结构元素动态生成与融合，并限制高亮区域影响"""
#     result = np.zeros_like(image, dtype=np.float64)
#     for size in sizes:
#         se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
#         morph_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)
#
#         # 限制高亮区域的权重
#         brightness_weight = np.clip(1 - image / high_brightness_threshold, 0, 1)
#         weight = gradient * brightness_weight  # 权重结合梯度和亮度约束
#
#         result += weight * morph_open
#     result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
#     return result
#
#
# def adaptive_contrast_enhancement(image, gradient, beta=1.0):
#     """自适应对比度增强，并对高亮区域做非线性压制"""
#     mean_filter = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5))
#
#     # 非线性压制高亮区域
#     #brightness_weight = np.clip(1 - image / high_brightness_threshold, 0, 1)
#
#     enhanced = image + beta * gradient * (image - mean_filter)
#     enhanced = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX)
#     return enhanced
#
#
# def nonlinear_adjustment(image, gamma=2.0):
#     """非线性调节（Sigmoid调整）"""
#     sigmoid = 1 / (1 + np.exp(-gamma * (image - 0.5)))
#     return cv2.normalize(sigmoid, None, 0, 1, cv2.NORM_MINMAX)
#
#
# def improved_top_hat(image, sizes=[3, 5, 7], beta=1.0, gamma=2.0, high_brightness_threshold=0.8):
#     """
#     改进 Top-Hat 算法（微调）
#     :param image: 输入灰度图像
#     :param sizes: 动态结构元素的大小列表
#     :param beta: 对比度增强系数
#     :param gamma: 非线性调整系数
#     :param high_brightness_threshold: 高亮区域阈值
#     :return: 改进的弱小目标检测结果
#     """
#     # 1. 计算梯度幅值作为感知权重
#     gradient = gradient_magnitude(image)
#
#     # 2. 多结构元素动态融合，限制高亮区域影响
#     morph_result = dynamic_structural_element(image, gradient, sizes, high_brightness_threshold)
#
#     # 3. 自适应对比度增强，加入高亮区域非线性压制
#     enhanced_image = adaptive_contrast_enhancement(image, gradient, beta)
#
#     # 4. 非线性调整
#     final_result = nonlinear_adjustment(morph_result * enhanced_image, gamma)
#
#     return final_result
#
#
# # 测试改进算法
# if __name__ == "__main__":
#     # 读取输入图像（灰度）
#     image_path = "./output_video_png/frame_0000.jpg"  # 替换为实际路径
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.normalize(image.astype("float64"), None, 0, 1, cv2.NORM_MINMAX)
#
#     # 改进算法的检测
#     result = improved_top_hat(image, sizes=[3, 5, 7], beta=1.0, gamma=2.0, high_brightness_threshold=0.8)
#
#     # 可视化结果
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1), plt.title("Original Image"), plt.imshow(image, cmap="gray"), plt.axis("off")
#     plt.subplot(1, 2, 2), plt.title("Improved Top-Hat Result"), plt.imshow(result, cmap="gray"), plt.axis("off")
#     plt.tight_layout()
#     plt.show()
