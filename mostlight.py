import cv2
import numpy as np
import pandas as pd

def local_contrast_measure(image, kernel_size=9):
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

def apply_border_mask(image, padding=1):
    """
    消除边框像素值的影响，将边缘像素设为 0。
    """
    mask = np.ones_like(image, dtype=np.uint8)
    mask[:padding, :] = 0  # 顶部
    mask[-padding:, :] = 0  # 底部
    mask[:, :padding] = 0  # 左侧
    mask[:, -padding:] = 0  # 右侧
    return image * mask  # 应用掩膜

def apply_inner_mask(image, padding=2):
    """
    将 ROI 内部边缘像素设为 0，以减少对边缘的干扰。
    """
    h, w = image.shape
    mask = np.ones_like(image, dtype=np.uint8)

    # 将四周的 padding 区域设为 0
    mask[:padding, :] = 0  # 顶部边界
    mask[-padding:, :] = 0  # 底部边界
    mask[:, :padding] = 0  # 左侧边界
    mask[:, -padding:] = 0  # 右侧边界

    return image * mask  # 应用掩膜

def calculate_centroid(image):
    """
    根据公式计算质心：
    C_x = (sum(x * I(x, y))) / (sum(I(x, y)))
    C_y = (sum(y * I(x, y))) / (sum(I(x, y)))
    """
    h, w, _ = image.shape  # 获取图像的高度和宽度

    # 生成 x, y 坐标矩阵
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # 分别计算每个通道的权重矩阵
    r_channel = image[:, :, 0].astype(np.float32)
    g_channel = image[:, :, 1].astype(np.float32)
    b_channel = image[:, :, 2].astype(np.float32)

    # 计算总的颜色权重矩阵：W(x, y) = R + G + B
    weight_matrix = r_channel + g_channel + b_channel

    # 计算权重的总和，避免除以 0
    total_weight = np.sum(weight_matrix)
    print(total_weight)
    if total_weight == 0:
        return None, None  # 如果权重为 0，返回 None

    # 计算质心的 x 和 y 坐标
    centroid_x = np.sum(x_coords * weight_matrix) / total_weight
    centroid_y = np.sum(y_coords * weight_matrix) / total_weight

    return centroid_x, centroid_y

def process_video(video_path, roi_top_left, roi_bottom_right, output_excel='brightest_points.xlsx'):
    """
    处理视频，计算局部对比度、最亮点和质心并保存结果。

    参数:
        video_path: 视频文件路径
        roi_top_left: ROI 区域的左上角 (x1, y1)
        roi_bottom_right: ROI 区域的右下角 (x2, y2)
        output_excel: 输出的 Excel 文件名称
    """
    x1, y1 = roi_top_left
    x2, y2 = roi_bottom_right

    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 创建 DataFrame 来保存每一帧的结果
    data = {'Frame': [], 'Brightest_X': [], 'Brightest_Y': [], 'Centroid_X': [], 'Centroid_Y': []}
    frame_count = 0  # 帧计数器

    while True:
        ret, frame = video.read()
        if not ret:
            break  # 如果读取失败（视频结束），退出循环

        frame_count += 1  # 更新帧计数

        # 在原始帧上绘制感兴趣区域（绿色矩形框）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 提取 ROI 区域
        roi = frame[y1:y2, x1:x2]

        # 使用 LCM 算法计算局部对比度
        local_contrast = local_contrast_measure(roi)

        # 应用内部掩膜，将 ROI 边界区域设为 0
        local_contrast = apply_inner_mask(local_contrast, padding=2)

        # 在对比度图像中找到最亮点
        _, _, _, max_loc = cv2.minMaxLoc(local_contrast)
        brightest_x_local, brightest_y_local = max_loc

        # 将局部坐标转换为全局坐标
        brightest_x = brightest_x_local + x1
        brightest_y = brightest_y_local + y1

        # 定义矩形框的上下左右边界（确保不越界）
        top = max(0, brightest_y - 15)
        bottom = min(frame.shape[0], brightest_y + 15)
        left = max(0, brightest_x - 15)
        right = min(frame.shape[1], brightest_x + 15)

        # 提取矩形框内的区域
        roi_rect= frame[top:bottom, left:right]

        # 应用边框掩膜，消除边缘影响
        roi_rect = apply_border_mask(roi_rect, padding=2)

        # 计算质心
        centroid_x_local, centroid_y_local = calculate_centroid(roi_rect)

        if centroid_x_local is not None and centroid_y_local is not None:
            centroid_x = centroid_x_local + left
            centroid_y = centroid_y_local + top
        else:
            centroid_x, centroid_y = brightest_x, brightest_y

        # 在视频帧上绘制最亮点和矩形框
        cv2.circle(frame, (brightest_x, brightest_y), 5, (0, 0, 255), -1)  # 红色圆点标记最亮点
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # 绿色矩形框
        cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (255, 255, 0), -1)  # 黄色圆点标记质心

        # 显示原始帧和 ROI 区域
        cv2.imshow('Original Frame with ROI', frame)

        # 将数据存储在 DataFrame 中
        data['Frame'].append(frame_count)
        data['Brightest_X'].append(brightest_x)
        data['Brightest_Y'].append(brightest_y)
        data['Centroid_X'].append(round(centroid_x, 3))
        data['Centroid_Y'].append(round(centroid_y, 3))

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 保存结果到 Excel 文件
    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False)

    # 释放资源并关闭窗口
    video.release()
    cv2.destroyAllWindows()

# 调用示例
process_video("D:\\output1.avi", (105, 190), (665, 605), 'brightest_points1.xlsx')
