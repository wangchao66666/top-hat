import cv2
import numpy as np
import time
from scipy.signal import convolve2d


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


def pcm_algorithm(image, kernel_size=9):
    # 计算局部对比度
    lcm_image = local_contrast_measure(image, kernel_size)

    # 定义三层窗口
    inner_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    outer_kernel = np.ones((kernel_size * 3, kernel_size * 3), np.float32) / (kernel_size * 3 * kernel_size * 3)

    # 计算内层和外层的局部均值
    inner_mean = cv2.filter2D(lcm_image, -1, inner_kernel)
    outer_mean = cv2.filter2D(lcm_image, -1, outer_kernel)

    # 计算对比度
    contrast = cv2.absdiff(inner_mean, outer_mean)

    # 归一化对比度图像
    cv2.normalize(contrast, contrast, 0, 255, cv2.NORM_MINMAX)

    return contrast

def ilcm(image, kernel_size=9):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 定义卷积核
    #kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # 应用卷积核以获得局部均值
    #local_mean = cv2.filter2D(gray, -1, kernel)
    # 定义分离卷积核
    kernel_1d = np.ones((kernel_size,), np.float32) / kernel_size

    # 应用分离卷积核以获得局部均值
    local_mean = cv2.filter2D(gray, -1, kernel_1d[:, None])
    local_mean = cv2.filter2D(local_mean, -1, kernel_1d[None, :])
    # 计算ILCM值
    ilcm_image = np.divide(gray, local_mean, out=np.zeros_like(gray, dtype=np.float32), where=local_mean != 0)

    # 归一化ILCM图像
    cv2.normalize(ilcm_image, ilcm_image, 0, 255, cv2.NORM_MINMAX)
    ilcm_image = np.uint8(ilcm_image)

    return ilcm_image

def gradient_contrast_measure(image, kernel_size=9):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # 计算局部均值
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(gradient_magnitude, -1, kernel)

    # 计算改进的对比度度量
    contrast_measure = np.divide(gradient_magnitude, local_mean, out=np.zeros_like(gradient_magnitude), where=local_mean!=0)

    # 归一化对比度图像
    cv2.normalize(contrast_measure, contrast_measure, 0, 255, cv2.NORM_MINMAX)
    contrast_measure = np.uint8(contrast_measure)

    return contrast_measure

def combined_contrast_ilcm(image, kernel_size=9):
    # 计算改进的对比度度量
    contrast_image = gradient_contrast_measure(image, kernel_size)

    # 计算ILCM值
    ilcm_image = ilcm(image, kernel_size)

    # 结合改进的对比度度量和ILCM值
    combined_image = cv2.addWeighted(contrast_image, 0.2, ilcm_image, 0.8, 0)

    return combined_image


def ilcm_multiscale(image, kernel_sizes=[3, 5, 7, 9]):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ilcm_images = []

    for kernel_size in kernel_sizes:
        # 定义卷积核
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

        # 应用FFT卷积以获得局部均值
        local_mean = cv2.filter2D(gray, -1, kernel)

        # 计算ILCM值
        ilcm_image = np.divide(gray, local_mean, out=np.zeros_like(gray, dtype=np.float32), where=local_mean != 0)
        ilcm_images.append(ilcm_image)

    # 将多尺度结果融合
    ilcm_multiscale_image = np.mean(ilcm_images, axis=0)

    # 归一化ILCM图像
    cv2.normalize(ilcm_multiscale_image, ilcm_multiscale_image, 0, 255, cv2.NORM_MINMAX)
    ilcm_multiscale_image = np.uint8(ilcm_multiscale_image)

    return ilcm_multiscale_image


def gau_ilcm(image, kernel_size=9, sigma=2):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 定义高斯卷积核
    gauss_kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)

    # 应用分离高斯卷积核以获得局部均值
    local_mean = cv2.filter2D(gray, -1, gauss_kernel_1d)
    local_mean = cv2.filter2D(local_mean, -1, gauss_kernel_1d.T)

    # 计算ILCM值
    ilcm_image = np.divide(gray, local_mean, out=np.zeros_like(gray, dtype=np.float32), where=local_mean != 0)

    # 归一化ILCM图像
    cv2.normalize(ilcm_image, ilcm_image, 0, 255, cv2.NORM_MINMAX)
    ilcm_image = np.uint8(ilcm_image)

    return ilcm_image


def find_brightest_point(image):
    # 找到图像中最亮的点
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    return max_loc


def process_video(video_path, algorithm='LCM', window_size=(800, 600)):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化计时器和帧计数器
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 选择算法
        if algorithm == 'LCM':
            processed_frame = local_contrast_measure(frame)
        elif algorithm == 'PCM':
            processed_frame = pcm_algorithm(frame)
        elif algorithm=='ILCM':
            processed_frame=ilcm(frame)
        elif algorithm=='f_ILCM':
            processed_frame=ilcm_multiscale(frame)
        elif algorithm=='C_ILCM':
            processed_frame=combined_contrast_ilcm(frame)
        elif algorithm=='G_ILCM':
            processed_frame=gau_ilcm(frame)
        else:
            raise ValueError("Unsupported algorithm. Use 'LCM' or 'PCM'.")

        # 找到最亮的点
        brightest_point = find_brightest_point(processed_frame)


        # 在原始帧上圈出最亮点
        x, y = brightest_point
        cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
        #print(f"the position of an object:{x},{y} ")
        cv2.putText(frame, f"original Coordinates: ({x}, {y})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)


        # 显示在当前分辨率下的图像位置
        original_height, original_width = frame.shape[:2]
        x_resized = int(x * window_size[0] / original_width)
        y_resized = int(y * window_size[1] / original_height)
        cv2.putText(frame, f"Coordinates: ({x_resized}, {y_resized})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)


        # 计算并显示帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps_display = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 调整窗口大小
        cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Video", window_size[0], window_size[1])

        # 实时显示处理后的帧
        cv2.imshow('Processed Video', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


# 使用示例
video_path = '1.mp4'
process_video(video_path, algorithm='G_ILCM', window_size=(1080,640))  # 使用LCM算法
#process_video(video_path, algorithm='PCM', window_size=(1080, 600))  # 使用PCM算法
