import cv2
import numpy as np


def local_contrast_measure(image, kernel_size=9):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 定义卷积核
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # 应用卷积核以获得局部均值
    local_mean = cv2.filter2D(gray, -1, kernel)

    # 计算局部对比度
    local_contrast = cv2.absdiff(gray, local_mean)

    # 归一化对比度图像
    cv2.normalize(local_contrast, local_contrast, 0, 255, cv2.NORM_MINMAX)

    return local_contrast


def pcm_algorithm(image, kernel_size=4):
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


def find_and_draw_brightest_points(image, pcm_image, num_points=5):
    # 找到图像中最亮的点
    flat_image = pcm_image.flatten()
    brightest_indices = np.argpartition(flat_image, -num_points)[-num_points:]
    brightest_points = [np.unravel_index(idx, pcm_image.shape) for idx in brightest_indices]

    # 在图像中圈出最亮点
    for point in brightest_points:
        x, y = point[1], point[0]
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)

    return image, brightest_points


# 加载图像
image = cv2.imread('./output_frames/Misc_404.jpg')

# 应用PCM算法
pcm_image = pcm_algorithm(image)

# 找到并圈出最亮的点
result_image, brightest_points = find_and_draw_brightest_points(image, pcm_image, num_points=5)

# 打印最亮点的坐标
print("最亮点的坐标:", brightest_points)

# 保存结果
cv2.imwrite('pcm_result_image.jpg', result_image)

# 显示结果
cv2.imshow('PCM Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
