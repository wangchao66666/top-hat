import cv2
import numpy as np

# 读取图像并转换为灰度图
image = cv2.imread('../output_frames/first_frame.jpg', cv2.IMREAD_GRAYSCALE)

# 使用高斯滤波减少噪声
blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

# 应用经典Canny边缘检测
edges = cv2.Canny(blurred, 5, 30)

# 获取亚像素级别的边缘信息
# 使用OpenCV的goodFeaturesToTrack函数来精确检测亚像素边缘点
corners = cv2.goodFeaturesToTrack(np.float32(edges), 100, 0.01, 10)

# 提取亚像素位置
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
corners_subpix = cv2.cornerSubPix(np.float32(image), corners, (5,5), (-1,-1), criteria)

# 在图像上绘制亚像素级别的角点
image_with_subpix = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for i in range(corners_subpix.shape[0]):
    cv2.circle(image_with_subpix, (int(corners_subpix[i,0,0]), int(corners_subpix[i,0,1])), 3, (0, 255, 0), -1)

# 显示原始图像、经典Canny边缘检测结果和亚像素级边缘检测结果
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', edges)
cv2.imshow('Subpixel Edge Detection', image_with_subpix)

# 按任意键关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
