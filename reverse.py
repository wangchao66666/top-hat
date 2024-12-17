import cv2
import numpy as np
import matplotlib.pyplot as plt

# 对数变换
def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output

# 读取原始图像
img = cv2.imread('./output_video_png2/frame_0050.jpg')

# 图像灰度转换
grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 获取图像高度和宽度
height = grayimage.shape[0]
width = grayimage.shape[1]

# 创建一幅图像
result = np.zeros((height, width), np.uint8)

# 创建一幅图像
result2 = np.zeros((height, width), np.uint8)

# 图像对比度增强变换  DB = DA * 1.5
for i in range(height):
    for j in range(width):
        if (int(grayimage[i, j] * 2) > 255):
            gray = 255
        else:
            gray = int(grayimage[i, j] * 2)
        result2[i, j] = np.uint8(gray)

# 图像灰度反色变换  DB = 255 - DA
for i in range(height):
    for j in range(width):
        gray = 255 - grayimage[i, j]
        result[i, j] = np.uint8(gray)


result3 = log(42, img)
# 显示图像 
cv2.imshow("Gray Image", grayimage)
cv2.imshow("Result1", result)
cv2.imshow("Result2", result2)
cv2.imshow("Result3", result3)


# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()