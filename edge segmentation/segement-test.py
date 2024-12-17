import cv2
import numpy as np


def segment_and_display_objects(image_path, max_width, max_height, min_width, min_height):
    """
    对图像进行分割，并在原图像上显示符合大小要求的目标物体及其中心坐标。

    参数:
    image_path: 输入图像的路径。
    max_width: 目标物体允许的最大宽度。
    max_height: 目标物体允许的最大高度。

    返回:
    center_points: 每个分割物体的中心坐标列表。
    """

    # 读取图像
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 去噪 (例如使用高斯滤波)
    denoised = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 增强对比度 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(denoised)

    # 边缘检测 (Canny)
    edges = cv2.Canny(enhanced_img, threshold1=50, threshold2=150)

    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    morph_img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 连通区域分析
    num_labels, labels_im = cv2.connectedComponents(morph_img)

    # 存储中心坐标的列表
    center_points = []

    # 遍历所有连通区域，筛选符合要求的区域
    for label in range(1, num_labels):  # 忽略背景标签 0
        mask = np.array(labels_im == label, dtype="uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # 获取每个连通区域的边界框
            if w <= max_width and h <= max_height and w >= min_width and h >= min_height:  # 根据定义的宽度和高度筛选目标
                # 在原图像上画出矩形框
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 计算中心坐标
                center_x = x + w // 2
                center_y = y + h // 2
                center_points.append((center_x, center_y))

                # 在图像上标记中心点
                #cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)  # 蓝色中心点

                # 在中心点上显示坐标
                cv2.putText(img, f"({center_x}, {center_y})", (center_x - 20, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示结果
    cv2.imshow('Segmented Image with Bounding Boxes and Centers', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return center_points  # 返回中心坐标列表


# 示例调用
image_path = '../output_video_png2/frame_0000.jpg'  # 输入图像路径
max_width = 20  # 可调整的最大宽度
max_height = 20  # 可调整的最大高度
min_width= 10
min_height= 10
# 调用函数并获取中心坐标
centers = segment_and_display_objects(image_path, max_width, max_height, min_width, min_height)

# 输出中心坐标
print("物体的中心坐标:", centers)
