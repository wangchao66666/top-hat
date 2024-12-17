import cv2
import numpy as np

def initialize_background_subtractor():
    """使用自适应高斯混合模型（GMM）初始化背景建模器。"""
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True)

def apply_morphological_operations(mask):
    """对掩模进行形态学操作，去除噪声并增强目标。"""
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 去除小噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填补小孔洞
    return mask

def detect_and_track_targets(frame, mask, min_area=8):
    """检测并跟踪运动目标，标注其位置和中心。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            # 绘制目标矩形框和中心点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(frame, center, 2, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: {center}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

def process_video(video_path):
    """处理视频，检测并跟踪小目标。"""
    cap = cv2.VideoCapture(video_path)
    background_subtractor = initialize_background_subtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图像并应用背景减除
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = background_subtractor.apply(gray_frame)

        # 应用形态学操作增强掩模效果
        cleaned_mask = apply_morphological_operations(fg_mask)

        # 检测并跟踪运动目标
        detect_and_track_targets(frame, cleaned_mask)

        # 调整窗口大小并显示结果
        resized_frame = cv2.resize(frame, (800, 600))
        resized_mask = cv2.resize(cleaned_mask, (800, 600))

        cv2.imshow('Target Detection', resized_frame)
        cv2.imshow('Foreground Mask', resized_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 调用检测函数处理视频
video_path = "1.mp4"
process_video(video_path)
