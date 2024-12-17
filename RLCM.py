import cv2
import numpy as np
import time


def calculate_local_contrast(image, window_size):
    height, width = image.shape
    contrast_map = np.zeros((height, width))

    for i in range(window_size // 2, height - window_size // 2):
        for j in range(window_size // 2, width - window_size // 2):
            local_patch = image[i - window_size // 2:i + window_size // 2 + 1,
                          j - window_size // 2:j + window_size // 2 + 1]
            center_value = local_patch[window_size // 2, window_size // 2]
            local_mean = np.mean(local_patch)
            contrast_map[i, j] = center_value - local_mean

    return contrast_map


def multiscale_contrast(image, scales):
    contrast_maps = []
    for scale in scales:
        contrast_map = calculate_local_contrast(image, scale)
        contrast_maps.append(contrast_map)

    final_contrast_map = np.max(contrast_maps, axis=0)
    return final_contrast_map


def threshold_segmentation(contrast_map, threshold):
    _, binary_map = cv2.threshold(contrast_map, threshold, 255, cv2.THRESH_BINARY)
    return binary_map


def process_video(video_path, scales, threshold, window_size=(800, 600)):
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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast_map = multiscale_contrast(gray_frame, scales)
        binary_map = threshold_segmentation(contrast_map, threshold)

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(contrast_map)

        # 转换坐标到设定窗口大小
        original_height, original_width = gray_frame.shape
        x_resized = int(maxLoc[0] * window_size[0] / original_width)
        y_resized = int(maxLoc[1] * window_size[1] / original_height)

        cv2.circle(frame, maxLoc, 5, (255, 0, 0), 2)
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
        #cv2.imshow('Contrast Map', contrast_map)
        #cv2.imshow('Binary Map', binary_map)
        cv2.imshow('Processed Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = '1.mp4'  # 替换为你的视频路径
    scales = [3, 5, 7]  # 定义不同的尺度
    threshold = 10  # 定义阈值

    process_video(video_path, scales, threshold)
