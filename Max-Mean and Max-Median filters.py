import cv2
import numpy as np


def fast_max_median_filter(image, window_size):
    # 使用 OpenCV 中的中值滤波，模拟 Max-Median 处理
    vertical = cv2.medianBlur(image, window_size)
    horizontal = cv2.medianBlur(image.T, window_size).T
    diag1 = cv2.medianBlur(np.fliplr(image), window_size)
    diag2 = cv2.medianBlur(np.flipud(image), window_size)

    # 选择最大中值
    filtered_image = np.maximum.reduce([vertical, horizontal, diag1, diag2])
    return filtered_image


def fast_max_mean_filter(image, window_size):
    # 使用 OpenCV 的均值滤波，模拟 Max-Mean 处理
    vertical = cv2.blur(image, (1, window_size))
    horizontal = cv2.blur(image, (window_size, 1))
    diag1 = cv2.blur(image, (window_size, window_size))
    diag2 = cv2.blur(np.fliplr(image), (window_size, window_size))

    # 选择最大均值
    filtered_image = np.maximum.reduce([vertical, horizontal, diag1, diag2])
    return filtered_image


def detect_targets(video_path, filter_type='max_median', window_size=5, k=2.0):
    cap = cv2.VideoCapture(video_path)
    background_accum = None  # Accumulate background information over frames
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply chosen filter
        if filter_type == 'max_median':
            filtered_frame = fast_max_median_filter(gray_frame, window_size)
        elif filter_type == 'max_mean':
            filtered_frame = fast_max_mean_filter(gray_frame, window_size)
        else:
            raise ValueError("Invalid filter type. Use 'max_median' or 'max_mean'")

        # Subtract background clutter
        subtracted_frame = cv2.absdiff(gray_frame, filtered_frame)

        # Calculate threshold dynamically based on local mean and standard deviation
        mean, stddev = cv2.meanStdDev(subtracted_frame)
        threshold_value = float(mean[0][0] + k * stddev[0][0])
        _, binary_frame = cv2.threshold(subtracted_frame, threshold_value, 255, cv2.THRESH_BINARY)

        # Accumulate target detections over frames to identify moving targets
        if background_accum is None:
            background_accum = np.zeros_like(binary_frame, dtype=np.uint8)

        background_accum = cv2.add(background_accum, binary_frame)

        # Detect contours and draw bounding boxes around detected targets
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Filter small noise contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result every few frames to reduce I/O overhead
        if frame_count % 2 == 0:
            cv2.imshow('Target Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# 使用示例
detect_targets('1.mp4', filter_type='max_median', window_size=5, k=2.0)
