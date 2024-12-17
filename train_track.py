import cv2
import numpy as np
from scipy.spatial import distance
import time

# 聚类近似坐标并取平均值
def cluster_and_average_coordinates(coords, threshold=5):
    if len(coords) == 0:
        return [], 0

    coords = np.array(coords)
    clusters = []

    for coord in coords:
        coord = np.array(coord)
        distances = np.array([distance.euclidean(cluster['center'], coord) for cluster in clusters])
        matched_indices = np.where(distances <= threshold)[0]

        if matched_indices.size > 0:
            first_match_index = matched_indices[0]
            clusters[first_match_index]['coords'].append(coord)
            clusters[first_match_index]['center'] = np.mean(clusters[first_match_index]['coords'], axis=0)
        else:
            clusters.append({'coords': [coord], 'center': coord})

    max_cluster = max(clusters, key=lambda c: len(c['coords']))
    return max_cluster['center'], len(max_cluster['coords'])

# 创建背景减除器函数，支持多种方法
def create_background_subtractor(method='NONE'):
    if method == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    elif method == 'GMM':
        return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    elif method == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=800, detectShadows=False)
    elif method == 'VIBE':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200)  # VIBE近似实现
    elif method == 'SUBSENSE':
        return cv2.bgsegm.createBackgroundSubtractorGSOC()  # SuBSENSE实现
    elif method == 'NONE':
        return None  # 不使用背景减除器
    else:
        raise ValueError(f"未知的背景减除方法: {method}")

# **你的原始process_frame_and_detect_motion函数**
def process_frame_and_detect_motion(frame, background, width, height):
    if frame is None or frame.size == 0:
        return None, None, []

    frame = cv2.resize(frame, (width, height))
    diff = cv2.absdiff(background, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    processed_frame = frame.copy()
    processed_frame[mask == 0] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_coordinates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        detected_coordinates.append(center)

    return frame, processed_frame, detected_coordinates

# **使用背景减除器的函数**
def process_frame_and_detect_motion_with_subtractor(frame, subtractor, width, height):
    if frame is None or frame.size == 0:
        return None, None, []

    frame = cv2.resize(frame, (width, height))
    fg_mask = subtractor.apply(frame)

    _, mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

    processed_frame = frame.copy()
    processed_frame[mask == 0] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_coordinates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        detected_coordinates.append(center)

    return frame, processed_frame, detected_coordinates

# **检测运动目标并获取坐标**
def detect_motion_and_get_coordinates(cap, max_frames=10, width=1080, height=640, threshold=5, method='NONE'):
    ret, background = cap.read()
    if not ret:
        print("无法读取视频或背景帧")
        return None, -1

    background = cv2.resize(background, (width, height))

    subtractor = create_background_subtractor(method)

    all_coordinates = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("无法读取视频帧，退出循环")
            return None, -1

        if subtractor is not None:
            _, processed_frame, coordinates = process_frame_and_detect_motion_with_subtractor(
                frame, subtractor, width, height
            )
        else:
            _, processed_frame, coordinates = process_frame_and_detect_motion(
                frame, background, width, height
            )

        if processed_frame is None:
            continue

        all_coordinates.extend(coordinates)
        frame_count += 1

    if len(all_coordinates) > 0:
        final_coordinate, count = cluster_and_average_coordinates(all_coordinates, threshold=threshold)
        print(f"最终坐标: {final_coordinate}, 出现次数: {count}")
        return final_coordinate, cap.get(cv2.CAP_PROP_POS_FRAMES)
    else:
        print("未检测到任何坐标")
        return None, -1

# **跟踪器创建函数**
def create_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        return cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        return cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    else:
        print(f"Unknown tracker type: {tracker_type}")
        return None

# 跟踪函数：使用检测到的初始坐标作为跟踪器的初始位置，并调整视频尺寸
def track_object(cap, initial_coordinate, start_frame, threshold, tracker_type='KCF', width=1080, height=640):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    initial_x, initial_y = int(initial_coordinate[0] - threshold/2), int(initial_coordinate[1] - threshold/2)
    bbox = (initial_x, initial_y, threshold, threshold)

    tracker = create_tracker(tracker_type)
    if tracker is None:
        print("无法创建跟踪器")
        return

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return

    frame = cv2.resize(frame, (width, height))

    tracker.init(frame, bbox)

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_height, original_width = frame.shape[:2]
        frame = cv2.resize(frame, (width, height))

        ret, bbox = tracker.update(frame)
        if ret:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            x_resized = int(x * original_width / width)
            y_resized = int(y * original_height / height)
            cv2.putText(frame, f"Coordinates: ({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"original Coordinates: ({x_resized}, {y_resized})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps_display = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "Tracking failure", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# **主函数**
def main(video_source, max_frames=10, width=1080, height=640, threshold=30, tracker_type='KCF', method='NONE'):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("无法打开视频源")
        return

    final_coordinate, start_frame = detect_motion_and_get_coordinates(
        cap, max_frames=max_frames, width=width, height=height, threshold=5, method=method
    )

    if final_coordinate is not None:
        track_object(cap, final_coordinate, start_frame, threshold=threshold, tracker_type=tracker_type, width=width, height=height)

# **调用主函数**
main('../untitled/output/GT2000 NIR-16-27-06_1X1.avi', max_frames=10, width=1080, height=640, threshold=30, tracker_type='KCF', method='GMM')
