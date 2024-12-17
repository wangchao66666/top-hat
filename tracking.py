import cv2
import sys
import os

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # 建立追踪器
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # 读取视频
    video = cv2.VideoCapture("1.mp4")

    # 如果视频没有打开，退出。
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # 设定显示器大小
    display_width = 1080  # 设定宽度
    display_height = 600  # 设定高度

    # 读第一帧。
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # 调整第一帧大小
    frame = cv2.resize(frame, (display_width, display_height))

    # 定义一个初始边界框
    bbox = cv2.selectROI(frame, False)

    # 用第一帧和包围框初始化跟踪器
    ok = tracker.init(frame, bbox)

    print(f"First object at{bbox[0],bbox[1]}")
    while True:
        # 读取一个新的帧
        ok, frame = video.read()
        if not ok:
            break

        # 调整帧大小
        frame = cv2.resize(frame, (display_width, display_height))

        # 启动计时器
        timer = cv2.getTickCount()

        # 更新跟踪器
        ok, bbox = tracker.update(frame)

        # 计算帧率(FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 绘制包围框
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            # 计算中心坐标
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            # 输出跟踪物体的位置信息
            print(f"Tracking object at position: center_x={center_x}, center_y={center_y}")
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failure detected")

        # 在帧上显示跟踪器类型名字
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # 在帧上显示帧率FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # 显示结果
        cv2.imshow("Tracking", frame)

        # 按ESC键退出
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
