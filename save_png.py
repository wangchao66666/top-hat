import cv2
import os


def extract_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    frame_count = 0

    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果读取失败，结束循环
        if not ret:
            break

        # 设置帧文件名
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # 保存帧
        cv2.imwrite(frame_filename, frame)

        print(f"Saved {frame_filename}")

        frame_count += 1

    # 释放视频捕捉对象
    cap.release()
    print("Done extracting frames.")


# 示例用法
video_path = '../untitled/output/GT2000 NIR-16-27-06_1X1.avi'
output_folder = 'output_video_png2'
extract_frames(video_path, output_folder)
