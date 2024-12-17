import cv2
import numpy as np
import matplotlib

matplotlib.use('TKAgg')  # 设置为无界面的后端
import matplotlib.pyplot as plt

# 获取视频路径
video_path = '1.mp4'
cap = cv2.VideoCapture(video_path)

# 设置视频帧大小
target_width = 1080
target_height = 640

# 设置视频开始和结束时间（秒）
start_time = int(input("请输入视频的开始时间（秒）："))
end_time = int(input("请输入视频的结束时间（秒）："))

# 设置视频开始帧和结束帧
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# 跳转到开始帧
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 初始化 KCF 跟踪器
tracker = cv2.TrackerKCF_create()

# 读取第一帧并调整大小，让用户选择要跟踪的对象区域
ret, frame = cap.read()
if not ret:
    print("无法读取视频！")
    cap.release()
    exit()

# 调整帧大小为1080x640
frame = cv2.resize(frame, (target_width, target_height))

# 选择跟踪区域
bbox = cv2.selectROI("选择跟踪区域", frame, False)
tracker.init(frame, bbox)
cv2.destroyWindow("选择跟踪区域")

# 用于存储路径点
path_points = []
box_interval = 50  # 每隔3帧绘制一个方框

# 遍历视频帧并跟踪目标
frame_num = start_frame
while ret and frame_num <= end_frame:
    # 读取并调整每一帧的大小
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (target_width, target_height))

    # 跟踪目标
    success, bbox = tracker.update(frame)
    if success:
        # 获取边界框的中心坐标
        x, y, w, h = map(int, bbox)
        cx = x + w // 2
        cy = y + h // 2

        # 每隔3帧记录一次路径点
        if (frame_num - start_frame) % box_interval == 0:
            path_points.append((x, y, w, h))  # 记录方框位置

    frame_num += 1

cap.release()  # 释放视频

# 在结束帧上绘制每隔3帧的方框
for (x, y, w, h) in path_points:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色方框

# 保存最终带路径的图片
output_path = 'output_with_path_boxes.jpg'
cv2.imwrite(output_path, frame)
print(f"带路径的图片已保存为 {output_path}")

# 显示最终结果
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Tracked Path with Boxes Every 3 Frames on Last Frame")
plt.axis('off')
plt.show()
