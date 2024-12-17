import cv2
import numpy as np
import random


class ViBe:
    def __init__(self, num_samples=20, min_matches=2, radius=20, subsampling_factor=16):
        self.num_samples = num_samples
        self.min_matches = min_matches
        self.radius = radius
        self.subsampling_factor = subsampling_factor

    def initialize(self, frame):
        # 初始化样本库
        height, width = frame.shape[:2]
        self.samples = np.zeros((self.num_samples, height, width), dtype=np.uint8)

        # 生成初始样本库
        for i in range(self.num_samples):
            self.samples[i] = self._get_neighbor_sample(frame)

    def _get_neighbor_sample(self, frame):
        # 随机采样邻域像素
        height, width = frame.shape[:2]
        neighbor_sample = np.zeros_like(frame)

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i in range(height):
            for j in range(width):
                offset = random.choice(offsets)
                x = min(width - 1, max(0, j + offset[1]))
                y = min(height - 1, max(0, i + offset[0]))
                neighbor_sample[i, j] = frame[y, x]

        return neighbor_sample

    def process_frame(self, frame):
        # 前景检测
        height, width = frame.shape[:2]

        # 对每个像素，与样本库比较计算绝对差值
        diff = np.abs(self.samples - frame)
        matches = (diff < self.radius).sum(axis=0)  # 匹配样本的数量

        # 前景掩膜：匹配数量小于阈值的为前景
        foreground_mask = (matches < self.min_matches).astype(np.uint8) * 255

        # 更新背景模型
        update_mask = (foreground_mask == 0) & (
                    np.random.randint(0, self.subsampling_factor, size=(height, width)) == 0)
        sample_indices = np.random.randint(0, self.num_samples, size=(height, width))
        self.samples[sample_indices, np.arange(height)[:, None], np.arange(width)] = frame[update_mask]

        # 邻域更新
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbors = random.choice(offsets)
        neighbor_y = np.clip(np.arange(height)[:, None] + neighbors[0], 0, height - 1)
        neighbor_x = np.clip(np.arange(width) + neighbors[1], 0, width - 1)
        self.samples[sample_indices, neighbor_y, neighbor_x] = frame[update_mask]

        return foreground_mask


# 使用示例
cap = cv2.VideoCapture('1.mp4')
ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 初始化 ViBe 模型
vibe = ViBe()
vibe.initialize(gray_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    foreground_mask = vibe.process_frame(gray_frame)

    # 显示前景检测结果
    cv2.imshow("Foreground Mask", foreground_mask)
    cv2.imshow("Original Frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
