import numpy as np
import cv2
import cvxpy as cp
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt

# 读取红外图像
image_path = './output_frame/frame_0001.jpg'  # 替换为红外图像的路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32) / 255.0  # 归一化到0-1之间


# 定义 RPCA 函数
def rpca_decomposition(X, lambda_=None, mu=None, tol=1e-7, max_iter=100):
    # 初始化
    if lambda_ is None:
        lambda_ = 1 / np.sqrt(max(X.shape))
    if mu is None:
        mu = 0.5 * np.prod(X.shape) / np.sum(np.abs(X))

    # 定义优化变量
    L = cp.Variable(X.shape)  # 低秩矩阵
    S = cp.Variable(X.shape)  # 稀疏矩阵
    # 定义目标函数：最小化核范数 + 稀疏矩阵的L1范数
    objective = cp.Minimize(cp.norm(L, "nuc") + lambda_ * cp.norm1(S))
    constraints = [X == L + S]  # 约束条件
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=True, max_iters=max_iter, eps=tol)  # 求解

    return L.value, S.value


# 执行 RPCA 分解
L, S = rpca_decomposition(image)

# 取出稀疏矩阵 S 中的目标区域（去除微小噪声）
threshold = 0.1  # 阈值（可以根据实际情况调整）
binary_sparse_image = np.abs(S) > threshold

# 提取目标区域的连通区域信息
labeled_array, num_features = label(binary_sparse_image)
center_points = center_of_mass(binary_sparse_image, labeled_array, range(1, num_features + 1))

# 显示分割后的结果和物体中心点
plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

# 低秩背景
plt.subplot(1, 3, 2)
plt.imshow(L, cmap='gray')
plt.title("Low-Rank Background")

# 稀疏目标 + 中心点标注
plt.subplot(1, 3, 3)
plt.imshow(binary_sparse_image, cmap='gray')
plt.title("Sparse Targets and Centers")

# 标注中心点
for y, x in center_points:
    plt.plot(x, y, 'r+', markersize=10)

plt.tight_layout()
plt.show()

# 输出中心点坐标
for i, center in enumerate(center_points):
    print(f"Object {i + 1} center at: {center}")
