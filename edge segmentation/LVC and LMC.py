import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
import matplotlib
matplotlib.use('TkAgg')
class LocalContrastSegmentation:
    def __init__(self, image, kernel_size=15):
        """
        初始化局部对比度分割类
        :param kernel_size: 局部对比度计算的窗口大小
        :param lmc_threshold: 局部均值对比度的分割阈值
        :param lvc_threshold: 局部方差对比度的分割阈值
        """
        self.kernel_size = kernel_size
        global_mean = np.mean(image)  # 全局均值
        global_variance = np.var(image)  # 全局方差
        self.lmc_threshold = 0.5 * global_mean
        self.lvc_threshold = 1.0 * global_variance

    def local_mean_contrast(self, image):
        """
        局部均值对比度（LMC）计算方法
        :param image: 输入图像
        :return: 局部均值对比度图像
        """
        local_mean = cv2.blur(image, (self.kernel_size, self.kernel_size))
        lmc = image - local_mean
        return lmc

    def local_variance_contrast(self, image):
        """
        局部方差对比度（LVC）计算方法
        :param image: 输入图像
        :return: 局部方差对比度图像
        """
        local_mean = cv2.blur(image, (self.kernel_size, self.kernel_size))
        local_mean_sq = cv2.blur(image**2, (self.kernel_size, self.kernel_size))
        local_variance = local_mean_sq - local_mean**2
        global_variance = np.var(image)
        lvc = local_variance / global_variance
        return lvc

    def segment(self, image, method='lmc'):
        """
        使用选择的算法进行图像分割
        :param image: 输入图像
        :param method: 选择分割算法：'lmc'（局部均值对比度） 或 'lvc'（局部方差对比度）
        :return: 分割后的二值图像和目标中心点列表
        """
        if method == 'lmc':
            # 计算局部均值对比度
            contrast_image = self.local_mean_contrast(image)
            # 根据阈值分割
            binary_image = (np.abs(contrast_image) > self.lmc_threshold).astype(np.uint8)
        elif method == 'lvc':
            # 计算局部方差对比度
            contrast_image = self.local_variance_contrast(image)
            # 根据阈值分割
            binary_image = (contrast_image > self.lvc_threshold).astype(np.uint8)
        else:
            raise ValueError("Invalid method. Choose 'lmc' or 'lvc'.")

        # 提取目标区域的连通区域及中心点
        labeled_array, num_features = label(binary_image)
        centers = center_of_mass(binary_image, labeled_array, range(1, num_features + 1))

        return binary_image, centers

    def plot_results(self, image, binary_image, centers, method_name):
        """
        显示分割结果和目标中心点
        :param image: 原图
        :param binary_image: 分割后的二值图像
        :param centers: 目标中心点坐标列表
        :param method_name: 使用的分割算法名称
        """
        plt.figure(figsize=(12, 6))

        # 原图
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")

        # 分割后的二值图像及中心点
        plt.subplot(1, 3, 2)
        plt.imshow(binary_image, cmap='gray')
        plt.title(f"Binary Segmentation ({method_name})")
        #for y, x in centers:
        #    plt.plot(x, y, 'r+', markersize=10)

        plt.tight_layout()
        plt.show()

        # 输出中心点坐标
        print(f"{method_name.upper()} Segmentation Centers:")
        for i, center in enumerate(centers):
            print(f"Object {i + 1} center at: {center}")


def main():
    # 读取红外图像
    image_path = '../output_video_png/frame_0001.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0  # 归一化到0-1之间

    # 初始化局部对比度分割类
    segmenter = LocalContrastSegmentation(image, kernel_size=15)

    # 选择分割算法： 'lmc'（局部均值对比度） 或 'lvc'（局部方差对比度）
    method = 'lvc'  # 替换为 'lvc' 以使用局部方差对比度分割

    # 分割图像
    binary_image, centers = segmenter.segment(image, method=method)

    # 显示结果
    segmenter.plot_results(image, binary_image, centers, method_name=method)


if __name__ == "__main__":
    main()
