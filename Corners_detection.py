import cv2
import numpy as np


def detect_corners(image, method):
    if method == 'harris':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        image[dst > 0.01 * dst.max()] = [0, 0, 255]  # 标注角点
        corners = np.argwhere(dst > 0.01 * dst.max())

    elif method == 'shi_tomasi':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (int(x), int(y)), 3, 255, -1)  # 确保 x 和 y 是整数
        return image, corners

    elif method == 'sift':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(gray, None)
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        corners = np.array([kp.pt for kp in keypoints])

    elif method == 'surf':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, _ = surf.detectAndCompute(gray, None)
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        corners = np.array([kp.pt for kp in keypoints])

    elif method == 'fast':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
        image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))
        corners = np.array([kp.pt for kp in keypoints])

    elif method == 'orb':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, _ = orb.detectAndCompute(gray, None)
        image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
        corners = np.array([kp.pt for kp in keypoints])

    elif method == 'brief':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints = orb.detect(gray, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        keypoints, descriptors = brief.compute(gray, keypoints)
        image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 255))
        corners = np.array([kp.pt for kp in keypoints])

    else:
        raise ValueError("无效的角点检测方法！")

    return image, corners


def main(image_path, method):
    image = cv2.imread(image_path)

    result_image, corners = detect_corners(image.copy(), method)
    title = f'{method.capitalize()} Corners'

    cv2.imshow(title, result_image)
    print(f"{title} corners:", corners)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有窗口


if __name__ == "__main__":
    main("./output_video_png2/frame_0050.jpg", 'orb')
