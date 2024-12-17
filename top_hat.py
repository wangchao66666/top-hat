import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def top_hat_detection(image_path, kernel_size=15, mode="white"):
    """
    Perform Top-Hat transformation for feature detection.

    Args:
        image_path (str): Path to the input image.
        kernel_size (int): Size of the structuring element.
        mode (str): "white" for white top-hat, "black" for black top-hat.

    Returns:
        top_hat_result (ndarray): Result of Top-Hat transformation.
    """
    # Load the input image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found at the provided path!")

    # Define the structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Perform the morphological opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Compute the Top-Hat transformation
    if mode == "white":
        top_hat_result = cv2.subtract(image, opened_image)  # White Top-Hat
    elif mode == "black":
        top_hat_result = cv2.subtract(opened_image, image)  # Black Top-Hat
    else:
        raise ValueError("Invalid mode! Choose 'white' or 'black'.")

    return image, top_hat_result

# Example usage
image_path = "C:\\Users\\localhost\\Desktop\\open-sirst-v2-master\\images\\targets\\Misc_52.png"  # Replace with the path to your image
original, top_hat_result = top_hat_detection(image_path, kernel_size=15, mode="white")

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Top-Hat Result")
plt.imshow(top_hat_result, cmap="gray")
plt.axis("off")

plt.show()

# 转换图像到 0-255 并转换为 uint8 类型
result = cv2.normalize(top_hat_result, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

# 保存结果
output_dir = "top_hat_image"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output_image3.jpg")
cv2.imwrite(output_path, result)
