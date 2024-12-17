% 创建一个接近于0的图像（比如设置为一个小的常数）
img = 0.0001 * ones(100, 100);  % 图像中的值接近于 0

% 归一化图像
normalized_img = mat2gray(img);

% 显示原始图像与归一化后的图像
subplot(1, 2, 1);
imshow(img);
title('原始图像');

subplot(1, 2, 2);
imshow(normalized_img);
title('归一化后的图像');

% 显示归一化后图像的像素值范围
disp(['归一化后图像的最小值: ', num2str(min(normalized_img(:)))]);
disp(['归一化后图像的最大值: ', num2str(max(normalized_img(:)))]);
disp(['归一化后图像的均值: ', num2str(mean(normalized_img(:)))]);
