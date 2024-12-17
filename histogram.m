% 1. 读取图像并转换为灰度
img = imread('./top_hat_image/output_image.jpg');  % 加载上传的图像
%gray_img = rgb2gray(img);  % 转换为灰度图像

% 2. 计算灰度直方图
hist_counts = imhist(img);  % 计算原始图像的灰度直方图

% 3. 计算显著性直方图
gray_levels = 0:255;  % 灰度值范围
saliency_values = zeros(size(gray_levels));  % 初始化显著性值数组

% 对每个灰度值计算其显著性值
for p = 1:256
    Ip = gray_levels(p);  % 当前灰度值
    % 计算当前灰度值与所有灰度值的对比度加权和
    contrast_sum = 0;
    for q = 1:256
        Iq = gray_levels(q);  % 另一个灰度值
        contrast = abs(Ip - Iq);  % 计算对比度（欧式距离）
        contrast_sum = contrast_sum + hist_counts(q) * contrast;  % 加权和
    end
    saliency_values(p) = contrast_sum;  % 存储显著性值
end

% 4. 归一化显著性直方图
saliency_values = saliency_values / max(saliency_values) * max(hist_counts);

% 5. 绘制原始图像直方图和显著性直方图
figure;
subplot(1, 2, 1);
bar(gray_levels, hist_counts, 'FaceColor', [0, 0, 0.5]);
title('(a) 原始图像直方图');
xlabel('灰度值');
ylabel('像素数');

subplot(1,2, 2);
bar(gray_levels, saliency_values, 'FaceColor', [0, 0, 0.5]);
title('(b) 显著性直方图');
xlabel('灰度值');
ylabel('显著性值');

% 6. 生成视觉显著性图
saliency_map = zeros(size(gray_img));  % 初始化显著性图
for x = 1:size(gray_img, 1)
    for y = 1:size(gray_img, 2)
        pixel_value = gray_img(x, y);  % 获取像素灰度值
        saliency_map(x, y) = saliency_values(pixel_value + 1);  % 将显著性值映射到显著性图
    end
end

% 7. 归一化显著性图以增强对比度
saliency_map = mat2gray(saliency_map);  % 将显著性图归一化到 [0, 1]

% 8. 显示视觉显著性图
figure;
imshow(saliency_map);
title('(c) 视觉显著性图');


