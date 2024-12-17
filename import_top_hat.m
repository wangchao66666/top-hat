clc; clear; close all;

% 读取图像
img = imread('./output_frames/Misc_391.jpg');

% 设置结构元素的参数
B_i = strel('rectangle', [2, 1]);   % 小矩形，用于初步去噪
B_b = strel('disk', 2);             % 中等椭圆，用于目标形态恢复
B_0 = strel('rectangle', [4, 1]);  % 大矩形，用于压制背景

% 执行 ESTH 算法
img_ESTH = ESTH(img, B_i, B_b, B_0);

% 显示原始图像和处理结果
figure;
subplot(221);
imshow(img);
title('Original Image');

subplot(223);
Draw3DGrayGraph(img, 0);

subplot(222);
imshow(img_ESTH);
title('ESTH Result');

subplot(224);
Draw3DGrayGraph(img_ESTH, 0);

% ----------------------- 函数部分 -----------------------

function [out] = ESTH(img, B_i, B_b, B_0)
    % ESTH 算法，检测复杂背景下的弱小目标
    % img: 待检测图像
    % B_i, B_b, B_0: 不同的结构元素

    % 如果图像为彩色，转为灰度图像
    if size(img, 3) > 1
        img = rgb2gray(img);
    end

    % 第一步：小矩形结构腐蚀 (去噪)
    img_eroded1 = imdilate(img, B_i);

    % 第二步：中等椭圆结构膨胀 (恢复目标)
    img_dilated = imerode(img_eroded1, B_b);

    

    % 第四步：原图与处理结果相减 (增强目标)
    out = img - img_dilated;

    % 防止负值，将负值设为 0
    out(out < 0) = 0;

    % 归一化处理
    out = mat2gray(out);
end

function [out] = Draw3DGrayGraph(img, isShowLine)
    % 绘制三维灰度图
    % img: 输入图像
    % isShowLine: 是否显示网格
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    [y, x] = size(img);
    [X, Y] = meshgrid(1:x, 1:y);
    surf(X, Y, double(img));
    if ~isShowLine
        shading interp;
    end
    out = 0;
end
