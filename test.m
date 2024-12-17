clc; clear; close all;
addpath('../');
img = imread('C:\\Users\\localhost\\Desktop\\open-sirst-v2-master\\images\\targets\\Misc_52.png');
img1 = imread('C:\\Users\\localhost\\Desktop\\deepsort\\test\\\top_hat_image\\output_image1.jpg');
img2 = imread('C:\\Users\\localhost\\Desktop\\deepsort\\test\\\top_hat_image\\output_image3.jpg');
R_o = 9;
R_i = 4;
delta_B = newRingStrel(R_o, R_i);
B_b = ones(R_i);
img_MNWTH = MNWTH(img, delta_B, B_b);

% 调整子图布局为 2 行 3 列
subplot(2, 4, 1);
imshow(img); title('Original Image');

subplot(2, 4, 5);
Draw3DGrayGraph(img, 0);

subplot(2, 4, 2);
imshow(img2); title('TH Result');

subplot(2, 4, 6);
Draw3DGrayGraph(img2, 0);

subplot(2, 4, 3);
imshow(img_MNWTH); title('MNWTH Result');

subplot(2, 4, 7);
Draw3DGrayGraph(img_MNWTH, 0);

subplot(2, 4, 4);
imshow(img1); title('Multi-scale Result');

subplot(2, 4, 8);
Draw3DGrayGraph(img1, 0);


function [SE] = newRingStrel(R_o, R_i)
    % 构造矩形环状结构元素
    % R_o : the radius of out
    % R_i : the radius of inner
    % delta_R = R_o - R_i
    d = 2 * R_o + 1;
    SE = ones(d);
    start_index = R_o + 1 - R_i;
    end_index = R_o + 1 + R_i;
    SE(start_index:end_index, start_index:end_index) = 0;
    
end

function [out] = MNWTH(img, delta_B, B_b)
    % MNWTH 算法，检测亮目标
    % img: 待检测图像
    % delta_B, B_b: 结构元素
    
    if (size(img, 3) > 1)

        img = rgb2gray(img);
    end
    
    % 先膨胀
    
    img_d = imdilate(img, delta_B);
  
    % 后腐蚀
    img_e = imerode(img_d, B_b);
   
    % 图像相减
    out = img-img_e;
    

    out = mat2gray(out);
    
end

function [out] = Draw3DGrayGraph(img, isShowLine)
    % 绘制三维灰度图
    % img: 输入图像
    % isShowLine: 是否显示网格
    if (size(img, 3) > 1)
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

function [out] = ESTH(img, B_i, B_b, B_0)
    % ESTH 算法，检测复杂背景下的弱小目标
    % img: 待检测图像
    % B_i, B_b, B_0: 不同的结构元素

    % 如果图像为彩色，转为灰度图像
    if size(img, 3) > 1
        img = rgb2gray(img);
    end

    % 第一步：小矩形结构腐蚀 (去噪)
    img_eroded1 = imerode(img, B_i);

    % 第二步：中等椭圆结构膨胀 (恢复目标)
    img_dilated = imdilate(img_eroded1, B_b);

    % 第三步：大矩形结构腐蚀 (压制背景)
    img_eroded2 = imerode(img_dilated, B_0);

    % 第四步：原图与处理结果相减 (增强目标)
    out = img - img_eroded2;

    % 防止负值，将负值设为 0
    out(out < 0) = 0;

    % 归一化处理
    out = mat2gray(out);
end

