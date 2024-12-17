% 1. 读取图像
img = imread('./top_hat_image/output_image1.jpg');  % 加载上传的图像
%gray_img = rgb2gray(img);  % 转换为灰度图像

Draw3DGrayGraph(img, 0);



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



