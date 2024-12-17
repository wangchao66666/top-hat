% 1. ��ȡͼ��
img = imread('./top_hat_image/output_image1.jpg');  % �����ϴ���ͼ��
%gray_img = rgb2gray(img);  % ת��Ϊ�Ҷ�ͼ��

Draw3DGrayGraph(img, 0);



function [out] = Draw3DGrayGraph(img, isShowLine)
    % ������ά�Ҷ�ͼ
    % img: ����ͼ��
    % isShowLine: �Ƿ���ʾ����
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



