clc; clear; close all;

% ��ȡͼ��
img = imread('./output_frames/Misc_391.jpg');

% ���ýṹԪ�صĲ���
B_i = strel('rectangle', [2, 1]);   % С���Σ����ڳ���ȥ��
B_b = strel('disk', 2);             % �е���Բ������Ŀ����̬�ָ�
B_0 = strel('rectangle', [4, 1]);  % ����Σ�����ѹ�Ʊ���

% ִ�� ESTH �㷨
img_ESTH = ESTH(img, B_i, B_b, B_0);

% ��ʾԭʼͼ��ʹ�����
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

% ----------------------- �������� -----------------------

function [out] = ESTH(img, B_i, B_b, B_0)
    % ESTH �㷨����⸴�ӱ����µ���СĿ��
    % img: �����ͼ��
    % B_i, B_b, B_0: ��ͬ�ĽṹԪ��

    % ���ͼ��Ϊ��ɫ��תΪ�Ҷ�ͼ��
    if size(img, 3) > 1
        img = rgb2gray(img);
    end

    % ��һ����С���νṹ��ʴ (ȥ��)
    img_eroded1 = imdilate(img, B_i);

    % �ڶ������е���Բ�ṹ���� (�ָ�Ŀ��)
    img_dilated = imerode(img_eroded1, B_b);

    

    % ���Ĳ���ԭͼ�봦������� (��ǿĿ��)
    out = img - img_dilated;

    % ��ֹ��ֵ������ֵ��Ϊ 0
    out(out < 0) = 0;

    % ��һ������
    out = mat2gray(out);
end

function [out] = Draw3DGrayGraph(img, isShowLine)
    % ������ά�Ҷ�ͼ
    % img: ����ͼ��
    % isShowLine: �Ƿ���ʾ����
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
