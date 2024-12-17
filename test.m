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

% ������ͼ����Ϊ 2 �� 3 ��
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
    % ������λ�״�ṹԪ��
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
    % MNWTH �㷨�������Ŀ��
    % img: �����ͼ��
    % delta_B, B_b: �ṹԪ��
    
    if (size(img, 3) > 1)

        img = rgb2gray(img);
    end
    
    % ������
    
    img_d = imdilate(img, delta_B);
  
    % ��ʴ
    img_e = imerode(img_d, B_b);
   
    % ͼ�����
    out = img-img_e;
    

    out = mat2gray(out);
    
end

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

function [out] = ESTH(img, B_i, B_b, B_0)
    % ESTH �㷨����⸴�ӱ����µ���СĿ��
    % img: �����ͼ��
    % B_i, B_b, B_0: ��ͬ�ĽṹԪ��

    % ���ͼ��Ϊ��ɫ��תΪ�Ҷ�ͼ��
    if size(img, 3) > 1
        img = rgb2gray(img);
    end

    % ��һ����С���νṹ��ʴ (ȥ��)
    img_eroded1 = imerode(img, B_i);

    % �ڶ������е���Բ�ṹ���� (�ָ�Ŀ��)
    img_dilated = imdilate(img_eroded1, B_b);

    % ������������νṹ��ʴ (ѹ�Ʊ���)
    img_eroded2 = imerode(img_dilated, B_0);

    % ���Ĳ���ԭͼ�봦������� (��ǿĿ��)
    out = img - img_eroded2;

    % ��ֹ��ֵ������ֵ��Ϊ 0
    out(out < 0) = 0;

    % ��һ������
    out = mat2gray(out);
end

