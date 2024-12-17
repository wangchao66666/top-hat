% 1. ��ȡͼ��ת��Ϊ�Ҷ�
img = imread('./top_hat_image/output_image.jpg');  % �����ϴ���ͼ��
%gray_img = rgb2gray(img);  % ת��Ϊ�Ҷ�ͼ��

% 2. ����Ҷ�ֱ��ͼ
hist_counts = imhist(img);  % ����ԭʼͼ��ĻҶ�ֱ��ͼ

% 3. ����������ֱ��ͼ
gray_levels = 0:255;  % �Ҷ�ֵ��Χ
saliency_values = zeros(size(gray_levels));  % ��ʼ��������ֵ����

% ��ÿ���Ҷ�ֵ������������ֵ
for p = 1:256
    Ip = gray_levels(p);  % ��ǰ�Ҷ�ֵ
    % ���㵱ǰ�Ҷ�ֵ�����лҶ�ֵ�ĶԱȶȼ�Ȩ��
    contrast_sum = 0;
    for q = 1:256
        Iq = gray_levels(q);  % ��һ���Ҷ�ֵ
        contrast = abs(Ip - Iq);  % ����Աȶȣ�ŷʽ���룩
        contrast_sum = contrast_sum + hist_counts(q) * contrast;  % ��Ȩ��
    end
    saliency_values(p) = contrast_sum;  % �洢������ֵ
end

% 4. ��һ��������ֱ��ͼ
saliency_values = saliency_values / max(saliency_values) * max(hist_counts);

% 5. ����ԭʼͼ��ֱ��ͼ��������ֱ��ͼ
figure;
subplot(1, 2, 1);
bar(gray_levels, hist_counts, 'FaceColor', [0, 0, 0.5]);
title('(a) ԭʼͼ��ֱ��ͼ');
xlabel('�Ҷ�ֵ');
ylabel('������');

subplot(1,2, 2);
bar(gray_levels, saliency_values, 'FaceColor', [0, 0, 0.5]);
title('(b) ������ֱ��ͼ');
xlabel('�Ҷ�ֵ');
ylabel('������ֵ');

% 6. �����Ӿ�������ͼ
saliency_map = zeros(size(gray_img));  % ��ʼ��������ͼ
for x = 1:size(gray_img, 1)
    for y = 1:size(gray_img, 2)
        pixel_value = gray_img(x, y);  % ��ȡ���ػҶ�ֵ
        saliency_map(x, y) = saliency_values(pixel_value + 1);  % ��������ֵӳ�䵽������ͼ
    end
end

% 7. ��һ��������ͼ����ǿ�Աȶ�
saliency_map = mat2gray(saliency_map);  % ��������ͼ��һ���� [0, 1]

% 8. ��ʾ�Ӿ�������ͼ
figure;
imshow(saliency_map);
title('(c) �Ӿ�������ͼ');


