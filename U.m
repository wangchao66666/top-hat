% ����һ���ӽ���0��ͼ�񣨱�������Ϊһ��С�ĳ�����
img = 0.0001 * ones(100, 100);  % ͼ���е�ֵ�ӽ��� 0

% ��һ��ͼ��
normalized_img = mat2gray(img);

% ��ʾԭʼͼ�����һ�����ͼ��
subplot(1, 2, 1);
imshow(img);
title('ԭʼͼ��');

subplot(1, 2, 2);
imshow(normalized_img);
title('��һ�����ͼ��');

% ��ʾ��һ����ͼ�������ֵ��Χ
disp(['��һ����ͼ�����Сֵ: ', num2str(min(normalized_img(:)))]);
disp(['��һ����ͼ������ֵ: ', num2str(max(normalized_img(:)))]);
disp(['��һ����ͼ��ľ�ֵ: ', num2str(mean(normalized_img(:)))]);
