function [R] = lazy_random_walk(A)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
D = sum(A, 2);
V = 1 ./ (sqrt(D + 1e-10));
R = (eye(size(A)) + V' .* A .* V) ./ 2.0;
end

