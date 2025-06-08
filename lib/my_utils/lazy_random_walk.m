function [R] = lazy_random_walk(A)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
D = sum(A, 2);
V = 1 ./ (sqrt(D + 1e-10));
R = (eye(size(A)) + V' .* A .* V) ./ 2.0;
end

