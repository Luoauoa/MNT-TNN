load('recovered_kernel.mat');
K1 = kernel;
K2 = permute(kernel, [1,3,2]);
K3 = permute(kernel, [2,3,1]);

[n1, n2, n3] = size(kernel);
tmps{1} = [];
tmps{2} = [];
tmps{3} = [];

tmp = [];
for i=1:n3
    [~,S1,~] = svd(K1(:,:,i), 'econ');
    tmp = cat(1, tmp, diag(S1));
end
plot(tmp, '-r', 'LineWidth', 1.5);

