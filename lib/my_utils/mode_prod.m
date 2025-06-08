function Y = mode_prod(X, A, dim0, k)
%MODE_PRO 
%   
X_mat = Unfold(X, size(X), k);
Y = Fold(A * X_mat, dim0, k);
end

