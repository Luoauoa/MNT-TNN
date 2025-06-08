function Y = mode_prodm(X, A, dim0, k)
%MODE_PRO 
%   
X_mat = Unfold(X, size(X), k);
Y = Fold(X_mat * A', dim0, k);
end

