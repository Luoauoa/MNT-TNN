function [U] = fast_polar(X,delta,max_iter)
% FAST_POLAR 
%   
X_k = X;
for i=1:max_iter
    X_pre = X_k;   % X_(k-1)
    X_inv = inv(X_k);
    g = (norm(X_inv, 1) * norm(X_inv, inf) / (norm(X_k, 1) * norm(X_k, inf))) ^ (1/4);
    X_k = (g * X_k + X_inv' / g) / 2;
    
    cond = norm(X_k - X_pre, 1) / norm(X, 1);
    if cond <= delta
        break
    end
end
U = X_k;
% H = (U * X + X * U) / 2;
end

