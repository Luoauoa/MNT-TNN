function [MAPE, RMSE, MAE, MSE] = metrics(X, Y, sample_inds)
%   Compute values of MAPE and RMSE
%   Args:
%          X -- inputs
%          Y -- ground truth 
%   Returns:
%          MAPE -- 
%          RMSE --

% mask sampled points
X(sample_inds) = 0;
Y(sample_inds) = 0;

X_flat = reshape(X, [], 1);
Y_flat = reshape(Y, [], 1);


mask_inds = find(Y_flat);
Y_mask = Y_flat(mask_inds); 
X_mask = X_flat(mask_inds);

MAE = mean(abs(X_mask - Y_mask), 'all');
MAPE = mean(abs(X_mask - Y_mask) ./ (Y_mask), 'all') * 100;
RMSE = sqrt(mean((X_mask - Y_mask).^2, 'all'));
MSE = RMSE ^ 2;

end

