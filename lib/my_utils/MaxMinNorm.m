function [Y, RP] = MaxMinNorm(M, dim)
% MAXMINNORM 
% Max-Min normalization for three order tensor input
% 

% [m, n] = size(M);

max_num = max(M, [], dim);
min_num = min(M, [], dim);
Y = (M - min_num) ./ ((max_num - min_num) + 1e-10);


RP.x = min_num;
RP.y = max_num - min_num;

end

