function [L] = geo_scatt_wave(S, scale)
% GEO_SCATT_WAVE 
%   
j = scale;
L0 = S^(2^(j-1));
L = L0 * (eye(size(L0)) - L0);
end

