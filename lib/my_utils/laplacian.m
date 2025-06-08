function L = laplacian(adj_mx)
% Compute the nomarlized laplacian
% L = I - D^(-1/2) * A * D^(-1/2) 
D = sum(adj_mx, 2);
V = 1 ./ (sqrt(D + 1e-10));
L = eye(size(adj_mx)) - V' .* adj_mx .* V;
