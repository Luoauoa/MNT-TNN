function [best_out, best_iter] = TC_TBTTNN_linear(opts, M0, M1, omega, M_t)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

if isfield(opts, 'tol');        tol         =  opts.tol;         end
if isfield(opts, 'max_iter');   max_iter    =  opts.max_iter;	 end
if isfield(opts, 'inner');      inner_iter  =  opts.inner;  end
% regular factors
if isfield(opts, 'alpha');      alpha       =  opts.alpha;       end
if isfield(opts, 'beta');       beta        =  opts.beta;        end
if isfield(opts, 'gamma');      gamma       =  opts.gamma;       end
if isfield(opts, 'lambda');     lambda      =  opts.lambda;      end
if isfield(opts, 'rho');        rho         =  opts.rho;         end
if isfield(opts, 'tau');        tau         =  opts.tau;         end
if isfield(opts, 'G');          G           = opts.G;            end
if isfield(opts, 'F');          F           = opts.F;            end
if isfield(opts, 'Q');          Q           = opts.Q;            end
if isfield(opts, 't_dim');      t_dim       = opts.t_dim;        end


% initialization       
M = M0;
dimM = size(M0);
Z = Fold(F * Unfold(mode_prod(mode_prodm(M, G, dimM, 3), Q, dimM, 2), dimM, 3), t_dim, 3);  
Y = Z;
best_iter = 0;
loss = inf;

%% optimize start
for j = 1:max_iter
    M_j = M;
    Z_j = Z;
    Y_j = Y; 
    G_j = G;
    F_j = F;
    Q_j = Q;
    %% M step; k=1,2
    if j > 0
        T = mode_prodm(mode_prod(Fold(F' * Unfold(Z, size(Z), 3), dimM, 3), Q', dimM, 2), G', dimM, 3);
        M = (alpha .* T + rho .* M_j) / (alpha + rho);
        M(omega) = M1(omega);  % replace M_t by the resampled temple tensor
    end
    %% Y step; frontal slice-wise SVT
    H = (beta .* Z + rho .* Y_j) / (beta + rho);
    Y = SVT(H, 1.0 / (beta + rho));
    %% Z step
%     H = Fold(F * Unfold(mode_prod(mode_prodm(M, G, dimM, 3), Q, dimM, 2), dimM, 3), t_dim, 3);
%     P = (alpha .* H + rho .* Z_j) / (alpha + rho); 
%     Z = Newton(Y, P, Z, inner_iter, (alpha + rho), beta);
    %% G step
    for k = 1
        S = F' * Unfold(Z, t_dim, 3);  % (n3, n1n2)
%         S = S';
        M_mat = Unfold(M, dimM, 3);          % (n3, n1n2)
        M_mat = M_mat';
        tmp = rho * G_j' + alpha * M_mat * S;    % (n1n2, n1n2)
        [Uk,~,Vk] = svd(tmp, 'econ');
        G = Vk * Uk';
%         U = fast_polar(tmp, 1e-3, 8);
%         G = U';
    end
    %% Q step
    for k = 1
        A = Fold(F' * Unfold(Z, size(Z), 3), dimM, 3); 
        A_mat = Unfold(A, size(A), 2);                   % (n2, n1n3)
        B = Fold(Unfold(M, dimM, 3) * G', dimM, 3);  % (n2, n1n3)
        B_mat = Unfold(B, size(B), 2);
        tmp = alpha * B_mat * A_mat' + rho * Q_j';  %(n2n3, n2n3)
        [Uk,~,Vk] = svd(tmp, 'econ');
        Q = Vk * Uk';
    end
    %% F step
    for k = 1
        S = Unfold(M, dimM, 3) * G';
        Z_mat = Unfold(Z, t_dim, 3);
        tmp = alpha * S * Z_mat' + rho * F_j';
        [Uk,~,Vk] = svd(tmp, 'econ');
        F = Vk * Uk';
    end
    %% check convergence
    chgX = norm(M(:) - M_j(:)) / norm(M(:));
    if (j > 10) && (chgX < tol)
        best_out.kernel = Fold(F * Unfold(mode_prod(mode_prodm(M, G, dimM, 3), Q, dimM, 2), dimM, 3), t_dim, 3);
        break;
    end
    %% check score
    loss_tmp = metrics(M, M_t, omega);
    if loss_tmp <= loss
        loss = loss_tmp;
        best_iter = j;
        best_out.M = M;
    end
end
end

function Z  = Newton(Y,P,Z,inner, alpha, beta)
    i=0;
    relchg=1;
    tol=10^(-4);  
    while  i <= inner  &&  relchg > tol 
            Zp=Z;
            % first-order deriviative
            Numer = beta .* (1 - tanh(Z).^2) .* (tanh(Z) - Y) + alpha .* (Z - P);  
            % second-order deriviative
            Denom = -2 .* beta .* tanh(Z) .* ( 1 - tanh(Z).^2) .* (tanh(Z) - Y) + beta .* ( 1 - tanh(Z).^2).^2 + alpha;
            Z  = Z - Numer ./ (Denom + 1e-12);
            relchg = norm(Z(:) - Zp(:)) / norm(Z(:));
            i=i+1;
    end
end

function output = tanh(x)
  output = (exp(x) - exp(-x))./(exp(x) + exp(-x));
end

function Y = SVT(Y, rho)
    [n1, n2, n3] = size(Y);
    n12 = min(n1, n2);
    Uf = zeros(n1, n12, n3);
    Vf = zeros(n2, n12, n3);
    Sf = zeros(n12,n12, n3);
    trank = 0;
%     Y = fillmissing(Y, 'constant', 0.0);
    
    for i = 1 : n3
        [Uf(:,:,i), Sf(:,:,i), Vf(:,:,i)] = svd(Y(:,:,i), 'econ');
        s = diag(Sf(:, :, i));
        s = max(s - rho, 0);
        Sf(:, :, i) = diag(s);
        temp = length(find(s>0));
        trank = max(temp, trank);
        Y(:,:,i) = Uf(:,:,i) * Sf(:,:,i) * Vf(:,:,i)';
    end
end

function Y = soft_threshold(X, thr)
    Y = max(X - thr, 0) - max(-X - thr, 0);
end
function Y = threshold(X)
    Y = (max(X, 0) - max(-X, 0)) ./ (abs(X) + 1e-12);
end
