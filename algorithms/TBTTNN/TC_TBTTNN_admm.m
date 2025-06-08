function [best_out, best_iter] = TC_TBTTNN_admm(opts, M0, omega, M_t)
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
if isfield(opts, 'd');          d           = opts.d;            end

% initialization
M = M0;
dimM = size(M0);
Z = Fold(F * (Unfold(M, size(M), 3) * G'), [dimM(1), dimM(2), d], 3);  
dim = size(Z);          % (n1, n2, d)
Y = tanh(Z);
L{1} = zeros(size(Y));
L{2} = zeros(size(Y));
best_iter = 0;
loss = inf;

%% optimize start
for j = 1:max_iter
    M_j = M;
    Z_j = Z;
%     Y_j = Y; 
%     G_j = G;
%     L_j = L;
    %% M step; k=1,2
    k = 3;
    H = Fold(F' * Unfold(Z, size(Z), k) * G, dimM, k);
    D = Fold(F' * Unfold(L{2}, size(L{2}), k) * G ,dimM, k);
    M = (beta * H + D) / beta;
    M(omega) = M_t(omega);
    %% Y step; frontal slice-wise SVT
    P = tanh(Z) - L{1} / alpha;
    Y = SVT(P, 1.0 / alpha);
    %% Z step
    k = 3;
    H = Fold(F * (Unfold(M, size(M), k) * G') ,dim, k);
    P = H - L{2} / beta;
    X = Y + L{1} / alpha;
    Z = Newton(X, P, Z, inner_iter, alpha, beta);

%     %% L step
%     L = (gamma .* (M - E) + rho .* L_j) / (gamma + rho);
%     %% E Step
%     tmp = (gamma .* (M - L) + rho .* E_j) / (gamma + rho);
%     E = soft_threshold(tmp, tau/(gamma + rho));
    %% G step
    k = 3;
    S = F' * Unfold(Z, size(Z), k);
    T = F' * Unfold(L{2}, size(L{2}), k);
    K_mat = beta * S + T ;           % (n3, n1n2)
    M_mat = Unfold(M, size(M), k);        
    M_mat = M_mat';                  % (n1n2, n3)
    tmp = M_mat * K_mat;             % (n1n2, n1n2)
    [Uk,~,Vk] = svd(tmp, 'econ');
    G = Vk * Uk';
    %% F step
    k = 3;
    A = Unfold(M, size(M), k) * G';  % (n3, n1n2)
    Z_mat = Unfold(Z, size(Z), k);
    L_mat = Unfold(L{2}, size(L{2}), k);
    tmp = A * (beta * Z_mat' + L_mat');
    [Uf,~,Vf] = svd(tmp, 'econ');
    F = Vf * Uf';
    %% L update
    L{1} = L{1} + alpha * (Y - tanh(Z));
    L{2} = L{2} + beta * (Z - H);
    %% check convergence
    chgX = norm(M(:) - M_j(:)) / norm(M(:));
    if (j > 10) && (chgX < tol)
        break;
    end
    %% check score
    loss_tmp = metrics(M, M_t);
    if loss_tmp <= loss
        loss = loss_tmp;
        best_iter = j;
        best_out.M = M;
    end
%     %% update the lagrangian parameter
%     r1 = Y - tanh(Z);
%     s1 = alpha * (Z - Z_j);
%     r2 = Z - H;
%     s2 = beta * (M - M_j);
%     
%     if norm(r1(:)) > 10 * norm(s1(:))
%         alpha = 2 * alpha;
%     elseif norm(s1(:)) > 10 * norm(r1(:))
%         alpha = alpha / 2;
%     end
%     
%     if norm(r2(:)) > 10 * norm(s2(:))
%        beta = 2 * beta;
%     elseif norm(s2(:)) > 10 * norm(r2(:))
%         beta = beta / 2;
%     end
end
end

function Z  = Newton(Y,P,Z,inner,alpha, beta)
    i=0;
    relchg=1;
    tol=10^(-4);  
    while  i < inner  &&  relchg > tol 
            Zp=Z;
            % first-order deriviative
            Numer = alpha .* (1 - tanh(Z).^2) .* (tanh(Z) - Y) + beta .* (Z - P);  
            % second-order deriviative
            Denom = -2 .* alpha .* tanh(Z) .* ( 1 - tanh(Z).^2) .* (tanh(Z) - Y) + alpha .* ( 1 - tanh(Z).^2).^2 + beta; 
            Z  = Z - Numer ./ (Denom + 1e-12);
            relchg = norm(Z(:) - Zp(:)) / norm(Z(:));
            i=i+1;
    end
end

function output = tanh(x)
  output = (exp(x) - exp(-x))./(exp(x) + exp(-x));
%     output = sigmoid(x);
end

function Y = SVT(Y, rho)
    [n1, n2, n3] = size(Y);
    n12 = min(n1, n2);
    Uf = zeros(n1, n12, n3);
    Vf = zeros(n2, n12, n3);
    Sf = zeros(n12,n12, n3);
    trank = 0;

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

