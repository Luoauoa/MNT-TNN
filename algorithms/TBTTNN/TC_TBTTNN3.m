function [best_out, best_iter] = TC_TBTTNN3(opts, M0, M1, omega, M_t)
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
Y = tanh(Z);
best_iter = 0;
loss = inf;

%% optimize start
for j = 1:max_iter
    M_j = M;    % X in the paper
    Z_j = Z;    % C in the paper
    Y_j = Y;    % Z in the paper
    G_j = G;    % G in the paper
    F_j = F;    % T in the paper
    Q_j = Q;    % H in the paper
    %% M step; k=1,2
    if j > 0
        T = mode_prodm(mode_prod(Fold(F' * Unfold(Z, size(Z), 3), dimM, 3), Q', dimM, 2), G', dimM, 3);
        M = (alpha .* T + rho .* M_j) / (alpha + rho);
        M(omega) = M1(omega);  % replace M_t by the resampled temple tensor
    end
    %% Y step; frontal slice-wise SVT
    H = (beta .* tanh(Z) + rho .* Y_j) / (beta + rho);
    Y = SVT(H, 1.0 / (beta + rho));
    %% Z step
    H = Fold(F * Unfold(mode_prod(mode_prodm(M, G, dimM, 3), Q, dimM, 2), dimM, 3), t_dim, 3);
    P = (alpha .* H + rho .* Z_j) / (alpha + rho); 
    Z = NewtonTanh(Y, P, Z, inner_iter, (alpha + rho), beta);
    %% G step
    for k = 1
        S = Unfold(mode_prod(Fold(F' * Unfold(Z, t_dim, 3), dimM, 3), Q', dimM, 2), dimM, 3);  
%         S = S';
        M_mat = Unfold(M, dimM, 3);          % (n3, n1n2)
        M_mat = M_mat';
        tmp = rho * G_j' + alpha * M_mat * S;    % (n1n2, n1n2)
        [Uk,~,Vk] = svd(tmp, 'econ');
        G = Vk * Uk';
%         U = fast_polar(tmp, 1e-3, 8);
%         G = U';
    end
%%  Q step
    for k = 1
        A = Fold(F' * Unfold(Z, size(Z), 3), dimM, 3); 
        A_mat = Unfold(A, size(A), 2);                   % (n2, n1n3)
        B = Fold(Unfold(M, dimM, 3) * G', dimM, 3);  % (n2, n1n3)
        B_mat = Unfold(B, size(B), 2);
        tmp = alpha * B_mat * A_mat' + rho * Q_j';  %(n2, n2)
        [Uk,~,Vk] = svd(tmp, 'econ');
        Q = Vk * Uk';
    end
    %% F step
    for k = 1
        S = Unfold(mode_prod(Fold(Unfold(M, dimM, 3) * G', dimM, 3), Q, dimM, 2), dimM, 3);
        Z_mat = Unfold(Z, t_dim, 3);
        tmp = alpha * S * Z_mat' + rho * F_j';
        [Uk,~,Vk] = svd(tmp, 'econ');
        F = Vk * Uk';
    end
    %% check convergence
    chgX = norm(M(:) - M_j(:)) / norm(M(:));
    if (j > 10) && (chgX < tol)
        best_out.kernel = tanh(Fold(F * Unfold(mode_prod(mode_prodm(M, G, dimM, 3), Q, dimM, 2), dimM, 3), t_dim, 3));
        break;
    end
    %% check score
    loss_tmp = metrics(M, M_t, omega);
    best_out.iter = j;
    if loss_tmp <= loss
        loss = loss_tmp;
        best_iter = j;
        best_out.M = M;
    end
end
end
% One can integrate the nonlinear activations into one function like activate(..., func='Tanh')
function output = tanh(x)
  output = (exp(x) - exp(-x))./(exp(x) + exp(-x));
end

function output = sigmoid(x)
  output =  1 ./ (1 + exp(x));
end

function output = softplus(x)
     exp_Z = exp(x);
     one_plus_exp_Z = 1 + exp_Z;
     output = log(one_plus_exp_Z); 
end

function Z = NewtonSigmoid(Y, P, Z, inner, alpha, beta)
% Newton's method to optimize Z for an objective function
% L(Z) = (alpha/2) * (Z - P)^2 + (beta/2) * (Y - sigmoid(Z))^2
% Output:
%     Z  - Optimized Z

    i = 0;
    relchg = 1;
    tol = 10^(-4);  % Tolerance for convergence

    while i < inner && relchg > tol % Corrected loop condition from <= to < for typical usage
        Zp = Z; % Store previous Z to calculate relative change

        % Sigmoid activation and its derivatives
        sig_Z = 1 ./ (1 + exp(-Z));          % sigma(Z)
        sig_prime_Z = sig_Z .* (1 - sig_Z);  % sigma'(Z) = sigma(Z)(1-sigma(Z))
        % sigma''(Z) = sigma'(Z)(1-2*sigma(Z))
        sig_double_prime_Z = sig_prime_Z .* (1 - 2 .* sig_Z); 

        % First-order derivative of the objective function L(Z)
        % L'(Z) = beta * (sigmoid(Z) - Y) * sigmoid'(Z) + alpha * (Z - P)
        Numer = beta .* (sig_Z - Y) .* sig_prime_Z + alpha .* (Z - P);

        % Second-order derivative of the objective function L(Z)
        % L''(Z) = beta * [ (sigmoid'(Z))^2 + (sigmoid(Z) - Y) * sigmoid''(Z) ] + alpha
        Denom = beta .* ( sig_prime_Z.^2 + (sig_Z - Y) .* sig_double_prime_Z ) + alpha;
        
        % Newton's method update step
        % Add a small epsilon to Denom for numerical stability
        Z = Z - Numer ./ (Denom + 1e-12); 
        
        % Calculate relative change for convergence check
        if norm(Zp(:)) > 1e-9 % Avoid division by zero if Zp is close to zero
            relchg = norm(Z(:) - Zp(:)) / norm(Zp(:));
        else
            relchg = norm(Z(:) - Zp(:)); % Use absolute change if Zp is very small
        end
        
        i = i + 1;
    end

%     if i == inner && relchg > tol
%         warning('NewtonSigmoid:MaxIterReached', ...
%                 'Maximum number of iterations reached without satisfying tolerance. Relative change: %f', relchg);
%     end
end

function Z  = NewtonTanh(Y,P,Z,inner,alpha,beta)
% Newton's method to optimize Z for an objective function
% L(Z) = (alpha/2) * (Z - P)^2 + (beta/2) * (Y - tanh(Z))^2
% Output:
%     Z  - Optimized Z
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

function Z = NewtonSoftplus(Y, P, Z, inner, alpha, beta)
% Newton's method for L(Z) = (beta/2)*(softplus(Z)-Y)^2 + (alpha/2)*(Z-P)^2

    i = 0;
    relchg = 1;
    tol = 10^(-4);

    while i < inner && relchg > tol
        Zp = Z;

        % Softplus and its derivatives
        % f(Z) = log(1 + exp(Z))
        exp_Z = exp(Z);
        one_plus_exp_Z = 1 + exp_Z;
        f_Z = log(one_plus_exp_Z); 

        sig_Z = 1 ./ (1 + exp(-Z));      % f'(Z) = sigmoid(Z)
        f_prime_Z = sig_Z;
        f_double_prime_Z = sig_Z .* (1 - sig_Z); % f''(Z) = sigmoid'(Z)
        
        f_Z_stable = zeros(size(Z));
        idx_pos_large = Z > 30; % Heuristic for Z where exp(Z) might be large
        idx_neg_large = Z < -30; % Heuristic for Z where exp(Z) is tiny
        idx_mid = ~(idx_pos_large | idx_neg_large);

        f_Z_stable(idx_pos_large) = Z(idx_pos_large);
        f_Z_stable(idx_neg_large) = exp(Z(idx_neg_large)); % log(1+x) ~ x for small x
        if any(idx_mid(:))
             f_Z_stable(idx_mid) = log(1 + exp(Z(idx_mid)));
        end
        f_Z = f_Z_stable;


        Numer = beta .* (f_Z - Y) .* f_prime_Z + alpha .* (Z - P);
        Denom = beta .* ( f_prime_Z.^2 + (f_Z - Y) .* f_double_prime_Z ) + alpha;
        
        Z = Z - Numer ./ (Denom + 1e-12);
        
        if norm(Zp(:)) > 1e-9
            relchg = norm(Z(:) - Zp(:)) / norm(Zp(:));
        else
            relchg = norm(Z(:) - Zp(:));
        end
        
        i = i + 1;
    end
    if i == inner && relchg > tol
        warning('NewtonSoftplus:MaxIterReached', 'Maximum iterations reached. Relchg: %f', relchg);
    end
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
