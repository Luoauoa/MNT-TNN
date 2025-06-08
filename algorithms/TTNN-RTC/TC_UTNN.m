function [Z, k] = TC_UTNN(UU,X,Omega,opts)

%% Robust tensor completion using transformed tensor singular value decomposition
% by Guangjing Song  Michael K. Ng  Xiongjun Zhang, Numerical Linear Algebra with Applications,
%  27(3):e2299, 2020.
% sGS-ADMM for robust tensor completion
%%
% Input:
%       UU     -     The initialized unitary transform  
%       X      -     The observed n1*n2*n3 tensor
%       Omega  -     The observed index set
%       opts   -     Some parameters
% Output:
%       L      -     The recovered n1*n2*n3 tensor
%       k      -     The iteration number
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'beta');        beta = opts.beta;            end
if isfield(opts, 'MaxIte');      MaxIte = opts.MaxIte;        end
if isfield(opts, 'Z0');          Z0 = opts.Z0;                end
if isfield(opts, 'Xtrue');       Xtrue = opts.Xtrue;                end
if isfield(opts, 'dim');         dim = opts.dim;              end
if isfield(opts, 'gamma');       gamma = opts.gamma;          end
if isfield(opts, 'max_beta');    max_beta = opts.max_beta;            end

O = X;
M = zeros(size(X));
Z = X;

for k = 1:MaxIte    
    Xold = X;
    
    % updata the X
    
    X = Z - M / beta;
    X(Omega) = O(Omega);
    
    % updata Z
    
    Z = prox_utnn(UU, X + M /beta ,1/beta);
        

    
    % update Lagrange multiplier
    M = M + beta * (X - Z);
    beta = min(gamma * beta, max_beta);

    %% stopping criterion
%     if isfield(opts, 'Xtrue')
%         XT   = opts.Xtrue;
%         resT = norm(X(:)-XT(:))/norm(XT(:));
%         [psnr,ssim] = quality(X * 255, XT * 255);
%     end
    res=norm(X(:)-Xold(:))/norm(Xold(:));
    
    if (res < tol) && (k >= 10) 
        break;
    end
end

end
