function [X, Out,k] = TC_TNN(B, Omega, opts)
%%% Solve the Low-Rank Tensor Completion (LRTC) based on Tensor Nuclear Norm (TNN) problem by M-ADMM
%
% min_X ||X||_*, s.t. P_Omega(X) = P_Omega(B)
%
% ---------------------------------------------
% Input:
%       B       -    d1*d2*d3 tensor
%       Omega   -    index of the observed entries
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.maxit      -   maximum number of iterations
%           opts.beta       -   stepsize for dual variable updating in ADMM
%           opts.max_beta   -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase beta
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    d1*d2*d3 tensor
%       Out
%         Out.PSNR
%         Out.Res
%         Out.ResT


% by Zemin Zhang 
% @INPROCEEDINGS{ZhangCVPR,
%   author={Z. Zhang and G. Ely and S. Aeron and N. Hao and M. Kilmer},
%   booktitle={the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
%   title={Novel Methods for Multilinear Data Completion and De-noising Based on Tensor-{SVD}},
%   year={2014},
%   volume={},
%   number={},
%   pages={3842-3849},
% }
% @article{zhang2017exact,
%   title={Exact tensor completion using t-{SVD}},
%   author={Zhang, Zemin and Aeron, Shuchin},
%   journal={IEEE Transactions on Signal Processing},
%   volume={65},
%   number={6},
%   pages={1511--1526},
%   year={2017},
% }
%%

if ~exist('opts', 'var')
    opts = [];
end
if isfield(opts, 'tol');                tol = opts.tol;             end
if isfield(opts, 'maxit');            maxit = opts.maxit;           end
if isfield(opts, 'rho');                rho = opts.rho;             end
if isfield(opts, 'beta');              beta = opts.beta;            end     %%1e-2
if isfield(opts, 'max_beta');      max_beta = opts.max_beta;        end
if isfield(opts, 'Xtrue');            Xtrue = opts.Xtrue;           end

Nway = size(B);
X = B;
Z = X; %% 辅助变量
M = zeros(Nway); %% 拉格朗日乘子
Out.Res=[];
for k = 1:maxit
    Xold = X;
    %% solve X-subproblem
    X = Z - M/beta;
    X(Omega) = B(Omega); %% observed data constrain
    
    %% solve Y-subproblem
    [Z,~,~] = prox_tnn(X + M/beta,1/beta);
    
    %% check the convergence
%     if isfield(opts, 'Xtrue')
%         XT   = Xtrue;
%         resT = norm(X(:)-XT(:))/norm(XT(:));
%         [psnr,~] = quality(X * 255, XT * 255);
%         Out.ResT = [Out.ResT,resT];
%         Out.PSNR = [Out.PSNR,psnr];
%     end
    res=norm(X(:)-Xold(:))/norm(Xold(:));
    Out.Res = [Out.Res,res];
    
     if (res < tol) && (k >= 10) 
        break;
    end
    
    %% update Lagrange multiplier
    M = M + beta * (X - Z);
    beta = min(rho * beta, max_beta);
end
end