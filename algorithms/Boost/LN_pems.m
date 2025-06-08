%% =================================================================
%clc;
clear;
close all;
addpath(genpath('data'));
addpath(genpath('lib'));
addpath(genpath('algorithms'));
CurrentPath = pwd;
%% Set enable bits
EN_TNN        = 1;
EN_UTNN       = 1;
EN_FTNN       = 1;
EN_NTTNN      = 1;
EN_TBTTNN     = 1;
method_name   = {'Observed','TNN','UTNN','FTNN','NTTNN', 'TBTTNN'};
Mnum          = length(method_name);
Re_tensor     =  cell(Mnum,1);
MPSNR_ALL     =  zeros(Mnum,1);
SSIM_ALL      =  zeros(Mnum,1);
MAPE_ALL      =  zeros(Mnum, 1);
RMSE_ALL      =  zeros(Mnum, 1);
time          =  zeros(Mnum,1);
RP.y = 823;
RP.x = 0;
%% Load initial data
load('pems04.mat'); 
load('lap_mx_pems.mat');
X = values;                   % transmit output key Y to X  
% X = permute(X, [2,1,1]);
fprintf('SR   & Method  & MPSNR   & MSSIM  & MFSIM  & Time    \n');
%% Sampling with random position
sample_ratio = 0.07;
fprintf('\n');
fprintf('================Results=p=%f======================\n',sample_ratio);
Y_tensorT   = X;
Y_tensor_ori = X .* RP.y + RP.x;
Nway        = size(Y_tensorT);
[n1,n2,n3]  = size(Y_tensorT);
Ndim        = ndims(Y_tensorT);
rng(2024);
Omega       = find(rand(numel(Y_tensorT),1) < sample_ratio);  %% sample indices
Ind         = zeros(Nway);
Ind(Omega)  = 1;  
Y_tensor0   = zeros(Nway);
Y_tensor0(Omega) = Y_tensorT(Omega);  %% sampled / observed data
 %% Observed
i  = 1;
Re_tensor{i} = Y_tensor0;
Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
[MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
[MAPE_ALL(i), RMSE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
time(i) = 0;
enList = 1;
fprintf(' %8.8s    %5.4s    %5.4s    %5.4s    %5.4s    %5.4s     %5.4s \n','method','PSNR', 'SSIM', 'MAPE', 'RMSE', 'iter','time');
fprintf(' %8.8s    %5.3f    %5.3f    %5.3f    %5.3f    %3.3d     %.1f \n',...
    method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), 0, time(i));
 %% Perform  algorithms
% interpolation
A = Y_tensor0;
B = padarray(A,[10,10,10],'symmetric','both');
C = padarray(Ind,[10,10,10],'symmetric','both');
a1 = interpolate(shiftdim(B,1),shiftdim(C,1));
a1(a1<0) = 0;
a1(a1>1) = 1;
a1 = a1(11:end-10,11:end-10,11:end-10);
a1 = shiftdim(a1,2);

a2 = interpolate(shiftdim(B,2),shiftdim(C,2));
a2(a2<0) = 0;
a2(a2>1) = 1;
a2 = a2(11:end-10,11:end-10,11:end-10);
a2 = shiftdim(a2,1);
a = 0.5*a1+0.5*a2;
%     a = zeros(size(Y_tensor0));
X0 = a;
X0(Omega) = Y_tensorT(Omega);
%% N-L model start
%% TNN
i = 2;
if EN_TNN
    enList = [enList,i];
    % initialization of the parameters
    kkk = 0;
    opts = [];
    opts.DEBUG = 0;
    opts.Xtrue = Y_tensorT;
    opts.max_beta = 1e10;
    opts.maxit = 200;
    
    for op_beta = 1e-2
        kkk =  kkk+1;
        opts.beta = op_beta;
        opts.tol = 1e-5;
        opts.rho = 1.1;
        opts_All{kkk} = opts;
    end
    
    for ii = 1:kkk  %% kkk = 1
        tStart = tic;
        opts = opts_All{ii};
        [Re_tensor{i},~,iterations] = TC_TNN(X0,Omega,opts);
        time(i)= toc(tStart);
        iters(i) = iterations;
        Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
        fprintf(' %8.8s    %5.3f    %5.3f    %5.3f    %5.3f    %3d     %.1f | my = %.3f   rho = %.3f \n',...
            method_name{i},MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), iters(i), time(i), opts.beta,opts.rho);
    end
end
%% UTNN
i = 3;
if EN_UTNN
    enList = [enList,i];
    % initialization of the parameters
    kkk = 0;
    opts = [];
    opts.gamma = 1.1;
    opts.Xtrue = Y_tensorT;
    opts.max_beta = 1e10;
    opts.MaxIte = 200;
    % Initialize the unitary transform
    X = Re_tensor{i-1} / RP.y - RP.x;
    O = tenmat(X,3); % unfolding
    O = O.data;
    [U, ~ ,~] = svd(O);
    for beta = 1              
            kkk =  kkk+1;
            opts.beta = beta;
            opts.tol = 5e-4;
            opts_All{kkk} = opts;
    end            
    for ii = 1:kkk
        tStart = tic;
        opts = opts_All{ii};
        opts.Xtrue = Y_tensorT;
        
        [M,iterations] = TC_UTNN(U,X,Omega,opts);
        Re_tensor{i} = M;
        time(i)= toc(tStart);
        iters(i) = iterations;
        Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);

        fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f   %3d     %.1f | beta = %.3f \n',...
            method_name{i},MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), iters(i), time(i), opts.beta);
    end
end
%% Use FTNN
i = 4;
if EN_FTNN
    enList = [enList,i];
    %parameters
    opts.Frame    = 3;  % (0,1,3)
    opts.Level    = 6;  % [1,2,3,4,5,6]
    opts.wLevel   = -1;
    opts.lambda1  = 1;
    opts.beta     = 1;
    opts.tol      = 1e-2;
    opts.rho      = 1;
    opts.DEBUG    = 0;
    opts.max_iter = 100;
    opts.max_beta = 1e10;
    tStart = tic;
    X = Re_tensor{i-1} / RP.y - RP.x;
    [Re_tensor{i},iter] = TC_FTNN(Y_tensor0,Omega,opts,X);
    time(i)= toc(tStart);
    iters(i) = iter;
    Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
    [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
    [MAPE_ALL(i), RMSE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
    fprintf(' %8.8s    %5.3f    %5.3f     %5.3f    %5.3f    %3.3d   %.1f | Frame =  %d, Level = %d, beta = %.2f\n',...
        method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), iters(i), time(i),...
        opts.Frame, opts.Level, opts.beta);        
end
%% NTTNN
i = 5;
if EN_NTTNN
    enList = [enList,i];
    
    kkk = 0;
    opts   = [];
    for alpha = 10
        for beta = 10
            if sample_ratio < 0.3
                d = 10;
            else
                d = 20;
            end
                rho = 1e-2;
                kkk = kkk+1;
                X            = (Re_tensor{i-1} - RP.x) / RP.y;
                X_m = Unfold(X,size(X),3);
                [DI,~,~] = svds(X_m, d);
                opts = [];
                opts.D0  = DI;
                opts.d   = d;
                opts.rho = rho;
                opts.tol = 10^-4;
                opts.max_iter = 2200;  %% deserve to be fine-tuned
                opts.inner = 10;
                opts.alpha = alpha;
                opts.beta = beta;
                opts_All{kkk} = opts;
        end
    end

    for ii = 1:kkk
        tStart = tic;
        opts = opts_All{ii};
        [Re_tensor{i},iter] = TC_NTTNN(Omega,opts,Y_tensorT,X);
        
        time(i)= toc(tStart);
        iters(i) = iter;
        Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
        fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f    %3d     %.1f | alpha = %.1f beta = %.3f  d = %.0f  rho = %.2f  \n',...
            method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), iters(i), time(i), opts.alpha,opts.beta,opts.d,opts.rho);
    end    
end
%% Use TBTTNN
i = 6;
if EN_TBTTNN
    enList = [enList,i];
    kkk = 0;
    opts   = [];
    kkk = kkk+1;
    [U1, ~, ~] = svd(lap_mx, 'econ');
    opts.G = U1;
%     [U2, ~, ~] = svd(l_wave, 'econ'); % V = U, V * U'=I
    opts.Q = eye(n1,n2);
    d = 60;
    opts.t_dim = [n1, n2, d];
    X0 = (Re_tensor{i-1} - RP.x) / RP.y;
    X_m = Unfold(X0,size(X0),3);
    [U3,~,~] = svds(X_m, d);
    opts.F = U3';
    
    opts.tol = 10^-4;
    opts.max_iter = 1000;  %% deserve tuning
    opts.inner = 10;
    opts.alpha = 10;
    opts.beta  = 10;
    opts.gamma = 10;
    opts.rho = 10;
    opts_All{kkk} = opts;
    
    for ii = 1:kkk
        tStart = tic;
        opts = opts_All{ii};
        [best_out, best_iter] = TC_TBTTNN3(opts, X0, X0, Omega, Y_tensorT);
        M = best_out.M;
        Re_tensor{i} = M;  
        time(i)= toc(tStart);
        iters(i) = best_iter;
        Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
        fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f    %3d     %.1f | alpha = %.1f beta = %.3f tau = %.2f rho = %.2f  d = %3d\n',...
            method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), iters(i), time(i), opts.alpha,opts.beta, 0, opts.rho, d);
    end    
end
%% Show results
fprintf('\n');
fprintf('================== Result =====================\n');
fprintf(' %8.8s    %5.4s    %5.4s    %5.4s    %5.4s    %5.4s    %5.4s\n','method','PSNR', 'SSIM', 'MAPE', 'RMSE', 'iter','time');
for i = enList
    fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f  %3.3d   %.1f \n',method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), iters(i), time(i));
end
%%
colormap(parula(7));
subplot(2,4,1);imagesc(squeeze(Re_tensor{1}(:,:,342)), 'AlphaData', .8);title('Observed')
if EN_TNN 
    subplot(2,4,2);imagesc(squeeze(Re_tensor{2}(:,:,342)),  'AlphaData', .8);title('TNN')
end

if EN_UTNN
    subplot(2,4,3);imagesc(squeeze(Re_tensor{3}(:,:,342)),  'AlphaData', .8);title('UTNN')
end

if EN_NTTNN
    subplot(2,4,5);imagesc(squeeze(Re_tensor{4}(:,:,342)),  'AlphaData', .8);title('NTTNN')
end

if EN_TBTTNN
    subplot(2,4,6);imagesc(squeeze(Re_tensor{5}(:,:,342)), 'AlphaData', .8);title('TBTTNN')
end

subplot(2,4,7);imagesc(squeeze(Y_tensor_ori(:,:,342)),  'AlphaData', .8);title('Original')

subplot(2,4,8);imagesc(squeeze(a(:,:,342)),  'AlphaData', .8);title('Interpolated')
