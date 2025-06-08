%% =================================================================
% This script compares transform-based tensor nuclear norm (TNN) methods for low-rank tensor completion
% listed as follows:
%     1. TNN           t-SVD base-line method
%     2. UTNN          unitary tranform-based TNN method
%     3. FTNN          framelet transform-based TNN method
%     4. NTTNN         nonlinear transform-based TNN method
%     5. MNT-TNN       Multimode Nonlinear Transform-based TNN method
%     6. HaLRTC

% References: 
% [1] C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin and S. Yan, 
% "Tensor Robust Principal Component Analysis with a New Tensor Nuclear Norm" ,
% in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 925-938, 2020.
% [2] G.-J. Song, M. K. Ng, and X.-J. Zhang, "Robust tensor completion
% using transformed tensor singular value decomposition", Numerical Linear
% Algebra with Applications, vol. 27, p. e2299, 2020.
% [3] T.-X. Jiang, M. K. Ng, X.-L. Zhao and T.-Z. Huang, 
% "Framelet Representation of Tensor Nuclear Norm for Third-Order Tensor Completion",
% IEEE Transactions on Image Processing, vol. 29, pp. 7233-7244, 2020.
% [4] B.-Z. Li, X.-L Zhao, T.-Y. Ji, X.-J. Zhang, and T.-Z. Huang, 
% "Nonlinear Transform Induced Tensor Nuclear Norm for Tensor Completion",
% Journal of Scientific Computing, vol. 92, no. 3, 2020.
% You can:
%     1. Type 'Demo' to to run various methods and see the pre-computed results.
%     2. Select competing methods by turn on/off the enable-bits in Demo.m
% Please make sure your data is in range [0, 1].
%
% Created by Ben-Zheng Li £¨lbz1604179601@126.com£©
% 23/4/2023
% Modified by Yihang Lu(lyhsa22@mail.ustc.edu.cn)
% 23/11/2023

%% =================================================================
%clc;
clear;
close all;
addpath(genpath('data'));
addpath(genpath('lib'));
addpath(genpath('algorithms'));
CurrentPath = pwd;

%% Set enable bits
EN_TNN        = 0;
EN_UTNN       = 0;
EN_FTNN       = 0;
EN_NTTNN      = 0;
EN_TBTTNN     = 1;
EN_HaLRTC     = 1;
method_name   = {'Observed','TNN','UTNN','FTNN','NTTNN', 'TBTTNN', 'HaLRTC'};
Mnum          = length(method_name);
Re_tensor     =  cell(Mnum,1);
MPSNR_ALL     =  zeros(Mnum,1);
SSIM_ALL      =  zeros(Mnum,1);
MAPE_ALL      =  zeros(Mnum, 1);
RMSE_ALL      =  zeros(Mnum, 1);
MAE_ALL      =  zeros(Mnum, 1);
time          =  zeros(Mnum,1);
%% Load initial data
load('RP_30x30_60min.mat');
load('30x30_60min_norm.mat'); 
% load('test_data_ori.mat')
% load('test_data_mask.mat')
load('lap_mx_60min2.mat');
load('lon_filt10.mat');
X = Y;
% X = double(Yo);                   % transmit output key Y to X  
fprintf('SR   & Method  & MPSNR   & MSSIM  & MFSIM  & Time    \n');
%% Sampling with random position
sample_ratio = 0.5;
rng(2025);
fprintf('\n');
fprintf('================Results=p=%f======================\n',sample_ratio);
Y_tensorT   = X;
Y_tensor_ori = X .* RP.y + RP.x;
Nway        = size(Y_tensorT);
[n1,n2,n3]  = size(Y_tensorT);
Ndim        = ndims(Y_tensorT);
Omega       = find(rand(numel(Y_tensorT),1) < sample_ratio);  %% sample indices
% Omega = find(Ym);
Ind         = zeros(Nway);
Ind(Omega)  = 1;  
Y_tensor0   = zeros(Nway);
Y_tensor0(Omega) = Y_tensorT(Omega);  %% sampled / observed data
save("sampled_tensor", "Y_tensor0");
%% Non-random missing
% [I, J, K] = size(Y_tensorT);
% total_fibers = I * J;
% num_occluded = round(total_fibers * (1 - sample_ratio));
% occluded_fibers = randperm(total_fibers, num_occluded); % linear index
% [i_id, j_id] = ind2sub([I, J], occluded_fibers);
% Y_tensor0 = Y_tensorT;
% for k = 1:num_occluded
%     Y_tensor0(i_id(k), j_id(k), :) = 0;
% end
% Omega = find(Y_tensor0);
% Ind = zeros(Nway);
% Ind(Omega) = 1;
 %% Observed
i  = 1;
Re_tensor{i} = Y_tensor0;
Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
[MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
[MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
time(i) = 0;
enList = 1;
fprintf(' %8.8s    %5.4s    %5.4s    %5.4s    %5.4s    %5.4s   %5.4s    %5.4s \n','method','PSNR', 'SSIM', 'MAPE', 'RMSE', 'MAE', 'iter','time');
fprintf(' %8.8s    %5.3f    %5.3f    %5.3f    %5.3f    %3.3d     %.1f \n',...
    method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), 0, time(i));
 %% Perform  algorithms
% interpolation
A = Y_tensor0;
B = padarray(A,[20,20,20],'symmetric','both');
C = padarray(Ind,[20,20,20],'symmetric','both');
a1 = interpolate(shiftdim(B,1),shiftdim(C,1));
a1(a1<0) = 0;
a1(a1>1) = 1;
a1 = a1(21:end-20,21:end-20,21:end-20);
a1 = shiftdim(a1,2);

a2 = interpolate(shiftdim(B,2),shiftdim(C,2));
a2(a2<0) = 0;
a2(a2>1) = 1;
a2 = a2(21:end-20,21:end-20,21:end-20);
a2 = shiftdim(a2,1);
a = 0.5*a1+0.5*a2;
% a = zeros(size(Y_tensor0));
X0 = a;
X0(Omega) = Y_tensorT(Omega);
%% Use TNN
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
        [MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
        fprintf(' %8.8s    %5.3f    %5.3f    %5.3f   %5.3f   %5.3f    %3d     %.3f | my = %.3f   rho = %.3f \n',...
            method_name{i},MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i), iters(i), time(i)/iters(i), opts.beta,opts.rho);
    end
end
%% Use UTNN
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
    U0 = Re_tensor{2};
    O = tenmat(U0,3); % unfolding
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
  
        [M,iterations] = TC_UTNN(U,X0,Omega,opts);
        Re_tensor{i} = M;
        time(i)= toc(tStart);
        iters(i) = iterations;
        Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
     fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f  %5.3f  %3d     %.3f | beta = %.3f \n',...
            method_name{i},MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i), iters(i), time(i)/iters(i), opts.beta);
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
    [Re_tensor{i},iter] = TC_FTNN(Y_tensor0,Omega,opts,X0);
    time(i)= toc(tStart);
    iters(i) = iter;
    Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
    [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
    [MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
    fprintf(' %8.8s    %5.3f    %5.3f     %5.3f    %5.3f  %5.3f  %3.3d   %.3f | Frame =  %d, Level = %d, beta = %.2f\n',...
        method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i), iters(i), time(i)/iters(i),...
        opts.Frame, opts.Level, opts.beta);        
end
%% Use NTTNN
i = 5;
if EN_NTTNN
    enList = [enList,i];
    
    kkk = 0;
    opts   = [];
    for alpha = 10
        for beta = 10
            if sample_ratio < 0.3
                d = 112;
            else
                d = 224;
            end
            rho = 1e-2;
            kkk = kkk+1;
            X_m = Unfold(X0,size(X0),3);
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
        [Re_tensor{i},iter] = TC_NTTNN(Omega,opts,Y_tensorT,X0);
        time(i)= toc(tStart);
        iters(i) = iter;
        Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
        fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f  %5.3f   %3d     %.3f | alpha = %.1f beta = %.3f  d = %.0f  rho = %.2f  \n',...
            method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i), iters(i), time(i)/iters(i), opts.alpha,opts.beta,opts.d,opts.rho);
    end    
end
%% Use MNTTNN
i = 6;
if EN_TBTTNN
    enList = [enList,i];
    kkk = 0;
    opts   = [];
    kkk = kkk+1;
    [U1, ~, ~] = svd(lap_mx, 'econ');
    opts.G = U1; 
%     opts.G = eye(size(U1)); % ablation on G
    [U2, ~, ~] = svd(l_wave, 'econ'); % V = U, V * U'=I
    opts.Q = U2;
%     opts.Q = eye(size(U2));   % ablation on G
    if sample_ratio <= 0.05
        d = 112;
    else
        d = 224;
    end
    opts.t_dim = [n1, n2, d];
%     X0 = (Re_tensor{i-1} - RP.x) / RP.y;
    X_m = Unfold(X0,size(X0),3);
    [U3,~,~] = svds(X_m, d);
    opts.F = U3';
    
    opts.tol = 10^-4;
    opts.max_iter = 1000;  %% deserve to be fine-tuned
    opts.inner = 10;
    opts.alpha = 10;
    opts.beta  = 10;
    opts.gamma = 10;
%     if sample_ratio < 0.3
%         opts.rho = 10;   %% accelerate the convergence
%     else
%         opts.rho = 1e-3; %% improve the accuracy
%     end
    opts.rho = 1e-3;
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
%         kernel = best_out.kernel * RP.y + RP.x;
%         save('recovered_kernel', 'kernel');
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
        fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f  %5.3f   %3d     %.3f | alpha = %.1f beta = %.3f tau = %.2f rho = %.2f  d = %3d\n',...
            method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i), iters(i), time(i)/best_out.iter, opts.alpha,opts.beta, 0, opts.rho, d);
    end    
end
%% Use HaLRTC
i = 7;
if EN_HaLRTC
    enList = [enList,i];
    kkk = 0;
    kkk = kkk+1;
    
   % initialization of the parameters
    opts=[];
    alpha = [1, 1, 1];
    opts.alpha=alpha/sum(alpha);
    opts.tol = 1e-4;
    opts.maxit = 500;
    opts.rho = 1.1;
    opts.beta = 1e-1;
    opts.max_beta = 1e10;
    opts.Xtrue = Y_tensor_ori;
    t0= tic;

    for ii = 1:kkk
        tStart = tic;
        [Re_tensor{i}, Out] = LRTC_HaLRTC(Y_tensor0, Omega, opts);
        time(i)= toc(tStart);
        iters(i) = Out.iter;
        Re_tensor{i} = Re_tensor{i} .* RP.y + RP.x;
        [MPSNR_ALL(i), SSIM_ALL(i)] = quality(Y_tensor_ori, Re_tensor{i});
        [MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i)]  = metrics(Re_tensor{i}, Y_tensor_ori, Omega);
        fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f   %5.3f   %3d     %.3f | alpha = %.1f beta = %.3f rho = %.2f \n',...
            method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), MAE_ALL(i), iters(i), time(i)/iters(i), opts.alpha,opts.beta,opts.rho);
    end    
end
%% Show results
fprintf('\n');
fprintf('================== Result =====================\n');
fprintf(' %8.8s    %5.4s    %5.4s    %5.4s    %5.4s    %5.4s    %5.4s\n','method','PSNR', 'SSIM', 'MAPE', 'RMSE', 'iter','time');
for i = enList
    fprintf(' %8.8s    %5.3f    %5.3f   %5.3f    %5.3f  %3.3d   %.1f \n',method_name{i}, MPSNR_ALL(i), SSIM_ALL(i), MAPE_ALL(i), RMSE_ALL(i), iters(i), time(i));
end
%% new rec_data
% load('rec_data')
% Y_rec = double(Yr);
% [A, B] = metrics(Y_rec * 108, Y_tensor_ori, Omega)
%% Spatial visualization
colormap(parula(7));
subplot(2,4,1);imagesc(squeeze(Re_tensor{1}(:,:,269)), 'AlphaData', .8);title('Observed')

% subplot(2,4,1);imagesc(squeeze(Y_rec(:,:,269)), 'AlphaData', .8);title('DUTTNN')
if EN_TNN 
    subplot(2,4,2);imagesc(squeeze(Re_tensor{2}(:,:,269)),  'AlphaData', .8);title('TNN')
end

if EN_UTNN
    subplot(2,4,3);imagesc(squeeze(Re_tensor{3}(:,:,269)),  'AlphaData', .8);title('UTNN')
end

if EN_FTNN
    subplot(2,4,4);imagesc(squeeze(Re_tensor{4}(:,:,269)),  'AlphaData', .8);title('FTNN')
end

if EN_NTTNN
    subplot(2,4,5);imagesc(squeeze(Re_tensor{5}(:,:,269)),  'AlphaData', .8);title('NTTNN')
end

if EN_TBTTNN
    subplot(2,4,6);imagesc(squeeze(Re_tensor{6}(:,:,269)), 'AlphaData', .8);title('MNT-TNN')
end

if EN_HaLRTC
    subplot(2,4,8);imagesc(squeeze(Re_tensor{7}(:,:,269)), 'AlphaData', .8);title('HaLRTC')
end

subplot(2,4,7);imagesc(squeeze(Y_tensor_ori(:,:,269)),  'AlphaData', .8);title('Original')

% subplot(2,4,8);imagesc(squeeze(a(:,:,341)),  'AlphaData', .8);title('Interpolated')
% % subplot(2,4,3);imagesc(squeeze(X_drop(:,:,341)),  'AlphaData', .8);title('droppped')

%% Temporal visualization
% colormap(parula(7));
% % subplot(7,1,1);plot(squeeze(Re_tensor{1}(10,10,1:300)),'.', 'linewidth', 1.5);title('Observed')
% if EN_TNN 
%     subplot(6,1,2);plot(squeeze(Re_tensor{2}(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','Imputed Value');title('TNN'); hold on
%     plot(squeeze(Y_tensor_ori(6,6,1:300)),'linewidth', 1.5, 'DisplayName','original');legend
% end
% 
% if EN_UTNN
%     subplot(6,1,3);plot(squeeze(Re_tensor{3}(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','Imputed Value');title('UTNN'); hold on
%     plot(squeeze(Y_tensor_ori(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','original');legend
% end
% 
% if EN_FTNN
%     subplot(6,1,4);plot(squeeze(Re_tensor{4}(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','Imputed Value');title('FTNN'); hold on
%     plot(squeeze(Y_tensor_ori(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','original');legend
% end
% 
% if EN_NTTNN
%     subplot(6,1,5);plot(squeeze((Re_tensor{5}(6,6,1:300))), 'linewidth', 1.5, 'DisplayName','Imputed Value');title('NTTNN'); hold on
%     plot(squeeze(Y_tensor_ori(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','original');legend
% end
% 
% if EN_TBTTNN
%     subplot(6,1,6);plot(squeeze(Re_tensor{6}(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','Imputed Value'); title('MNT-TNN');  hold on
%     plot(squeeze(Y_tensor_ori(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','original');legend
% end
% 
% if EN_HaLRTC
%     subplot(6,1,1);plot(squeeze(Re_tensor{7}(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','Imputed Value');title('HaLRTC'); hold on
%     plot(squeeze(Y_tensor_ori(6,6,1:300)), 'linewidth', 1.5, 'DisplayName','original');legend
% end

% subplot(2,4,8);imagesc(squeeze(a(:,:,341)),  'AlphaData', .8);title('Interpolated')
% subplot(2,4,3);imagesc(squeeze(X_drop(:,:,341)),  'AlphaData', .8);title('droppped')
