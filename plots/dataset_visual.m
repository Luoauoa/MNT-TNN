clear;
close all;
addpath(genpath('data'));
addpath(genpath('lib'));
addpath(genpath('algorithms'));
CurrentPath = pwd;

colormap(parula(7));
%% CHSP
load('RP_30x30_60min.mat');
load('30x30_60min_norm.mat'); 
X = Y;                
chsp_ori = X .* RP.y + RP.x;
subplot(1, 3, 1); imagesc(squeeze(chsp_ori(:,:,275)), 'AlphaData', .8);title('CHSP')
%% PEMS04
RP.y = 823;
RP.x = 0;
load('pems04.mat'); 
X = values;                
pems04_ori = X .* RP.y + RP.x;
subplot(1, 3, 2); imagesc(squeeze(pems04_ori(:,:,341)), 'AlphaData', .8);title('PEMS04')
%% PEMS-BAY
RP.y = 80;
RP.x = 3.5;
load('bay_norm.mat'); 
X = Y;                
pemsbay_ori = X .* RP.y + RP.x;
subplot(1, 3, 3); imagesc(squeeze(pemsbay_ori(:,:,341)), 'AlphaData', .8);title('PEMS-BAY')