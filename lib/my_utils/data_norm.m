clear;
dir = "./";
dir2 = "../data";
file_name = fullfile(dir2, "bay.mat");
% file_name = fullfile(dir2, "num_tensor_30min_good.mat");
whos("-file", file_name)

%%
X = load(file_name);
t_flow = cell2mat(X.values);
t_flow = t_flow(:,:,1:4343);
% t_flow = double(cell2mat(values(:,:,3))); %% traffic flow
% time_axis = datetime(cell2mat(values(1,:,2)),...
%     'ConvertFrom','posixtime','Format','dd-MMM-yyyy HH:mm'); %% time conversion in millisetime

clear X;
[m, n, t] = size(t_flow);

% flow_re = reshape(t_flow, sqrt(m), sqrt(m), []);

[Y, RP] = MaxMinNorm(t_flow, 'all');

% figure;
% end_t = 3 * 144;
% plot(time_axis(1:end_t), squeeze(Y(12, 12, 1:end_t)));
% check by the denormalization 
X_hat = RP.y .* Y + RP.x;  
error = sum(t_flow - X_hat, 'all');
% hold on;
% figure;
% plot(time_axis(1:end_t), squeeze(flow_re(18, 12, 1:end_t)));
% xtickformat("MM-dd HH:mm");
save("bay_norm","Y");
save("RP_bay", "RP")


