%% bar chart - w/o transform G, H
X = categorical({'90%', '70%', '50%'});
X = reordercats(X,{'90%', '70%', '50%'});
RMSE = [5.98; 4.45; 3.76];
MAPE = [33.50; 28.51; 25.50];

RMSE_wo_G = [12.98; 8.71; 7.25];
MAPE_wo_G = [76.39; 45.45; 37.17];

RMSE_wo_H = [6.00; 4.50; 3.80];
MAPE_wo_H = [33.70; 28.61; 25.60];
y1 = [RMSE, RMSE_wo_H, RMSE_wo_G];
y2 = [MAPE, MAPE_wo_H, MAPE_wo_G];

% RMSE
subplot(1,2,1); a = bar(X, y1, 'BarWidth', 0.85); xlabel('MRs'); ylabel('RMSE')
ax=gca;grid on
ax.LineWidth=1.2;
ax.XMinorTick='on';
ax.YMinorTick='on';
ax.ZMinorTick='on';
ax.GridLineStyle=':';
% Legend   
hYLabel = ylabel('RMSE');
hLegend = legend([a(1),a(2), a(3)], ...
                 'MNT-TNN', 'MNT-TNN w/o H', 'MNT-TNN w/o G', ...
                 'Location', 'northeast');

% Font of label and Legend
set([hYLabel,hLegend], 'FontName',  'Helvetica')
set(hLegend, 'FontSize', 6)

% MAPE
subplot(1,2,2); b = bar(X, y2, 'BarWidth', 0.85); xlabel('MRs'); ylabel('MAPE')
ax=gca;grid on
ax.LineWidth=1.2;
ax.XMinorTick='on';
ax.YMinorTick='on';
ax.ZMinorTick='on';
ax.GridLineStyle=':';
% Legend   
hYLabel = ylabel('MAPE (%)');
hLegend = legend([b(1),b(2), b(3)], ...
                 'MNT-TNN', 'MNT-TNN w/o H', 'MNT-TNN w/o G', ...
                 'Location', 'northeast');

% Font of label and Legend
set([hYLabel,hLegend], 'FontName',  'Helvetica')
set(hLegend, 'FontSize', 6)

%% bar chart - w/o nonlinearity
% RMSE
RMSE_linear = [12.87; 8.56; 7.13];
MAPE_linear = [74.58; 44.76; 36.82];

MAPE_sigmoid = [35.568;35.75;34.23];
RMSE_sigmoid = [7.23;7.20;7.54];

MAPE_softplus = [62.91;37.00;30.83];
RMSE_softplus = [11.46;7.37;5.65];
subplot(1,2,1); a = bar(X, [RMSE, RMSE_linear, RMSE_sigmoid, RMSE_softplus]);xlabel('MRs'); ylabel('RMSE');
a(1).FaceColor='#000000';
a(2).FaceColor='#be0027';
ax=gca;hold on;grid on
ax.LineWidth=1.2;
ax.XMinorTick='on';
ax.YMinorTick='on';
ax.ZMinorTick='on';
ax.GridLineStyle=':';
% 标签及Legend 设置    
hYLabel = ylabel('RMSE');
hLegend = legend([a(1),a(2),a(3),a(4)], ...
                  'MNT-TNN','MNT-TNN\_linear','MNT-TNN\_sigmoid','MNT-TNN\_softplus', ...
                 'Location', 'northeast');

% % 刻度标签字体和字号
% set(gca, 'FontName', 'Helvetica', 'FontSize', 9)
% 标签及Legend的字体字号 
set([hYLabel,hLegend], 'FontName',  'Helvetica')
set(hLegend, 'FontSize', 8)

% MAPE
subplot(1,2,2); b=bar(X, [MAPE, MAPE_linear, MAPE_sigmoid, MAPE_softplus]);xlabel('MRs'); ylabel('MAPE(%)');
b(1).FaceColor='#000000';
b(2).FaceColor='#be0027';
ax=gca;hold on;grid on
ax.LineWidth=1.2;
ax.XMinorTick='on';
ax.YMinorTick='on';
ax.ZMinorTick='on';
ax.GridLineStyle=':';
% 标签及Legend 设置    
hYLabel = ylabel('MAPE (%)');
hLegend = legend([b(1),b(2),b(3),b(4)], ...
                 'MNT-TNN','MNT-TNN\_linear','MNT-TNN\_sigmoid','MNT-TNN\_softplus', ...
                 'Location', 'northeast');

% % 刻度标签字体和字号
% set(gca, 'FontName', 'Helvetica', 'FontSize', 9)
% 标签及Legend的字体字号 
set([hYLabel,hLegend], 'FontName',  'Helvetica')
set(hLegend, 'FontSize', 8)
