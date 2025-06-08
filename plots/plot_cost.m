mr=[30, 50, 70, 90, 110];
%% PEMS-BAY
t_TNN = [167, 160, 166, 157, 150];
t_UTNN = [165, 155, 163, 160, 159];
t_FTNN = [2986, 3007, 3399, 3441, 3407];
t_NTTNN = [25, 27, 29, 27, 23];
t_MNTTNN = [102, 146, 147, 150, 85];
t_HaLRTC = [127, 222, 219, 223, 150];
%% CHPS
t_TNN_2 = [162, 160, 188, 193, 174];
t_UTNN_2 = [114, 111, 141, 145, 131];
t_FTNN_2 = [2969, 3646, 4015, 4301, 3894];
t_NTTNN_2 = [99, 113, 113, 63, 56];
t_MNTTNN_2 = [310, 493, 433, 461, 352];
t_HaLRTC_2 = [139, 221, 162, 195, 201];

figure
subplot(2, 2, 3);
plot(mr, t_TNN,'-x', 'linewidth', 2, 'DisplayName','TNN');
grid on
xlim([25, 115])
xticks([30, 50, 70, 90, 110])
xticklabels({'30%','50%','70%','93%','95%'})
xlabel('Missing Rate')
ylabel('Computation Time (ms/iter)')
hold on
plot(mr, t_UTNN,'-x', 'linewidth', 2, 'DisplayName','UTNN')
hold on
plot(mr, t_NTTNN,'-x', 'linewidth', 2, 'DisplayName','NTTNN')
hold on
plot(mr, t_MNTTNN,'-x', 'linewidth', 2, 'DisplayName','MNTTNN')
hold on
plot(mr, t_HaLRTC,'-x', 'linewidth', 2, 'DisplayName','HaLRTC')
legend

subplot(2, 2, 4);
plot(mr, t_TNN_2,'-x', 'linewidth', 2, 'DisplayName','TNN');
grid on
xlim([25, 115])
xticks([30, 50, 70, 90, 110])
xticklabels({'30%','50%','70%','93%','95%'})
xlabel('Missing Rate')
ylabel('Computation Time (ms/iter)')
hold on
plot(mr, t_UTNN_2,'-x', 'linewidth', 2, 'DisplayName','UTNN')
hold on
plot(mr, t_NTTNN_2,'-x', 'linewidth', 2, 'DisplayName','NTTNN')
hold on
plot(mr, t_MNTTNN_2,'-x', 'linewidth', 2, 'DisplayName','MNTTNN')
hold on
plot(mr, t_HaLRTC_2,'-x', 'linewidth', 2, 'DisplayName','HaLRTC')
legend

hold off
subplot(2, 2, 1);
plot(mr, t_FTNN, '-x', 'linewidth', 2, 'color', 	"#A2142F", 'DisplayName','FTNN')
grid on
xlim([25, 120])
xticks([30, 50, 70, 90, 110])
xticklabels({'30%','50%','70%','93%','95%'})
ylabel('Computation Time (ms/iter)')
legend

subplot(2, 2, 2);
plot(mr, t_FTNN_2, '-x', 'linewidth', 2, 'color', 	"#A2142F", 'DisplayName','FTNN')
grid on
xlim([25, 120])
xticks([30, 50, 70, 90, 110])
xticklabels({'30%','50%','70%','93%','95%'})
ylabel('Computation Time (ms/iter)')
legend