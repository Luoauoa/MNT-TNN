rho = [1,2,3,4,5]; %% x-axis
mr = [0.1, 0.5]; %% two subplots
MAPE = [33.550, 33.550, 33.564, 33.568, 33.557;
        25.541, 25.545, 25.560, 25.651, 26.057]; %% (2, 5)

figure
for i = 1:2
    subplot(1,2,i);
    plot(rho, MAPE(i, :), "-*", 'linewidth', 1.)
    xticks(rho)
    xticklabels({'1e-3', '1e-2', '1e-1', '1', '10'})
    xlabel('\rho')
    ylabel('MAPE (%)')
%     legend
    title(['Varation of MAPE under MR = ', num2str(mr(i))])
end