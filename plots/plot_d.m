d = [0, 1, 2, 3, 4]; %% x-axis
mr = [0.05, 0.1, 0.3, 0.5]; %% four subplots
MAPE_N = [42.724 43.298 43.586 45.617 46.997; 
            35.869 36.371 37.224 38.794 40.029;
            29.530 29.493 28.917 30.373 31.320;
            27.511 26.012 25.885 26.392 27.606;]; %% (4, 5)

MAPE_M = [37.222 36.828 36.814 36.827 36.761; 
            34.378 33.924 33.637 33.499 33.486;
            29.684 29.803 29.004 28.249 28.701;
            27.826 26.965 26.067 25.550 25.765;];

figure
for i = 1:4
    subplot(2,2,i);
    plot(d, MAPE_N(i,:), "--*", 'linewidth', 1., 'DisplayName','NTTNN')
    xticks(d)
    xticklabels({'28','56','112','224','336'})
    xlabel('d')
    ylabel('MAPE (%)')
    hold on 
    plot(d, MAPE_M(i,:), "--o", 'linewidth', 1., 'DisplayName','MNTTNN')
    legend
    title(['Varation of MAPE under MR = ', num2str(mr(i))])
end