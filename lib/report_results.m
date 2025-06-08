SRs = {'0.03', '0.05', '0.07', '0.1', '0.3', '0.5'};
methods = {'TNN', 'UTNN', 'FTNN', 'NTTNN', 'MNT-TNN', 'ATNNs', 'HaLRTC', 'LRTC_TNN'};
means = cell(length(SRs),1);
vars = cell(length(SRs),1);
for i = 1:length(SRs)
    SR = SRs(i);
    file = fullfile(join([SR,'.txt'], ''));
    ID = fopen(file{1});
    C = textscan(ID, '%s %f %f %f %f %f %f');
    fclose(ID);
    C = C(4:5);
    M = cell2mat(C);
    Mt = reshape(M', 2, length(methods), []);  % (metrics, models, trais)
    means{i,1} = mean(Mt, 3);
    vars{i,1} = std(Mt, 0, 3);
end
for i = 1:length(SRs)
    fprintf('================Results=p=%s======================\n',SRs{i});
    fprintf('    Method         MAPE            RMSE    \n');
    ms = means{i,1};
    vs = vars{i,1};
    for j = 1:length(methods)
        fprintf(' %7s    %6.3f+%6.3f     %6.3f+%6.3f', methods{j}, ms(1,j), vs(1,j), ms(2,j), vs(2,j));
        fprintf('\n');
    end
    fprintf('\n');
end
