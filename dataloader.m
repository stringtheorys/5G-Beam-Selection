%You will have to replace these with your own file locations
gpu = load('\\filestore.soton.ac.uk\users\hh3g17\mydocuments\MATLAB\Imperial federated_beam run 1 results\top10_cpu.mat');
gpu_old = load('\\filestore.soton.ac.uk\users\hh3g17\mydocuments\GitHub\5G-Beam-Selection\Imperial federated GPU run 2\top10.mat');
cpu = load('\\filestore.soton.ac.uk\users\hh3g17\mydocuments\MATLAB\Imperial federated_beam GPU run\top10.mat');


figure
hold on
plot(smooth(gpu.top10), '-s')
plot(smooth(cpu.top10), '-d')
plot(smooth(gpu_old.top10), '-^')

xlabel('Number of Beams');
ylabel('Top 10 Categorical Accuracy');
legend('New GPU results', 'CPU Generated Results', 'Old GPU results', 'Location','southeast');
title('Performance of Imperial Code');
grid on;
box on;