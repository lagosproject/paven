
%% HVECTest.m 
% All the codes in "code_forMetrics" are from MIT Saliency Benchmark (https://github.com/cvzoya/saliency). Please refer to their webpage for more details.

% load global parameters, you should set up the "ROOT_DIR" to your own path
% for data.
clear
% METRIC_DIR = 'code_forMetrics';
% addpath(genpath(METRIC_DIR));

%% path to store evaluation results
CACHE = 'cache/';
if ~exist(CACHE, 'dir')
    mkdir(CACHE);
end

Metrics{1} = 'CC'; 
Metrics{2} = 'similarity'; 
Metrics{3} = 'NSS';
Metrics{4} = 'AUC_Judd';
Metrics{5} = 'AUC_shuffled';

% saliency prediction results
options.SALIENCY_DIR = '/media/beegfs/home/v582/SCRATCH/ModelToQPoutput/';
options.DS_GT_DIR = ['/media/beegfs/home/v582/SCRATCH/HVCEyeTracking' '/'];
options.IMG_DIR = [options.DS_GT_DIR, '*/'];
% Print of options.IMG_DIR to check if the path is correct
disp(options.IMG_DIR);
% We exit

%%
frames = dir(fullfile([options.IMG_DIR '*.jpg']));
nframe = length(frames);

metricRequiresSaliency = ismember(Metrics, {'similarity', 'CC', 'EMD'});
metricRequiresFixation = ismember(Metrics, {'AUC_Judd', 'AUC_Borji', 'NSS'});

for j = 5:5 %length(Metrics) 
    metric = Metrics{j};
    [result, allMetric, folderMetrics] = evaluationFuncHVEC(options, metric, frames, metricRequiresSaliency(j), metricRequiresFixation(j));  
    %meanMetric{i}(j) = result;
    fprintf('%s :%.4f \n', Metrics{j}, result);
end
