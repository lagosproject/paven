
%% Demo.m 
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
%% Path normal
% options.Result_path = '/media/beegfs/home/v582/SCRATCH/2024-03-04_21-23_vinet/DHF1K';
%% Path YYY
options.Result_path = '/media/beegfs/home/v582/SCRATCH/2024-06-20_10-00_vinet_yuv/DHF1K';
options.DS_path = '/media/beegfs/home/v582/PROJECT/';

Datasets{1} = 'UCF sports';
Datasets{2} = 'Hollywood-2';
Datasets{3} = 'DHF1K';


Metrics{1} = 'CC'; 
Metrics{2} = 'similarity'; 
Metrics{3} = 'NSS';
Metrics{4} = 'AUC_Judd';
Metrics{5} = 'AUC_shuffled';

postfixFrames = '.jpg';

for i = 3:3 %length(Datasets)


    % saliency prediction results
    options.SALIENCY_DIR = [options.Result_path '/'];
    
    % dataset path
    if isequal(Datasets{i}, 'DHF1K')
        options.DS_GT_DIR = [options.DS_path Datasets{i} '/'];
    else
        options.DS_GT_DIR = [options.DS_path Datasets{i} '/test/'];
    end
    options.IMG_DIR = [options.DS_GT_DIR, 'videoframes_test/*/'];
    % Print of options.IMG_DIR to check if the path is correct
    disp(options.IMG_DIR);
    % We exit

    %%
    frames = dir(fullfile([options.IMG_DIR '*' postfixFrames]));
    nframe = length(frames);
        
    %% we evaluate at most 50000 randomly selected frames in one dataset for efficiency
    if nframe>50000
        k = randperm(nframe);
        frames = frames(k(1:50000));
    end

    for j = 1:length(Metrics)
        metric = Metrics{j};
        cacheFile = [CACHE Datasets{i} '_' metric '.mat'];
        if exist(cacheFile, 'file')
            load(cacheFile, 'result');
        else
            [result, allMetric] = evaluationFunc(options, metric, frames, metricRequiresSaliency(j), metricRequiresFixation(j));
            save(cacheFile, 'result');
        end
        %meanMetric{i}(j) = result;
        fprintf('%s :%.4f \n', Metrics{j}, result);
    end

end

%%
