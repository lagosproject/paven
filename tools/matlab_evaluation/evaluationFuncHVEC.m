function [meanMetric, allMetrics, folderMetrics] = evaluationFuncHVEC(options, metricName, frames, reqSaliency, reqFixation)
 
%EVALUATIONFUNC Evaluate result with the metric
%   result: array of cells containing the predicted saliency map
%   data: the ground truth data
%   metricName: the name of metric
%       -"similarity": Similarity
%       -"CC": CC
%       -"AUC_Borji": AUC_Borji
%       -"AUC_Judd": AUC_Judd
%       -"AUC_shuffled": sAUC
%   if the ground truth cannot be found, e.g. testing data, the central
%   gaussian will be taken as ground truth automatically.

fh = str2func(metricName);

allMetrics = zeros(length(frames),1);
folderMetrics = struct();

folder_idx = 0;
lastFolderName = '';
sanitizedFolderName = '';

for i = 1:length(frames)
    if mod(i, 100) == 0
        disp(['Processing frame: ', num2str(i), ' / ', num2str(length(frames))]);
    end
    gt_fold = strrep(frames(i).folder, '\','/');

    if ~strcmp(lastFolderName, frames(i).folder)
        folder_idx = 1;
        lastFolderName = frames(i).folder;
        sanitizedFolderName = matlab.lang.makeValidName(frames(i).folder);
        folderMetrics.(sanitizedFolderName) = [];
    end

    map_eval_path = [strrep(gt_fold, 'HVCEyeTracking', 'ModelToQPoutput') '.cfg'];
    map_eval_file_path = [map_eval_path, '/' , num2str(folder_idx-1), '.png'];

    saliency_path = [frames(i).folder '/frame_' sprintf('%04d', folder_idx) '.jpg'];

    fixation_path = [frames(i).folder '_fixmaps.mat'];

    % for first frame print all paths to check if they are formed correctly
    if i == 1
        disp('Folder');
        disp(frames(i).folder);
        disp('Name');
        disp(frames(i).name);
        disp('Salient path');
        disp(saliency_path); % Ground thruth
        disp('Fixation path');
        disp(fixation_path); % Ground thruth
        disp('Map eval path');
        disp(map_eval_path); % Prediction
        disp(map_eval_file_path);
    end

    if ~exist(map_eval_file_path, 'file')
        allMetrics(i) = nan;
        folderMetrics.(sanitizedFolderName) = [folderMetrics.(sanitizedFolderName); nan];
        continue;
    end

    result = double(imread(map_eval_file_path));
    result = result(:,:,1);

    %%%%% Ver tamanos
    %disp('Result size');
    %disp(size(result));
    %
    %newdata = load(fixation_path);
    %newdata = newdata.fixmaps(:,:,folder_idx);
    %disp('Fixation size');
    %disp(size(newdata));
    %return
    %result = imresize(result, [size(I,1) size(I,2)]);
    if reqSaliency && exist(saliency_path, 'file')
        I = double(imread(saliency_path))/255;
        
        allMetrics(i) = fh( result, I);

    elseif reqFixation && exist(fixation_path, 'file')
        newdata = load(fixation_path);
        newdata = newdata.fixmaps(:,:,folder_idx);
        I = double(newdata);
        allMetrics(i) = fh( result, I);
 
    elseif strcmp(metricName, 'AUC_shuffled')
        if exist(fixation_path, 'file')
            newdata = load(fixation_path);
            newdata = newdata.fixmaps(:,:,folder_idx);
            I = newdata > 0;
            I = double(I);
            ids = randsample(length(frames), min(10,length(frames)));
            fixation_point = zeros(0,2);
            for k = 1:min(10,length(frames))
                valid_file = false;
                while ~valid_file
                    fx_name = frames(ids(k)).name;
                    %disp('Fixation name');
                    %disp([frames(ids(k)).folder '/' fx_name]);
    
                    fx_fold = frames(ids(k)).folder;
                    fx_fold = strrep(fx_fold, '\','/');
    
                    fixation_path = [fx_fold '_fixmaps.mat'];
                    %disp('Fixation path');
                    %disp(fixation_path);
    
                    if exist(fixation_path, 'file')
                        valid_file = true;
                        %disp('File exists');
                    else
                        ids(k) = randsample(length(frames), 1);
                    end
                end

                number = extractNumber(fx_name);

                newdata = load(fixation_path);
                newdata = newdata.fixmaps(:,:,number);
                Ix = double(newdata);
                
                training_resolution = size(Ix);
                rescale = size(result)./training_resolution;
                [fx, fy]= find(Ix);
                pts = vertcat([fy fx]);
                fixation_point = [fixation_point; pts.*repmat(rescale, size(pts,1), 1)];
            end
            otherMap = makeFixationMap(size(result), fixation_point);
            allMetrics(i) = fh( result, I, otherMap);
        else       
            allMetrics(i) = nan;
        end 
    else
        allMetrics(i) = nan;
    end
    folderMetrics.(sanitizedFolderName) = [folderMetrics.(sanitizedFolderName); allMetrics(i)];
    folder_idx = folder_idx + 1;
end
allMetrics(isnan(allMetrics)) = [];
meanMetric = mean(allMetrics);

folderNames = fieldnames(folderMetrics);
for j = 1:length(folderNames)
    folderName = folderNames{j};
    sanitazedName = matlab.lang.makeValidName(folderName);
    folderMetrics.(sanitazedName) = mean(folderMetrics.(sanitazedName), 'omitnan');
    disp(['Folder: ', folderName, ' Mean Value: ', num2str(folderMetrics.(sanitazedName))]);
end

end


function number = extractNumber(filename)
    % Use regular expression to extract the number from the filename
    tokens = regexp(filename, 'frame_(\d+)\.jpg', 'tokens');
    if ~isempty(tokens)
        number = str2double(tokens{1}{1});
        number = int32(number);
    else
        error('Filename does not match the expected pattern.');
    end
end