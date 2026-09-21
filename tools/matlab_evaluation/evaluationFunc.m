function [meanMetric, allMetrics] = evaluationFunc(options, metricName, frames, reqSaliency, reqFixation)
 
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


postfixFrames = '.jpg';

fh = str2func(metricName);



allMetrics = zeros(length(frames),1);
for i = 1:length(frames)

    gt_fold = strrep(frames(i).folder, '\','/');
    gt_name = frames(i).name;

    map_gt_path = strrep(gt_fold,'/videoframes_test', '/annotation_test');
    map_gt_path = [map_gt_path, '/maps/'];

    fix_gt_path = strrep(gt_fold,'/videoframes_test', '/annotation_test');
    fix_gt_path = [fix_gt_path, '/fixation/maps/'];
    map_eval_path = strrep(gt_fold, options.DS_GT_DIR, options.SALIENCY_DIR);

    % Obtenemos el nombre de la imagen de fijacion y la ajustamos
    gt_fixation_name = gt_name(1:end-4);
    %disp(gt_fixation_name);
    gt_fixation_number = str2double(gt_fixation_name) + 1;
    gt_fixation_name = sprintf('%04d', gt_fixation_number);


    gt_fixation_name_png = [gt_fixation_name, '.png'];
    saliency_path = [map_gt_path, gt_fixation_name_png]; 


    gt_fixation_name_mat = [gt_fixation_name, '.mat'];

    fixation_path = [fix_gt_path, strrep(gt_name, postfixFrames, '.mat')];
    fixation_path = strrep(fixation_path, strrep(gt_name, postfixFrames, '.mat'), gt_fixation_name_mat);

    % for first frame print all paths to check if they are formed correctly
    if i == 1
        disp(saliency_path);
        disp(fixation_path);
        disp(map_eval_path);
    end

    load(fixation_path);
      
    result = double(imread([map_eval_path, '/' , gt_name]));

    result = result(:,:,1);
    result = imresize(result, [size(I,1) size(I,2)]);
    if reqSaliency && exist(saliency_path, 'file')
        I = double(imread(saliency_path))/255;
        
        allMetrics(i) = fh( result, I);

    elseif reqFixation && exist(fixation_path, 'file')
        data = load(fixation_path, 'I');
        I = double(data.I);
        allMetrics(i) = fh( result, I);
 
    elseif strcmp(metricName, 'AUC_shuffled')
        if exist(fixation_path, 'file')
            data = load(fixation_path, 'I');
            I = data.I > 0;
            I = double(I);
            ids = randsample(length(frames), min(10,length(frames)));
            fixation_point = zeros(0,2);
            for k = 1:min(10,length(frames))
                fx_name = frames(ids(k)).name;
                fx_fold = frames(ids(k)).folder;
                fx_fold = strrep(fx_fold, '\','/');

                %fix_path = strrep(fx_fold,'/images', '/fixation/maps/');
                fix_path = strrep(fx_fold,'/videoframes_test', '/annotation_test');
                fix_path = [fix_path, '/fixation/maps/'];
                
                gt_fixation_namex = fx_name(1:end-4);
                gt_fixation_numberx = str2double(gt_fixation_namex) + 1;
                gt_fixation_namex = sprintf('%04d', gt_fixation_numberx);
                gt_fixation_namex = [gt_fixation_namex, '.mat'];
                
                fixation_pathx = [fix_path, strrep(fx_name, postfixFrames, '.mat')];
                fixation_pathx = strrep(fixation_pathx, strrep(fx_name, postfixFrames, '.mat'), gt_fixation_namex);
                
                Ix = load(fixation_pathx);
                Ix = double(Ix.I);
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
end
allMetrics(isnan(allMetrics)) = [];
meanMetric = mean(allMetrics);
end

