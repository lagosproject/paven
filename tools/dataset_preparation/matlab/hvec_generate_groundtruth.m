%% HVEC / Eye-Tracking Dataset Ground Truth Generation
% Generates per-frame fixation heatmaps (JPEG) and discrete fixation matrices (.mat).
% Used for evaluation against predicted saliency maps (e.g. via HVECTest.m).

close all; clear; clc;
tic;

% Paths - adjust to local environment
DATABASE_MAT = 'video_database.mat';
OUTPUT_DIR = './ground_truth_fixations';

if ~exist(DATABASE_MAT, 'file')
    error('Eye-tracking database file not found: %s', DATABASE_MAT);
end

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

load(DATABASE_MAT, 'video_database');
num_videos = length(video_database.video_names_list);

for v = 1:num_videos
    video_name = video_database.video_names_list{v};
    fprintf('Processing video %d/%d: %s\n', v, num_videos, video_name);
    
    num_frames = video_database.videos_info.frames(v);
    fps = video_database.videos_info.framerate_fps(v);
    fix_data = video_database.fixdata(video_database.fixdata(:, 2) == v, :);
    video_size = video_database.videos_info.size(v, :); % [width, height]
    
    width = video_size(1);
    height = video_size(2);
    frame_duration_ms = 1000 / fps;
    
    fixation_per_frame = cell(1, num_frames);
    for k = 1:size(fix_data, 1)
        fix_pos = [fix_data(k, 5); fix_data(k, 6)];
        start_f = max(1, ceil(fix_data(k, 3) / frame_duration_ms));
        end_f = min(num_frames, ceil((fix_data(k, 3) + fix_data(k, 4)) / frame_duration_ms));
        
        for f = start_f:end_f
            fixation_per_frame{f} = [fixation_per_frame{f} fix_pos];
        end
    end
    
    % 1. Create binary discrete fixation matrix (.mat)
    fixmaps = zeros(height, width, num_frames);
    
    % Video-specific image folder
    video_out_dir = fullfile(OUTPUT_DIR, video_name);
    if ~exist(video_out_dir, 'dir')
        mkdir(video_out_dir);
    end
    
    for f = 1:num_frames
        if ~isempty(fixation_per_frame{f})
            x = fixation_per_frame{f}(1, :);
            y = fixation_per_frame{f}(2, :);
            
            % Clamp coordinates
            xc = max(1, min(width, round(x)));
            yc = max(1, min(height, round(y)));
            
            for i = 1:length(xc)
                fixmaps(yc(i), xc(i), f) = 1;
            end
            
            % 2. Create smooth continuous heatmap
            heatmap = make_gauss_edited(x, y, [height width]);
            heatmap = heatmap';
            heatmap = (heatmap - min(heatmap(:))) / (max(heatmap(:)) - min(heatmap(:)) + eps);
            heatmap_uint8 = uint8(heatmap * 255)';
            
            imwrite(heatmap_uint8, fullfile(video_out_dir, sprintf('frame_%04d.jpg', f)));
        end
    end
    
    % Save discrete binary fixation matrix
    mat_out = fullfile(OUTPUT_DIR, [video_name '_fixmaps.mat']);
    save(mat_out, 'fixmaps', '-v7.3');
end

toc;
fprintf('All ground truth heatmaps and fixation matrices generated successfully.\n');
