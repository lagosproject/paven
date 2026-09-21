%% LEDOV Dataset - Gaussian Saliency Mask Generation
% Generates per-frame saliency maps from gaze fixation records.
% Based on the methodology in Pablo Fernández Lagos's Master's Thesis (UPM).

clear; close all; clc;
tic;

% Path configurations - Update these for your environment
LEDOV_ROOT = './LEDOV';
VIDEO_NAME_LIST_FILE = fullfile(LEDOV_ROOT, 'VideoNameList.mat');
FIX_SIZE_GAUSS = 100; % Window radius for Gaussian kernel

if ~exist(VIDEO_NAME_LIST_FILE, 'file')
    error('VideoNameList.mat not found at: %s', VIDEO_NAME_LIST_FILE);
end

load(VIDEO_NAME_LIST_FILE, 'VideoNameList');
num_videos = length(VideoNameList);
fprintf('Found %d videos in dataset\n', num_videos);

for m = 1:num_videos
    video_short = VideoNameList{m};
    mat_file = fullfile(LEDOV_ROOT, video_short, 'Data.mat');
    
    if ~exist(mat_file, 'file')
        fprintf('Warning: %s not found. Skipping...\n', mat_file);
        continue;
    end
    
    load(mat_file, 'Data');
    
    output_dir = fullfile(LEDOV_ROOT, video_short, 'maps');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    num_frames = Data.VideoFrames;
    fps = Data.VideoFrameRate;
    video_size = Data.VideoSize;
    width = video_size(1);
    height = video_size(2);
    video_fixation = Data.fixdata;
    
    frame_duration_ms = 1000 / fps;
    fixation_per_frame = cell(1, num_frames);
    centermask = round(height / 20);
    begin_flag = 1;
    
    for k = 1:size(video_fixation, 1)
        if k == 1 || video_fixation(k, 2) < video_fixation(k-1, 2)
            begin_flag = 1;
        end
        
        vx = video_fixation(k, 4);
        vy = video_fixation(k, 5);
        
        if begin_flag
            if (vx > (width/2 + centermask) || vx < (width/2 - centermask)) && ...
               (vy > (height/2 + centermask) || vy < (height/2 - centermask)) && ...
               (vx > 0 && vx < width) && (vy > 0 && vy < height)
                fix_pos = [vx; vy];
                begin_flag = 0;
            else
                continue;
            end
        else
            if (vx > 0 && vx < width) && (vy > 0 && vy < height)
                fix_pos = [vx; vy];
            else
                continue;
            end
        end
        
        start_frame = max(1, ceil(video_fixation(k, 2) / frame_duration_ms));
        end_frame = min(num_frames, ceil((video_fixation(k, 2) + video_fixation(k, 3)) / frame_duration_ms));
        
        for f = start_frame:end_frame
            fixation_per_frame{f} = [fixation_per_frame{f} fix_pos];
        end
    end
    
    % Generate and save per-frame saliency maps
    for f = 1:num_frames
        if ~isempty(fixation_per_frame{f})
            x = fixation_per_frame{f}(1, :);
            y = fixation_per_frame{f}(2, :);
            mask = make_gauss_masks4(x, y, [height width], FIX_SIZE_GAUSS);
            mask = mask ./ max(mask(:));
            mask_gray = mat2gray(mask);
            
            out_file = fullfile(output_dir, sprintf('%04d.jpg', f));
            imwrite(mask_gray, out_file);
        end
    end
    
    fprintf('Processed video %d/%d: %s\n', m, num_videos, video_short);
end

toc;
fprintf('Mask generation complete.\n');
