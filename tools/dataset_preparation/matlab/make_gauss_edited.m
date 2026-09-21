function my_mask = make_gauss_edited(x, y, my_size, fixTime)
    if (nargin < 4) 
        fixTime = ones(size(x));
    end

    % Parameters
    REF_WIDTH = 640; % Reference width for scaling
    current_width = my_size(2); % Get current video width
    scale_factor = current_width / REF_WIDTH;
    
    W = 1; %Full width at half max in visual degrees
    W = W * 30 * scale_factor; %Scale based on video width (30 pixels per degree at 640 width)
  
    my_sigma = W/(2*sqrt(2*log(2))); %Convert to variance

    R = my_size(1);
    C = my_size(2);
    big_R = 2*R+1;
    big_C = 2*C+1;

    [X, Y] = meshgrid(0:big_C-1, 0:big_R-1);
    my_mask = zeros(R, C);

    temp_num = (X-C).^2 + (Y-R).^2;
    temp = exp(-temp_num/(2*my_sigma^2));
    big_gauss = temp/(2*pi*my_sigma);

    for i = 1:length(x)
        % Ensure coordinates are within bounds
        temp_x = max(1, min(C, round(x(i))));
        temp_y = max(1, min(R, round(y(i))));
        
        % Calculate valid ranges for indexing
        y_start = max(1, R-temp_y);
        y_end = min(big_R, R-temp_y+R-1);
        x_start = max(1, C-temp_x);
        x_end = min(big_C, C-temp_x+C-1);
        
        % Calculate corresponding ranges for output mask
        out_y_start = max(1, temp_y-(R-1));
        out_x_start = max(1, temp_x-(C-1));
        
        % Extract and add valid portion of gaussian
        temp = big_gauss(y_start:y_end, x_start:x_end);
        mask_height = size(temp, 1);
        mask_width = size(temp, 2);
        
        my_mask(out_y_start:out_y_start+mask_height-1, ...
                out_x_start:out_x_start+mask_width-1) = ...
                my_mask(out_y_start:out_y_start+mask_height-1, ...
                       out_x_start:out_x_start+mask_width-1) + temp*fixTime(i);
    end

    % Normalize
    my_mask = my_mask./max(my_mask(:));
end