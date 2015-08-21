function batch_crop(src_folder, dst_folder, jitter)

t_begin = clock;
fprintf('Collecting *.png images in \"%s\" folder...', src_folder);
src_image_list = rdir([src_folder '/**/*.png']);
image_num = length(src_image_list);
t_end = clock;
fprintf('done (%d images, %f seconds)!\n', image_num, etime(t_end, t_begin));

fprintf('Start croping at time %s...it takes for a while!!\n', datestr(now, 'HH:MM:SS'));
poolobj=parpool;
report_num = 80;
fprintf(['\n' repmat('.',1,report_num) '\n\n']);
report_step = floor((image_num+report_num-1)/report_num);
t_begin = clock;
%for i = 1:length(src_image_list)
parfor i = 1:image_num
    src_image_file = src_image_list(i).name;
    try
        [I, ~, alpha] = imread(src_image_file);       
    catch
        fprintf('Failed to read %s\n', src_image_file);
    end

    [alpha, top, bottom, left, right] = crop_gray(alpha, 0, jitter);      
    I = I(top:bottom, left:right, :);

    if numel(I) == 0
        fprintf('Failed to crop %s (empty image after crop)\n', src_image_file);
    else
        dst_image_file = strrep(src_image_file, src_folder, dst_folder);
        [dst_image_file_folder, ~, ~] = fileparts(dst_image_file);
        if ~exist(dst_image_file_folder, 'dir')
            mkdir(dst_image_file_folder);
        end
        imwrite(I, dst_image_file, 'png', 'Alpha', alpha);
    end
    
    if mod(i, report_step) == 0
        fprintf('\b|\n');
    end
end      
delete(poolobj);
t_end = clock;
fprintf('%f seconds spent on cropping!\n', etime(t_end, t_begin));
