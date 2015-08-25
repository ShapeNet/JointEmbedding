addpath('../');
global_variables;

addpath(genpath(g_piotr_toolbox_path));

%% collect all LFD images
t_begin = clock;
fprintf('Collecting LFD images of shapes listed in \"%s\"...', g_shape_list_file);
view_image_lists = cell(g_lfd_view_num, 1);
shape_list_fid = fopen(g_shape_list_file);
line = fgetl(shape_list_fid);
shape_count = 0;
while ischar(line)
    shape_count = shape_count + 1;
    shape_property = strsplit(line, ' ');
    lfd_images_folder = fullfile(g_lfd_images_cropped_folder, shape_property{1}, shape_property{2});
    lfd_images_struct = dir(fullfile(lfd_images_folder, '*.png'));
    lfd_images_cell = struct2cell(lfd_images_struct);
    lfd_images_sorted = sort(lfd_images_cell(1, :));
    for i = 1:g_lfd_view_num
        view_image_lists{i, shape_count} = fullfile(lfd_images_folder, lfd_images_sorted{i});
    end
    line = fgetl(shape_list_fid);
end
fclose(shape_list_fid);
t_end = clock;
fprintf('done (%d shapes, %f seconds)!\n', shape_count, etime(t_end, t_begin));

%% try to get dimension of the HoG feature
hog_dimension = numel(extract_pyramid_hog(view_image_lists{1, 1}, g_lfd_hog_image_size));
fprintf('Each shape will be converted into HoG feature of %d dimensions!\n', hog_dimension);

%% compute the HoG feature for the LFD images
fprintf('Start LFD HoG feature extraction at time...it takes for a while!!\n', datestr(now, 'HH:MM:SS'));
poolobj=parpool('local', g_lfd_cropping_thread_num);
report_num = 80; 
report_step = floor((shape_count+report_num-1)/report_num);
t_begin = clock;
view_hog_features = cell(g_lfd_view_num, 1);
for i = 1:g_lfd_view_num
    fprintf('Extracting HoG feature from LFD images of view %d of %d...\n', i, g_lfd_view_num);
    fprintf([repmat('.', 1, report_num) '\n']);
    view_hog_feature = zeros(shape_count, hog_dimension);
    parfor j = 1:shape_count
        view_hog_feature(j, :) = extract_pyramid_hog(view_image_lists{i, j}, g_lfd_hog_image_size);
        if mod(j, report_step) == 0
            fprintf('\b|');
        end
    end    
    fprintf('\n');
    view_hog_features{i} = view_hog_feature;
end
delete(poolobj);
t_end = clock;
fprintf('%f seconds spent on LFD HoG feature extraction!\n', etime(t_end, t_begin));


%% save HoG features
t_begin = clock;
fprintf('Save HoG features to \"%s\"...', g_lfd_hog_features_file);
lfd_hog_features = cat(2, view_hog_features{:});
clearvars view_hog_features;
save(g_lfd_hog_features_file, 'lfd_hog_features', '-v7.3');
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

exit;
