function [ image_list ] = collect_image_list( folder, shape_list_file )
%COLLECT_IMAGE_LIST Summary of this function goes here
%   Detailed explanation goes here

t_begin = clock;
fprintf('Collecting synthetic images of shapes listed in \"%s\"...', shape_list_file);
shape_list_fid = fopen(shape_list_file);
line = fgetl(shape_list_fid);
image_count = 0;
while ischar(line)
    shape_property = strsplit(line, ' ');
    shape_images_folder = fullfile(folder, shape_property{1}, shape_property{2});
    image_count = image_count + length(dir(fullfile(shape_images_folder, '*.png')));
    line = fgetl(shape_list_fid);
end
fclose(shape_list_fid);

image_list = cell(image_count, 1);
shape_list_fid = fopen(shape_list_file);
line = fgetl(shape_list_fid);
image_count = 0;
while ischar(line)
    shape_property = strsplit(line, ' ');
    shape_images_folder = fullfile(folder, shape_property{1}, shape_property{2});
    shape_images = extractfield(dir(fullfile(shape_images_folder, '*.png')), 'name');
    shape_image_count = length(shape_images);
    for i = 1:shape_image_count
        image_list{image_count+i} = fullfile(shape_images_folder, shape_images{i});
    end
    image_count = image_count + shape_image_count;
    line = fgetl(shape_list_fid);
end
fclose(shape_list_fid);
t_end = clock;
fprintf('done (%d images, %f seconds)!\n', image_count, etime(t_end, t_begin));

end

