addpath('../');

global_variables;

%% Collect LFD images according to shape list
image_list = collect_image_list(g_lfd_images_folder, g_shape_list_file);

local_cluster = parcluster('local');
poolobj = parpool('local', min(g_lfd_cropping_thread_num, local_cluster.NumWorkers));
fprintf('Batch cropping LFD images from \"%s\" to \"%s\" ...\n', g_lfd_images_folder, g_lfd_images_cropped_folder);
batch_crop(fullfile(g_lfd_images_folder, synset), fullfile(g_lfd_images_cropped_folder, synset), 0, image_list);
delete(poolobj);

exit;
