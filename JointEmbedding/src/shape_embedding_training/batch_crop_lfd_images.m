addpath('../');

global_variables;

local_cluster = parcluster('local');
poolobj = parpool('local', min(g_lfd_cropping_thread_num, local_cluster.NumWorkers));
for i = 1:length(g_shapenet_synset_set)
    synset = g_shapenet_synset_set{i};
    fprintf('Batch cropping \"%s\" to \"%s\" ...\n', fullfile(g_lfd_images_folder, synset), fullfile(g_lfd_images_cropped_folder, synset));
    batch_crop(fullfile(g_lfd_images_folder, synset), fullfile(g_lfd_images_cropped_folder, synset), 0);
end
delete(poolobj);

exit;
