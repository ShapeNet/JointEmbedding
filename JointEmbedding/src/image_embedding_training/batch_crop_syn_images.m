addpath('../');

global_variables;

local_cluster = parcluster('local');
poolobj = parpool('local', min(g_syn_cropping_thread_num, local_cluster.NumWorkers));
for i = 1:length(g_shapenet_synset_set)
    synset = g_shapenet_synset_set{i};
    fprintf('Batch cropping \"%s\" to \"%s\" ...\n', fullfile(g_syn_images_folder, synset), fullfile(g_syn_images_cropped_folder, synset));
    batch_crop(fullfile(g_syn_images_folder, synset), fullfile(g_syn_images_cropped_folder, synset), 1);
end
delete(poolobj);

exit;
