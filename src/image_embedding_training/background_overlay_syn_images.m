addpath('../');

global_variables;

local_cluster = parcluster('local');
poolobj = parpool('local', min(g_syn_bkg_overlay_thread_num, local_cluster.NumWorkers));
for i = 1:length(g_shapenet_synset_set)
    synset = g_shapenet_synset_set{i};
    fprintf('Background overlaying \"%s\" to \"%s\" ...\n', fullfile(g_syn_images_cropped_folder, synset), fullfile(g_syn_images_bkg_overlaid_folder, synset));
    background_overlay(fullfile(g_syn_images_cropped_folder, synset), fullfile(g_syn_images_bkg_overlaid_folder, synset), g_syn_bkg_filelist, g_syn_bkg_folder, g_syn_cluttered_bkg_ratio);
end
delete(poolobj);

exit;
