addpath('../');

global_variables;

for i = 1:length(g_shapenet_synset_set)
    synset = g_shapenet_synset_set{i};
    fprintf('Batch cropping \"%s\" to \"%s\" ...\n', fullfile(g_lfd_images_folder, synset), fullfile(g_lfd_images_cropped_folder, synset));
    batch_crop(fullfile(g_lfd_images_folder, synset), fullfile(g_lfd_images_cropped_folder, synset), 0);
end

exit;
