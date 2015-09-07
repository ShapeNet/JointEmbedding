addpath('../');
global_variables;

%% Load shape distance matrix
t_begin = clock;
fprintf('Loading shape distance matrix from \"%s\"...', g_shape_distance_matrix_file_mat);
load(g_shape_distance_matrix_file_mat);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

shape_distance_matrix = squareform(shape_distance_matrix);

shapenet_synset_set_handle_mirror = [g_shapenet_synset_set_handle '_' g_mirror_name];
shape_distance_matrix_file_mat_mirror = fullfile(g_data_folder, ['shape_embedding/shape_distance_matrix' shapenet_synset_set_handle_mirror '.mat']);
shape_distance_matrix_file_txt_mirror = fullfile(g_data_folder, ['shape_embedding/shape_distance_matrix' shapenet_synset_set_handle_mirror '.txt']);

shape_list_mirror_mapping_filename = fullfile(g_data_folder, ['shape_list_mapping' shapenet_synset_set_handle_mirror '.txt']);
shape_list_mirror_mapping = logical(importdata(shape_list_mirror_mapping_filename));
shape_distance_matrix = shape_distance_matrix(shape_list_mirror_mapping, shape_list_mirror_mapping);
shape_distance_matrix = squareform(shape_distance_matrix);

%% Save shape distance matrix
t_begin = clock;
fprintf('Save shape distance matrix to \"%s\"...', shape_distance_matrix_file_mat_mirror);
save(shape_distance_matrix_file_mat_mirror, 'shape_distance_matrix', '-v7.3');
shape_distance_matrix_NxN = squareform(shape_distance_matrix);
dlmwrite(shape_distance_matrix_file_txt_mirror, shape_distance_matrix_NxN, 'delimiter', ' ');
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

exit;
