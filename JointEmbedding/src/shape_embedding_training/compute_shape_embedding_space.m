addpath('../');
global_variables;

%% Load shape distance matrix
t_begin = clock;
fprintf('Loading shape distance matrix from \"%s\"...', g_shape_distance_matrix_file);
load(g_shape_distance_matrix_file);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

%% Compute shape embedding space
t_begin = clock;
fprintf('Computing shape embedding space, it takes for a while...');
options = statset('Display', 'iter', 'MaxIter', 32);
[shape_embedding_space, stress, disparities] = mdscale(shape_distance_matrix, g_shape_embedding_space_dimension, 'criterion', 'sammon', 'options', options); 
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

%% Save embedding space
t_begin = clock;
fprintf('Save shape embedding space to \"%s\"...', g_shape_embedding_space_file_mat);
save(g_shape_embedding_space_file_mat, 'shape_embedding_space', '-v7.3');
dlmwrite(g_shape_embedding_space_file_txt, shape_embedding_space, 'delimiter', ' ');
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

exit;
