addpath('../');
global_variables;

%% Load HoG features
t_begin = clock;
fprintf('Loading HoG features from \"%s\"...', g_lfd_hog_features_file);
load(g_lfd_hog_features_file);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Compute distance matrix
t_begin = clock;
fprintf('Computing shape distance matrix, it takes for a while...');
shape_distance_matrix = pdist(lfd_hog_features);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Save shape distance matrix
t_begin = clock;
fprintf('Save shape distance matrix to \"%s\"...', g_shape_distance_matrix_file);
save(g_shape_distance_matrix_file, 'shape_distance_matrix', '-v7.3');
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

exit;
