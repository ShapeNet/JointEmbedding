function display_result(result_id)

BASEDIR = fullfile(mfilename('fullpath'),'../');
addpath(fullfile(BASEDIR, 'results'));

result_id_clutter = strcat(result_id,'_clutter');
result_id_all = strcat(result_id, '_all');

% image to shape
clutter_105models_accuracies = importdata(strcat(result_id_clutter,'_image2shape_topK_accuracy_105models.txt'));
figure, plot(1:30, clutter_105models_accuracies(1:30), 'r', 'LineWidth', 2);

% shape to image
fprintf('shape2image median rank of first and last matched images.\n');
all_105models_rank = importdata(strcat(result_id_all,'_first_last_appearance_median_rank.txt'))

end
