clear all; close all; clc;

BASEDIR = fullfile(mfilename('fullpath'),'../');
addpath(fullfile(BASEDIR, 'results'));

% Image to shape results
sammon100_clutter_105models = importdata('sammon100_clutter_image2shape_topK_accuracy_105models.txt');
hog_without_view_clutter_oracle_105models = importdata('hog_withoutview_clutter_image2shape_topK_accuracy_105models.txt');
cnn_pool5_without_view_clutter_105models = importdata('cnn_pool5_withoutview_clutter_image2shape_topK_accuracy_105models.txt');
cnn_fc7_without_view_clutter_105models = importdata('cnn_fc7_withoutview_clutter_image2shape_topK_accuracy_105models.txt');
siamese_clutter_105models = importdata('siamese_20view_clutter_image2shape_topK_accuracy_105models.txt');


figure, 
plot(1:30, sammon100_clutter_105models(1:30), 'r', 'LineWidth', 2); hold on;
plot(1:30,hog_without_view_clutter_oracle_105models(1:30), 'b', 'LineWidth',2);
plot(1:30, cnn_pool5_without_view_clutter_105models(1:30), 'k', 'LineWidth', 2);
plot(1:30, cnn_fc7_without_view_clutter_105models(1:30), 'c', 'LineWidth', 2);
plot(1:30, siamese_clutter_105models(1:30), 'g', 'LineWidth', 2);
plot(1:30, [1:30]/105, 'k--', 'LineWidth', 1);
xlabel('Top-k','FontName', 'Times New Roman', 'FontSize', 14);
ylabel('Accuracy', 'FontName', 'Times New Roman', 'FontSize', 14);
mylegend =legend('Embedding (ours)', 'HoG', 'AlexNet pool5', 'AlexNet fc7', 'Siamese embedding', 'Chance','FontName', 'Times New Roman', 'FontSize', 14);
set(mylegend,'FontSize',12, 'FontName', 'Times New Roman');

% Shape to image results
fprintf('shape2image median rank of first and last matched images.\n');
sammon100_all_105models_rank = importdata('sammon100_tmp_first_last_appearance_median_rank.txt')
hog_all_105models_rank = importdata('hog_withoutview_tmp_first_last_appearance_median_rank.txt')
cnn_pool5_all_105models_rank = importdata('cnn_pool5_withoutview_tmp_first_last_appearance_median_rank.txt')
cnn_fc7_all_105models_rank = importdata('cnn_fc7_withoutview_tmp_first_last_appearance_median_rank.txt')
siamese_all_105models_rank = importdata('siamese_20view_tmp_first_last_appearance_median_rank.txt')
