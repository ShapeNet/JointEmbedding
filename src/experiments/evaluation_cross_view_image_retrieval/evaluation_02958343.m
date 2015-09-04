evaluation_variables;

embedding = importdata(fullfile(evaluation_folder, 'image_embeddings_02958343.txt'));
num_imgs = size(embedding, 1);

%% construct ranking matrix with L2 distances
tic;
ranking_mat = pdist(embedding);
ranking_mat = squareform(ranking_mat);
ranking_mat = -ranking_mat;
toc;

%%
load(fullfile(evaluation_folder, 'ground_truth_02958343.mat'));
image_list = [1:num_imgs];

%% compute pr curve
pr_curve_all = get_pr_curve_all(-ranking_mat, ground_truth, image_list);
avg_pr_curve_all = pr_curve_all(end, :) / length(image_list);
figure(1);
hold on;
plot([0.001:0.002:1], avg_pr_curve_all);
auc = trapz([0.001:0.002:1], avg_pr_curve_all);
fprintf('AUC: %f\n', auc);
