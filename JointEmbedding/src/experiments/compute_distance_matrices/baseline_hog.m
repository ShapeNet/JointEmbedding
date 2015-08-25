%%
test_img_filelist = importdata('/orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_filelist.txt');
load('/orions3-zfs/projects/haosu/Image2Scene/code/matlab/evaluation/exact_match_chairs_img_hog.mat');
load('/orions3-zfs/projects/haosu/Image2Scene/code/matlab/evaluation/pure_img_hog.mat');
modelFeatures = pure_features';

% addpath(genpath('/orions4-zfs/software/matlab-package/vlfeat-0.9.20/toolbox'));
% %%
% kdtree = vl_kdtreebuild(modelFeatures);
% 
% %%
% for i = 1:length(filelist)
%     [ind, dist] = vl_kdtreequery(kdtree, modelFeatures, query_features(1, :)');
% end


addpath('/orions3-zfs/projects/rqi/Code/extract_feature/');
test_img_3dviews = importdata('/orions3-zfs/projects/rqi/Code/new_extract_feature/shape2image_and_image2shape_evaluation/exact_match_estimated_3dview.txt');
[closest_view_indices, closest_views] = get_closest_views(test_img_3dviews(:,1:2));

%% WITH VIEW ORACLE
% addpath(genpath('/orions-zfs/software/matlab-package/sltoolbox_r101/sltoolbox/'));
D = zeros(length(test_img_filelist), 6777);
for i = 1:length(test_img_filelist)
    tic;
    i
    selects = linspace(closest_view_indices(i),closest_view_indices(i)+6776*100,6777);
    dist = pdist2(query_features(i, :), modelFeatures(:,selects)');
    D(i, :) = dist;
    toc;
end

dlmwrite('./hogImageModelDist_withViewOracle.txt', D, 'delimiter', ' ');


%% WITHOUT VIEW ORACLE
% addpath(genpath('/orions-zfs/software/matlab-package/sltoolbox_r101/sltoolbox/'));
D = zeros(length(test_img_filelist), size(modelFeatures, 2));
for i = 1:length(test_img_filelist)
    tic;
    i
    dist = pdist2(query_features(i, :), modelFeatures');
    D(i, :) = dist;
    toc;
end

%%
% load('./hogImageModelDist.mat');
imageModelDist = zeros(length(test_img_filelist), 6777);
bestModelView = zeros(length(test_img_filelist), 6777);
for i = 1:length(test_img_filelist)
    i
    distToModel = D(i, :);
    distToModel = reshape(distToModel, [100, 6777]); 
    [distToModel, bestview] = min(distToModel, [], 1);
    imageModelDist(i, :) = distToModel;
    bestModelView = bestview;
end

dlmwrite('./hogImageModelDist_withoutViewOracle.txt', imageModelDist, 'delimiter', ' ');











%%
%fp = fopen('./test_img_filelist_with_path.txt', 'w');
%for i = 1:length(test_img_filelist)
%    fprintf(fp, '%s\n', fullfile('/orions3-zfs/projects/rqi/Data/Chair_Imgs_IKEA_rqi_cropped_resized/', test_img_filelist{i}));
%end
%fclose(fp);
%%
% python visualize_model_ranking.py -d /orions3-zfs/projects/haosu/Image2Scene/code/matlab/evaluation/hogImageModelDist.txt -q /orions3-zfs/projects/haosu/Image2Scene/code/matlab/evaluation/IKEA_filelist_with_path.txt -m /orions3-zfs/projects/haosu/Image2Scene/data/pretrained_models/filelist_chair_6777.txt -e /orions3-zfs/projects/haosu/Image2Scene/data/data_chair_exemplars_7k -s /orions3-zfs/projects/haosu/Image2Scene/data/baseline_hog -k 10
%%%%%%%%%%%%%%%%%
%%
% hao_filelist = importdata('/orions3-zfs/projects/haosu/Image2Scene/data/chair_hao_filelist.txt');
% D = zeros(length(hao_filelist), size(modelFeatures, 2));
% for i = 1:length(hao_filelist)
%     tic;
%     i
%     dist = slmetric_pw(query_features(i, :)', modelFeatures, 'eucdist');
%     D(i, :) = dist;    
%     toc;
% end
% imageModelDist = zeros(length(hao_filelist), 6777);
% bestModelView = zeros(length(hao_filelist), 6777);
% for i = 1:length(hao_filelist)
%     distToModel = D(i, :);
%     distToModel = reshape(distToModel, [100, 6777]); 
%     [distToModel, bestview] = min(distToModel, [], 1);
%     imageModelDist(i, :) = distToModel;
%     bestModelView = bestview;
% end
% dlmwrite('./hogImageModelDist_haochair.txt', imageModelDist, 'delimiter', ' ');
% python visualize_model_ranking.py -d /orions3-zfs/projects/haosu/Image2Scene/code/matlab/evaluation/hogImageModelDist_haochair.txt -q /orions3-zfs/projects/haosu/Image2Scene/data/chair_hao_filelist.txt -m /orions3-zfs/projects/haosu/Image2Scene/data/pretrained_models/filelist_chair_6777.txt -e /orions3-zfs/projects/haosu/Image2Scene/data/data_chair_exemplars_7k -s /orions3-zfs/projects/haosu/Image2Scene/data/baseline_hog_haochair -k 10
