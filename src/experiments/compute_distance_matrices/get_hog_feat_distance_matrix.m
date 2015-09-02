clear all; close all; clc;

BASEDIR = fullfile(mfilename('fullpath'),'../');
EXACTMATCH_DATASET = fullfile(BASEDIR, '../ExactMatchChairsDataset');
addpath(fullfile(BASEIDR, '../../'));
global_variables;
addpath(genpath(g_piotr_toolbox_path));


%%
exact_match_chairs_filelist = importdata(fullfile(EXACTMATCH_DATASET, 'exact_match_chairs_img_filelist.txt'));
query_features = zeros(length(exact_match_chairs_filelist), 1764);
parfor i = 1:length(exact_match_chairs_filelist)
    display(i); 
    I = imread(exact_match_chairs_filelist{i});
    I = imresize(I, [60, 60]);
    h0 = hog(single(I));
    query_features(i, :) = h0(:);
end
%save(fullfile(BASEDIR, 'exact_match_chairs_img_hog.mat'), 'query_features');


%%
pure_img_filelist = importdata(fullfile(BASEDIR,'pure_img_filelist.txt'));
pure_features = zeros(length(pure_img_filelist), 1764);
parfor i = 1:length(pure_img_filelist)
    display(i);
    I = imread(pure_img_filelist{i});
    I = imresize(I, [60, 60]);
    h0 = hog(single(I));
    pure_features(i, :) = h0(:);
end
%save(fullfile(BASEDIR, 'pure_img_hog.mat'), 'pure_features');

% --------------------------------------------
% --------------------------------------------
%% compute distance matrix
test_img_filelist = importdata(fullfile(EXACTMATCH_DATASET, 'exact_match_chairs_img_filelist.txt'));
%load(fullfile(BASEDIR, 'exact_match_chairs_img_hog.mat'));
%load(fullfile(BASEDIR, 'pure_img_hog.mat'));
modelFeatures = pure_features';


%% WITHOUT VIEW ORACLE
D = zeros(length(test_img_filelist), size(modelFeatures, 2));
for i = 1:length(test_img_filelist)
    tic;
    i
    dist = pdist2(query_features(i, :), modelFeatures');
    D(i, :) = dist;
    toc;
end

%%
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

dlmwrite(fullfile(BASEDIR, 'hogImageModelDist_withoutViewOracle.txt'), imageModelDist, 'delimiter', ' ');
