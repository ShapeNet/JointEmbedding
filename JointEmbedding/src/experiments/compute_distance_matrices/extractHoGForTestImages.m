addpath(genpath('/orions3-zfs/software/matlab-package/piotr_toolbox'));
%IKEA_filelist = importdata('/orions3-zfs/projects/rqi/Data/labels/Chair_Imgs_IKEA_rqi.txt');
%
%folder = '/orions3-zfs/projects/rqi/Data/Chair_Imgs_IKEA_rqi_cropped_resized';
%query_features = zeros(length(IKEA_filelist), 1764);
%parfor i = 1:length(filelist)
%    i    
%    I = imread(fullfile(folder, IKEA_filelist{i}));
%    I = imresize(I, [60, 60]);
%    h0 = hog(single(I));
%    query_features(i, :) = h0(:);    
%end
%save('./Chair_Imgs_IKEA_rqi_cropped_resized_hog.mat', 'query_features');
%
%%%
%hao_filelist = importdata('/orions3-zfs/projects/haosu/Image2Scene/data/chair_hao_filelist.txt');
%
%query_features = zeros(length(hao_filelist), 1764);
%for i = 1:length(hao_filelist)
%    i    
%    I = imread(hao_filelist{i});
%    I = imresize(I, [60, 60]);
%    h0 = hog(single(I));
%    query_features(i, :) = h0(:);    
%end
%save('./chair_hao_hog.mat', 'query_features');



%%
exact_match_chairs_filelist = importdata('/orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/exact_match_chairs_img_filelist.txt');
query_features = zeros(length(exact_match_chairs_filelist), 1764);
parfor i = 1:length(exact_match_chairs_filelist)
    i    
    I = imread(exact_match_chairs_filelist{i});
    I = imresize(I, [60, 60]);
    h0 = hog(single(I));
    query_features(i, :) = h0(:);
end
save('./exact_match_chairs_img_hog.mat', 'query_features');


%%
pure_img_filelist = importdata('/orions3-zfs/projects/rqi/Code/extract_feature/pure_img_filelist.txt');
pure_features = zeros(length(pure_img_filelist), 1764);
parfor i = 1:length(pure_img_filelist)
    i    
    I = imread(pure_img_filelist{i});
    I = imresize(I, [60, 60]);
    h0 = hog(single(I));
    pure_features(i, :) = h0(:);
end
save('./pure_img_hog.mat', 'pure_features');


