function pr_curve_all = get_pr_curve_all(distMat, gt, imlist)
% distMat: n*n distance matrix, n is total image number
% gt: M*n binary matrix, M is total class number
% imlist: you can choose to only evaluate performance on a subset of images; k*1 vector for the selected indices;

sampling = [0.001:0.002:1]; % pr curve granularity
pr_curve = zeros(1, length(sampling));
distMat_here = distMat(imlist,imlist);
nImgs_selected = length(imlist);
pr_curve_all = zeros(nImgs_selected, length(sampling));
for i = 1:length(imlist)
    fprintf(1, '%d\n', i);
    [sorted indices] = sort(distMat_here(i,:),'ascend');
    query_gt = gt(:,imlist(i));
    
    precision = zeros(1, size(distMat_here,1));
    for j = 1:nImgs_selected
        result = gt(:, imlist(indices(j)));
        if sum(query_gt.*result) > 0
            precision(j) = 1;
        end
    end
    precision = precision(1:find(precision == 1, 1,'last'));
    recall = cumsum(precision)/sum(precision); 
    precision = cumsum(precision)./[1:length(precision)];
    removelist = [];
    for j = 2:length(recall)
        if recall(j) == recall(j-1)
            removelist = [removelist j];
        end
    end
    list = setdiff(1:length(recall), removelist);
    recall = recall(list); precision = precision(list);
    resampled = interp1(recall, precision, sampling);
    resampled(isnan(resampled)) = 1;
    pr_curve = pr_curve + resampled;
    pr_curve_all(i,:) = pr_curve;
end

