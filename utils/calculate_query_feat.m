function query_feat = calculate_query_feat(query_dir)

% load parameters for bow descriptor
codebooksize = 350;
codebook = importdata(['utils/BOW_params/codebook_' num2str(codebooksize) '.mat']);
par = importdata(['utils/BOW_params/params_' num2str(codebooksize) '.mat']);
w2c = importdata('utils/BOW_params/w2c.mat'); % used in CN extraction

query_files = dir([query_dir '*.jpg']);
% bow feature extraction
if ~exist('cache/bow_query.mat');
    query_feat = zeros(5600, length(query_files));
    for n = 1:length(query_files)
        n
        box_img = imread([query_dir query_files(n).name]);
        box_img = imresize(box_img, [128, 64]);
        descriptor = calculateDescriptor(box_img, par, w2c, codebook, 'CN');
        query_feat(:, n) = descriptor;
    end
    sum_val = sqrt(sum(query_feat.^2));
    sum_val = repmat(sum_val, [size(query_feat,1), 1]);
    query_feat = query_feat./sum_val;
    query_feat = single(query_feat);
    save('cache/bow_query.mat', 'query_feat', '-v7.3');
else
    query_feat = single(importdata('cache/bow_query.mat'));
end