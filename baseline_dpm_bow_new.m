% Evaluation code of the PRW dataset.
% This code will evaluate the DPM detector (pre-trained on INRIA dataset, denoted as DPM_Inria in our paper) plus the BoW+XQDA recognizer.
% If you find our code helpful in your research, please kindly cite our
% paper as
% Liang Zheng, Hengheng Zhang, Shaoyan Sun, Manmohan Chandraker, Qi Tian,
% "Person Re-identification in the Wild",  arXiv:1604.02531, 2016.

% adding systems paths, loading detection results, training results
dpm_test = importdata('data/dpm_test.mat');
addpath 'utils/LOMO_XQDA/bin'
addpath 'utils/LOMO_XQDA/code'
addpath 'utils'
anno_dir = 'PRW/annotations/';
img_dir = 'PRW/frames/';
query_dir = 'PRW/query_box/';
img_index_train = importdata('PRW/frame_train.mat');
img_index_test = importdata('PRW/frame_test.mat');
data = load('cache/bow_xqda_param.mat');
M = data.M;
W = data.W;

% calculate the ID and camera of each detected box in the gallery
[ID_cam_gallery, miss_ID_cam] = calculate_ID_cam_test(dpm_test, anno_dir, img_index_test);

% calculate the ID and camera of each image in the query folder
ID_cam_query = calculate_ID_cam_query(query_dir);

% calculate query features
query_feat = calculate_query_feat(query_dir);

% calculate gallery features
gallery_feat = calculate_gallery_feat(img_dir, dpm_test, img_index_test, ID_cam_gallery);

% calculate MahDist between query and gallery boxes with learnt subspace
dist = MahDist(M, gallery_feat' * W, query_feat' * W); % smaller distance means larger similarity

% prepare some data for evaluation
testID = ID_cam_gallery(:, 1);
queryID = ID_cam_query(:, 1);
testCAM = ID_cam_gallery(:, 2);
queryCAM = ID_cam_query(:, 2);

% detection thresholds used for gallery generation
piece_thre = 30;
[m,n_gallery] = sort(ID_cam_gallery(:, 3) );
thre = floor(size(n_gallery,1)/piece_thre);

% evaluation for each query
for k = 1:length(ID_cam_query)
    k
    % calculate ground truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    miss_index = intersect(find(miss_ID_cam(:, 1) == queryID(k)), find(miss_ID_cam(:, 2) ~= queryCAM(k))); % ground truth images missed by detection
    miss_index = miss_index + 1000000; % A large index is assigned to ground truths that have been missed by detection
    good_index = [good_index, miss_index'];
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index3 = find(testID == -2); % images with ambiguous ID (-2 in the paper)
    junk_index = [junk_index1; junk_index2; junk_index3]'; % junk images have zero impact on re-id accuracy
    
    tic
    score = dist(:, k);
    
    for i = 1:piece_thre% for each detection threshold, we calculate ap and cmc
        pos = n_gallery(1: (i-1)*thre-1, :);
        score2 = score;
        score2(pos) = 1000;
        score2(junk_index) = 5000;
        [A, index] = sort(score2, 'ascend');
        index = index(A < 1000);
        [ap{i}(k), cmc{i}(:, k)] = compute_AP(good_index, index);
    end
    toc
end

% calculate mAP and cmc against # windows per image
map = zeros(piece_thre, 1);
cmc_avg = cell(piece_thre, 1);
r1 = zeros(piece_thre, 1);
r20 = zeros(piece_thre, 1);
nwindow = zeros(piece_thre, 1);
for i = 1: piece_thre
    pos2 = n_gallery(max((i-1)*thre, 1) : end, :);
    nwindow(i) = length(pos2)/length(dpm_test);
    map(i) = mean(ap{i});
    cmc_avg{i} = mean(cmc{i}, 2);
    r1(i) = cmc_avg{i}(1);
    r20(i) = cmc_avg{i}(20);
end

% draw figures
figure;
plot(nwindow, map);
figure;
plot(nwindow, r1);
figure;
plot(nwindow, r20);




