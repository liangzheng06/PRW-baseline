% Training code on PRW dataset with BoW and XQDA

% system paths, data loading
addpath 'utils/LOMO_XQDA/bin'
addpath 'utils/LOMO_XQDA/code'
addpath 'utils'
anno_dir = 'PRW/annotations/';
img_dir = 'PRW/frames/';
img_index_train = importdata('PRW/frame_train.mat');
dpm_train = importdata('data/dpm_train.mat'); % load training dpm boxes

% calculate the ID and camera of each detected box in the training data
[ID_cam_train0, ID_cam_train] = calculate_ID_cam_train(dpm_train, anno_dir, img_index_train);

% calculate training features
train_feat = calculate_train_feat(img_dir, dpm_train, img_index_train, ID_cam_train, ID_cam_train0);

% generate pairwise training features for XQDA
[train_sample1, train_sample2, label1, label2] = gen_train_sample_xqda(ID_cam_train, train_feat);

% train XQDA
[W, M] = XQDA(train_sample1, train_sample2, label1, label2);
save('cache/bow_xqda_param.mat', 'M', 'W');


