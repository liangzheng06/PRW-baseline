function train_feat = calculate_train_feat(img_dir, dpm_train, img_index_train, ID_cam_train, ID_cam_train0)

% load parameters for bow descriptor
codebooksize = 350;
codebook = importdata(['utils/BOW_params/codebook_' num2str(codebooksize) '.mat']);
par = importdata(['utils/BOW_params/params_' num2str(codebooksize) '.mat']);
w2c = importdata('utils/BOW_params/w2c.mat'); % used in CN extraction

% bow feature extraction
if ~exist('cache/dpm_bow_train.mat')
    train_feat = zeros(5600, size(ID_cam_train, 1));
    count = 0;
    for n = 1:length(img_index_train)
        n
        box = dpm_train{n}; % dpm_train corresponds to ID_cam_train0, not ID_cam_train
        pos = find(ID_cam_train0(:, 4) == n);
        img = imread([img_dir img_index_train{n} '.jpg']);
        for m = 1:length(pos)
            if ID_cam_train0(pos(m), 1) > 0
                count = count + 1;
                coord = box(m, 1:4);
                box_img = imcrop(img, [coord(1), coord(2), max(1, coord(3)-coord(1)), max(1, coord(4)-coord(2))]);
                box_img = imresize(box_img, [128, 64]);
                descriptor = calculateDescriptor(box_img, par, w2c, codebook, 'CN');
                train_feat(:, count) = descriptor;
            end
        end
    end
    train_feat = single(train_feat);
    save('cache/dpm_bow_train.mat', 'train_feat');
else
    train_feat = importdata('cache/dpm_bow_train.mat');
end
% l2 normalization 
sum_val = sqrt(sum(train_feat.^2));
sum_val = repmat(sum_val, [size(train_feat,1), 1]);
train_feat = train_feat./sum_val;