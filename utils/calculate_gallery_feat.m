function gallery_feat = calculate_gallery_feat(img_dir, dpm_test, img_index_test, ID_cam_gallery)

% load parameters for bow descriptor
codebooksize = 350;
codebook = importdata(['utils/BOW_params/codebook_' num2str(codebooksize) '.mat']);
par = importdata(['utils/BOW_params/params_' num2str(codebooksize) '.mat']);
w2c = importdata('utils/BOW_params/w2c.mat'); % used in CN extraction

if ~exist('cache/dpm_bow_gallery.mat');
    gallery_feat = zeros(5600, size(ID_cam_gallery, 1));
    count = 0;
    for n = 1:length(img_index_test)
        n
        box = dpm_test{n};
        img = imread([img_dir img_index_test{n} '.jpg']);
        for m = 1:size(box, 1)
            count = count + 1;
            coord = box(m, 1:4);
            box_img = imcrop(img, [coord(1), coord(2), max(1, coord(3)-coord(1)), max(1, coord(4)-coord(2))]);
            box_img = imresize(box_img, [128, 64]);
            descriptor = calculateDescriptor(box_img, par, w2c, codebook, 'CN');
            gallery_feat(:, count) = descriptor;
        end
    end
    sum_val = sqrt(sum(gallery_feat.^2));
    sum_val = repmat(sum_val, [size(gallery_feat,1), 1]);
    gallery_feat = gallery_feat./sum_val;
    gallery_feat = single(gallery_feat);
    save('cache/dpm_bow_gallery.mat', 'gallery_feat', '-v7.3');
else
    gallery_feat = importdata('cache/dpm_bow_gallery.mat');
end
