function [ID_cam_gallery, miss] = calculate_ID_cam_test(box_test, anno_dir, img_index_test)

% calculate the ID of all the detected boxes
if ~exist('cache/ID_cam_dpm_gallery.mat', 'file')
    ID_cam_gallery = [];
    miss = [];
    for n = 1:length(img_index_test)
        if round(n/1000) == n/1000
            n
        end
        box_data = box_test{n};
        if isempty(box_data)
            continue;
        end
        anno_data = importdata([anno_dir img_index_test{n} '.jpg.mat']);
        anno_data(:, 4:5) = anno_data(:, 4:5) + anno_data(:, 2:3);
        cam = str2double(img_index_test{n}(2));
        flag = zeros(size(anno_data, 1), 1);
        ID_cam_tmp = zeros(size(box_data, 1), 2);
        ov_tmp = zeros(size(box_data, 1), 1);
        for m = 1:size(box_data, 1)
            o = boxoverlap(anno_data(:, 2:5), box_data(m, 1:4));
            [m_v, pos] = max(o);
            ov_tmp(m) = m_v;
            if m_v >= 0.5
                ID_cam_tmp(m, :) = [anno_data(pos, 1), cam];
                flag(pos) = 1;
            end
            if m_v < 0.5 && m_v >= 0.3
                ID_cam_tmp(m, :) = [-1, cam];
            end
            if m_v < 0.3
                ID_cam_tmp(m, :) = [0, cam];
            end
        end
        ID_cam_tmp(:, 3) = box_data(:, 5);
        ID_cam_tmp(:, 4) = n;
        ID_uni = unique(ID_cam_tmp(:, 1));
        for m = 1:length(ID_uni)
            pos = find(ID_cam_tmp(:, 1) == ID_uni(m) & ID_uni(m) > 0);
            if length(pos) > 1
                ov_now = ov_tmp(pos);
                [mm, pp] = max(ov_now);
                pos(pp) = [];
                ID_cam_tmp(pos, 1) = -1;
            end
        end
        ID_cam_gallery = [ID_cam_gallery; ID_cam_tmp];
        pos = find(flag == 0);
        anno_data = anno_data(pos, :);
        pos = find(anno_data(:, 1) > 0);
        anno_data = anno_data(pos, :);
        tmp = [anno_data(: ,1), ones(size(anno_data, 1), 1).*cam];
        miss = [miss; tmp];
    end
    save('cache/ID_cam_dpm_gallery.mat', 'ID_cam_gallery', 'miss');
else
    tmp = load('cache/ID_cam_dpm_gallery.mat');
    ID_cam_gallery = tmp.ID_cam_gallery;
    miss = tmp.miss; 
end
