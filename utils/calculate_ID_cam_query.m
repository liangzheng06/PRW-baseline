function ID_cam = calculate_ID_cam_query(query_dir)

if ~exist('cache/ID_cam_query.mat', 'file');
    query_files = dir([query_dir '*.jpg']);
    % calculate the ID of all the detected boxes
    ID_cam = [];
    for n = 1:length(query_files)
        if round(n/1000) == n/1000
            n
        end
        query_name = query_files(n).name;
        ID = str2double(query_name(1:3));
        cam = str2double(query_name(6));
        ID_cam = [ID_cam; [ID, cam]];
    end
    save('cache/ID_cam_query.mat', 'ID_cam');
else
    ID_cam = importdata('cache/ID_cam_query.mat');
end