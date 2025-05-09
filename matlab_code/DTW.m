% Loop through different values of K
K_values = [2, 4, 8, 16, 32, 64, 128, 256, 512];

% Distance metrics to use in DTW
distance_metrics = {'euclidean', 'absolute', 'squared'};

% Initialize a flag to determine if it's the first iteration
first_iteration = true;

for K = K_values
    for distance_metric_idx = 1:length(distance_metrics)
        dtw_search_and_evaluate(K, distance_metrics{distance_metric_idx}, first_iteration);
        first_iteration = false; % Update the flag after the first iteration
    end
end

function [meanMpjse, meanPck] = dtw_search_and_evaluate(K, distance_metric, first_iteration)
    % Load the .mat files
    queryData = load('queryDataset.mat');  % Load the query poses
    datasetData = load('wholeDataset.mat');  % Load the whole dataset poses

    % Transpose the data
    queryPoses = queryData.queryDataset.pos';  % Transpose the query data
    datasetPoses = datasetData.wholeDataset.pos';  % Transpose the whole dataset data

    % Take the first 500 rows from queryPoses
    queryPoses = queryPoses';

    % Take the first 1000 rows from datasetPoses
    datasetPoses = datasetPoses';

    % Perform DTW-based search for the query matrix with specified distance metric
    start_time = tic;

    % Using MATLAB's built-in dtw function
    [min_distance, warping_path_query, warping_path_dataset] = dtw(queryPoses, datasetPoses, distance_metric);

    retrieval_time = toc(start_time);

    disp(['Minimum Distance: ', num2str(min_distance)]);

    % Calculate evaluation metrics for each query pose
    mpjseList = [];
    pckList = [];

    for i = 1:size(queryPoses, 1)
        % Get the indices from the warping paths
        index_query = warping_path_query(i, :);
        index_dataset = warping_path_dataset(i, :);

        % Get the aligned signals
        aligned_query = queryPoses(i, index_query);
        aligned_dataset = datasetPoses(index_dataset, :);

        % Calculate Mean Per Joint Squared Error (MPJSE)
        mpjse = mean(sqrt(sum((aligned_query - aligned_dataset).^2, 2)));
        mpjseList = [mpjseList; mpjse];

        % Compute PCK (Percentage of Correct Keypoints)
        threshold = 30; % Set a threshold for correctness
        correctKeypoints = sqrt(sum((aligned_query - aligned_dataset).^2, 2)) < threshold;
        pck = sum(correctKeypoints) / length(correctKeypoints);
        pckList = [pckList; pck];
    end

    % Calculate the mean of each evaluation metric
    meanMpjse = mean(mpjseList);
    meanPck = mean(pckList);

    % Display results
    disp(['Mean Per Joint Squared Error (MPJSE): ', num2str(meanMpjse)]);
    disp(['Percentage of Correct Keypoints (PCK): ', num2str(meanPck)]);
    disp(['Retrieval Time: ', num2str(retrieval_time), ' seconds']);

    % Save data to CSV file
    if first_iteration
        header = {'K_value', 'Distance_Metric', 'Min_Distance', 'Retrieval_Time'};
    end
    data = {K, distance_metric, min_distance, retrieval_time};

    % Check if the file already exists to write the header only once
    if first_iteration
        writecell(header, 'results4.csv', 'WriteMode', 'overwrite');
    end

    % Append data to the CSV file
    writecell(data, 'results4.csv', 'WriteMode', 'append');
end
