% Loop through different values of K
K_values = [2, 4, 8, 16, 32, 64, 128, 256, 512,1024];

% Distance metrics to use for the first value of K
distance_metrics = {'euclidean', 'cityblock', 'minkowski', 'chebychev'};

% Radius values for KD-tree search
radius_values = [2, 3 ,4 ,5];

% Initialize a flag to determine if it's the first iteration
first_iteration = true;

for K = K_values
    knn_search_and_evaluate(K, distance_metrics, radius_values, first_iteration);
    first_iteration = false; % Update the flag after the first iteration
end

function [meanMpjse, meanPck] = knn_search_and_evaluate(K, distance_metrics, radius_values, first_iteration)
    % Load the .mat files
    queryData = load('queryDataset.mat');  % Load the query poses
    datasetData = load('wholeDataset.mat');  % Load the whole dataset poses

    % Transpose the data
    queryPoses = queryData.queryDataset.pos';  % Transpose the query data
    datasetPoses = datasetData.wholeDataset.pos';  % Transpose the whole dataset data

    % Take the first 500 rows from queryPoses
    queryPoses = queryPoses;

    % Take the first 1000 rows from datasetPoses
    datasetPoses = datasetPoses;

    % Initialize variables to store evaluation metrics
    mpjseList = [];
    pckList = [];

    % Loop through distance metrics
    for metric_idx = 1:length(distance_metrics)
        % Loop through radius values
        for radius_idx = 1:length(radius_values)
            % Build KD-tree
            kdtree = KDTreeSearcher(datasetPoses, 'Distance', distance_metrics{metric_idx});

            % Perform k-nearest neighbor search for 
            
            
%             the query matrix with KD-tree and radius
            start_time = tic;
            [indices, distances] = rangesearch(kdtree, queryPoses, radius_values(radius_idx));
            retrieval_time = toc(start_time);

            disp(['Indices Shape: ', num2str(size(indices))]);

            % Calculate evaluation metrics for each query pose
            for i = 1:size(queryPoses, 1)
                % Get the neighbors from the dataset within the specified radius
                nearestNeighbors = datasetPoses(indices{i}, :);

                % Calculate Mean Per Joint Squared Error (MPJSE)
                mpjse = mean(sqrt(sum((queryPoses(i, :) - nearestNeighbors).^2, 2)));
                mpjseList = [mpjseList; mpjse];

                % Compute PCK (Percentage of Correct Keypoints)
                threshold = 30; % Set a threshold for correctness
                correctKeypoints = sqrt(sum((queryPoses(i, :) - nearestNeighbors).^2, 2)) < threshold;
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

            % Save data to CSV filechr
            if first_iteration && radius_idx == 1
                header = {'K_value', 'Distance_Metric', 'Radius', 'MPJSE', 'PCK', 'Retrieval_Time'};
            end
            data = {K, distance_metrics{metric_idx}, radius_values(radius_idx), meanMpjse, meanPck, retrieval_time};

            % Check if the file already exists to write the header only once
            if first_iteration && radius_idx == 1
                writecell(header, 'results2.csv', 'WriteMode', 'overwrite');
            end

            % Append data to the CSV file
            writecell(data, 'results2.csv', 'WriteMode', 'append');

            % Clear variables for the next iteration
            mpjseList = [];
            pckList = [];
        end
    end
end
