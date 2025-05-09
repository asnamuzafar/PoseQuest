from sklearn.neighbors import NearestNeighbors, BallTree
import numpy as np
import time
import h5py

# Open the .mat file in read mode
mat_file = h5py.File('queryDataset.mat', 'r')
mat_file2 = h5py.File('wholeDataset.mat', 'r')

Q = np.array(mat_file['queryDataset']['pos'])
D = np.array(mat_file2['wholeDataset']['pos'])

query = Q
dataset = D
print("Query shape: ", query.shape)
print("Dataset shape: ", dataset.shape)


def knn_M(query, dataset, K, method='knn', metric='euclidean'):
    if method == 'knn':
        knn = NearestNeighbors(n_neighbors=K, algorithm='auto', metric=metric)
        knn.fit(dataset)

    elif method == 'range':
        knn = BallTree(dataset, metric=metric)

    # Perform K-nearest neighbor search or range search for the query matrix
    start_time = time.time()
    if method == 'knn':
        distances, indices = knn.kneighbors(query)
    elif method == 'range':
        indices = knn.query_radius(query, r=55.0)
    retrieval_time = time.time() - start_time

    # Initialize variables to store evaluation metrics
    # mse_list = []
    mpjse_list = []
    pck_list = []

    # Calculate evaluation metrics for each query pose
    for i in range(len(query)):
        if method == 'knn':
            # Get the K-nearest neighbors from the dataset
            nearest_neighbors = dataset[indices[i]]
        elif method == 'range':
            # Get the neighbors within the specified radius
            nearest_neighbors = dataset[indices[i]]

        # Calculate Mean Squared Error (MSE)
        # mse = mean_squared_error(query[i], nearest_neighbors)
        # mse_list.append(mse)

        # Calculate Mean Per Joint Squared Error (MPJSE)
        mpjse = np.mean(np.linalg.norm(query[i] - nearest_neighbors, axis=1))
        mpjse_list.append(mpjse)

        # Compute PCK (Percentage of Correct Keypoints)
        threshold = 30  # Set a threshold for correctness
        A = np.linalg.norm(query[i] - nearest_neighbors, axis=1)
        correct_keypoints = A < threshold
        pck = np.sum(correct_keypoints) / len(correct_keypoints)
        pck_list.append(pck)

    # mean_msee = np.mean(mse_list)
    mean_mpjse = np.mean(mpjse_list)
    mean_pck = np.mean(pck_list)

    print(f"Method: {method}")
    print(f"Metric: {metric}")
    # print(f"Mean Squared Error (MSE): {mean_mse}")
    print(f"Mean Per Joint Squared Error (MPJSE): {mean_mpjse}")
    print(f"Percentage of Correct Keypoints (PCK): {mean_pck}")
    print(f"Retrieval Time: {retrieval_time} seconds")


# calling the function
knn_M(query, dataset, 32, method='knn', metric='euclidean')
