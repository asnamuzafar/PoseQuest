import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import time
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import os
import sys
import numpy as np
import librosa
from numba import jit
from matplotlib import patches
import libfmp.b
import libfmp.c3
import libfmp.c7
import networkx as nx
from dtw import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

import h5py

def mean_squared_error(vec1, vec2):
    """Compute the Mean Squared Error (MSE) between two vector arrays."""
    return np.mean((vec1 - vec2) ** 2)

def search_and_evaluate(query, dataset, K):
    kdtree = KDTree(dataset)
    
    # Perform K-nearest neighbor search for the query matrix
    start_time = time.time()
    K = int(K)
    print("TYPE OF K   ",type(K))
    distances, indices = kdtree.query(query, k=K)
    retrieval_time = time.time() - start_time
    print(indices.shape)
    
    # Initialize variables to store evaluation metrics
    mse_list = []
    mpjse_list = []
    pck_list = []
    
    # Calculate evaluation metrics for each query pose
    for i in range(len(query)):
        # Get the K-nearest neighbors from the dataset
        nearest_neighbors = dataset[indices[i]]
        
        # Calculate Mean Squared Error (MSE)
#         mse = mean_squared_error(query[i], nearest_neighbors)
#         mse_list.append(mse)
        
        # Calculate Mean Per Joint Squared Error (MPJSE)
        mpjse = np.mean(np.linalg.norm(query[i] - nearest_neighbors))
        mpjse_list.append(mpjse)
        
        # Compute PCK (Percentage of Correct Keypoints)
        threshold = 30  # Set a threshold for correctness
        correct_keypoints = np.linalg.norm(query[i] - nearest_neighbors, axis=1) < threshold
        pck = np.sum(correct_keypoints) / len(correct_keypoints)
        pck_list.append(pck)
        
        mse_list.append(mean_squared_error(query[i], nearest_neighbors))
        
    # Calculate the mean of each evaluation metric
    mean_mse = np.mean(mse_list)
    mean_mpjse = np.mean(mpjse_list)
    mean_pck = np.mean(pck_list)
    
    data_eval = []
    print(f"Mean Squared Error (MSE): {mean_mse}")
    print(f"Mean Per Joint Squared Error (MPJSE): {mean_mpjse}")
    print(f"Percentage of Correct Keypoints (PCK): {mean_pck}")
    print(f"Retrieval Time: {retrieval_time} seconds")
    
    data_eval.append(f"Mean Squared Error (MSE): {mean_mse}")
    data_eval.append(f"Mean Per Joint Squared Error (MPJSE): {mean_mpjse}")
    data_eval.append(f"Percentage of Correct Keypoints (PCK): {mean_pck}")
    data_eval.append(f"Retrieval Time: {retrieval_time} seconds")
    
    return indices, data_eval , mean_mse, mean_mpjse, mean_pck , retrieval_time


def sliding_window(matrix, window_size, stride):
    num_rows, num_cols = matrix.shape
    num_row_windows = (num_rows - window_size) // stride + 1
    num_col_windows = (num_cols - window_size) // stride + 1
    windowed_matrix = []

    for i in range(num_row_windows):
        for j in range(num_col_windows):
            start_row_idx = i * stride
            end_row_idx = start_row_idx + window_size
            start_col_idx = j * stride
            
            end_col_idx = start_col_idx + window_size
            windowed_matrix.append(matrix[i, start_col_idx:end_col_idx])

    return np.array(windowed_matrix)



def mean_squared_error(node1, node2):
    """Calculate Mean Squared Error (MSE) between two nodes."""
    return np.mean((node1 - node2) ** 2)



import networkx as nx
import numpy as np

def connect_neighbors(matrix):
    """Connect neighboring nodes in a grid-like manner."""
    rows, cols, h = matrix.shape
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Dictionary for node indexing
    node_indexes = {}
    
    # Add nodes and populate node indexing dictionary
    node_count = 0
    for i in range(rows):
        for j in range(cols):
            node = (i, j)
            G.add_node(node)
            node_indexes[node] = node_count
            node_count += 1
    
    # Connect nodes according to grid-like logic
    for i in range(rows):
        for j in range(cols):
            current_node = (i, j)
            current_node_index = node_indexes[current_node]
            
            # Connect to the right neighbor
            if j < cols - 1:
                neighbor_node = (i, j + 1)
                neighbor_node_index = node_indexes[neighbor_node]
                edge_weight = np.mean(np.square(matrix[i, j] - matrix[i, j + 1]))  # Example of edge weight calculation
                G.add_edge(current_node, neighbor_node, weight=edge_weight)
            
            # Connect to the bottom neighbor
            if i < rows - 1:
                neighbor_node = (i + 1, j)
                neighbor_node_index = node_indexes[neighbor_node]
                edge_weight = np.mean(np.square(matrix[i, j] - matrix[i + 1, j]))  # Example of edge weight calculation
                G.add_edge(current_node, neighbor_node, weight=edge_weight)
            
            # Connect diagonally to the bottom-right neighbor
            if i < rows - 1 and j < cols - 1:
                neighbor_node = (i + 1, j + 1)
                neighbor_node_index = node_indexes[neighbor_node]
                edge_weight = np.mean(np.square(matrix[i, j] - matrix[i + 1, j + 1]))  # Example of edge weight calculation
                G.add_edge(current_node, neighbor_node, weight=edge_weight)
            
            # Connect diagonally to the bottom-left neighbor
            if i < rows - 1 and j > 0:
                neighbor_node = (i + 1, j - 1)
                neighbor_node_index = node_indexes[neighbor_node]
                edge_weight = np.mean(np.square(matrix[i, j] - matrix[i + 1, j - 1]))  # Example of edge weight calculation
                G.add_edge(current_node, neighbor_node, weight=edge_weight)
    
    # Add starting node 'S' at positions (0, 1), (1, 0), (2, 0), ...
    for i in range(rows):
        start_node = (i, 0)
        G.add_edge('S', start_node, weight=1)
    
    # Add end node 'E' at the last element of each row
    for i in range(rows):
        end_node = (i, cols - 1)
        G.add_edge(end_node, 'E', weight=1)
    
    return G


import time 


def find_all_shortestP(matrix):
    # Create graph and connect neighbors
    graph = connect_neighbors(matrix)
    
    total_costs = []
    S_neighbors = []
    ASP = []
    
    start_t = time.time()
    print("Shortest path search started now..")
    
    for successor in graph.successors('S'):
        S_neighbors.append(successor)
    
        # Find all shortest paths from 'S' to 'E'
        all_shortest_paths = list(nx.all_shortest_paths(graph, source=successor, target='E', weight='weight'))
        ASP.extend(all_shortest_paths)  # Extend instead of append to flatten the list
    
        # Compute the total cost of each path
        for path in all_shortest_paths:
            total_cost = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
            total_costs.append(total_cost)
    
    # print("Total Costs:", total_costs)
    # print("Number of Paths:", len(total_costs))
    
    # Ensure the lengths of total_costs and ASP are the same
    assert len(total_costs) == len(ASP), "Mismatch between total costs and paths"
    
    # Find the index of the shortest path with the lowest cost
    min_cost_index = total_costs.index(min(total_costs))
    
    return ASP, total_costs, min_cost_index, graph

def find_all_shortest_paths(matrix):
    # Create graph and connect neighbors
    graph = connect_neighbors(matrix)
    
    total_costs = []
    # Get nodes connected only to 'S'
    S_neighbors = []
    ASP = []
    
    start_t = time.time()
    print("Shortest path search started now..")
    for successor in graph.successors('S'):
        S_neighbors.append(successor)
    
        # Find all shortest paths from 'S' to 'E'
        all_shortest_paths = list(nx.all_shortest_paths(graph, source=successor, target='E', weight='weight'))
        ASP.append(all_shortest_paths)
    
        # Compute the total cost of each path
        for path in all_shortest_paths:
            total_cost = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
            total_costs.append(total_cost)
    
    print(total_costs)
    print(len(total_costs))
    # Find the index of the shortest path with the lowest cost
    min_cost_index = total_costs.index(min(total_costs))
    
    return ASP, total_costs, min_cost_index,graph


def compute_accumulated_cost_matrix_subsequence_dtw(C):
    """Given the cost matrix, compute the accumulated cost matrix for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}

    Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N, M))
    D[:, 0] = np.cumsum(C[:, 0])
    D[0, :] = C[0, :]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D


def compute_optimal_warping_path_subsequence_dtw(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}

    Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P



def analysis(query,B_data,P_Data):

    T_b, graph_html_B, shortest_path_nodes_B, minimum_cost_B, motionB = GGH(B_data)

    T_p, graph_html_p, shortest_path_nodes_p, minimum_cost_p, motionP = GGH(P_Data)

    print("Time taken to construct alignments using Base Paper Method: ", T_b)
    print("Time taken to construct alignments using  Pose Quest Method: ", T_p)
    
    mse_b = np.mean((query[:40] - B_data[:40,:40]) ** 2)

    mse_p = np.mean((query[:40] - P_Data[:40,:40]) ** 2)

    print("MSE Base paper: ",mse_b)
    print("MSE PoseQuest: ",mse_p)


    return T_b, mse_b ,T_p , mse_p 


## evaluation
import numpy as np

def mean_squared_error(vec1, vec2):
    """Compute the Mean Squared Error (MSE) between two vector arrays."""
    return np.mean((vec1 - vec2) ** 2)

def mean_pixelwise_joint_squared_error(vec1, vec2):
    """Compute the Mean Pixel-wise Joint Squared Error (MPJSE) between two vector arrays."""
    return np.mean(np.mean((vec1 - vec2) ** 2, axis=1))

def percentage_correct_keypoints(vec1, vec2, threshold):
    """Compute the Percentage of Correct Keypoints (PCK) between two vector arrays."""
    distances = np.linalg.norm(vec1 - vec2, axis=1)
    return np.mean(distances < threshold) * 100



def load_data(query_path,dataset_path):
    mat_file = h5py.File(query_path, 'r')
    mat_file2 = h5py.File(dataset_path, 'r')
    Q = mat_file['queryDataset']['pos']
    D = mat_file2['wholeDataset']['pos']
    Qnp = np.array(Q)
    Dnp = np.array(D)
    query = Qnp
    dataset = Dnp
    return query, dataset


def load_data_CMU(query_path,dataset_path):
    query = np.load(query_path)
    dataset = np.load(dataset_path)

    return query, dataset

def GGH(matrix):
    # Generate the graph based on the options
    # Return HTML code for the graph
    # part 3 Lazy neighborhood graph construction
    # Draw the graph
    matrix = np.array(matrix)
    plt.figure(figsize=(40, 40))

    # Positioning nodes
    pos = {'S': (-1, 0)}  # Set position for 'S' node
    for i, example_index in enumerate(range(matrix.shape[0])):
        for j, neighbor_index in enumerate(range(matrix.shape[1])):
            pos[(example_index, neighbor_index)] = (j, -i)  # Position each example and its neighbors in a square-like shape
    pos['E'] = (matrix.shape[1] + 1, 0)  # Set position for 'E' node

    # Create graph and connect neighbors
    graph = connect_neighbors(matrix)

    
    start_t = time.time()
    # part 4 Shortest path search
    # Find all shortest paths, display total cost of each, and highlight the one with the lowest cost
    all_shortest_paths, total_costs, min_cost_index, graph = find_all_shortestP(matrix)
    
    endtime = time.time()
    
    print("Time taken to find alignements using graph:", endtime - start_t)

#     # Highlight the shortest path
    print("Min cost index ",min_cost_index)
    print("Length of all shortests paths list ",len(all_shortest_paths))
    shortest_path_nodes = all_shortest_paths[min_cost_index]

#     # Drawing nodes and edges
#     nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="skyblue")
#     nx.draw_networkx_edges(graph, pos, edge_color="gray")
#     nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): round(d["weight"], 2) for u, v, d in graph.edges(data=True)})

    # Drawing the shortest path nodes with a different color
#     nx.draw_networkx_nodes(graph, pos, nodelist=shortest_path_nodes, node_color="red", node_size=1000)

#     plt.title("Interconnected Graph with Shortest Path Highlighted")
#     plt.axis("off")  # Turn off axis

    # Save the graph as an image
#     plt.savefig("./GGG.png")
    
#     plt.close()  # Close the plot to avoid displaying it

    # part 5 DTW analysis
    retireved_motion = shortest_path_nodes[:-1]

    print(shortest_path_nodes)

    motion = []
    for x, y in retireved_motion:
        print(x, y)
        motion.append(matrix[x][y])

    motion = np.array(motion)

    return endtime - start_t,'<img src="./graph.png" alt="Graph">' , shortest_path_nodes, total_costs[min_cost_index], motion 

def generate_graph_html(matrix):
    # Generate the graph based on the options
    # Return HTML code for the graph
    # part 3 Lazy neighborhood graph construction
    # Draw the graph
    matrix = np.array(matrix)
    plt.figure(figsize=(40, 40))

    # Positioning nodes
    pos = {'S': (-1, 0)}  # Set position for 'S' node
    for i, example_index in enumerate(range(matrix.shape[0])):
        for j, neighbor_index in enumerate(range(matrix.shape[1])):
            pos[(example_index, neighbor_index)] = (j, -i)  # Position each example and its neighbors in a square-like shape
    pos['E'] = (matrix.shape[1] + 1, 0)  # Set position for 'E' node

    # Create graph and connect neighbors
    graph = connect_neighbors(matrix)

    
    start_t = time.time()
    # part 4 Shortest path search
    # Find all shortest paths, display total cost of each, and highlight the one with the lowest cost
    all_shortest_paths, total_costs, min_cost_index,graph = find_all_shortestP(matrix)
    
    endtime = time.time()
    
    print("Time taken to find alignements using graph:", endtime - start_t)

#     # Highlight the shortest path
    # print("Min cost index ",min_cost_index)
    # print("Length of all shortests paths list ",len(all_shortest_paths))
    shortest_path_nodes = all_shortest_paths[min_cost_index]

    # Drawing nodes and edges
    # nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="skyblue")
    # nx.draw_networkx_edges(graph, pos, edge_color="gray")
    # nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): round(d["weight"], 2) for u, v, d in graph.edges(data=True)})

    # # Drawing the shortest path nodes with a different color
    # nx.draw_networkx_nodes(graph, pos, nodelist=shortest_path_nodes, node_color="red", node_size=1000)

    # plt.title("Interconnected Graph with Shortest Path Highlighted")
    # plt.axis("off")  # Turn off axis

    # # Save the graph as an image
    # plt.savefig("./GGG.png")
    
    # plt.close()  # Close the plot to avoid displaying it

    # part 5 DTW analysis
    retireved_motion = shortest_path_nodes[:-1]

    print(shortest_path_nodes)

    motion = []
    for x, y in retireved_motion:
        print(x, y)
        motion.append(matrix[x][y])

    motion = np.array(motion)

    return endtime - start_t, '<img src="./graph.png" alt="Graph">' , shortest_path_nodes, total_costs[min_cost_index], motion 


import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import os

# Set the environment variable to avoid memory leak on Windows with MKL
os.environ["OMP_NUM_THREADS"] = "1"

def kmeans_clustering_nearest_neighbors(data, kmeans_nn):
    """
    Perform K-means clustering on each row's nearest neighbors and select nearest points.

    Parameters:
    data (numpy.ndarray): Input data of shape (rows, neighbors, features).
    kmeans_nn (int): Desired number of nearest neighbors after clustering.

    Returns:
    numpy.ndarray: New data of shape (rows, kmeans_nn, features).
    """
    new_data = []

    for i in range(data.shape[0]):  # Iterate over each row
        # Extract the nearest neighbors for the current row
        nearest_neighbors_data = data[i]

        # Ensure kmeans_nn is less than or equal to the number of nearest neighbors
        if kmeans_nn > nearest_neighbors_data.shape[0]:
            kmeans_nn = nearest_neighbors_data.shape[0]

        # Perform K-means clustering on the nearest neighbors
        kmeans_model_nn = KMeans(n_clusters=kmeans_nn, random_state=0, n_init=10)
        kmeans_labels_nn = kmeans_model_nn.fit_predict(nearest_neighbors_data)

        # Calculate the cluster centers (new nearest neighbors)
        centroids = kmeans_model_nn.cluster_centers_

        # Find the nearest points in the original data to the centroids
        nearest_points = []
        for centroid in centroids:
            distances = distance.cdist([centroid], nearest_neighbors_data, 'euclidean')
            nearest_point_index = np.argmin(distances)
            nearest_points.append(nearest_neighbors_data[nearest_point_index])

        new_data.append(nearest_points)

    # Convert list to numpy array
    new_data = np.array(new_data)
    
    return new_data

# Example usage
# data = np.random.rand(128, 128, 93)  # Replace this with your actual data
# kmeans_nn = 40
# new_data = kmeans_clustering_nearest_neighbors(data, kmeans_nn)
# print("New data shape:", new_data.shape)
