# PoseQuest: Efficient 3D Human Pose Retrieval

PoseQuest is a hybrid system that optimizes the retrieval of human poses from large motion capture databases. Combining **KD-Tree nearest neighbor searches**, **K-means clustering**, **lazy neighborhood graph construction**, and **Dijkstra’s algorithm**, PoseQuest achieves faster and more accurate pose retrieval compared to traditional methods like Subsequence DTW.

> ⚡️ PoseQuest enhances pose retrieval efficiency, making it highly suitable for **real-time applications** like animation, virtual reality, and gaming.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Experiments and Results](#experiments-and-results)
- [Base Paper References](#base-paper-references)
- [Future Work](#future-work)

---

## Introduction

Human Pose Estimation (HPE) plays a key role in various computer vision applications. However, retrieving specific poses from vast motion databases is challenging due to high dimensionality and complex non-Euclidean data structures.

PoseQuest proposes an enhanced approach for efficient pose retrieval, targeting improvements in **retrieval speed**, **accuracy**, and **computational efficiency**.

---

## Methodology

The core components of PoseQuest include:
- **KD-Tree Nearest Neighbor Search:** Quickly narrows down candidate poses.
- **K-Means Clustering:** Reduces dataset size while preserving important motion patterns.
- **Lazy Neighborhood Graph Construction:** Connects poses based on similarity using Mean Squared Error (MSE).
- **Shortest Path Search (Dijkstra’s Algorithm):** Finds optimal motion sequences in the graph.

> 📈 PoseQuest achieves **faster alignment times** than traditional Subsequence DTW and the baseline method.

---

## Dataset

We used the **HDM05 Motion Capture Dataset**:

- **Source:** Hdm05 Dataset from [KIT Database](https://resources.mpi-inf.mpg.de/HDM05/)
- **Query Size:** 2463 samples
- **Dataset Size:** 378,694 samples
- **Features per Sample:** 93 (3D joint coordinates)

**Preprocessing:**
- Normalization (removing translation and rotation)
- KD-Tree building
- Data reduction via K-means clustering

---

## Experiments and Results

- **Speed Improvements:** PoseQuest provides significantly faster retrieval times compared to both Subsequence DTW and the base paper algorithm.
- **Accuracy:** Maintains competitive accuracy using MPJSE (Mean per Joint Square Error) and PCK (Percentage of Correct Keypoints).
- **Efficiency:** Integration of K-means clustering boosts retrieval speed without compromising essential pose features.

---

### Table 1: Comparison of PoseQuest with Subsequence DTW

| Data Size  | Subsequence DTW (seconds) | PoseQuest Algorithm (seconds) |
|------------|----------------------------|-------------------------------|
| 30 frames  | 0.3022                     | 0.0490                        |
| 48 frames  | 0.3333                     | 0.1720                        |
| 64 frames  | 0.4489                     | 0.1700                        |

---

### Table 2: Comparison of PoseQuest with Base Paper Algorithm

| Data Size  | Base Paper Algorithm (seconds) | PoseQuest Algorithm (seconds) |
|------------|--------------------------------|-------------------------------|
| 30 frames  | 0.1108                         | 0.0490                        |
| 48 frames  | 0.2020                         | 0.1720                        |
| 64 frames  | 0.1210                         | 0.1700                        |

---

## Base Paper Reference

- **Krüger, B., Tautges, J., Weber, A., & Zinke, A. (2010). Fast Local and Global Similarity Searches in Large Motion Capture Databases** [[Paper Link]](https://cg.cs.uni-bonn.de/backend/v1/files/publications/FastSimilaritySearch.pdf)

## References

Our work builds upon and critically analyzes the following studies:

- **Efficient Motion Retrieval in Large Motion Databases** — Kapadia et al., 2013 [[Paper Link]](https://doi.org/10.1145/2448196.2448199)
- **A Dual-Source Approach for 3D Pose Estimation from a Single Image** — Yasin et al., 2016 [[CVPR 2016 Paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yasin_A_Dual-Source_Approach_CVPR_2016_paper.pdf)
- **View-Invariant, Occlusion-Robust Probabilistic Embedding for Human Pose** — Liu et al., 2022 [[IJCV Paper]](https://link.springer.com/article/10.1007/s11263-021-01508-0)
- **Multi-Modal Interactive Video Retrieval with Temporal Queries** — Heller et al., 2022 [[MMM Conference Paper]](https://link.springer.com/chapter/10.1007/978-3-030-98355-5_40)
- **ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model** — Zhang et al., 2023 [[arXiv Paper]](https://arxiv.org/abs/2304.01116)

We also took inspiration from:
- **Subsequence Dynamic Time Warping**, AudioLabs Erlangen [[Tutorial Link]](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html)


## Future Work

- Implement memory optimization for larger frame sets
- Integrate **parallel processing** and **data streaming**
- Extend PoseQuest to handle real-time 3D pose retrieval from live camera feeds
- Explore retrieval-augmented diffusion models for motion generation

---
