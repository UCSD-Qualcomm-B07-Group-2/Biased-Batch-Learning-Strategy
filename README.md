# Walking Cluster-GCN

This repository contains the code and supporting materials for a modified version of the Cluster-GCN algorithm. This work is part of a research project aimed at improving the performance and efficiency of graph convolutional networks via a batching technique used in Cluster-GCN (https://arxiv.org/pdf/1905.07953.pdf). We implement Cluster-GCN with different clustering and partitioning algorithms, as well as an improved batch sampling method that improves message passing.

## Paper

The accompanying paper, "Walking Cluster-GCN", provides a detailed explanation of the modifications made to the original Cluster-GCN algorithm and presents experimental results demonstrating the improvements achieved by our approach. You can read the paper [here](link-to-your-paper).

## Code

The `src` directory contains the Python code for our modified Cluster-GCN. Here's a brief overview of the main files and their purpose:

- `main.py`: The main script that runs the algorithm.
- `model.py`: Defines the model architecture.
- `utils.py`: Contains utility functions used by the main script and model.

## Installation

To run the code, you'll need to install the required Python packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```
The following combinations are supported:
| Operating System | Supported | GPU Support |
| ---------------- | --------- | ----------- |
| Windows          | No        | None   | 
| macOS            | Yes ✅       | None |
| Linux            | Yes ✅       | CUDA (NVIDIA) ✅ |
