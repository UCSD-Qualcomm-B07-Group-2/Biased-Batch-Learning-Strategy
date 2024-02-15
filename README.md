# Walking Cluster-GCN

This repository contains the code and supporting materials for a modified version of the Cluster-GCN algorithm. This work is part of a research project aimed at improving the performance and efficiency of graph convolutional networks via a batching technique used in Cluster-GCN (https://arxiv.org/pdf/1905.07953.pdf). We implement Cluster-GCN with different clustering and partitioning algorithms, as well as an improved batch sampling method that improves message passing.

## Paper

The accompanying paper, "Walking Cluster-GCN", provides a detailed explanation of the modifications made to the original Cluster-GCN algorithm and presents experimental results demonstrating the improvements achieved by our approach. You can read the paper [here](link-to-your-paper).

## Code

The `src` directory contains the Python code for our modified Cluster-GCN. Here's a brief overview of the main files and their purpose:

- `run.sh`: Experiements that can be ran with different configurations
- `main.py`: The main script that runs the algorithm.
- `load.py`: This script includes functions that loads and caches the data.
- `model.py`: Defines the model architectures
- `partitioning.py`: Defines the partitioning algorithms we have for our clustering approach.
- `batching.py`: Creates a generic Batcher class that can help cluster our data via different methods.
`arguments.py`: Argument flags that are accepted with running main.py

To run:

`sh run.sh`
or
`python main.py --task baseline --dataset xbar --learning-rate 0.001 --hidden-dim 10 --drop-rate 0.5 --adam-epsilon 1e-8 --n-epochs 10`


## Installation

To run the code, you'll need to install the required Python packages. You can do this by running the following command:

With pip:

```bash
pip install -r requirements.txt
```
With conda:

```bash
conda env create -f environment.yml
```

The following combinations are supported:
| Operating System | Supported | GPU Support |
| ---------------- | --------- | ----------- |
| Windows          | No        | None   | 
| macOS            | Yes ✅       | None |
| Linux            | Yes ✅       | CUDA (NVIDIA) ✅ |
