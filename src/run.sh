#!/bin/bash

# This script runs the main.py Python script with various combinations of command line arguments.
# Each line corresponds to a single run of the script with a specific set of arguments.

# The arguments are:
# --task: The task to perform. Can be 'baseline' or 'cluster'.
# --output-dir: The directory where the model predictions and checkpoints are written.
# --model: The model architecture to be trained or fine-tuned.
# --seed: The random seed for initialization.
# --dataset: The dataset to use. Can be 'xbar' or 'superblue'.
# --ignore-cache: If set, ignores cache and creates new input data.
# --do-train: If set, runs training.
# --do-eval: If set, runs evaluation on the dev set.
# --batch-size: The batch size for training and evaluation.
# --learning-rate: The starting learning rate for the model.
# --hidden-dim: The hidden dimension for the model.
# --drop-rate: The dropout rate for model training.
# --adam-epsilon: The epsilon for the Adam optimizer.
# --n-epochs: The total number of training epochs to perform.

# Run main.py with different combinations of arguments
python main.py --task cluster --batcher k_clique --dataset xbar --batch-size 32 --learning-rate 0.00001 --drop-rate 0.1 --n-epochs 1000
# python main.py --task cluster --dataset xbar --batch-size 1 --learning-rate 0.01 --hidden-dim 10 --drop-rate 0.5 --adam-epsilon 1e-8 --n-epochs 20