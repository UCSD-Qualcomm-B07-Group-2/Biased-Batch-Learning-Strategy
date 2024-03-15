import argparse
import os

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="baseline", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method",
                choices=['baseline','cluster'])

    parser.add_argument("--batcher", default="cluster_gcn", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method",
                choices=['cluster_gcn', 'k_clique', 'random'])
    
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='gcn', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="xbar", type=str,
                help="dataset", choices=['xbar', 'superblue'])
    

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=1, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=3, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=1, type=int,
                help="Total number of training epochs to perform.")

    # Arguments from AdvancedGCNRegression
    parser.add_argument("--num-node-features", type=int,
                        help="Number of node features")
    parser.add_argument("--num-edge-features", type=int,
                        help="Number of edge features")
    parser.add_argument("--conv1-out-features", default=16, type=int,
                        help="Number of output features for the first GCN convolution layer")
    parser.add_argument("--conv2-out-features", default=32, type=int,
                        help="Number of output features for the second GCN convolution layer")
    parser.add_argument("--conv3-out-features", default=32, type=int,
                        help="Number of output features for the third GCN convolution layer")
    parser.add_argument("--gat-out-features", default=32, type=int,
                        help="Number of output features for the GAT convolution layer")
    parser.add_argument("--gat-heads", default=4, type=int,
                        help="Number of heads for the GAT convolution layer")
    parser.add_argument("--dropout-rate", default=0.7, type=float,
                        help="Dropout rate for the model")

    args = parser.parse_args()
    return args