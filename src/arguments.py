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

    parser.add_argument("--batcher", default="cluster", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method",
                choices=['cluster', 'k_clique', 'random'])
    
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
    parser.add_argument("--hidden-dim", default=10, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.9, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=1, type=int,
                help="Total number of training epochs to perform.")

    args = parser.parse_args()
    return args