import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from gnn.model.link_level_model import LinkLevelGCN
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from gnn.model.link_level_runner import LinkLevelGNNRunner
from datetime import datetime
from loguru import logger
import sys

# Remove default logger
logger.remove()
# Add file handler with INFO level
logger.add("logs/link_level_training.log", 
           rotation="50 MB", 
           level="INFO",
           format="{time} {level} {message}",
           compression="zip")
# Add console handler with WARNING level
logger.add(sys.stderr, level="WARNING")


"""
Train the link-level model
"""
def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train Link Level Model')
    parser.add_argument('--model_type', type=str, required=True, choices=['gcn'], default='gcn', 
                       help='Type of model to use: gcn')
    parser.add_argument('--dataset_type', type=str, required=True, 
                       choices=['spider', 'bird', 'combined'], 
                       default='combined',
                       help='Type of dataset to train on (spider, bird, or combined)')
    parser.add_argument('--epochs', type=int, required=True, default=2, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embed_method', type=str, required=True,
                        choices=['sentence_transformer', 'bert', 'api_small', 'api_large', 'api_mock'], 
                        default='sentence_transformer', 
                        help='Embedding methods used to create the current graph dataset')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    parser.add_argument('--in_channels', type=int, default=384, help='Number of input channels, aligned with embedding dimension: 384 for sentence_transformer, 768 for bert, 1536 for text-embedding-3-small (api_small), 3072 for text-embedding-3-large (api_large).')
    parser.add_argument('--val_dataset_type', type=str, 
                       choices=['spider', 'bird', 'combined'], 
                       default='combined',
                       help='Type of validation dataset to use')
    parser.add_argument('--prediction_method', 
                       type=str,
                       choices=['dot_product', 'concat_mlp'],
                       default='dot_product',
                       help='Method for link prediction: dot_product or concat_mlp')
    parser.add_argument('--negative_sampling', 
                       action='store_true',
                       help='Whether to use negative sampling during training')
    parser.add_argument('--negative_sampling_ratio', 
                       type=float, 
                       default=2.0,
                       help='Ratio of negative to positive samples (e.g., 3.0 means 3 negative samples for each positive)')
    parser.add_argument('--negative_sampling_method',
                       type=str,
                       choices=['random', 'hard'],
                       default='random',
                       help='Method for negative sampling: random or hard negative mining')
    parser.add_argument('--metric', 
                       type=str,
                       choices=['auc', 'f1'],
                       default='auc',
                       help='Metric to use for saving the best model: auc or f1')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Training on {args.dataset_type} dataset: {args.epochs} epochs, batch size: {args.batch_size}, validation ratio: {args.val_ratio}, learning rate: {args.lr}")

    # Hyperparameters (from arguments)
    num_epochs = args.epochs
    batch_size = args.batch_size
    val_ratio = args.val_ratio
    lr = args.lr

    # Model hyperparameters
    in_channels = args.in_channels
    hidden_channels = 128
    num_layers = 2
    dropout = 0.1
    
    # Load dataset based on argument
    embed_method = args.embed_method
    
    if args.dataset_type == 'combined':
        # Check if combined dataset exists, if not create it
        combined_filename = 'combined_train_link_level_graph.pt'
        combined_path = os.path.join('data/schema_linking_graph_dataset/link_level_graph_dataset', 
                                    embed_method, 
                                    combined_filename)
        if not os.path.exists(combined_path):
            logger.warning("Combined dataset not found. Creating it by combining and shuffling Spider and BIRD datasets...")
            # Load individual datasets
            spider_train = LinkLevelGraphDataset(
                root='data/schema_linking_graph_dataset/',
                dataset_type='spider',
                split='train',
                embed_method=embed_method
            )
            bird_train = LinkLevelGraphDataset(
                root='data/schema_linking_graph_dataset/',
                dataset_type='bird',
                split='train',
                embed_method=embed_method
            )
            
            # Load and combine the data
            spider_train_data = torch.load(os.path.join(spider_train.graph_data_dir, spider_train.processed_file_names[0]))
            bird_train_data = torch.load(os.path.join(bird_train.graph_data_dir, bird_train.processed_file_names[0]))
            combined_train_data = spider_train_data + bird_train_data
            
            # Shuffle the combined dataset
            random_indices = torch.randperm(len(combined_train_data))
            combined_train_data = [combined_train_data[i] for i in random_indices]
            
            # Save the combined dataset
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            torch.save(combined_train_data, combined_path)
            logger.info(f"Created and saved combined dataset with {len(combined_train_data)} graphs")

    # Load the specified dataset
    train_dataset = LinkLevelGraphDataset(
        root='data/schema_linking_graph_dataset/',
        dataset_type=args.dataset_type,
        split='train',
        embed_method=embed_method
    )
    logger.info(f"Loaded {args.dataset_type} training dataset with {len(train_dataset)} samples")
    
    # Initialize model
    if args.model_type == 'gcn':
        model = LinkLevelGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            prediction_method=args.prediction_method
        )
        model.print_model_structure()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model_name = f'{args.model_type}_train_{args.dataset_type}_{args.metric}_{num_epochs}ep_{args.negative_sampling_method}_neg_samp_{args.negative_sampling_ratio}' \
                if args.negative_sampling \
                else f'{args.model_type}_train_{args.dataset_type}_{args.metric}_{num_epochs}ep_no_neg_samp'

    # Initialize runner (for training)
    runner = LinkLevelGNNRunner(
        model=model,
        train_dataset=train_dataset,  # Use the selected dataset
        test_dataset=None,
        device=device,
        val_ratio=val_ratio,
        val_dataset_type=args.val_dataset_type,
        lr=lr,
        batch_size=batch_size,
        threshold=args.threshold,
        tensorboard_dir=f'gnn/tensorboard/link_level_{args.model_type}/train_{args.dataset_type}/{embed_method}/{model_name}',  # Add subdirectory
        negative_sampling=args.negative_sampling,
        negative_sampling_ratio=args.negative_sampling_ratio,
        negative_sampling_method=args.negative_sampling_method
    )
    
    # Train model
    checkpoint_dir = f'checkpoints/link_level_{args.model_type}/{embed_method}/'
    checkpoint_name = f'{model_name}_{datetime.now().strftime("%m%d_%H%M")}.pt'
    runner.train(num_epochs=num_epochs, 
                 checkpoint_dir=checkpoint_dir, 
                 checkpoint_name=checkpoint_name, 
                 resume_from=args.resume_from, 
                 metric=args.metric)



if __name__ == "__main__":
    main()

    # NOTE: To check the tensorboard, run the following command: tensorboard --logdir=xxx
    # For example: tensorboard --logdir=gnn/tensorboard/link_level/train_combined/