import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from gnn.model.link_level_model import LinkLevelGNN
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from gnn.model.link_level_runner import LinkLevelGNNRunner
from datetime import datetime
from loguru import logger

logger.add("logs/link_level_training.log", rotation="50 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


"""
Train the link-level model
"""
def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train Link Level Model')
    parser.add_argument('--dataset_type', type=str, 
                       choices=['spider', 'bird', 'combined'], 
                       default='combined',
                       help='Type of dataset to train on (spider, bird, or combined)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embed_method', type=str, 
                        choices=['sentence_transformer', 'bert', 'api_small', 'api_large', 'api_mock'], 
                        default='sentence_transformer', 
                        help='Embedding methods used to create the current graph dataset')
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
    in_channels = 384
    hidden_channels = 256
    num_heads = 4
    num_layers = 5
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
    model = LinkLevelGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Initialize runner (for training)
    runner = LinkLevelGNNRunner(
        model=model,
        train_dataset=train_dataset,  # Use the selected dataset
        test_dataset=None,
        device=device,
        val_ratio=val_ratio,
        lr=lr,
        batch_size=batch_size,
        tensorboard_dir=f'gnn/tensorboard/link_level/train_{args.dataset_type}'  # Add dataset type to tensorboard dir
    )
    
    # Train model
    checkpoint_dir = f'checkpoints/link_level_model/{embed_method}/'
    checkpoint_name = f'link_level_model_{args.dataset_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    runner.train(num_epochs=num_epochs, checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name)



if __name__ == "__main__":
    main()