from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gnn.model.link_level_model import LinkLevelGNN
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from tqdm import tqdm
import os
import time
from datetime import datetime, timedelta
from loguru import logger
from gnn.model.link_level_runner import LinkLevelGNNRunner

logger.add("logs/link_level_training.log", rotation="50 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


"""
Train the link-level model
"""
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 1
    batch_size = 16
    val_ratio = 0.1
    lr = 1e-4
    # Model hyperparameters
    in_channels = 384
    hidden_channels = 256
    num_heads = 4
    num_layers = 5
    dropout = 0.1
    
    # Load datasets
    embed_method = 'sentence_transformer'
    train_dataset_spider = LinkLevelGraphDataset(
        root='data/schema_linking_graph_dataset/',
        dataset_type='spider',
        split='train',
        embed_method=embed_method
    )
    
    # Initialize model
    model = LinkLevelGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Initialize trainer
    trainer = LinkLevelGNNRunner(
        model=model,
        train_dataset=train_dataset_spider,
        test_dataset=None,
        device=device,
        val_ratio=val_ratio,
        lr=lr,
        batch_size=batch_size,
        tensorboard_dir='gnn/tensorboard/link_level/train'
    )
    
    # Train model
    checkpoint_dir = f'checkpoints/link_level_model/{embed_method}/'
    checkpoint_name = f'link_level_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    trainer.train(num_epochs=num_epochs, checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name)



if __name__ == "__main__":
    main()