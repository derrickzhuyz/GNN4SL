import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from gnn.model.homo_model import SchemaLinkingHomoGNN
from gnn.graph_data.homo_graph_dataset import SchemaLinkingHomoGraphDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from loguru import logger
from tqdm import tqdm
import json
import copy
import time
from pathlib import Path

logger.add("logs/homo_training.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


"""
Train one epoch
:param model: model to train
:param loader: data loader
:param optimizer: optimizer
:param device: device to train on
:return: average loss
"""
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc='Training', leave=False)
    
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        
        # Ensure loss calculation is on same device
        loss = F.binary_cross_entropy(out, batch.y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar description with current loss
        progress_bar.set_description(f'Training (loss={loss.item():.4f})')
    
    return total_loss / len(loader)




def train(model, train_loader, num_epochs, optimizer, device):
    """
    Training loop for the model
    """
    logger.info("[i] Starting training...")
    start_time = time.time()  # Start time tracking
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f'[i] Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    end_time = time.time()  # End time tracking
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logger.info(f"[i] Training completed in {elapsed_time:.2f} seconds")  # Log elapsed time




def main():
    # Hyperparameters
    input_dim = 3  # Should match embedding dimension
    hidden_channels = 256
    num_layers = 3
    batch_size = 64
    num_epochs = 3
    lr = 0.001
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the device to use GPUs 4, 5, 6, and 7
    # device_ids = [6, 7]
    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')  # Default to first GPU
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(device_ids[0])
    #     logger.info(f"[i] Using GPUs: {device_ids}")
    
    # Modified device setup
    device_ids = [1, 2]
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])
        logger.info(f"[i] Using GPUs: {device_ids}")
        
        # Print the current device(s) being used
        print(f"[i] Current device: {device}")
        print(f"[i] Available devices: {torch.cuda.device_count()}")

    # Load datasets
    logger.info("[i] Loading datasets...")
    spider_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='train')
    spider_test = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='dev')  # dev set used as test set
    # spider_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='dev')
    # bird_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='train')
    # bird_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='dev')
    
    # Create data loaders
    # train_loader = DataLoader(spider_train + bird_train, batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(spider_train + bird_train, batch_size=batch_size, shuffle=True)
    # spider_dev_loader = DataLoader(spider_dev, batch_size=batch_size)
    # bird_dev_loader = DataLoader(bird_dev, batch_size=batch_size)
    train_loader = DataLoader(spider_train, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(spider_test, batch_size=batch_size)
    
    # Initialize model
    logger.info("[i] Initializing model...")
    model = SchemaLinkingHomoGNN(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    # # Move model to primary GPU first, then wrap with DataParallel
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create results directory if it doesn't exist
    results_dir = Path('gnn/results')
    results_dir.mkdir(exist_ok=True)
    
    # Train the model
    train(model, train_loader, num_epochs, optimizer, device)


if __name__ == '__main__':
    main()