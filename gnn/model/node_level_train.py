import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from gnn.model.node_level_model import NodeLevelGNN
from gnn.graph_data.node_level_graph_dataset import NodeLevelGraphDataset
import numpy as np
from loguru import logger
from tqdm import tqdm
import time
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

logger.add("logs/node_level_training.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


"""
Train one epoch
:param model: model to train
:param loader: data loader
:param optimizer: optimizer
:param device: device to train on
:return: average loss
"""
def train_epoch(model: NodeLevelGNN, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    epoch_start_time = time.time()
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
    
    epoch_time = time.time() - epoch_start_time
    logger.info(f"[i] Epoch time: {epoch_time:.2f} seconds")
    return total_loss / len(loader)


"""
Training loop for the model
:param model: model to train
:param train_loader: data loader for training
:param num_epochs: number of epochs to train
:param optimizer: optimizer
:param device: device to train on
:return: None
"""
def train(model: NodeLevelGNN, train_loader: DataLoader, num_epochs: int, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    logger.info("[i] Starting training...")
    start_time = time.time()
    best_loss = float('inf')
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir='logs/tensorboard')

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f'[i] Epoch {epoch:03d}, Loss: {loss:.4f}')
        
        # Log loss to TensorBoard
        writer.add_scalar('Training/Loss', loss, epoch)

        # Save checkpoint if loss improved
        if loss < best_loss:
            best_loss = loss
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch}_loss_{loss:.4f}_{current_time}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            logger.info(f'[i] New best loss: {loss:.4f}. Saved checkpoint to {checkpoint_path}')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"[i] Training completed in {elapsed_time:.2f} seconds")
    
    # Close the TensorBoard writer
    writer.close()




def main():
    # Hyperparameters
    input_dim = 3  # Should match embedding dimension
    hidden_channels = 64
    num_layers = 2
    batch_size = 128
    num_epochs = 1
    lr = 0.001
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the device to use GPUs 4, 5, 6, and 7
    # device_ids = [6, 7]
    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')  # Default to first GPU
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(device_ids[0])
    #     logger.info(f"[i] Using GPUs: {device_ids}")
    
    # Modified device setup
    device_ids = [7]
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])
        logger.info(f"[i] Using GPUs: {device_ids}")
        
        # Print the current device(s) being used
        print(f"[i] Current device: {device}")
        print(f"[i] Available devices: {torch.cuda.device_count()}")

    # Load datasets
    logger.info("[i] Loading datasets...")
    spider_train = NodeLevelGraphDataset(root='data/schema_linking_graph_dataset/', dataset_type='spider', split='train')
    # bird_train = NodeLevelGraphDataset(root='data/schema_linking_graph_dataset/', dataset_type='bird', split='train')
    
    # Create data loaders
    # train_loader = DataLoader(spider_train + bird_train, batch_size=batch_size, shuffle=True) # For multi-dataset training
    train_loader = DataLoader(spider_train, batch_size=batch_size, shuffle=True) # For single-dataset training on Spider training set
    
    # Initialize model
    logger.info("[i] Initializing model...")
    model = NodeLevelGNN(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # Move model to primary GPU first, then wrap with DataParallel
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create results directory if it doesn't exist
    # results_dir = Path('gnn/results')
    # results_dir.mkdir(exist_ok=True)
    
    # Train the model
    train(model, train_loader, num_epochs, optimizer, device)


if __name__ == '__main__':
    main()