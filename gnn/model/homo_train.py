import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from gnn.model.homo_model import SchemaLinkingHomoGNN
from gnn.graph_data.homo_graph_dataset import SchemaLinkingHomoGraphDataset
from sklearn.metrics import f1_score
import numpy as np
from loguru import logger
from tqdm import tqdm

logger.add("logs/training.log", rotation="1 MB", level="INFO",
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
        
        # Calculate loss
        loss = F.binary_cross_entropy(out, batch.y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar description with current loss
        progress_bar.set_description(f'Training (loss={loss.item():.4f})')
    
    return total_loss / len(loader)


"""
Evaluate model on dev set. Note to disable gradient calculation for evaluation
:param model: model to evaluate
:param loader: data loader
:param device: device to evaluate on
:return: F1 score
"""
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    
    progress_bar = tqdm(loader, desc='Evaluating', leave=False)
    for batch in progress_bar:
        batch = batch.to(device)
        out = model(batch)
        
        # Convert predictions to binary (0 or 1)
        pred = (out > 0.5).float()
        
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())
    
    # Calculate F1 score
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return f1_score(labels, preds, average='binary')



def main():
    # Hyperparameters
    input_dim = 3  # Should match embedding dimension
    hidden_channels = 256
    num_layers = 3
    batch_size = 32
    num_epochs = 100
    lr = 0.001
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the device to use GPUs 4, 5, 6, and 7
    device_ids = [6, 7]
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')  # Default to first GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])
        logger.info(f"[i] Using GPUs: {device_ids}")

    
    logger.info(f"[i] Using device: {device}")
    
    # Load datasets
    logger.info("[i] Loading datasets...")
    spider_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='train')
    spider_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='dev')
    # bird_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='train')
    # bird_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='dev')
    
    # Create data loaders
    # train_loader = DataLoader(spider_train + bird_train, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(spider_train, batch_size=batch_size, shuffle=True)
    spider_dev_loader = DataLoader(spider_dev, batch_size=batch_size)
    # bird_dev_loader = DataLoader(bird_dev, batch_size=batch_size)
    
    # Initialize model
    logger.info("[i] Initializing model...")
    model = SchemaLinkingHomoGNN(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_f1 = 0
    for epoch in range(num_epochs):
        # Train
        loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        spider_f1 = evaluate(model, spider_dev_loader, device)
        # bird_f1 = evaluate(model, bird_dev_loader, device)
        
        # Calculate average F1 score
        # avg_f1 = (spider_f1 + bird_f1) / 2
        
        logger.info(f'[i] Epoch {epoch:03d}, Loss: {loss:.4f}')
        logger.info(f'[i] Spider Dev F1: {spider_f1:.4f}')
        # logger.info(f'[i] BIRD Dev F1: {bird_f1:.4f}')
        
        # Save best model
        # if avg_f1 > best_f1:
        #     best_f1 = avg_f1
        #     torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        #     logger.info(f'New best model saved with F1: {best_f1:.4f}')
        if spider_f1 > best_f1:
            best_f1 = spider_f1
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            logger.info(f'New best model saved with F1: {best_f1:.4f}')

if __name__ == '__main__':
    main()