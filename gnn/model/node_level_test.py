from typing import Tuple, Dict, List, Any
import torch
from torch_geometric.loader import DataLoader
from gnn.model.node_level_model import NodeLevelGNN
from gnn.graph_data.node_level_graph_dataset import NodeLevelGraphDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
from loguru import logger
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

logger.add("logs/node_level_testing.log", rotation="50 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


"""
Evaluate model on dev set and save predictions
:param model: model to evaluate
:param loader: data loader
:param device: device to evaluate on
:param dataset: original dataset for accessing example data
:return: metrics dictionary and predictions
"""
@torch.no_grad()
def evaluate(
    model: NodeLevelGNN, loader: DataLoader, device: torch.device, dataset: NodeLevelGraphDataset
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()  
    preds, labels = [], []
    predictions_json = []
    batch_start_idx = 0  # Track the starting index for each batch
    
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        out = model(batch)
        pred = (out > 0.5).float()
        
        preds.append(pred)
        labels.append(batch.y)
        
        # Process each graph in the batch
        for batch_idx, graph_idx in enumerate(batch.batch.unique()):
            mask = batch.batch == graph_idx
            graph_pred = pred[mask]
            
            # Get the original data example
            example_idx = batch_start_idx + batch_idx
            if example_idx >= len(dataset.raw_data):
                continue
                
            curr_example = dataset.raw_data[example_idx]
            example = curr_example.copy()
            
            # Update tables with predictions
            node_idx = 0
            for table_idx, table in enumerate(example['tables']):
                # Add predicted relevance while keeping original relevance
                table['predict_relevant'] = int(graph_pred[node_idx].cpu().item())
                node_idx += 1
                
                # Process columns
                for col_idx, col in enumerate(table['columns']):
                    col['predict_relevant'] = int(graph_pred[node_idx].cpu().item())
                    node_idx += 1
            
            predictions_json.append(example)
        
        batch_start_idx += len(batch.batch.unique())
    
    # Calculate metrics
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Move to CPU for metrics calculation
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(labels_np, preds_np),
        'precision': precision_score(labels_np, preds_np, average='binary'),
        'recall': recall_score(labels_np, preds_np, average='binary'),
        'f1': f1_score(labels_np, preds_np, average='binary')
    }
    
    return metrics, predictions_json


def main():
    # Hyperparameters
    input_dim = 3
    hidden_channels = 64
    num_layers = 2
    batch_size = 128
    model_path = 'checkpoints/model_epoch_0_loss_0.4089_20241118_083841.pt'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir='logs/tensorboard')
    
    # Load datasets
    logger.info("[i] Loading datasets...")
    spider_test = NodeLevelGraphDataset(root='data/schema_linking_graph_dataset/', dataset_type='spider', split='dev')
    test_loader = DataLoader(spider_test, batch_size=batch_size)
    
    # Initialize model
    logger.info("[i] Initializing model...")
    model = NodeLevelGNN(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    
    # Load model weights - Modified this part
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"[i] Model weights loaded from epoch {checkpoint['epoch']}")
    
    # Evaluate the model
    metrics, predictions = evaluate(model, test_loader, device, spider_test)
    
    # Log metrics
    logger.info(f'[i] Test Metrics:')
    logger.info(f'    Accuracy:  {metrics["accuracy"]:.4f}')
    logger.info(f'    Precision: {metrics["precision"]:.4f}')
    logger.info(f'    Recall:    {metrics["recall"]:.4f}')
    logger.info(f'    F1:        {metrics["f1"]:.4f}')
    
    # Log metrics to TensorBoard
    writer.add_scalar('Test/Accuracy', metrics['accuracy'])
    writer.add_scalar('Test/Precision', metrics['precision'])
    writer.add_scalar('Test/Recall', metrics['recall'])
    writer.add_scalar('Test/F1', metrics['f1'])
    
    # Save predictions
    results_dir = Path('gnn/results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f'Predictions saved to {results_dir}/predictions.json')
    
    # Close the TensorBoard writer
    writer.close()



if __name__ == '__main__':
    main() 