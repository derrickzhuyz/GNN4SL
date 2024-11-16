import torch
from torch_geometric.loader import DataLoader
from gnn.model.homo_model import SchemaLinkingHomoGNN
from gnn.graph_data.homo_graph_dataset import SchemaLinkingHomoGraphDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
from loguru import logger
from pathlib import Path

logger.add("logs/homo_testing.log", rotation="1 MB", level="INFO",
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
def evaluate(model, loader, device, dataset):
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
    hidden_channels = 256
    num_layers = 3
    batch_size = 64
    model_path = 'checkpoints/final_model.pt'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    logger.info("[i] Loading datasets...")
    spider_test = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='dev')
    test_loader = DataLoader(spider_test, batch_size=batch_size)
    
    # Initialize model
    logger.info("[i] Initializing model...")
    model = SchemaLinkingHomoGNN(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    logger.info("[i] Model weights loaded.")
    
    # Evaluate the model
    metrics, predictions = evaluate(model, test_loader, device, spider_test)
    
    # Log metrics
    logger.info(f'[i] Test Metrics:')
    logger.info(f'    Accuracy:  {metrics["accuracy"]:.4f}')
    logger.info(f'    Precision: {metrics["precision"]:.4f}')
    logger.info(f'    Recall:    {metrics["recall"]:.4f}')
    logger.info(f'    F1:        {metrics["f1"]:.4f}')
    
    # Save predictions
    results_dir = Path('gnn/results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f'Predictions saved to {results_dir}/predictions.json')

if __name__ == '__main__':
    main() 