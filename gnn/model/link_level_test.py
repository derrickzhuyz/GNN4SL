import torch
from gnn.model.link_level_model import LinkLevelGNN
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from gnn.model.link_level_runner import LinkLevelGNNRunner
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from loguru import logger
import datetime
import argparse

logger.add("logs/link_level_test.log", rotation="10 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


"""
Format metrics results for logging
:param metrics: metrics results
:return: formatted metrics results
"""
def format_metrics(metrics):
    return (
        f"Loss: {metrics['loss']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"Positive Ratio: {metrics['pos_ratio']:.4f}"
    )


"""
Test the link-level model
"""
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Argument parsing for model checkpoint and model name
    parser = argparse.ArgumentParser(description='Test Link Level Model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the model checkpoint to be tested')
    args = parser.parse_args()

    # Extract model name from the checkpoint path
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    logger.info(f"Testing model: {model_name}")
    
    # Model hyperparameters
    in_channels = 384
    hidden_channels = 256
    num_heads = 4
    num_layers = 5
    dropout = 0.1
    
    # Load datasets
    embed_method = 'sentence_transformer'
    try:
        test_dataset = LinkLevelGraphDataset(
            root='data/schema_linking_graph_dataset/',
            dataset_type='spider',
            split='dev',
            embed_method=embed_method
        )
        logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
        
        # Print sample data structure
        sample_data = test_dataset[0]
        logger.info("Sample data structure:")
        logger.info(f"Node types shape: {len(sample_data.node_types)}")
        logger.info(f"Edge index shape: {sample_data.edge_index.shape}")
        if hasattr(sample_data, 'node_names'):
            logger.info(f"Node names shape: {len(sample_data.node_names)}")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Initialize model
    model = LinkLevelGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load the best model checkpoint
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model checkpoint from {args.model_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.model_path}")

    # Initialize trainer with TensorBoard support
    trainer = LinkLevelGNNRunner(
        model=model,
        train_dataset=None,
        test_dataset=test_dataset,
        device=device,
        val_ratio=None,
        lr=1e-4,
        batch_size=16,
        tensorboard_dir='gnn/tensorboard/link_level/test'
    )

    # Test and save predictions with error handling
    try:
        logger.info("Generating predictions for test set...")
        predictions, test_metrics = trainer.test()
        
        if predictions:
            # Save predictions and metrics
            output_dir = 'gnn/results/link_level/'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save predictions
            model.save_predictions(
                predictions=predictions,
                output_dir=output_dir,
                split='dev',
                dataset_type='spider',
                model_name=model_name
            )
            
            # Log metrics
            logger.info(f"Final Test Metrics - {format_metrics(test_metrics)}")
            
            # Save metrics to file
            metrics_path = os.path.join(output_dir, f'test_metrics_{model_name}.txt')
            with open(metrics_path, 'w') as f:
                f.write(format_metrics(test_metrics))
            logger.info(f"Saved metrics to {metrics_path}")
            
        else:
            logger.warning("No predictions generated!")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.exception("Full traceback:")
        raise

    logger.info("Testing completed!")



if __name__ == "__main__":
    main()