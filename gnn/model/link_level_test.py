import torch
from gnn.model.link_level_model import LinkLevelGCN
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from gnn.model.link_level_runner import LinkLevelGNNRunner
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from loguru import logger
import datetime
import argparse

logger.add("logs/link_level_test.log", rotation="10 MB", level="WARNING",
           format="{time} {level} {message}", compression="zip")


"""
Format metrics results for logging
:param metrics: metrics results
:return: formatted metrics results
"""
def format_metrics(metrics):
    return (
        f"  Loss: {metrics['loss']:.6f}, "
        f"Accuracy: {metrics['accuracy']:.6f}, "
        f"Precision: {metrics['precision']:.6f}, "
        f"Recall: {metrics['recall']:.6f}, "
        f"F1: {metrics['f1']:.6f}, "
        f"AUC: {metrics['auc']:.6f}, \n"
        f"  Ground Truth Positive Ratio: {metrics['true_pos_ratio']:.6f}, "
        f"Ground Truth Negative Ratio: {metrics['true_neg_ratio']:.6f}, \n"
        f"  Prediction Positive Ratio: {metrics['pred_pos_ratio']:.6f}, "
        f"Prediction Negative Ratio: {metrics['pred_neg_ratio']:.6f}"
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
    parser.add_argument('--model_type', type=str, required=True, choices=['gcn'], default='gcn',
                       help='Type of model to use: gcn')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the model checkpoint to be tested')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['spider', 'bird'], default='spider',
                       help='Type of dataset to test on (spider or bird)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    parser.add_argument('--embed_method', type=str, required=True,
                        choices=['sentence_transformer', 'bert', 'api_small', 'api_large', 'api_mock'], 
                        default='sentence_transformer',
                        help='Embedding methods used to create the current graph dataset')
    parser.add_argument('--in_channels', type=int, default=384,
                        help='Number of input channels, aligned with embedding dimension: 384 for sentence_transformer, 768 for bert, 1536 for text-embedding-3-small (api_small), 3072 for text-embedding-3-large (api_large).')
    parser.add_argument('--prediction_method', 
                       type=str,
                       choices=['dot_product', 'concat_mlp'],
                       default='dot_product',
                       help='Method for link prediction: dot_product or concat_mlp')
    args = parser.parse_args()

    # Extract model name from the checkpoint path
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    logger.info(f"Testing model: {model_name} on {args.dataset_type} dataset")
    
    # Model hyperparameters
    in_channels = args.in_channels
    hidden_channels = 128
    num_layers = 2
    dropout = 0.1
    
    # Load datasets
    embed_method = args.embed_method
    try:
        test_dataset = LinkLevelGraphDataset(
            root='data/schema_linking_graph_dataset/',
            dataset_type=args.dataset_type,
            split='dev',
            embed_method=embed_method
        )
        logger.info(f"Loaded {args.dataset_type} test dataset with {len(test_dataset)} samples")
        
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
    if args.model_type == 'gcn':
        model = LinkLevelGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            prediction_method=args.prediction_method,
            threshold=args.threshold
        )
        model.print_model_structure()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Load the best model checkpoint
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract additional information from checkpoint
        training_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'best_model_f1': checkpoint.get('f1', 'N/A'),
            'best_model_auc': checkpoint.get('auc', 'N/A'),
            'val_metrics': checkpoint.get('val_metrics', {}),
            'train_metrics': checkpoint.get('train_metrics', {}),
            'resumed_from': checkpoint.get('resumed_from', 'N/A'),
            'train_true_pos_ratio': checkpoint.get('train_true_pos_ratio', 'N/A'),
            'train_true_neg_ratio': checkpoint.get('train_true_neg_ratio', 'N/A'),
            'train_pred_pos_ratio': checkpoint.get('train_pred_pos_ratio', 'N/A'),
            'train_pred_neg_ratio': checkpoint.get('train_pred_neg_ratio', 'N/A'),
            'val_true_pos_ratio': checkpoint.get('val_true_pos_ratio', 'N/A'),
            'val_true_neg_ratio': checkpoint.get('val_true_neg_ratio', 'N/A'),
            'val_pred_pos_ratio': checkpoint.get('val_pred_pos_ratio', 'N/A'),
            'val_pred_neg_ratio': checkpoint.get('val_pred_neg_ratio', 'N/A')
        }
        
        logger.info(f"[✓] Loaded model checkpoint from {args.model_path}")
        logger.info("Training Information:")
        logger.info(f"- Trained for {training_info['epoch']} epochs")
        logger.info(f"- Best F1: {training_info['best_model_f1']}")
        logger.info(f"- Best AUC: {training_info['best_model_auc']}")
        if training_info['resumed_from']:
            logger.info(f"- Model was resumed from: {training_info['resumed_from']}")

    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.model_path}")

    # Initialize runner (for test)
    runner = LinkLevelGNNRunner(
        model=model,
        train_dataset=None,
        test_dataset=test_dataset,
        device=device,
        val_ratio=None,
        lr=1e-4,
        batch_size=args.batch_size,
        tensorboard_dir=f'gnn/tensorboard/link_level_{args.model_type}/test/{embed_method}/'
    )

    # Test and save predictions with error handling
    try:
        logger.info("Generating predictions for test set...")
        predictions, test_metrics = runner.test()
        
        if predictions:
            # Save predictions and metrics
            output_dir = f'gnn/results/link_level_{args.model_type}/{embed_method}/'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save predictions
            model.save_predictions(
                predictions=predictions,
                output_dir=output_dir,
                split='dev',
                dataset_type=args.dataset_type,
                model_name=model_name
            )
            # Log metrics
            logger.info(f"Final Test Metrics - {format_metrics(test_metrics)}")
            
            # Save training information along with test metrics
            metrics_path = os.path.join(output_dir, 'test_metrics_results.txt')
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare the metrics summary text
            metrics_text = (
                f"\n{'='*120}\n"
                f"Model: {model_name}, Test Time: {current_time}\n"
                f"Dataset Tested on: {args.dataset_type}, Embedding Method: {embed_method}, Prediction Method: {args.prediction_method}\n"
                f"{'-'*120}\n"
                f"Training Information and Metrics:\n"
                f"  Epochs: {training_info['epoch']}, Best F1: {training_info['best_model_f1']}, Best AUC: {training_info['best_model_auc']}\n"
                f"{format_metrics(training_info['train_metrics'])}\n"
            )
            
            if training_info['resumed_from']:
                metrics_text += f"Resumed from: {training_info['resumed_from']}\n"
            
            if training_info['val_metrics']:
                metrics_text += "Final Validation Metrics:\n"
                metrics_text += format_metrics(training_info['val_metrics'])
                
            metrics_text += (
                f"\n{'-'*120}\n"
                f"Test Metrics:\n"
                f"{format_metrics(test_metrics)}\n"
                f"{'='*120}\n"
            )
            
            # Print to console and write to file
            print(metrics_text)
            with open(metrics_path, 'a') as f:
                f.write(metrics_text)

        else:
            logger.warning("No predictions generated!")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.exception("Full traceback:")
        raise

    logger.info("Testing completed!")



if __name__ == "__main__":
    main()