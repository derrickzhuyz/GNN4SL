import torch
from gnn.model.link_level_model import LinkLevelModel
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from gnn.model.link_level_train import LinkLevelTrainer
from tqdm import tqdm
import os
from loguru import logger

logger.add("logs/link_level_test.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    model = LinkLevelModel(
        in_channels=384,
        hidden_channels=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    )
    
    # Load the best model checkpoint
    checkpoint_path = 'checkpoints/link_level_model/best_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded model checkpoint.")

    # Initialize trainer with the loaded model
    trainer = LinkLevelTrainer(
        model=model,
        train_dataset=None,  # No training dataset needed for testing
        test_dataset=test_dataset,
        device=device,
        val_ratio=0.1,  # Not used in testing
        lr=1e-4,
        batch_size=1
    )

    # Test and save predictions with error handling
    try:
        logger.info("Generating predictions for test set...")
        predictions, test_loss, test_acc = trainer.test()
        
        if predictions:
            output_dir = 'gnn/results/link_level/'
            model.save_predictions(
                predictions=predictions,
                output_dir=output_dir,
                split='dev',
                dataset_type='spider'
            )
            logger.info(f"Final Test Metrics - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        else:
            logger.warning("No predictions generated!")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.exception("Full traceback:")
        raise

    logger.info("Testing completed!")