import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from gnn.model.link_level_model import LinkLevelModel
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from tqdm import tqdm
import os
from loguru import logger

logger.add("logs/train_link_level_model.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")

class LinkLevelTrainer:
    def __init__(self, 
                 model: LinkLevelModel,
                 train_dataset: LinkLevelGraphDataset,
                 test_dataset: LinkLevelGraphDataset,
                 device: torch.device,
                 val_ratio: float = 0.1,  # 10% of train set for validation
                 lr: float = 1e-4,
                 batch_size: int = 1):
        """
        Trainer for LinkLevelModel
        
        Args:
            model: LinkLevelModel instance
            train_dataset: Training dataset
            test_dataset: Test dataset (dev set)
            device: torch device
            val_ratio: Ratio of training data to use for validation
            lr: Learning rate
            batch_size: Batch size (usually 1 for full graphs)
        """
        self.model = model.to(device)
        self.device = device
        
        # Split training data into train and validation
        train_size = len(train_dataset)
        val_size = int(train_size * val_ratio)
        train_size = train_size - val_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
        logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    def _extract_labels(self, data):
        """Extract ground truth labels from graph data"""
        logger.info("Extracting labels...")
        logger.info(f"Edge index shape: {data.edge_index.shape}")
        
        labels = []
        question_indices = [i for i, type_ in enumerate(data.node_types[0]) 
                           if type_ == 'question']
        schema_elements = [(i, type_) for i, type_ in enumerate(data.node_types[0]) 
                          if type_ in ['table', 'column']]
        
        logger.info(f"Found {len(question_indices)} questions and {len(schema_elements)} schema elements")
        
        for q_idx in question_indices:
            # Find edges connected to this question node
            edge_mask = (data.edge_index[0] == q_idx) | (data.edge_index[1] == q_idx)
            connected_edges = data.edge_index[:, edge_mask]
            
            # logger.info(f"Question {q_idx} has {connected_edges.shape[1]} connected edges")
            
            # Create labels for all schema elements
            for i, type_ in schema_elements:
                # Check if there's an edge between question and this node
                is_connected = torch.any(
                    ((connected_edges[0] == q_idx) & (connected_edges[1] == i)) |
                    ((connected_edges[0] == i) & (connected_edges[1] == q_idx))
                ).item()  # Convert to Python boolean
                labels.append(float(is_connected))
        
        labels_tensor = torch.tensor(labels, device=self.device)
        logger.info(f"Generated {len(labels)} labels")
        return labels_tensor

    def _calculate_metrics(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Calculate loss and accuracy metrics"""
        loss = self.criterion(predictions, labels)
        
        # Convert predictions to binary (0 or 1) using 0.5 as threshold
        binary_preds = (torch.sigmoid(predictions) > 0.5).float()
        
        # Calculate accuracy
        correct = (binary_preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        
        return loss, accuracy

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        # Wrap the training loop with tqdm for progress bar
        for batch_idx, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training"):
            logger.info(f"\nProcessing batch {batch_idx}")
            
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            node_embeddings = self.model(data.x, data.edge_index)
            
            # Find question nodes
            question_indices = [i for i, type_ in enumerate(data.node_types[0]) 
                              if type_ == 'question']
            # logger.info(f"Found {len(question_indices)} question nodes")
            
            # Find schema elements
            schema_elements = [(i, type_) for i, type_ in enumerate(data.node_types[0]) 
                              if type_ in ['table', 'column']]
            # logger.info(f"Found {len(schema_elements)} schema elements")
            
            if not schema_elements:
                logger.warning(f"Batch {batch_idx}: No schema elements found")
                continue
            
            if not question_indices:
                logger.warning(f"Batch {batch_idx}: No question nodes found")
                continue
            
            # Get predictions
            predictions = []
            for q_idx in question_indices:
                q_embedding = node_embeddings[q_idx]
                
                for i, type_ in schema_elements:
                    schema_embedding = node_embeddings[i]
                    pair_embedding = torch.cat([q_embedding, schema_embedding])
                    pred = self.model.link_predictor(pair_embedding)
                    predictions.append(pred)
            
            if not predictions:
                logger.warning(f"Batch {batch_idx}: No predictions generated")
                continue
            
            try:
                predictions = torch.cat(predictions)
                labels = self._extract_labels(data)
                
                logger.info(f"Predictions shape: {predictions.shape}")
                logger.info(f"Labels shape: {labels.shape}")
                
                if predictions.size(0) != labels.size(0):
                    logger.error(f"Batch {batch_idx}: Mismatch in predictions ({predictions.size(0)}) and labels ({labels.size(0)}) size")
                    continue
                
                # Calculate metrics
                loss, accuracy = self._calculate_metrics(predictions, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_acc += accuracy
                num_batches += 1
                
                logger.info(f"Batch {batch_idx} processed successfully - Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Batch {batch_idx}: Error processing batch: {str(e)}")
                logger.exception("Full traceback:")  # This will print the full traceback
                continue
        
        if num_batches == 0:
            logger.error("No valid batches found in training epoch!")
            return 0.0, 0.0
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        return avg_loss, avg_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validating"):
                data = data.to(self.device)
                
                # Forward pass
                node_embeddings = self.model(data.x, data.edge_index)
                
                # Find question nodes
                question_indices = [i for i, type_ in enumerate(data.node_types[0]) if type_ == 'question']
                logger.info(f"Found {len(question_indices)} question nodes")
                
                # Find schema elements
                schema_elements = [(i, type_) for i, type_ in enumerate(data.node_types[0]) if type_ in ['table', 'column']]
                logger.info(f"Found {len(schema_elements)} schema elements")
                
                if not schema_elements:
                    logger.warning("Validation batch is empty, skipping this batch.")
                    continue
                
                if not question_indices:
                    logger.warning("Validation batch is empty, skipping this batch.")
                    continue
                
                # Get predictions and labels
                predictions = []
                labels = []
                
                for q_idx in question_indices:
                    q_embedding = node_embeddings[q_idx]
                    
                    # Find edges connected to this question node
                    edge_mask = (data.edge_index[0] == q_idx) | (data.edge_index[1] == q_idx)
                    connected_edges = data.edge_index[:, edge_mask]
                    
                    # Get predictions for all schema elements
                    for i, type_ in schema_elements:
                        schema_embedding = node_embeddings[i]
                        pair_embedding = torch.cat([q_embedding, schema_embedding])
                        pred = self.model.link_predictor(pair_embedding)
                        predictions.append(pred)
                        
                        # Get label (1 if connected, 0 if not)
                        is_connected = torch.any(
                            ((connected_edges[0] == q_idx) & (connected_edges[1] == i)) |
                            ((connected_edges[0] == i) & (connected_edges[1] == q_idx))
                        ).item()  # Convert to Python boolean
                        labels.append(torch.tensor([float(is_connected)], device=self.device))
                
                if not predictions:  # Skip if no predictions
                    logger.warning("Validation batch is empty, skipping this batch.")
                    continue
                
                # Calculate metrics
                predictions = torch.cat(predictions)
                labels = torch.cat(labels)
                
                loss, accuracy = self._calculate_metrics(predictions, labels)
                
                total_loss += loss.item()
                total_acc += accuracy
                num_batches += 1
                
        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        logger.info(f"Validation Loss: {avg_loss:.4f}, Validation Acc: {avg_acc:.4f}")
        
        return avg_loss, avg_acc

    def train(self, num_epochs: int, checkpoint_dir: str):
        """Train the model"""
        best_val_acc = 0.0  # Changed from best_val_loss to best_val_acc
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoint if validation accuracy improved
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(checkpoint_dir, f"best_model.pt")
                logger.info(f"Saved new best model with val_acc: {val_acc:.4f}")
    
    def save_checkpoint(self, checkpoint_dir: str, filename: str):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def test(self):
        """Test the model and return predictions"""
        self.model.eval()
        all_predictions = []
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device)
                
                # Get predictions using predict_links method
                predictions, _ = self.model.predict_links(data)
                all_predictions.extend(predictions)
                
                # Calculate test metrics
                node_embeddings = self.model(data.x, data.edge_index)
                labels = []
                test_predictions = []
                
                question_indices = [i for i, type_ in enumerate(data.node_types[0]) if type_ == 'question']
                logger.info(f"Found {len(question_indices)} question nodes")
                
                for q_idx in question_indices:
                    q_embedding = node_embeddings[q_idx]
                    edge_mask = (data.edge_index[0] == q_idx) | (data.edge_index[1] == q_idx)
                    connected_edges = data.edge_index[:, edge_mask]
                    
                    schema_elements = [(i, type_) for i, type_ in enumerate(data.node_types[0]) if type_ in ['table', 'column']]
                    
                    if not schema_elements:
                        logger.warning("Test batch is empty, skipping this batch.")
                        continue
                    
                    for i, type_ in schema_elements:
                        schema_embedding = node_embeddings[i]
                        pair_embedding = torch.cat([q_embedding, schema_embedding])
                        pred = self.model.link_predictor(pair_embedding)
                        test_predictions.append(pred)
                        
                        is_connected = torch.any(
                            ((connected_edges[0] == q_idx) & (connected_edges[1] == i)) |
                            ((connected_edges[0] == i) & (connected_edges[1] == q_idx))
                        ).item()  # Convert to Python boolean
                        labels.append(torch.tensor([float(is_connected)], device=self.device))
                
                if test_predictions:
                    test_predictions = torch.cat(test_predictions)
                    labels = torch.cat(labels)
                    
                    loss, accuracy = self._calculate_metrics(test_predictions, labels)
                    
                    total_loss += loss.item()
                    total_acc += accuracy
                    num_batches += 1
        
        avg_test_loss = total_loss / max(num_batches, 1)
        avg_test_acc = total_acc / max(num_batches, 1)
        logger.info(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        
        return all_predictions, avg_test_loss, avg_test_acc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load datasets
    embed_method = 'sentence_transformer'
    train_dataset = LinkLevelGraphDataset(
        root='data/schema_linking_graph_dataset/',
        dataset_type='spider',
        split='train',
        embed_method=embed_method
    )
    
    test_dataset = LinkLevelGraphDataset(
        root='data/schema_linking_graph_dataset/',
        dataset_type='spider',
        split='dev',  # This is our test set
        embed_method=embed_method
    )
    
    # Initialize model
    model = LinkLevelModel(
        in_channels=384,
        hidden_channels=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = LinkLevelTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,  # Pass dev set as test set
        device=device,
        val_ratio=0.1,  # Use 10% of train set for validation
        lr=1e-4,
        batch_size=8
    )
    
    # Train model
    checkpoint_dir = 'checkpoints/link_level_model/'
    trainer.train(num_epochs=1, checkpoint_dir=checkpoint_dir)
    
    # Test and save predictions
    logger.info("Generating predictions for test set...")
    predictions, test_loss, test_acc = trainer.test()
    
    # Save predictions
    output_dir = 'outputs/schema_linking/'
    model.save_predictions(
        predictions=predictions,
        output_dir=output_dir,
        split='dev',
        dataset_type='spider'
    )
    
    logger.info(f"Final Test Metrics - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    logger.info("Training and testing completed!")

if __name__ == "__main__":
    main()