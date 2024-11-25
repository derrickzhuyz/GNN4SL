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


class LinkLevelGNNRunner:
    def __init__(self, 
                 model: LinkLevelGNN,
                 train_dataset: Optional[LinkLevelGraphDataset],
                 test_dataset: Optional[LinkLevelGraphDataset],
                 device: torch.device,
                 val_ratio: Optional[float] = 0.1,  # 10% of train set for validation
                 lr: float = 1e-4,
                 batch_size: int = 1,
                 tensorboard_dir: str = 'gnn/tensorboard/link_level/train'):
        """
        Runner for LinkLevelGNN: train, validate, and test
        
        Args:
            model: LinkLevelGNN instance
            train_dataset: Training dataset
            test_dataset: Test dataset (dev set for spider and bird)
            device: torch device
            val_ratio: Ratio of training data to use for validation
            lr: Learning rate
            batch_size: Batch size (usually 1 for full graphs)
            tensorboard_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter(tensorboard_dir)
        if train_dataset is not None:
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
        else:
            self.train_loader = None
            self.val_loader = None

        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.test_loader = None
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        if train_dataset is not None:
            logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}")
        if test_dataset is not None:
            logger.info(f"Dataset sizes - Test: {len(test_dataset)}")
        self.training_start_time = None
        self.epoch_start_time = None


    """
    Format seconds into human readable time string
    :param seconds: seconds to format
    :return: human readable time string
    """
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time string"""
        return str(timedelta(seconds=int(seconds)))


    """
    Extract ground truth labels from graph data
    :param data: graph data
    :return: ground truth labels
    """
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


    """
    Calculate loss and metrics: accuracy, precision, recall, f1, and positive ratio
    :param predictions: predictions
    :param labels: ground truth labels
    :return: loss and metrics
    """
    def _calculate_metrics(self, predictions: torch.Tensor, labels: torch.Tensor):
        # Use weighted BCE loss to handle class imbalance
        pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=1.0, max=10.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(predictions, labels)
        
        # Convert predictions to binary (0 or 1) using 0.5 as threshold
        binary_preds = (torch.sigmoid(predictions) > 0.5).float()
        
        # Calculate detailed metrics
        tp = ((binary_preds == 1) & (labels == 1)).sum().float()
        fp = ((binary_preds == 1) & (labels == 0)).sum().float()
        tn = ((binary_preds == 0) & (labels == 0)).sum().float()
        fn = ((binary_preds == 0) & (labels == 1)).sum().float()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Log class distribution
        pos_ratio = labels.mean().item()
        logger.info(f"Positive samples ratio: {pos_ratio:.3f}")
        logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return loss, {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'pos_ratio': pos_ratio
        }


    """
    Log metrics to TensorBoard
    :param metrics: metrics results
    :param step: step number
    :param prefix: prefix for the metrics
    """
    def _log_metrics(self, metrics: dict, step: int, prefix: str = ''):
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)


    """
    Train for one epoch with time tracking
    """
    def train_epoch(self):
        self.model.train()
        total_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'pos_ratio': 0}
        num_batches = 0
        
        epoch_start = time.time()
        
        # Wrap the training loop with tqdm for progress bar
        for batch_idx, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training"):
            batch_start = time.time()
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
                loss, metrics = self._calculate_metrics(predictions, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_metrics['loss'] += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1
                
                logger.info(f"Batch {batch_idx} - Loss: {loss.item():.4f}, F1: {metrics['f1']:.4f}")
                
                # Log batch metrics
                self._log_metrics(metrics, self.global_step, prefix='batch')
                self.global_step += 1
                
                batch_time = time.time() - batch_start
                logger.info(f"Batch {batch_idx} completed in {batch_time:.2f}s - "
                          f"Loss: {loss.item():.4f}, F1: {metrics['f1']:.4f}")
                
                # Log batch time to TensorBoard
                self.writer.add_scalar('time/batch_seconds', batch_time, self.global_step)
                
            except Exception as e:
                logger.error(f"Batch {batch_idx}: Error processing batch: {str(e)}")
                logger.exception("Full traceback:")
                continue
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch completed in {self._format_time(epoch_time)}")
        
        # Log epoch time to TensorBoard
        self.writer.add_scalar('time/epoch_seconds', epoch_time, self.current_epoch)
        
        if num_batches == 0:
            logger.error("No valid batches found in training epoch!")
            return {k: 0.0 for k in total_metrics}
        
        # Calculate and log epoch averages
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        self._log_metrics(avg_metrics, self.current_epoch, prefix='train')
        return avg_metrics, epoch_time
    

    """
    Validate the model with TensorBoard logging
    """
    def validate(self):
        self.model.eval()
        total_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'pos_ratio': 0}
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
                
                loss, metrics = self._calculate_metrics(predictions, labels)
                
                # Update metrics
                total_metrics['loss'] += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1
        
        # Calculate and log validation averages
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        self._log_metrics(avg_metrics, self.current_epoch, prefix='val')
        return avg_metrics


    """
    Train the model for a given number of epochs
    :param num_epochs: number of epochs to train
    :param checkpoint_dir: directory to save checkpoints
    :param checkpoint_name: name of the checkpoint file
    """
    def train(self, num_epochs: int, checkpoint_dir: str, checkpoint_name: str):
        best_f1 = 0.0
        self.global_step = 0
        total_start_time = time.time()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        epoch_times = []
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train and log metrics
            train_metrics, epoch_time = self.train_epoch()
            epoch_times.append(epoch_time)
            
            # Validate and log metrics
            val_metrics = self.validate()
            
            # Calculate ETA
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            eta = avg_epoch_time * (num_epochs - epoch - 1)
            
            logger.info(f"Epoch {epoch+1} completed in {self._format_time(epoch_time)}")
            logger.info(f"Average epoch time: {self._format_time(avg_epoch_time)}")
            logger.info(f"Estimated time remaining: {self._format_time(eta)}")
            
            # Save checkpoint if validation F1 improved
            if val_metrics['f1'] >= best_f1:
                best_f1 = val_metrics['f1']
                self.save_checkpoint(checkpoint_dir, checkpoint_name)
                logger.info(f"Saved new best model with F1: {best_f1:.4f}")
        
        total_time = time.time() - total_start_time
        logger.info(f"\nTraining completed in {self._format_time(total_time)}")
        logger.info(f"Average epoch time: {self._format_time(sum(epoch_times) / num_epochs)}")
        
        # Log final timing statistics to TensorBoard
        self.writer.add_scalar('time/total_training_hours', total_time / 3600, 0)
        self.writer.add_scalar('time/average_epoch_minutes', (sum(epoch_times) / num_epochs) / 60, 0)
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Save timing statistics to file (NOTE: No need to save timing txt any more.)
        # timing_stats = {
        #     'total_training_time': self._format_time(total_time),
        #     'average_epoch_time': self._format_time(sum(epoch_times) / num_epochs),
        #     'epoch_times': [self._format_time(t) for t in epoch_times],
        #     'start_time': datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        #     'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # }
        
        # timing_path = os.path.join(checkpoint_dir, 'training_timing.txt')
        # with open(timing_path, 'w') as f:
        #     for key, value in timing_stats.items():
        #         if key == 'epoch_times':
        #             f.write(f"Epoch times:\n")
        #             for i, t in enumerate(value, 1):
        #                 f.write(f"  Epoch {i}: {t}\n")
        #         else:
        #             f.write(f"{key}: {value}\n")
        
        # logger.info(f"Saved timing statistics to {timing_path}")
    

    """
    Save the model checkpoint
    :param checkpoint_dir: directory to save checkpoints
    :param filename: name of the checkpoint file
    """
    def save_checkpoint(self, checkpoint_dir: str, filename: str):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        

    """
    Test the model
    """
    def test(self):
        self.model.eval()
        all_predictions = []
        total_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'pos_ratio': 0}
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
                    
                    loss, metrics = self._calculate_metrics(test_predictions, labels)
                    
                    # Update metrics
                    total_metrics['loss'] += loss.item()
                    for k, v in metrics.items():
                        total_metrics[k] += v
                    num_batches += 1
        
        # Calculate and log test averages
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        self._log_metrics(avg_metrics, 0, prefix='test')  # Use step 0 for test metrics
        return all_predictions, avg_metrics