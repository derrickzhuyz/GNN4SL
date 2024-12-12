from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gnn.model.link_level_model import BaseLinkLevelGNN
from gnn.graph_data.link_level_graph_dataset import LinkLevelGraphDataset
from tqdm import tqdm
import os
import time
from datetime import datetime, timedelta
from loguru import logger
from sklearn.metrics import roc_auc_score
import sys
from collections import defaultdict
import torch.nn.functional as F

# Remove default logger
logger.remove()
# Add file handler with INFO level
logger.add("logs/link_level_runner.log", 
           rotation="50 MB", 
           level="INFO",
           format="{time} {level} {message}",
           compression="zip")
# Add console handler with WARNING level
logger.add(sys.stderr, level="WARNING")


class LinkLevelGNNRunner:
    def __init__(self, 
                 model: BaseLinkLevelGNN,
                 train_dataset: Optional[LinkLevelGraphDataset],
                 test_dataset: Optional[LinkLevelGraphDataset],
                 device: torch.device,
                 val_ratio: Optional[float] = 0.1,  # Proportion of train set for validation
                 val_dataset_type: str = 'bird',  # Use 'spider', 'bird', or 'combined' for validation
                 lr: float = 1e-4,
                 batch_size: int = 1,  # Force batch_size=1
                 threshold: float = 0.5,  # Threshold for binary prediction
                 tensorboard_dir: str = 'gnn/tensorboard/link_level',
                 negative_sampling: bool = False,
                 negative_sampling_ratio: float = 2.0,
                 negative_sampling_method: str = 'random'):
        """
        Runner for LinkLevelGNN: train, validate, and test
        
        Args:
            model: LinkLevelGNN instance
            train_dataset: Training dataset
            test_dataset: Test dataset (dev set for spider and bird)
            device: torch device
            val_ratio: Ratio of training data to use for validation
            val_dataset_type: Type of validation dataset to use ('spider', 'bird', or 'combined')
            lr: Learning rate
            batch_size: Batch size (usually 1 for full graphs)
            tensorboard_dir: Directory for TensorBoard logs
            negative_sampling: Whether to use negative sampling during training
            negative_sampling_ratio: Ratio of negative to positive samples (e.g., 3.0 means 3 negative samples for each positive)
            negative_sampling_method: Method for negative sampling: random or hard negative mining
        """
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter(tensorboard_dir)
        self.global_step = 0
        self.prediction_method = model.prediction_method
        self.threshold = threshold

        if train_dataset is not None:
            # Get validation dataset based on type
            if val_dataset_type not in ['spider', 'bird', 'combined']:
                raise ValueError("val_dataset_type must be 'spider', 'bird', or 'combined'")
            
            # Filter validation data based on dataset type
            all_data = [train_dataset[i] for i in range(len(train_dataset))]  # Access data using indexing
            if val_dataset_type != 'combined':
                # Filter graphs based on dataset type
                filtered_indices = [i for i, data in enumerate(all_data) 
                                  if data.dataset_type == val_dataset_type]
                filtered_data = [all_data[i] for i in filtered_indices]
                val_size = int(len(filtered_data) * val_ratio)
                train_size = len(all_data) - val_size
                
                # Create validation subset from filtered data
                val_subset = filtered_data[:val_size]
                # Use remaining data (including other dataset type) for training
                train_subset = [data for i, data in enumerate(all_data) 
                              if i not in filtered_indices[:val_size]]
            else:
                # Original combined validation logic
                train_size = len(train_dataset)
                val_size = int(train_size * val_ratio)
                train_size = train_size - val_size
                
                train_subset, val_subset = torch.utils.data.random_split(
                    train_dataset, 
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)  # For reproducibility
                )
            
            self.train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
            self.val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            
            logger.info(f"Dataset sizes - Training: {len(train_subset)}, "
                       f"Validation ({val_dataset_type}): {len(val_subset)}")
        else:
            self.train_loader = None
            self.val_loader = None

        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
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
        self.negative_sampling = negative_sampling
        self.negative_sampling_ratio = negative_sampling_ratio
        self.negative_sampling_method = negative_sampling_method


    """
    Format seconds into human readable time string
    :param seconds: seconds to format
    :return: human readable time string
    """
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time string"""
        return str(timedelta(seconds=int(seconds)))


    """
    Extract ground truth labels from graph data (including negative sampling during training if enabled)
    :param data: graph data
    :return: ground truth labels
    """
    def _extract_labels(self, data):
        logger.info("Extracting labels...")
        
        # First find positive pairs
        positive_pairs = []
        labels = []
        question_indices = [i for i, type_ in enumerate(data.node_types[0]) 
                           if type_ == 'question']
        schema_elements = [(i, type_) for i, type_ in enumerate(data.node_types[0]) 
                          if type_ in ['table', 'column']]
        
        # Get positive pairs
        for q_idx in question_indices:
            # Find edges connected to this question node
            edge_mask = (data.edge_index[0] == q_idx) | (data.edge_index[1] == q_idx)
            connected_edges = data.edge_index[:, edge_mask]
            # logger.info(f"Question {q_idx} has {connected_edges.shape[1]} connected edges")
            
            # Create labels for all schema elements (tables and columns) connected to this question node
            for i, type_ in schema_elements:
                # Check if there is an edge between question and this node
                is_connected = torch.any(
                    ((connected_edges[0] == q_idx) & (connected_edges[1] == i)) |
                    ((connected_edges[0] == i) & (connected_edges[1] == q_idx))
                ).item()
                
                if is_connected:
                    positive_pairs.append((q_idx, i))
                    labels.append(1.0)
        
        # If negative sampling is enabled during training, sample negative pairs
        if self.negative_sampling and self.model.training:
            num_positive = len(positive_pairs)
            num_negative = int(num_positive * self.negative_sampling_ratio)
            negative_pairs = self._sample_negative_pairs(data, positive_pairs, num_negative)
            
            # Add negative samples to labels: 0 for negative pairs, 1 for positive pairs
            for _ in negative_pairs:
                labels.append(0.0)
        else:
            # During validation/testing or if negative sampling is not enabled, use the full set of negative pairs
            for q_idx in question_indices:
                for i, type_ in schema_elements:
                    if (q_idx, i) not in positive_pairs:
                        labels.append(0.0)
        
        labels_tensor = torch.tensor(labels, device=self.device)
        logger.info(f"Generated {len(labels)} labels (Positive: {len(positive_pairs)}, "
                   f"Negative: {len(labels) - len(positive_pairs)})")
        return labels_tensor


    """
    Calculate loss and metrics: accuracy, precision, recall, f1, auc, and positive ratio
    :param predictions: predictions
    :param labels: ground truth labels
    :return: loss and metrics
    """
    def _calculate_metrics(self, predictions: torch.Tensor, labels: torch.Tensor):
        # Use weighted BCE loss to handle class imbalance
        pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=1.0, max=10.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(predictions, labels)
        
        # Get probabilities and binary predictions
        with torch.no_grad():  # Prevent gradient computation for metrics
            # Apply sigmoid here since we're not applying it in the prediction step
            probs = torch.sigmoid(predictions.detach())
            binary_preds = (probs > self.threshold).float()
            
            # Calculate detailed metrics
            tp = ((binary_preds == 1) & (labels == 1)).sum().float()
            fp = ((binary_preds == 1) & (labels == 0)).sum().float()
            tn = ((binary_preds == 0) & (labels == 0)).sum().float()
            fn = ((binary_preds == 0) & (labels == 1)).sum().float()
            
            total_samples = binary_preds.size(0)
            
            # Calculate ratios
            true_pos_ratio = (labels == 1).sum().float() / total_samples
            true_neg_ratio = (labels == 0).sum().float() / total_samples
            pred_pos_ratio = (binary_preds == 1).sum().float() / total_samples
            pred_neg_ratio = (binary_preds == 0).sum().float() / total_samples
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            # Calculate AUC-ROC using detached tensors
            try:
                auc = roc_auc_score(
                    labels.cpu().detach().numpy(),
                    probs.cpu().detach().numpy()
                )
            except ValueError:  # Handle case where all labels are of one class
                auc = 0.0
                logger.warning("Could not calculate AUC - possibly all labels are of one class")
            
            # Log class distribution and metrics
            logger.info(f"Ground Truth - Positive: {true_pos_ratio:.3f}, Negative: {true_neg_ratio:.3f}")
            logger.info(f"Predictions - Positive: {pred_pos_ratio:.3f}, Negative: {pred_neg_ratio:.3f}")
            logger.info(f"AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return loss, {
                'accuracy': accuracy.item(),
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'auc': auc,
                'true_pos_ratio': true_pos_ratio.item(),
                'true_neg_ratio': true_neg_ratio.item(),
                'pred_pos_ratio': pred_pos_ratio.item(),
                'pred_neg_ratio': pred_neg_ratio.item()
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
    :param current_epoch: current epoch number, to indicate the current progress
    :param num_epochs: total number of epochs, to indicate the total progress
    """
    def train_epoch(self, current_epoch: int, end_epoch: int):
        self.model.train()
        total_metrics = {
            'loss': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc': 0,
            'true_pos_ratio': 0,
            'true_neg_ratio': 0,
            'pred_pos_ratio': 0,
            'pred_neg_ratio': 0
        }
        num_batches = 0

        epoch_start = time.time()
        desc_info = f"Epoch {current_epoch + 1}/{end_epoch}"
        for data in tqdm(self.train_loader, total=len(self.train_loader), desc=desc_info):
            batch_start = time.time()
            logger.info("Processing batch")

            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            node_embeddings = self.model(data.x, data.edge_index, data.edge_type)
            
            # Normalize embeddings if using dot product
            if self.prediction_method == 'dot_product':
                node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)

            # Find question nodes and schema elements
            question_indices = [i for i, type_ in enumerate(data.node_types[0]) 
                              if type_ == 'question']
            schema_elements = [(i, type_) for i, type_ in enumerate(data.node_types[0]) 
                             if type_ in ['table', 'column']]

            if not schema_elements:
                logger.warning("No schema elements found")
                continue
            if not question_indices:
                logger.warning("No question nodes found")
                continue

            # First, get all positive pairs and their labels
            positive_pairs = []
            for q_idx in question_indices:
                edge_mask = (data.edge_index[0] == q_idx) | (data.edge_index[1] == q_idx)
                connected_edges = data.edge_index[:, edge_mask]
                
                for element, type_ in schema_elements:
                    is_connected = torch.any(
                        ((connected_edges[0] == q_idx) & (connected_edges[1] == element)) |
                        ((connected_edges[0] == element) & (connected_edges[1] == q_idx))
                    ).item()
                    
                    if is_connected:
                        positive_pairs.append((q_idx, element))

            # Get negative pairs if negative sampling is enabled
            pairs_to_evaluate = positive_pairs.copy() # Positive pairs are pairs we want to evaluate
            if self.negative_sampling and self.model.training:
                num_positive = len(positive_pairs)
                num_negative = int(num_positive * self.negative_sampling_ratio)
                negative_pairs = self._sample_negative_pairs(data, positive_pairs, num_negative)
                pairs_to_evaluate.extend(negative_pairs)
            else:
                # If negative sampling is not enabled, add all negative pairs
                for q_idx in question_indices:
                    for element, type_ in schema_elements:
                        if (q_idx, element) not in positive_pairs: # Not a positive pair
                            pairs_to_evaluate.append((q_idx, element))

            # Get predictions only for the pairs we want to evaluate
            predictions = []
            labels = []
            for q_idx, schema_idx in pairs_to_evaluate:
                q_embedding = node_embeddings[q_idx]
                schema_embedding = node_embeddings[schema_idx]
                if self.prediction_method == 'concat_mlp':
                    pair_embedding = torch.cat([q_embedding, schema_embedding])
                    pred = self.model.link_predictor(pair_embedding)
                else:  # dot_product
                    pred = (q_embedding * schema_embedding).sum(dim=-1, keepdim=True)
                predictions.append(pred)
                # Add label (1 for positive pairs, 0 for negative pairs)
                labels.append(1.0 if (q_idx, schema_idx) in positive_pairs else 0.0)

            if not predictions:
                logger.warning("No predictions generated")
                continue

            try:
                predictions = torch.cat(predictions)
                labels = torch.tensor(labels, device=self.device)

                logger.info(f"Predictions shape: {predictions.shape}")
                logger.info(f"Labels shape: {labels.shape}")

                if predictions.size(0) != labels.size(0):
                    logger.error(f"Mismatch in predictions ({predictions.size(0)}) and labels ({labels.size(0)}) size")
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

                logger.info(f"Loss: {loss.item():.6f}, F1: {metrics['f1']:.6f}, AUC: {metrics['auc']:.6f}")

                # Log batch metrics
                self._log_metrics(metrics, self.global_step, prefix='batch')
                self.global_step += 1

                batch_time = time.time() - batch_start
                logger.info(f"Batch completed in {batch_time:.2f}s - Loss: {loss.item():.6f}, F1: {metrics['f1']:.6f}, AUC: {metrics['auc']:.6f}")

                # Log batch time to TensorBoard
                self.writer.add_scalar('time/batch_seconds', batch_time, self.global_step)

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
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
        total_metrics = {
            'loss': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc': 0,
            'true_pos_ratio': 0,
            'true_neg_ratio': 0,
            'pred_pos_ratio': 0,
            'pred_neg_ratio': 0
        }
        num_batches = 0
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validating"):
                data = data.to(self.device)
                
                # Forward pass
                node_embeddings = self.model(data.x, data.edge_index, data.edge_type)
                
                # Find question nodes
                question_indices = [i for i, type_ in enumerate(data.node_types[0]) if type_ == 'question']
                logger.info(f"Found {len(question_indices)} question nodes")
                
                # Find schema elements
                schema_elements = [(i, type_) for i, type_ in enumerate(data.node_types[0]) if type_ in ['table', 'column']]
                logger.info(f"Found {len(schema_elements)} schema elements")
                
                if not schema_elements or not question_indices:
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
                    for element, type_ in schema_elements:
                        schema_embedding = node_embeddings[element]
                        if self.prediction_method == 'concat_mlp':
                            pair_embedding = torch.cat([q_embedding, schema_embedding])
                            pred = self.model.link_predictor(pair_embedding)
                        else:  # dot_product
                            pred = (q_embedding * schema_embedding).sum(dim=-1, keepdim=True)
                        
                        predictions.append(pred)
                        is_connected = torch.any(
                            ((connected_edges[0] == q_idx) & (connected_edges[1] == element)) |
                            ((connected_edges[0] == element) & (connected_edges[1] == q_idx))
                        ).item()  # Convert to Python boolean
                        labels.append(float(is_connected))
                
                if not predictions:  # Skip if no predictions
                    logger.warning("Validation batch is empty, skipping this batch.")
                    continue
                
                # Calculate metrics
                predictions = torch.cat(predictions)
                labels = torch.tensor(labels, device=self.device)
                loss, metrics = self._calculate_metrics(predictions, labels)

                # Calculate and log the ratio of predicted positive labels
                # predicted_positive_ratio = (torch.sigmoid(predictions) > 0.5).float().mean().item()
                # print(f"[Val INFO] Predicted positive ratio: {predicted_positive_ratio:.3f}")
                
                # Update metrics
                total_metrics['loss'] += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1
        
        # Calculate and log validation averages
        if num_batches > 0:
            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        else:
            avg_metrics = {k: 0.0 for k in total_metrics}
        
        self._log_metrics(avg_metrics, self.current_epoch, prefix='val')
        return avg_metrics


    """
    Train the model for a given number of epochs
    :param num_epochs: number of epochs to train
    :param checkpoint_dir: directory to save checkpoints
    :param checkpoint_name: name of the checkpoint file
    :param resume_from: path to checkpoint file to resume training from
    :param metric: metric to use for saving the best model: 'auc' or 'f1'
    """
    def train(self, num_epochs: int, checkpoint_dir: str, checkpoint_name: str, resume_from: str = None, metric: str = 'auc'):
        if metric not in ['auc', 'f1']:
            raise ValueError("Metric must be either 'auc' or 'f1'")

        # Load checkpoint if it is a resume training
        start_epoch = 0
        best_model_f1 = 0.0
        best_model_auc = 0.0
        if resume_from and os.path.exists(resume_from):
            try:
                logger.info(f"[IMPORTANT] Loading checkpoint from {resume_from} for resuming training: {num_epochs} epochs.")
                checkpoint = torch.load(resume_from)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Handle possible missing keys
                start_epoch = checkpoint.get('epoch', 0)
                best_model_f1 = checkpoint.get('f1', 0.0)
                best_model_auc = checkpoint.get('auc', 0.0)
                self.global_step = checkpoint.get('global_step', 0)
                logger.info(f"[IMPORTANT] Resuming training with best model F1: {best_model_f1:.6f}, AUC: {best_model_auc:.6f}")
                # Modify checkpoint name to indicate it's a continuation
                base_name, ext = os.path.splitext(checkpoint_name)
                checkpoint_name = f"{base_name}_resume_{num_epochs}ep{ext}"
                
                if start_epoch == 0:
                    logger.warning(f"No epoch information found in checkpoint, starting from epoch 0. Previous best model F1: {best_model_f1:.6f}, AUC: {best_model_auc:.6f}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.warning("Starting training from scratch")
                start_epoch = 0
                best_model_f1 = 0.0
                best_model_auc = 0.0
                self.global_step = 0
        
        self.model = self.model.to(self.device)
        total_start_time = time.time()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Starting/Resuming training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        epoch_times = []
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\nEpoch {epoch+1}/{start_epoch + num_epochs}")
            
            # Train and log metrics
            train_metrics, epoch_time = self.train_epoch(current_epoch=epoch, end_epoch=start_epoch + num_epochs)
            epoch_times.append(epoch_time)
            
            # Validate and log metrics
            val_metrics = self.validate()
            
            # Calculate ETA
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            eta = avg_epoch_time * (start_epoch + num_epochs - epoch - 1)
            
            logger.info(f"Epoch {epoch+1} completed in {self._format_time(epoch_time)}")
            logger.info(f"Average epoch time: {self._format_time(avg_epoch_time)}")
            logger.info(f"Estimated time remaining: {self._format_time(eta)}")
            
            # Print epoch summary
            print(f"\n{'='*120}")
            print(f"Epoch {epoch+1}/{start_epoch + num_epochs} Summary:")
            print(f"{'-'*120}")
            print("Training Metrics:")
            print(f" Loss: {train_metrics['loss']:.6f}, F1: {train_metrics['f1']:.6f}, AUC: {train_metrics['auc']:.6f}, Precision: {train_metrics['precision']:.6f}, Recall: {train_metrics['recall']:.6f}")
            print(f" True Positive Ratio: {train_metrics['true_pos_ratio']:.6f}, True Negative Ratio: {train_metrics['true_neg_ratio']:.6f}, Predicted Positive Ratio: {train_metrics['pred_pos_ratio']:.6f}, Predicted Negative Ratio: {train_metrics['pred_neg_ratio']:.6f}")
            print(f"Validation Metrics:")
            print(f" Loss: {val_metrics['loss']:.6f}, F1: {val_metrics['f1']:.6f}, AUC: {val_metrics['auc']:.6f}, Precision: {val_metrics['precision']:.6f}, Recall: {val_metrics['recall']:.6f}")
            print(f" True Positive Ratio: {val_metrics['true_pos_ratio']:.6f}, True Negative Ratio: {val_metrics['true_neg_ratio']:.6f}, Predicted Positive Ratio: {val_metrics['pred_pos_ratio']:.6f}, Predicted Negative Ratio: {val_metrics['pred_neg_ratio']:.6f}")
            print(f"Timing:")
            print(f" Epoch time: {self._format_time(epoch_time)}, Avg epoch time: {self._format_time(avg_epoch_time)}, Estimated time remaining: {self._format_time(eta)}")
            print(f"{'='*120}\n")
            
            # Update best metrics and save checkpoint
            current_f1 = val_metrics['f1']
            current_auc = val_metrics['auc']
            best_model_f1 = max(best_model_f1, current_f1)
            best_model_auc = max(best_model_auc, current_auc)
            
            # Save checkpoint if specified metric improved
            if val_metrics[metric] >= max(best_model_f1 if metric == 'f1' else best_model_auc, 0.0):
                checkpoint = {
                    'epoch': epoch + 1,  # Save the next epoch number
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1': best_model_f1 if metric == 'f1' else current_f1,
                    'auc': best_model_auc if metric == 'auc' else current_auc,
                    'global_step': self.global_step,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                    'resumed_from': resume_from if resume_from else None,  # Track original model
                    'train_true_pos_ratio': train_metrics['true_pos_ratio'],
                    'train_true_neg_ratio': train_metrics['true_neg_ratio'],
                    'train_pred_pos_ratio': train_metrics['pred_pos_ratio'],
                    'train_pred_neg_ratio': train_metrics['pred_neg_ratio'],
                    'val_true_pos_ratio': val_metrics['true_pos_ratio'],
                    'val_true_neg_ratio': val_metrics['true_neg_ratio'],
                    'val_pred_pos_ratio': val_metrics['pred_pos_ratio'],
                    'val_pred_neg_ratio': val_metrics['pred_neg_ratio']
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_name))
                logger.info(f"Saved new best model with validation F1: {checkpoint['f1']:.6f}, AUC: {checkpoint['auc']:.6f} (selected by {metric.upper()})")
                print(f"\n{'*'*120}")
                print(f"Saved new best model with validation F1: {checkpoint['f1']:.6f}, AUC: {checkpoint['auc']:.6f} (selected by {metric.upper()})")
                print(f"{'*'*120}\n")

        # Print final summary
        total_time = time.time() - total_start_time
        logger.info(f"\nTraining completed in {self._format_time(total_time)}")
        logger.info(f"Best Model F1: {checkpoint['f1']:.6f}, Best Model AUC: {checkpoint['auc']:.6f} (selected by {metric.upper()})")
        logger.info(f"Average epoch time: {self._format_time(sum(epoch_times) / num_epochs)}")
        print(f"\n{'*'*120}")
        print("Training Completed!")
        print(f" Best Model F1: {checkpoint['f1']:.6f}, Best Model AUC: {checkpoint['auc']:.6f} (selected by {metric.upper()})")
        print(f" Training: True Positive Ratio: {checkpoint['train_true_pos_ratio']:.6f}, True Negative Ratio: {checkpoint['train_true_neg_ratio']:.6f}, Predicted Positive Ratio: {checkpoint['train_pred_pos_ratio']:.6f}, Predicted Negative Ratio: {checkpoint['train_pred_neg_ratio']:.6f}")
        print(f" Validation: True Positive Ratio: {checkpoint['val_true_pos_ratio']:.6f}, True Negative Ratio: {checkpoint['val_true_neg_ratio']:.6f}, Predicted Positive Ratio: {checkpoint['val_pred_pos_ratio']:.6f}, Predicted Negative Ratio: {checkpoint['val_pred_neg_ratio']:.6f}")
        print(f" Total training time: {self._format_time(total_time)}, Average epoch time: {self._format_time(sum(epoch_times) / num_epochs)}")
        print(f"{'*'*120}\n")
        
        # Log final timing statistics to TensorBoard
        self.writer.add_scalar('time/total_training_hours', total_time / 3600, 0)
        self.writer.add_scalar('time/average_epoch_minutes', (sum(epoch_times) / num_epochs) / 60, 0)
        
        # Close TensorBoard writer
        self.writer.close()
    

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
        total_metrics = {
            'loss': 0, 
            'accuracy': 0, 
            'precision': 0, 
            'recall': 0, 
            'f1': 0, 
            'auc': 0, 
            'true_pos_ratio': 0, 
            'true_neg_ratio': 0, 
            'pred_pos_ratio': 0, 
            'pred_neg_ratio': 0
        }
        num_batches = 0
        
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device)
                
                # Get predictions using predict_links method
                predictions, scores_ = self.model.predict_links(data)
                all_predictions.extend(predictions)
                
                # Calculate test metrics
                node_embeddings = self.model(data.x, data.edge_index, data.edge_type)
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
                        if self.prediction_method == 'concat_mlp':
                            pair_embedding = torch.cat([q_embedding, schema_embedding])
                            pred = self.model.link_predictor(pair_embedding)
                        else:  # dot_product
                            pred = (q_embedding * schema_embedding).sum(dim=-1, keepdim=True)
                        
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


    """
    Sample negative pairs for training (if negative sampling is enabled)
    1. Random sampling
    2. Hard negative mining using embeddings similarity: 
        - Calculate similarities with all schema elements
        - Get top-k most similar schema elements that are not positive pairs
    :param data: graph data
    :param positive_pairs: list of positive (question, schema) pairs
    :param num_negative_samples: number of negative samples to generate
    :return: list of negative (question, schema) pairs
    """
    def _sample_negative_pairs(self, data, positive_pairs, num_negative_samples):
        question_indices = [i for i, type_ in enumerate(data.node_types[0]) 
                           if type_ == 'question']
        schema_indices = [i for i, type_ in enumerate(data.node_types[0]) 
                         if type_ in ['table', 'column']]
        
        negative_pairs = []
        positive_set = set((q, s) for q, s in positive_pairs)
        
        if self.negative_sampling_method == 'random':
            # Random sampling
            while len(negative_pairs) < num_negative_samples:
                q_idx = torch.randint(0, len(question_indices), (1,)).item()
                s_idx = torch.randint(0, len(schema_indices), (1,)).item()
                pair = (question_indices[q_idx], schema_indices[s_idx])
                
                if pair not in positive_set and pair not in negative_pairs:
                    negative_pairs.append(pair)
        
        elif self.negative_sampling_method == 'hard':
            # Hard negative mining using embeddings similarity
            with torch.no_grad():
                node_embeddings = self.model(data.x, data.edge_index)
                
                for q_idx in question_indices:
                    q_embedding = node_embeddings[q_idx]
                    
                    # Calculate similarities with all schema elements
                    schema_embeddings = node_embeddings[schema_indices]
                    similarities = F.cosine_similarity(
                        q_embedding.unsqueeze(0),
                        schema_embeddings,
                        dim=1
                    )
                    
                    # Get top-k most similar schema elements that are not positive pairs
                    k = min(num_negative_samples // len(question_indices) + 1, len(schema_indices))
                    _, top_k_indices = similarities.topk(k)
                    
                    for idx in top_k_indices:
                        pair = (q_idx, schema_indices[idx.item()])
                        if pair not in positive_set and pair not in negative_pairs:
                            negative_pairs.append(pair)
                            if len(negative_pairs) >= num_negative_samples:
                                break
        else:
            raise ValueError(f"Invalid sampling method: {self.negative_sampling_method}. Please select 'random' or 'hard'.")
        
        return negative_pairs