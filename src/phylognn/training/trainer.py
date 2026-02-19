"""
Training utilities for PhyloGNN models.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Literal, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import json

@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        scheduler: Learning rate scheduler type ('plateau', 'step', 'cosine', None)
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
        scheduler_factor: Factor for learning rate reduction
        early_stopping_patience: Patience for early stopping (None to disable)
        device: Device to train on ('cuda', 'cpu', or 'mps')
        save_dir: Directory to save checkpoints and logs
        save_best_only: Whether to save only the best model
        verbose: Whether to print training progress
        gradient_clip_val: Maximum gradient norm (None to disable)
    """
    
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: Literal['adam', 'adamw', 'sgd'] = 'adam'
    scheduler: Optional[Literal['plateau', 'step', 'cosine']] = 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: Optional[int] = 20
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str = './checkpoints'
    save_best_only: bool = True
    verbose: bool = True
    gradient_clip_val: Optional[float] = None

class Trainer:
    """
    Trainer for PhyloGNN models.

    This class handles the training loop, validation, checkpointing,
    and logging for GNN models on phylogenetic tree data. Supports both
    single-task and multi-task learning.

    Args:
        model: PyTorch model to train
        config: TrainingConfig object
        loss_fn: Loss function or dict of loss functions for multi-task
        metrics: Optional dictionary of metric functions
        
    Example:
        Single-task:
        >>> model = GATBiLSTMNet(input_dim=4, output_dim=2)
        >>> config = TrainingConfig(epochs=100, batch_size=32)
        >>> trainer = Trainer(model, config)
        >>> history = trainer.fit(train_dataset, val_dataset)
        
        Multi-task:
        >>> model = MultiTaskGATNet(input_dim=4, task_configs=[...])
        >>> loss_fns = {
        ...     'task1': nn.MSELoss(),
        ...     'task2': nn.MSELoss()
        ... }
        >>> trainer = Trainer(model, config, loss_fn=loss_fns)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Determine if multi-task
        self.is_multitask = isinstance(loss_fn, dict)
        
        # Set up loss function(s)
        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics = metrics or {}
        
        # Training history
        if self.is_multitask:
            task_names = list(self.loss_fn.keys())
            self.history = {
                **{f'train_loss_{task}': [] for task in task_names},
                **{f'val_loss_{task}': [] for task in task_names},
                'train_loss_total': [],
                'val_loss_total': []
            }
        else:
            self.history = {
                'train_loss': [],
                'val_loss': [],
                **{f'train_{name}': [] for name in self.metrics.keys()},
                **{f'val_{name}': [] for name in self.metrics.keys()}
            }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler based on config."""
        if self.config.scheduler is None:
            return None
        elif self.config.scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
                verbose=self.config.verbose
            )
        elif self.config.scheduler == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        elif self.config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        if self.is_multitask:
            return self._train_epoch_multitask(train_loader)
        else:
            return self._train_epoch_single(train_loader)

    def _train_epoch_single(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch for single-task learning."""
        total_loss = 0
        metric_values = {name: 0 for name in self.metrics.keys()}
        
        pbar = tqdm(train_loader, disable=not self.config.verbose, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute metrics
            with torch.no_grad():
                for name, metric_fn in self.metrics.items():
                    metric_values[name] += metric_fn(out, batch.y).item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Average metrics
        num_batches = len(train_loader)
        results = {'loss': total_loss / num_batches}
        for name in self.metrics.keys():
            results[name] = metric_values[name] / num_batches
        
        return results

    def _train_epoch_multitask(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch for multi-task learning."""
        task_names = list(self.loss_fn.keys())
        task_losses = {task: 0 for task in task_names}
        total_loss_sum = 0
        
        pbar = tqdm(train_loader, disable=not self.config.verbose, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute loss for each task
            total_loss = 0
            for task_name in task_names:
                task_loss = self.loss_fn[task_name](
                    outputs[task_name],
                    batch.y[task_name].to(self.device)
                )
                total_loss += task_loss
                task_losses[task_name] += task_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
            
            self.optimizer.step()
            
            total_loss_sum += total_loss.item()
            pbar.set_postfix({'total_loss': total_loss.item()})
        
        # Average losses
        num_batches = len(train_loader)
        results = {
            f'loss_{task}': task_losses[task] / num_batches
            for task in task_names
        }
        results['loss_total'] = total_loss_sum / num_batches
        
        return results

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        if self.is_multitask:
            return self._validate_multitask(val_loader)
        else:
            return self._validate_single(val_loader)

    def _validate_single(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for single-task learning."""
        total_loss = 0
        metric_values = {name: 0 for name in self.metrics.keys()}
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            
            total_loss += loss.item()
            
            # Compute metrics
            for name, metric_fn in self.metrics.items():
                metric_values[name] += metric_fn(out, batch.y).item()
        
        # Average metrics
        num_batches = len(val_loader)
        results = {'loss': total_loss / num_batches}
        for name in self.metrics.keys():
            results[name] = metric_values[name] / num_batches
        
        return results

    def _validate_multitask(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for multi-task learning."""
        task_names = list(self.loss_fn.keys())
        task_losses = {task: 0 for task in task_names}
        total_loss_sum = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            outputs = self.model(batch)
            
            # Compute loss for each task
            total_loss = 0
            for task_name in task_names:
                task_loss = self.loss_fn[task_name](
                    outputs[task_name],
                    batch.y[task_name].to(self.device)
                )
                total_loss += task_loss
                task_losses[task_name] += task_loss.item()
            
            total_loss_sum += total_loss.item()
        
        # Average losses
        num_batches = len(val_loader)
        results = {
            f'loss_{task}': task_losses[task] / num_batches
            for task in task_names
        }
        results['loss_total'] = total_loss_sum / num_batches
        
        return results

    def fit(
        self,
        train_dataset,
        val_dataset=None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            train_loader: Custom training DataLoader (optional)
            val_loader: Custom validation DataLoader (optional)
            
        Returns:
            Training history dictionary
        """
        # Create data loaders if not provided
        if train_loader is None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        if val_dataset is not None and val_loader is None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Training loop
        for epoch in range(self.config.epochs):
            if self.config.verbose:
                print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Update history
            if self.is_multitask:
                for key, value in train_metrics.items():
                    self.history[f'train_{key}'].append(value)
            else:
                self.history['train_loss'].append(train_metrics['loss'])
                for name, value in train_metrics.items():
                    if name != 'loss':
                        self.history[f'train_{name}'].append(value)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Update history
                if self.is_multitask:
                    for key, value in val_metrics.items():
                        self.history[f'val_{key}'].append(value)
                    current_val_loss = val_metrics['loss_total']
                else:
                    self.history['val_loss'].append(val_metrics['loss'])
                    for name, value in val_metrics.items():
                        if name != 'loss':
                            self.history[f'val_{name}'].append(value)
                    current_val_loss = val_metrics['loss']
                
                if self.config.verbose:
                    if self.is_multitask:
                        print(f"Train Loss: {train_metrics['loss_total']:.4f}, "
                            f"Val Loss: {val_metrics['loss_total']:.4f}")
                    else:
                        print(f"Train Loss: {train_metrics['loss']:.4f}, "
                            f"Val Loss: {val_metrics['loss']:.4f}")
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(current_val_loss)
                    else:
                        self.scheduler.step()
                
                # Save best model
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.epochs_without_improvement = 0
                    if self.config.save_best_only:
                        self.save_checkpoint('best_model.pt')
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if (self.config.early_stopping_patience is not None and
                    self.epochs_without_improvement >= self.config.early_stopping_patience):
                    if self.config.verbose:
                        print(f"\nEarly stopping after {epoch + 1} epochs")
                    break
            else:
                if self.config.verbose:
                    if self.is_multitask:
                        print(f"Train Loss: {train_metrics['loss_total']:.4f}")
                    else:
                        print(f"Train Loss: {train_metrics['loss']:.4f}")
                
                if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
            
            # Save checkpoint
            if not self.config.save_best_only:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save history
        self.save_history()
        
        return self.history

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = Path(self.config.save_dir) / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if self.config.verbose:
            print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = Path(self.config.save_dir) / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.config.verbose:
            print(f"Checkpoint loaded from {checkpoint_path}")

    def save_history(self):
        """Save training history to JSON file."""
        history_path = Path(self.config.save_dir) / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        if self.config.verbose:
            print(f"Training history saved to {history_path}")

    @torch.no_grad()
    def predict(
        self,
        dataset,
        batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make predictions on a dataset.
        
        Args:
            dataset: Dataset to predict on
            batch_size: Batch size for prediction (uses config batch_size if None)
            
        Returns:
            Predictions tensor for single-task, or dict of tensors for multi-task
        """
        self.model.eval()
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        if self.is_multitask:
            # Multi-task predictions
            task_names = list(self.loss_fn.keys())
            predictions = {task: [] for task in task_names}
            
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                for task_name in task_names:
                    predictions[task_name].append(outputs[task_name].cpu())
            
            # Concatenate predictions for each task
            return {
                task: torch.cat(preds, dim=0)
                for task, preds in predictions.items()
            }
        else:
            # Single-task predictions
            predictions = []
            
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                predictions.append(out.cpu())
            
            return torch.cat(predictions, dim=0)