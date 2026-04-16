"""
Training examples for PhyloGNN models.

This script demonstrates how to train GNN models on phylogenetic tree data,
including single-task and multi-task learning scenarios.
"""

import torch
import torch.nn as nn
from ete3 import Tree
from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter
from phylognn.models import GATBiLSTMNet, MultiTaskGATNet
from phylognn.training import (
    Trainer,
    TrainingConfig,
    SplitPhyloDataset,
    mse_metric,
    mae_metric,
    r2_metric,
)


def generate_sample_data(num_trees=100, origin_time=10.0, seed=42):
    """
    Generate sample phylogenetic tree data for demonstration.

    Args:
        num_trees: Number of trees to generate
        origin_time: Origin time for tree normalization
        seed: Random seed for reproducibility

    Returns:
        List of Data objects
    """
    torch.manual_seed(seed)
    print(f"Generating {num_trees} sample trees...")

    # Initialize feature engineer and converter
    engineer = TreeFeatureEngineer(num_time_bins=101, extant_sampling_probability=0.8)

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names, add_virtual_nodes=True, num_time_bins=101
    )

    data_list = []

    for i in range(num_trees):
        # Generate random tree
        tree = Tree()
        tree.populate(15, random_branches=True)

        # Add features
        tree_with_features = engineer.add_features(tree, origin_time=origin_time)

        # Convert to graph
        data = converter.convert(tree_with_features)
        data_list.append(data)

    print(f"Generated {len(data_list)} graphs")
    return data_list


def example_1_single_task_training():
    """Example 1: Single-task regression"""
    print("\n" + "=" * 70)
    print("Example 1: Single-Task Training (Parameter Estimation)")
    print("=" * 70)

    # Generate data
    data_list = generate_sample_data(num_trees=200)

    # Generate synthetic labels (2 parameters to estimate)
    labels = torch.randn(200, 2) * 0.5 + 1.0  # Mean ~1.0, std ~0.5

    # Split into train/val/test
    train_size = int(0.7 * len(data_list))
    val_size = int(0.15 * len(data_list))

    train_data = data_list[:train_size]
    train_labels = labels[:train_size]

    val_data = data_list[train_size : train_size + val_size]
    val_labels = labels[train_size : train_size + val_size]

    test_data = data_list[train_size + val_size :]
    test_labels = labels[train_size + val_size :]

    # Create datasets
    train_dataset = SplitPhyloDataset(train_data, labels=train_labels)
    val_dataset = SplitPhyloDataset(val_data, labels=val_labels)
    test_dataset = SplitPhyloDataset(test_data, labels=test_labels)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=2,
        preprocess_dim=32,
        gat_hidden_dim=64,
        temporal_hidden_dim=128,
        head_hidden_dim=64,
        gat_heads=4,
        dropout_prob=0.2,
        num_lstm_layers=2,
        output_positive=False,
    )

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure training
    config = TrainingConfig(
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=1e-5,
        optimizer="adam",
        scheduler="plateau",
        scheduler_patience=5,
        early_stopping_patience=15,
        save_dir="./checkpoints/single_task",
        verbose=True,
    )

    # Create trainer with metrics
    trainer = Trainer(model=model, config=config, metrics={"mae": mae_metric, "r2": r2_metric})

    # Train
    print("\nStarting training...")
    history = trainer.fit(train_dataset, val_dataset)

    # Test
    print("\nEvaluating on test set...")
    predictions = trainer.predict(test_dataset)
    test_mse = mse_metric(predictions, test_labels)
    test_mae = mae_metric(predictions, test_labels)
    test_r2 = r2_metric(predictions, test_labels)

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    return history, predictions


def example_2_multitask_training():
    """Example 2: Multi-task learning"""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Task Training (Joint Parameter Estimation)")
    print("=" * 70)

    # Generate data
    data_list = generate_sample_data(num_trees=200, seed=123)

    # Generate synthetic labels for 3 tasks
    labels = {
        "speciation_rate": torch.rand(200, 1) * 2.0,  # Range: [0, 2]
        "extinction_rate": torch.rand(200, 1) * 1.0,  # Range: [0, 1]
        "sampling_prob": torch.rand(200, 1) * 0.5 + 0.5,  # Range: [0.5, 1]
    }

    # Split data
    train_size = int(0.7 * len(data_list))
    val_size = int(0.15 * len(data_list))

    train_data = data_list[:train_size]
    train_labels = {task: labels[task][:train_size] for task in labels.keys()}

    val_data = data_list[train_size : train_size + val_size]
    val_labels = {task: labels[task][train_size : train_size + val_size] for task in labels.keys()}

    test_data = data_list[train_size + val_size :]
    test_labels = {task: labels[task][train_size + val_size :] for task in labels.keys()}

    # Create datasets
    train_dataset = SplitPhyloDataset(train_data, labels=train_labels)
    val_dataset = SplitPhyloDataset(val_data, labels=val_labels)
    test_dataset = SplitPhyloDataset(test_data, labels=test_labels)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Tasks: {train_dataset.get_task_names()}")

    # Create multi-task model
    task_configs = [
        {"name": "speciation_rate", "output_dim": 1, "lstm_hidden_dim": 128, "fc_hidden_dim": 64},
        {"name": "extinction_rate", "output_dim": 1, "lstm_hidden_dim": 128, "fc_hidden_dim": 64},
        {"name": "sampling_prob", "output_dim": 1, "lstm_hidden_dim": 128, "fc_hidden_dim": 64},
    ]

    model = MultiTaskGATNet(
        input_dim=4,
        task_configs=task_configs,
        preprocess_fc_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        num_gat_layers=3,
        num_lstm_layers=2,
        dropout_prob=0.2,
    )

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure training
    config = TrainingConfig(
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        optimizer="adam",
        scheduler="plateau",
        early_stopping_patience=15,
        save_dir="./checkpoints/multitask",
        verbose=True,
    )

    # Create trainer with task-specific loss functions
    loss_fns = {
        "speciation_rate": nn.MSELoss(),
        "extinction_rate": nn.MSELoss(),
        "sampling_prob": nn.MSELoss(),
    }

    trainer = Trainer(model=model, config=config, loss_fn=loss_fns)

    # Train
    print("\nStarting training...")
    history = trainer.fit(train_dataset, val_dataset)

    # Test
    print("\nEvaluating on test set...")
    predictions = trainer.predict(test_dataset)

    print("\nTest Results:")
    for task_name in predictions.keys():
        task_pred = predictions[task_name]
        task_true = test_labels[task_name]

        task_mse = mse_metric(task_pred, task_true)
        task_mae = mae_metric(task_pred, task_true)
        task_r2 = r2_metric(task_pred, task_true)

        print(f"\n{task_name}:")
        print(f"  MSE: {task_mse:.4f}")
        print(f"  MAE: {task_mae:.4f}")
        print(f"  R²: {task_r2:.4f}")

    return history, predictions


def example_3_custom_model():
    """Example 3: Using custom model configuration"""
    print("\n" + "=" * 70)
    print("Example 3: Custom Model Configuration")
    print("=" * 70)

    # Generate data
    data_list = generate_sample_data(num_trees=150, seed=456)
    labels = torch.randn(150, 3) * 0.3 + 0.5

    # Split data
    train_size = int(0.8 * len(data_list))
    train_dataset = SplitPhyloDataset(data_list[:train_size], labels=labels[:train_size])
    val_dataset = SplitPhyloDataset(data_list[train_size:], labels=labels[train_size:])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model with custom architecture
    # Example: No preprocessing, no LSTM, just GAT + FC
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=3,
        preprocess_dim=None,  # Disable preprocessing
        gat_hidden_dim=128,
        temporal_mode="none",  # Disable temporal LSTM aggregation
        head_hidden_dim=128,
        gat_heads=8,
        dropout_prob=0.3,
        output_positive=True,  # Enforce positive outputs
    )

    print(f"\nModel: {model.__class__.__name__} (Custom: No preprocessing, No LSTM)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure training with different settings
    config = TrainingConfig(
        epochs=30,
        batch_size=32,
        learning_rate=0.0005,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="cosine",
        early_stopping_patience=None,  # Disable early stopping
        save_dir="./checkpoints/custom",
        save_best_only=False,  # Save all checkpoints
        verbose=True,
        gradient_clip_val=1.0,  # Enable gradient clipping
    )

    trainer = Trainer(model=model, config=config)

    print("\nStarting training...")
    history = trainer.fit(train_dataset, val_dataset)

    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")

    return history
