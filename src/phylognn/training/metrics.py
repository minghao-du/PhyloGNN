"""
Evaluation metrics for phylogenetic GNN models.
"""

import torch
import torch.nn.functional as F
from typing import Union, Dict

def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error metric.

    Args:
        pred: Predictions [batch_size, output_dim]
        target: Ground truth [batch_size, output_dim]
        
    Returns:
        MSE value as scalar tensor
    """
    return F.mse_loss(pred, target)

def mae_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error metric.

    Args:
        pred: Predictions [batch_size, output_dim]
        target: Ground truth [batch_size, output_dim]
        
    Returns:
        MAE value as scalar tensor
    """
    return F.l1_loss(pred, target)

def r2_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    R-squared (coefficient of determination) metric.

    Args:
        pred: Predictions [batch_size, output_dim]
        target: Ground truth [batch_size, output_dim]
        
    Returns:
        R² value as scalar tensor
    """
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2

def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error metric.

    Args:
        pred: Predictions [batch_size, output_dim]
        target: Ground truth [batch_size, output_dim]
        
    Returns:
        RMSE value as scalar tensor
    """
    return torch.sqrt(F.mse_loss(pred, target))

def relative_error_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Relative Error metric.

    Args:
        pred: Predictions [batch_size, output_dim]
        target: Ground truth [batch_size, output_dim]
        
    Returns:
        Mean relative error as scalar tensor
    """
    relative_error = torch.abs((target - pred) / (target + 1e-8))
    return relative_error.mean()