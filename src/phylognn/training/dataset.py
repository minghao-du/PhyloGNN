"""
Dataset classes for phylogenetic tree data.
"""

from typing import List, Optional, Callable, Union
import torch
from torch_geometric.data import Dataset, Data

class PhyloDataset(Dataset):
    """
    Dataset for phylogenetic trees converted to PyTorch Geometric Data objects.

    This dataset handles loading and preprocessing of phylogenetic tree data
    for training GNN models. It supports both single-task and multi-task learning.

    Args:
        data_list: List of PyTorch Geometric Data objects
        labels: Labels/targets for each tree. Can be:
            - torch.Tensor for single-task learning [num_samples, output_dim]
            - Dict[str, torch.Tensor] for multi-task learning
        transform: Optional transform to be applied on each data object
        pre_transform: Optional transform to be applied once during initialization
        
    Example:
        Single-task:
        >>> data_list = [...]  # List of Data objects
        >>> labels = torch.randn(100, 2)  # 100 samples, 2 outputs
        >>> dataset = PhyloDataset(data_list, labels=labels)
        
        Multi-task:
        >>> labels = {
        ...     'speciation': torch.randn(100, 1),
        ...     'extinction': torch.randn(100, 1)
        ... }
        >>> dataset = PhyloDataset(data_list, labels=labels)
    """

    def __init__(
        self,
        data_list: List[Data],
        labels: Optional[Union[torch.Tensor, dict]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        self.data_list = data_list
        self.labels = labels
        self.is_multitask = isinstance(labels, dict)
        
        # Validate labels
        if labels is not None:
            if self.is_multitask:
                # Check all tasks have same number of samples
                num_samples = len(data_list)
                for task_name, task_labels in labels.items():
                    if len(task_labels) != num_samples:
                        raise ValueError(
                            f"Task '{task_name}' has {len(task_labels)} labels "
                            f"but dataset has {num_samples} samples"
                        )
            else:
                if len(data_list) != len(labels):
                    raise ValueError(
                        f"Number of data objects ({len(data_list)}) must match "
                        f"number of labels ({len(labels)})"
                    )
        
        super().__init__(None, transform, pre_transform)
        
        # Apply pre_transform if provided
        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        """
        Get a single data object.
        
        Args:
            idx: Index of the data object to retrieve
            
        Returns:
            Data object with attached label(s)
        """
        data = self.data_list[idx].clone()
        
        # Attach labels if available
        if self.labels is not None:
            if self.is_multitask:
                # Multi-task: attach all task labels
                data.y = {
                    task_name: task_labels[idx]
                    for task_name, task_labels in self.labels.items()
                }
            else:
                # Single-task
                data.y = self.labels[idx]
        
        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)
        
        return data

    def get_task_names(self) -> Optional[List[str]]:
        """
        Get list of task names for multi-task datasets.
        
        Returns:
            List of task names, or None for single-task datasets
        """
        if self.is_multitask:
            return list(self.labels.keys())
        return None
