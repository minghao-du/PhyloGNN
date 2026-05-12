"""
Training utilities for split-aware PhyloGNN datasets.

Specification
-------------
This module provides a production-oriented trainer for PyTorch / PyG models
working with the following dataset conventions:

Each sample exposes its target as `data.y`.

This trainer is designed to work cleanly with:
- `SplitPhyloDataset`
- `SplitPhyloDiskDataset`
- `SplitDatasetView`

Engineering goals
-----------------
- deterministic and checkpointable training flow
- strong validation of inputs and runtime assumptions
- explicit typing and clear docstrings
- robust support for single-output training
- minimal coupling to model implementation details
- ergonomic defaults for common training workflows

Notes
-----
- The trainer assumes the model returns a `Tensor`.
- Metrics are optional.
- Best-model checkpointing, history saving, and resume-loading are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union

import json
import time
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torchmetrics import Metric, R2Score

from phylognn.training.metrics import MetricRegistry
from phylognn.training.tracking import (
    TrackerProtocol,
    TrackingConfig,
    build_epoch_metrics,
    build_final_metrics,
    build_status_metrics,
    create_tracker,
    sanitize_config_metadata,
)

# ---------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------
LossFn = Callable[[Tensor, Tensor], Tensor]
MetricSpec = Union[str, Metric]
MetricsMap = Dict[str, MetricSpec]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Attributes
    ----------
    epochs : int, default=100
        Number of training epochs.

    batch_size : int, default=32
        Batch size used for train / validation / prediction loaders when a
        custom loader is not provided.

    learning_rate : float, default=1e-3
        Initial optimizer learning rate.

    weight_decay : float, default=1e-5
        Optimizer weight decay.

    optimizer : {"adam", "adamw", "sgd"}, default="adam"
        Optimizer type.

    scheduler : {"plateau", "step", "cosine"} or None, default="plateau"
        Learning rate scheduler type.

    scheduler_patience : int, default=10
        Used as:
        - patience for ReduceLROnPlateau
        - step_size for StepLR

    scheduler_factor : float, default=0.5
        Used as:
        - factor for ReduceLROnPlateau
        - gamma for StepLR

    early_stopping_patience : int or None, default=20
        Stop training when validation loss does not improve for this many
        epochs. If None, early stopping is disabled. TOML configuration values
        `"none"`, `0`, `-1`, and omitted keys are normalized to None by the
        TOML loader.

    device : str, default=auto
        Device string, e.g. "cuda", "cpu", or "mps".

    save_dir : str, default="./checkpoints"
        Directory for checkpoints and training history.

    save_best_only : bool, default=True
        If True, only the best validation checkpoint is saved during training.
        Without validation data, training instead overwrites
        `checkpoint_latest.pt` after each epoch and emits a UserWarning. The
        final checkpoint is always saved at the end.

    verbose : bool, default=True
        Whether to print progress.

    gradient_clip_val : float or None, default=None
        Max gradient norm. If None, clipping is disabled.

    num_workers : int, default=0
        Number of DataLoader worker processes.

    pin_memory : bool, default=False
        Whether DataLoader should pin memory.

    train_shuffle : bool, default=True
        Whether the automatically created train loader shuffles data.

    non_blocking : bool, default=True
        Whether tensor transfers to device are non-blocking when possible.
    """

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    scheduler: Optional[Literal["plateau", "step", "cosine"]] = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: Optional[int] = 20
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    save_dir: str = "./checkpoints"
    save_best_only: bool = True
    verbose: bool = True
    gradient_clip_val: Optional[float] = None
    num_workers: int = 0
    pin_memory: bool = False
    train_shuffle: bool = True
    non_blocking: bool = True

    def validate(self) -> None:
        """
        Validate configuration values.
        """
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError(
                f"optimizer must be one of ('adam', 'adamw', 'sgd'), got {self.optimizer!r}."
            )
        if self.scheduler not in {None, "plateau", "step", "cosine"}:
            raise ValueError(
                "scheduler must be one of (None, 'plateau', 'step', 'cosine'), "
                f"got {self.scheduler!r}."
            )
        if self.scheduler_patience <= 0:
            raise ValueError("scheduler_patience must be > 0.")
        if not (0.0 < self.scheduler_factor <= 1.0):
            raise ValueError("scheduler_factor must be in (0, 1].")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0 or None.")
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be > 0 when provided.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _detach_item(x: Union[Tensor, float, int]) -> float:
    """
    Convert a scalar tensor / numeric value into float.
    """
    if isinstance(x, Tensor):
        if x.numel() != 1:
            raise ValueError("Expected scalar tensor when converting metric/loss to float.")
        return float(x.detach().cpu().item())
    return float(x)


def _metric_compute_to_float(x: Union[Tensor, float, int]) -> float:
    """
    Convert a computed metric value to a float for history and logging.
    """
    if isinstance(x, Tensor):
        value = x.detach().float()
        if value.numel() == 0:
            raise ValueError("Metric compute returned an empty tensor.")
        if value.numel() > 1:
            value = value.mean()
        return float(value.cpu().item())
    return float(x)


def _safe_mean(total: float, count: int) -> float:
    """
    Mean with explicit empty-check.
    """
    if count <= 0:
        raise ValueError("Cannot compute average over zero batches.")
    return total / count


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class Trainer:
    """
    Generic trainer for split-aware PyG datasets.

    Specification
    -------------
    - Works with datasets exposing target as `data.y`
    - Maintains training history across epochs
    - Supports checkpoint save/load
    - Supports optional validation and early stopping
    - Supports optional metric computation
    - Uses PyG `DataLoader` by default when loaders are not provided

    Parameters
    ----------
    model : nn.Module
        Model to train.

    config : TrainingConfig
        Training configuration.

    loss_fn : callable, optional
        Loss function(s). If omitted, defaults to `nn.MSELoss()` for
        standard regression.

    metrics : dict[str, str | torchmetrics.Metric], optional
        Each value must be a supported built-in metric key or an instantiated
        `torchmetrics.Metric`.

    tracking_config : TrackingConfig, optional
        Optional experiment tracking settings. Disabled or omitted tracking
        keeps the existing local-only training workflow and does not import
        wandb.

    tracker : TrackerProtocol, optional
        Test or custom tracker implementation. When omitted, the tracker is
        created from `tracking_config`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[LossFn] = None,
        metrics: Optional[MetricsMap] = None,
        tracking_config: Optional[TrackingConfig] = None,
        tracking_metadata: Optional[Mapping[str, object]] = None,
        tracker: Optional[TrackerProtocol] = None,
    ) -> None:
        config.validate()

        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.loss_fn: LossFn = nn.MSELoss() if loss_fn is None else loss_fn

        self.metrics = self._resolve_metrics(metrics or {})
        self._validate_loss_and_metrics()
        self._move_metrics_to_device()

        self.tracking_config = tracking_config or TrackingConfig(enabled=False)
        self.tracking_config.validate()
        self.tracking_metadata = sanitize_config_metadata(tracking_metadata or {})
        self.tracker: TrackerProtocol = (
            tracker if tracker is not None else create_tracker(self.tracking_config)
        )
        self._tracking_started = False

        self.optimizer: Optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # `current_epoch` stores the next epoch index to execute.
        # This avoids rerunning the last completed epoch after resuming.
        self.current_epoch: int = 0
        self.best_epoch: Optional[int] = None
        self.best_val_loss: float = float("inf")
        self.epochs_without_improvement: int = 0

        self.history: Dict[str, List[float]] = self._init_history()

        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._save_config()

    # -----------------------------------------------------------------
    # Validation / initialization helpers
    # -----------------------------------------------------------------
    def _validate_loss_and_metrics(self) -> None:
        """
        Validate loss / metric configuration.
        """
        if not callable(self.loss_fn):
            raise TypeError("loss_fn must be callable.")

        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dict[str, str | torchmetrics.Metric] or None.")
        if not all(isinstance(k, str) for k in self.metrics.keys()):
            raise TypeError("All metric names must be strings.")
        for name, metric in self.metrics.items():
            self._validate_metric(name, metric)

    def _resolve_metrics(self, metrics: MetricsMap) -> Dict[str, Metric]:
        """
        Resolve configured metric keys to fresh TorchMetrics instances.
        """
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict[str, str | torchmetrics.Metric] or None.")

        resolved: Dict[str, Metric] = {}
        for display_name, metric_spec in metrics.items():
            if not isinstance(display_name, str):
                raise TypeError("All metric names must be strings.")
            if isinstance(metric_spec, str):
                resolved[display_name] = MetricRegistry.create(metric_spec)
            else:
                resolved[display_name] = metric_spec
        return resolved

    def _validate_metric(self, name: str, metric: Metric) -> None:
        if not isinstance(metric, Metric):
            raise TypeError(
                f"Metric {name!r} must be a torchmetrics.Metric instance or supported metric key."
            )
        if self._distributed_training_active() and getattr(metric, "dist_sync_on_step", None):
            raise ValueError(
                f"Metric {name!r} must use dist_sync_on_step=False for distributed training."
            )

    def _distributed_training_active(self) -> bool:
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def _move_metrics_to_device(self) -> None:
        """
        Move all metric state to the trainer device.
        """
        for metric in self.metrics.values():
            metric.to(self.device)

    def _validate_metric_inputs(
        self, name: str, metric: Metric, pred: Tensor, target: Tensor
    ) -> None:
        if pred.size(0) != target.size(0):
            raise ValueError(
                f"Metric {name!r} expected prediction and target batch dimensions to match, "
                f"got {pred.size(0)} and {target.size(0)}."
            )
        if not pred.is_floating_point() or not target.is_floating_point():
            raise TypeError(f"Metric {name!r} expects floating point predictions and targets.")

        expected_outputs = getattr(metric, "num_outputs", None)
        if isinstance(metric, R2Score) and expected_outputs is not None:
            observed_outputs = 1 if pred.ndim == 1 else pred.size(-1)
            target_outputs = 1 if target.ndim == 1 else target.size(-1)
            if observed_outputs != target_outputs:
                raise ValueError(
                    f"Metric {name!r} expected prediction and target output dimensions to match, "
                    f"got {observed_outputs} and {target_outputs}."
                )
            if observed_outputs != expected_outputs:
                raise ValueError(
                    f"Metric {name!r} was configured for {expected_outputs} output(s), "
                    f"but received {observed_outputs}."
                )

    def _update_metrics(self, pred: Tensor, target: Tensor) -> None:
        for name, metric in self.metrics.items():
            self._validate_metric_inputs(name, metric, pred, target)
            metric.update(pred, target)

    def _compute_metrics(self) -> Dict[str, float]:
        results = {}
        for name, metric in self.metrics.items():
            results[name] = _metric_compute_to_float(metric.compute())
            metric.reset()
        return results

    def _init_history(self) -> Dict[str, List[float]]:
        """
        Build the training history structure.
        """
        return {
            "train_loss": [],
            "val_loss": [],
            **{f"train_{name}": [] for name in self.metrics.keys()},
            **{f"val_{name}": [] for name in self.metrics.keys()},
            "lr": [],
            "epoch_time_sec": [],
        }

    def _save_config(self) -> None:
        """
        Persist config for reproducibility.
        """
        config_path = self.save_dir / "training_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    def _completed_epoch_count(self) -> int:
        """
        Return the number of fully completed epochs recorded in history.

        The learning-rate history is appended exactly once per finished epoch
        in every training mode, so it is a stable source of truth when
        restoring checkpoints created by older trainer versions.
        """
        lr_history = self.history.get("lr", [])
        if not isinstance(lr_history, list):
            raise TypeError("history['lr'] must be a list.")
        return len(lr_history)

    def _resolve_resume_epoch(self, checkpoint: Mapping[str, Any]) -> int:
        """
        Resolve the next epoch index to execute after loading a checkpoint.

        New checkpoints store `current_epoch` as the next epoch index.
        Older checkpoints stored the last completed epoch index, so we fall
        back to the recorded history length when it is ahead.
        """
        saved_epoch = checkpoint.get("current_epoch")
        completed_epochs = self._completed_epoch_count()

        if saved_epoch is None:
            return completed_epochs
        if not isinstance(saved_epoch, int):
            raise TypeError("Checkpoint field 'current_epoch' must be an int.")
        if saved_epoch < 0:
            raise ValueError("Checkpoint field 'current_epoch' must be >= 0.")

        return max(saved_epoch, completed_epochs)

    # -----------------------------------------------------------------
    # Optimizer / scheduler
    # -----------------------------------------------------------------
    def _create_optimizer(self) -> Optimizer:
        """
        Create optimizer from configuration.
        """
        if self.config.optimizer == "adam":
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        """
        Create LR scheduler from configuration.
        """
        if self.config.scheduler is None:
            return None

        if self.config.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            )

        if self.config.scheduler == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor,
            )

        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )

        raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    # -----------------------------------------------------------------
    # Data helpers
    # -----------------------------------------------------------------
    def _create_loader(
        self,
        dataset,
        *,
        shuffle: bool,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """
        Create a default PyG DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def _move_batch_to_device(self, batch: Data) -> Data:
        """
        Move a PyG batch to the configured device.
        """
        return batch.to(self.device, non_blocking=self.config.non_blocking)

    def _get_current_lr(self) -> float:
        """
        Read learning rate from the first optimizer param group.
        """
        return float(self.optimizer.param_groups[0]["lr"])

    # -----------------------------------------------------------------
    # Target extraction
    # -----------------------------------------------------------------
    def _extract_single_target(self, batch: Data) -> Tensor:
        """
        Extract target tensor from batch.

        Raises
        ------
        AttributeError
            If `batch.y` does not exist.
        """
        if not hasattr(batch, "y"):
            raise AttributeError("Training expects batch.y, but batch has no attribute 'y'.")
        target = batch.y
        if not isinstance(target, Tensor):
            raise TypeError(f"Target batch.y must be a Tensor, got {type(target).__name__}.")
        return target

    def _prepare_single_target(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Align a single-task target tensor with model predictions.
        """
        if pred.size(0) != target.size(0):
            raise ValueError(
                "Prediction and target batch dimensions must match before loss/metric "
                f"calculation, got {pred.size(0)} and {target.size(0)}."
            )

        if pred.shape == target.shape:
            return target

        if pred.ndim == 2 and pred.size(1) == 1 and target.ndim == 1:
            return target.reshape(-1, 1)

        raise ValueError(
            "Prediction and target shapes are incompatible for loss/metric calculation: "
            f"prediction shape {tuple(pred.shape)}, target shape {tuple(target.shape)}. "
            "Single-output predictions shaped [batch, 1] accept targets shaped [batch]; "
            "all other cases require identical shapes."
        )

    # -----------------------------------------------------------------
    # Core epoch loops
    # -----------------------------------------------------------------
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.

        Returns
        -------
        Dict[str, float]
            Aggregated epoch metrics.
        """
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty.")

        self.model.train()
        return self._train_epoch_single(train_loader)

    def _train_epoch_single(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch in single-task mode.
        """
        total_loss = 0.0
        for metric in self.metrics.values():
            metric.reset()

        pbar = tqdm(
            train_loader,
            disable=not self.config.verbose,
            desc="Training",
            leave=False,
        )

        for batch in pbar:
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(batch)
            if not isinstance(pred, Tensor):
                raise TypeError(f"Model output must be Tensor, got {type(pred).__name__}.")

            target = self._extract_single_target(batch)
            target = self._prepare_single_target(pred, target)
            loss = self.loss_fn(pred, target)

            if not isinstance(loss, Tensor):
                raise TypeError("Loss function must return a torch.Tensor.")

            loss.backward()

            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )

            self.optimizer.step()

            total_loss += _detach_item(loss)

            with torch.no_grad():
                self._update_metrics(pred.detach(), target.detach())

            pbar.set_postfix(loss=f"{_detach_item(loss):.4f}", lr=f"{self._get_current_lr():.2e}")

        num_batches = len(train_loader)
        results = {"loss": _safe_mean(total_loss, num_batches)}
        results.update(self._compute_metrics())
        return results

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run one validation epoch.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        Dict[str, float]
            Aggregated validation metrics.
        """
        if len(val_loader) == 0:
            raise ValueError("val_loader is empty.")

        self.model.eval()
        return self._validate_single(val_loader)

    def _validate_single(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate one epoch in single-task mode.
        """
        total_loss = 0.0
        for metric in self.metrics.values():
            metric.reset()

        for batch in val_loader:
            batch = self._move_batch_to_device(batch)

            pred = self.model(batch)
            if not isinstance(pred, Tensor):
                raise TypeError(f"Model output must be Tensor, got {type(pred).__name__}.")

            target = self._extract_single_target(batch)
            target = self._prepare_single_target(pred, target)
            loss = self.loss_fn(pred, target)

            total_loss += _detach_item(loss)

            self._update_metrics(pred.detach(), target.detach())

        num_batches = len(val_loader)
        results = {"loss": _safe_mean(total_loss, num_batches)}
        results.update(self._compute_metrics())
        return results

    # -----------------------------------------------------------------
    # Training orchestration
    # -----------------------------------------------------------------
    def fit(
        self,
        train_dataset=None,
        val_dataset=None,
        *,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Parameters
        ----------
        train_dataset : optional
            Training dataset used to auto-create a loader when `train_loader`
            is not provided.

        val_dataset : optional
            Validation dataset used to auto-create a loader when `val_loader`
            is not provided.

        train_loader : DataLoader, optional
            Explicit training loader. If provided, `train_dataset` is ignored.

        val_loader : DataLoader, optional
            Explicit validation loader. If provided, `val_dataset` is ignored.

        Returns
        -------
        Dict[str, List[float]]
            Training history.

        Notes
        -----
        If `load_checkpoint()` was called beforehand, training resumes from the
        next epoch recorded in the checkpoint instead of restarting.

        Raises
        ------
        ValueError
            If no training data source is provided.

        Notes
        -----
        If training fails after tracking has started, tracking finalization is
        attempted but any cleanup failure is emitted as a warning so the
        original training exception remains primary.
        """
        if train_loader is None:
            if train_dataset is None:
                raise ValueError("Either train_dataset or train_loader must be provided.")
            train_loader = self._create_loader(
                train_dataset,
                shuffle=self.config.train_shuffle,
            )

        if val_loader is None and val_dataset is not None:
            val_loader = self._create_loader(
                val_dataset,
                shuffle=False,
            )

        self._start_tracking()

        try:
            for epoch_idx in range(self.current_epoch, self.config.epochs):
                epoch_num = epoch_idx + 1
                epoch_start = time.time()

                if self.config.verbose:
                    print(f"\nEpoch {epoch_num}/{self.config.epochs}")

                train_metrics = self.train_epoch(train_loader)
                self._append_train_history(train_metrics)

                val_metrics: Optional[Dict[str, float]] = None
                current_val_loss: Optional[float] = None
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    self._append_val_history(val_metrics)
                    current_val_loss = val_metrics["loss"]

                epoch_time = time.time() - epoch_start
                current_lr = self._get_current_lr()
                self.history["lr"].append(current_lr)
                self.history["epoch_time_sec"].append(epoch_time)

                self._log_epoch_summary(train_metrics, val_metrics, epoch_time)
                self._step_scheduler(current_val_loss)

                # Store the next epoch index before saving checkpoints so resume
                # continues after the last fully processed epoch.
                self.current_epoch = epoch_idx + 1
                self._handle_checkpointing_and_early_stopping(current_val_loss, epoch_num)
                self._log_tracking_epoch(
                    train_metrics, val_metrics, current_lr, epoch_time, epoch_num
                )

                if not self.config.save_best_only:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch_num}.pt")

                if (
                    val_loader is not None
                    and self.config.early_stopping_patience is not None
                    and self.epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    if self.config.verbose:
                        print(f"Early stopping triggered at epoch {epoch_num}.")
                    break

            self.save_checkpoint("final_model.pt")
            self.save_history()
            self._finish_tracking("completed")
            return self.history
        except KeyboardInterrupt:
            self._finish_tracking_after_failure("interrupted")
            raise
        except Exception:
            self._finish_tracking_after_failure("failed")
            raise

    def _start_tracking(self) -> None:
        """
        Initialize experiment tracking before any epoch work begins.
        """
        if not self.tracking_config.enabled:
            return
        info = self.tracker.start(self.tracking_metadata)
        self._tracking_started = True
        identity_parts = [
            part
            for part in (
                f"id={info.run_id}" if info.run_id else None,
                f"name={info.run_name}" if info.run_name else None,
                f"url={info.run_url}" if info.run_url else None,
            )
            if part is not None
        ]
        if identity_parts:
            print("Tracking run: " + " ".join(identity_parts))

    def _log_tracking_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        lr: float,
        epoch_time: float,
        epoch_num: int,
    ) -> None:
        if not self.tracking_config.enabled:
            return
        payload = build_epoch_metrics(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=lr,
            epoch_time_sec=epoch_time,
        )
        self.tracker.log_metrics(payload, step=epoch_num)

    def _finish_tracking(self, status: Literal["completed", "failed", "interrupted"]) -> None:
        if not self.tracking_config.enabled or not self._tracking_started:
            return
        step = max(self.current_epoch, 0)
        final_metrics = build_final_metrics(
            best_val_loss=self.best_val_loss,
            best_epoch=self.best_epoch,
        )
        if final_metrics:
            self.tracker.log_metrics(final_metrics, step=step)
        self.tracker.log_metrics(build_status_metrics(status), step=step)
        self.tracker.finish(status)

    def _finish_tracking_after_failure(
        self,
        status: Literal["failed", "interrupted"],
    ) -> None:
        try:
            self._finish_tracking(status)
        except Exception as exc:
            warnings.warn(
                f"Tracking cleanup failed after training {status}: {exc}",
                UserWarning,
                stacklevel=2,
            )

    def _append_train_history(self, train_metrics: Dict[str, float]) -> None:
        """
        Append train metrics to history.
        """
        self.history["train_loss"].append(train_metrics["loss"])
        for name, value in train_metrics.items():
            if name != "loss":
                self.history[f"train_{name}"].append(value)

    def _append_val_history(self, val_metrics: Dict[str, float]) -> None:
        """
        Append validation metrics to history.
        """
        self.history["val_loss"].append(val_metrics["loss"])
        for name, value in val_metrics.items():
            if name != "loss":
                self.history[f"val_{name}"].append(value)

    def _log_epoch_summary(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        epoch_time: float,
    ) -> None:
        """
        Print epoch summary when verbose mode is enabled.
        """
        if not self.config.verbose:
            return

        lr_str = f"{self._get_current_lr():.2e}"

        train_loss = train_metrics["loss"]
        if val_metrics is not None:
            val_loss = val_metrics["loss"]
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {lr_str} | "
                f"Time: {epoch_time:.2f}s"
            )
        else:
            print(f"Train Loss: {train_loss:.4f} | " f"LR: {lr_str} | " f"Time: {epoch_time:.2f}s")

    def _step_scheduler(self, current_val_loss: Optional[float]) -> None:
        """
        Advance scheduler state.

        Behavior
        --------
        - ReduceLROnPlateau requires validation loss
        - StepLR / CosineAnnealingLR step every epoch
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if current_val_loss is not None:
                self.scheduler.step(current_val_loss)
            return

        self.scheduler.step()

    def _handle_checkpointing_and_early_stopping(
        self,
        current_val_loss: Optional[float],
        epoch_num: int,
    ) -> None:
        """
        Handle best-model tracking, latest-checkpoint fallback, and early stopping counters.
        """
        if current_val_loss is None:
            if self.config.save_best_only:
                warnings.warn(
                    "save_best_only=True requires validation data to identify the best "
                    "checkpoint; saving checkpoint_latest.pt instead.",
                    UserWarning,
                    stacklevel=2,
                )
                self.save_checkpoint("checkpoint_latest.pt")
            return

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch_num
            self.epochs_without_improvement = 0

            if self.config.save_best_only:
                self.save_checkpoint("best_model.pt")
        else:
            self.epochs_without_improvement += 1

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------
    def _checkpoint_state(self) -> Dict[str, Any]:
        """
        Build serializable checkpoint state.

        The stored `current_epoch` value represents the next epoch index to
        execute. This keeps resume behavior unambiguous.
        """
        state: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": asdict(self.config),
            "current_epoch": self.current_epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        return state

    def save_checkpoint(self, filename: str) -> Path:
        """
        Save a training checkpoint.

        Parameters
        ----------
        filename : str
            Checkpoint filename.

        Returns
        -------
        Path
            Saved checkpoint path.
        """
        checkpoint_path = self.save_dir / filename
        torch.save(self._checkpoint_state(), checkpoint_path)

        if self.config.verbose:
            print(f"Checkpoint saved to {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, filename: str) -> None:
        """
        Load a training checkpoint.

        Parameters
        ----------
        filename : str
            Checkpoint filename.

        Notes
        -----
        Checkpoints are loaded with `weights_only=False` because trainer
        checkpoints are complete project artifacts. Only load checkpoint files
        produced by trusted project workflows.

        This restores:
        - model weights
        - optimizer state
        - scheduler state (if present and scheduler exists)
        - history
        - epoch counters

        The restored `current_epoch` value is normalized to mean "the next
        epoch to run". Legacy checkpoints are upgraded automatically by using
        the recorded history length as the lower bound.
        """
        checkpoint_path = self.save_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.current_epoch = self._resolve_resume_epoch(checkpoint)
        self.best_epoch = checkpoint.get("best_epoch", None)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.config.verbose:
            print(f"Checkpoint loaded from {checkpoint_path}")

    def save_history(self, filename: str = "history.json") -> Path:
        """
        Save history to JSON.

        Parameters
        ----------
        filename : str, default="history.json"
            Output history filename.

        Returns
        -------
        Path
            Saved history path.
        """
        history_path = self.save_dir / filename
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if self.config.verbose:
            print(f"Training history saved to {history_path}")

        return history_path

    # -----------------------------------------------------------------
    # Prediction / inference
    # -----------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        dataset=None,
        *,
        loader: Optional[DataLoader] = None,
        batch_size: Optional[int] = None,
        return_sample_ids: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, List[str]],
    ]:
        """
        Run model prediction.

        Parameters
        ----------
        dataset : optional
            Dataset used to auto-create a loader when `loader` is not provided.

        loader : DataLoader, optional
            Explicit prediction loader.

        batch_size : int, optional
            Batch size for auto-created loader.

        return_sample_ids : bool, default=False
            If True, also return ordered sample IDs when available in batches.

        Returns
        -------
        Tensor
            Predictions.

        Or, if `return_sample_ids=True`:
            (predictions, sample_ids)

        Notes
        -----
        - Prediction order follows loader iteration order.
        - For `SplitDatasetView`, IDs should reflect split order when
          shuffle=False in the loader.
        """
        if loader is None:
            if dataset is None:
                raise ValueError("Either dataset or loader must be provided for prediction.")
            loader = self._create_loader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
            )

        self.model.eval()

        collected_sample_ids: List[str] = []

        pred_parts: List[Tensor] = []

        for batch in loader:
            if return_sample_ids:
                collected_sample_ids.extend(self._extract_batch_sample_ids(batch))

            batch = self._move_batch_to_device(batch)
            pred = self.model(batch)

            if not isinstance(pred, Tensor):
                raise TypeError(f"Model output must be Tensor, got {type(pred).__name__}.")

            pred_parts.append(pred.detach().cpu())

        predictions = torch.cat(pred_parts, dim=0) if pred_parts else torch.empty(0)
        return (predictions, collected_sample_ids) if return_sample_ids else predictions

    def _extract_batch_sample_ids(self, batch: Data) -> List[str]:
        """
        Best-effort extraction of sample IDs from a batch.

        Returns
        -------
        List[str]
            Sample IDs if present, otherwise an empty list.
        """
        if not hasattr(batch, "sample_id"):
            return []

        sample_id_obj = batch.sample_id

        if isinstance(sample_id_obj, (list, tuple)):
            return [str(x) for x in sample_id_obj]

        if isinstance(sample_id_obj, str):
            return [sample_id_obj]

        return []


# ---------------------------------------------------------------------
# Optional convenience factory
# ---------------------------------------------------------------------
def create_default_trainer(
    model: nn.Module,
    *,
    save_dir: str = "./checkpoints",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_fn: Optional[LossFn] = None,
    metrics: Optional[MetricsMap] = None,
    optimizer: Literal["adam", "adamw", "sgd"] = "adam",
    scheduler: Optional[Literal["plateau", "step", "cosine"]] = "plateau",
) -> Trainer:
    """
    Build a trainer with common defaults.

    This helper is purely ergonomic and does not add behavior beyond
    creating `TrainingConfig` and `Trainer`.
    """
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
    )
    return Trainer(
        model=model,
        config=config,
        loss_fn=loss_fn,
        metrics=metrics,
    )
