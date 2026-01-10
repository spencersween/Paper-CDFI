"""
m05_nn_training.py - Neural Network Training

Training loop with:
- On-the-fly outcome differencing
- Proper loss masking for (unit, g, t) combinations
- Early stopping and learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time

from .m00_config import Config, get_device
from .m01_utils import log_message
from .m02_data_loader import UnitData, GTInfo
from .m04_nn_architecture import MultiTaskDiDNet


@dataclass
class TrainingHistory:
    """Training history."""
    train_loss: List[float]
    val_loss: List[float]
    train_outcome_loss: List[float]
    train_propensity_loss: List[float]
    best_epoch: int
    best_val_loss: float


class DiDDataset:
    """Dataset for DiD estimation."""

    def __init__(
        self,
        X_full: np.ndarray,
        Y: np.ndarray,
        sample_mask: np.ndarray,
        treatment: np.ndarray,
        t_indices: np.ndarray,
        base_t_indices: np.ndarray,
        unit_indices: Optional[np.ndarray] = None
    ):
        self.X_full = X_full
        self.Y = Y
        self.sample_mask = sample_mask
        self.treatment = treatment
        self.t_indices = t_indices
        self.base_t_indices = base_t_indices

        if unit_indices is None:
            self.unit_indices = np.arange(X_full.shape[0])
        else:
            self.unit_indices = unit_indices

        self.n_units = len(self.unit_indices)

    def __len__(self):
        return self.n_units

    def get_batch(self, indices: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get a batch of data."""
        # Map to actual unit indices
        actual_indices = self.unit_indices[indices]

        X = torch.tensor(self.X_full[actual_indices], dtype=torch.float32, device=device)
        Y_batch = self.Y[actual_indices]  # (batch, n_times)

        # Compute outcome differences on-the-fly
        n_gt = len(self.t_indices)
        delta_Y = np.full((len(indices), n_gt), np.nan, dtype=np.float32)

        for gt_idx in range(n_gt):
            t_idx = self.t_indices[gt_idx]
            base_idx = self.base_t_indices[gt_idx]
            if t_idx >= 0 and base_idx >= 0 and t_idx < Y_batch.shape[1] and base_idx < Y_batch.shape[1]:
                delta_Y[:, gt_idx] = Y_batch[:, t_idx] - Y_batch[:, base_idx]

        delta_Y = torch.tensor(delta_Y, dtype=torch.float32, device=device)
        sample_mask = torch.tensor(self.sample_mask[actual_indices], dtype=torch.bool, device=device)
        treatment = torch.tensor(self.treatment[actual_indices], dtype=torch.float32, device=device)

        return {
            'X': X,
            'delta_Y': delta_Y,
            'sample_mask': sample_mask,
            'treatment': treatment,
            'indices': actual_indices
        }


def compute_loss(
    outcome_preds: torch.Tensor,
    propensity_preds: torch.Tensor,
    delta_Y: torch.Tensor,
    treatment: torch.Tensor,
    sample_mask: torch.Tensor,
    config: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute training loss.

    Outcome loss: MSE on CONTROLS only (D=0)
    Propensity loss: BCE on all in-sample units

    Args:
        outcome_preds: (batch, n_gt) predicted E[delta_Y | X, D=0]
        propensity_preds: (batch, n_gt) propensity logits
        delta_Y: (batch, n_gt) outcome differences
        treatment: (batch, n_gt) treatment indicator (1/0/NaN)
        sample_mask: (batch, n_gt) is unit in this (g,t) sample?

    Returns:
        total_loss, outcome_loss, propensity_loss
    """
    # Valid mask: in sample and has valid outcome
    valid = sample_mask & ~torch.isnan(delta_Y) & ~torch.isnan(treatment)

    # Outcome loss: MSE on controls only
    control_mask = valid & (treatment == 0)
    n_control = control_mask.sum()

    if n_control > 0:
        outcome_loss = F.mse_loss(
            outcome_preds[control_mask],
            delta_Y[control_mask]
        )
    else:
        outcome_loss = torch.tensor(0.0, device=outcome_preds.device)

    # Propensity loss: BCE on all valid
    n_valid = valid.sum()
    if n_valid > 0:
        propensity_loss = F.binary_cross_entropy_with_logits(
            propensity_preds[valid],
            treatment[valid]
        )
    else:
        propensity_loss = torch.tensor(0.0, device=outcome_preds.device)

    # Weighted sum
    total_loss = (
        config.loss.outcome_weight * outcome_loss +
        config.loss.propensity_weight * propensity_loss
    )

    return total_loss, outcome_loss, propensity_loss


def create_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Create optimizer."""
    opt_config = config.optimizer
    if opt_config.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas
        )
    elif opt_config.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_config.lr,
            betas=opt_config.betas
        )
    else:
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay
        )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    sched_config = config.scheduler
    if sched_config.name == "none":
        return None
    elif sched_config.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config.T_max,
            eta_min=sched_config.eta_min
        )
    elif sched_config.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    return None


def train_epoch(
    model: MultiTaskDiDNet,
    dataset: DiDDataset,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train one epoch."""
    model.train()

    batch_size = config.training.batch_size
    n_units = len(dataset)
    indices = np.random.permutation(n_units)

    total_loss = 0.0
    total_outcome_loss = 0.0
    total_propensity_loss = 0.0
    n_batches = 0

    for start_idx in range(0, n_units, batch_size):
        end_idx = min(start_idx + batch_size, n_units)
        batch_indices = indices[start_idx:end_idx]

        batch = dataset.get_batch(batch_indices, device)

        optimizer.zero_grad()

        outcome_preds, propensity_preds = model(batch['X'])

        loss, outcome_loss, propensity_loss = compute_loss(
            outcome_preds,
            propensity_preds,
            batch['delta_Y'],
            batch['treatment'],
            batch['sample_mask'],
            config
        )

        loss.backward()

        # Gradient clipping
        if config.training.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip_norm
            )

        optimizer.step()

        total_loss += loss.item()
        total_outcome_loss += outcome_loss.item()
        total_propensity_loss += propensity_loss.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_outcome_loss / n_batches,
        total_propensity_loss / n_batches
    )


def validate_epoch(
    model: MultiTaskDiDNet,
    dataset: DiDDataset,
    config: Config,
    device: torch.device
) -> float:
    """Validate one epoch."""
    model.eval()

    batch_size = config.training.batch_size
    n_units = len(dataset)

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for start_idx in range(0, n_units, batch_size):
            end_idx = min(start_idx + batch_size, n_units)
            batch_indices = np.arange(start_idx, end_idx)

            batch = dataset.get_batch(batch_indices, device)

            outcome_preds, propensity_preds = model(batch['X'])

            loss, _, _ = compute_loss(
                outcome_preds,
                propensity_preds,
                batch['delta_Y'],
                batch['treatment'],
                batch['sample_mask'],
                config
            )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_model(
    model: MultiTaskDiDNet,
    train_dataset: DiDDataset,
    val_dataset: DiDDataset,
    config: Config,
    device: torch.device
) -> Tuple[MultiTaskDiDNet, TrainingHistory]:
    """
    Train the model.

    Returns:
        Trained model and training history
    """
    log_message("Starting training...")

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    epochs = config.training.epochs
    patience = config.training.early_stopping_patience
    min_delta = config.training.early_stopping_min_delta

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_outcome_loss=[],
        train_propensity_loss=[],
        best_epoch=0,
        best_val_loss=float('inf')
    )

    best_state = None
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, outcome_loss, prop_loss = train_epoch(
            model, train_dataset, optimizer, config, device
        )

        # Validate
        val_loss = validate_epoch(model, val_dataset, config, device)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Record history
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_outcome_loss.append(outcome_loss)
        history.train_propensity_loss.append(prop_loss)

        # Check for improvement
        if val_loss < history.best_val_loss - min_delta:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        if config.monitoring.verbose and (epoch + 1) % config.monitoring.print_every == 0:
            elapsed = time.time() - start_time
            log_message(
                f"Epoch {epoch+1}/{epochs} - "
                f"train_loss: {train_loss:.4f} (out: {outcome_loss:.4f}, ps: {prop_loss:.4f}) - "
                f"val_loss: {val_loss:.4f} - "
                f"time: {elapsed:.1f}s"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            log_message(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        log_message(f"Restored best model from epoch {history.best_epoch + 1}")

    total_time = time.time() - start_time
    log_message(f"Training complete in {total_time:.1f}s ({total_time/60:.1f}m)")

    return model, history
