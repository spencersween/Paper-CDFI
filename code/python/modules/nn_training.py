"""
Neural Network Training Module for DiD Estimation

Training loop, loss functions, and optimization for multi-task DiD network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import gc

from .utils import log_message
from .config import Config, get_device


@dataclass
class TrainingHistory:
    """Training history."""
    train_loss: List[float]
    train_outcome_loss: List[float]
    train_propensity_loss: List[float]
    val_loss: List[float]
    val_outcome_loss: List[float]
    val_propensity_loss: List[float]
    lr: List[float]


def compute_loss(
    outcome_pred: torch.Tensor,
    propensity_pred: torch.Tensor,
    outcome_target: torch.Tensor,
    treatment_target: torch.Tensor,
    config: Config
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-task loss.

    Combines outcome regression loss (MSE) and propensity score loss (BCE).
    Outcome loss is computed only on control units (D=0).
    """
    # Outcome regression: MSE loss (only on control units)
    control_mask = treatment_target == 0

    if control_mask.any():
        outcome_loss = F.mse_loss(
            outcome_pred[control_mask],
            outcome_target[control_mask].unsqueeze(1)
        )
    else:
        outcome_loss = torch.tensor(0.0, device=outcome_pred.device)

    # Propensity score: Binary cross-entropy loss (on all units)
    propensity_loss = F.binary_cross_entropy(
        propensity_pred,
        treatment_target.unsqueeze(1),
        reduction='mean'
    )

    # Weighted combination
    total_loss = (config.loss.outcome_weight * outcome_loss +
                  config.loss.propensity_weight * propensity_loss)

    return {
        'total': total_loss,
        'outcome': outcome_loss,
        'propensity': propensity_loss
    }


def create_optimizer(model: nn.Module, config: Config):
    """Create optimizer."""
    params = model.parameters()

    if config.optimizer == "adamw":
        opt_config = config.optimizer_params.get('adamw', {})
        if hasattr(opt_config, 'lr'):
            # It's a dataclass
            optimizer = AdamW(
                params,
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps
            )
        else:
            # It's a dict
            optimizer = AdamW(
                params,
                lr=opt_config.get('lr', 0.001),
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=opt_config.get('betas', (0.9, 0.999)),
                eps=opt_config.get('eps', 1e-8)
            )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def create_scheduler(optimizer, config: Config):
    """Create learning rate scheduler."""
    sched = config.scheduler

    if sched.type == "none":
        return None
    elif sched.type == "step":
        return StepLR(optimizer, step_size=sched.step_size, gamma=sched.gamma)
    elif sched.type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=sched.T_max, eta_min=sched.eta_min)
    elif sched.type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched.factor,
            patience=sched.patience,
            min_lr=sched.min_lr
        )

    return None


class EarlyStopping:
    """Early stopping tracker."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None

    def check(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        """Check if should stop. Returns True if should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Restore best weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(
    model: nn.Module,
    dataloader_fn: Callable,
    optimizer,
    config: Config,
    device: torch.device
) -> Dict[str, float]:
    """Training step for one epoch."""
    model.train()

    total_loss = 0.0
    total_outcome_loss = 0.0
    total_propensity_loss = 0.0
    n_samples = 0

    batches = dataloader_fn()

    for batch in batches:
        x = batch['X']
        delta_y = batch['delta_y']
        D = batch['D']
        gt_idx = batch['gt_index']
        batch_size = batch['batch_size']

        optimizer.zero_grad()

        # Forward pass
        output = model.forward_gt(x, gt_idx)

        # Compute loss
        losses = compute_loss(
            output['outcome'],
            output['propensity'],
            delta_y,
            D,
            config
        )

        # Backward pass
        losses['total'].backward()

        # Gradient clipping
        if config.training.gradient_clipping.enabled:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clipping.max_norm
            )

        optimizer.step()

        # Accumulate metrics
        total_loss += losses['total'].item() * batch_size
        total_outcome_loss += losses['outcome'].item() * batch_size
        total_propensity_loss += losses['propensity'].item() * batch_size
        n_samples += batch_size

    return {
        'loss': total_loss / n_samples,
        'outcome_loss': total_outcome_loss / n_samples,
        'propensity_loss': total_propensity_loss / n_samples
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader_fn: Callable,
    config: Config,
    device: torch.device
) -> Dict[str, float]:
    """Validation step."""
    model.eval()

    total_loss = 0.0
    total_outcome_loss = 0.0
    total_propensity_loss = 0.0
    n_samples = 0

    batches = dataloader_fn()

    for batch in batches:
        x = batch['X']
        delta_y = batch['delta_y']
        D = batch['D']
        gt_idx = batch['gt_index']
        batch_size = batch['batch_size']

        output = model.forward_gt(x, gt_idx)

        losses = compute_loss(
            output['outcome'],
            output['propensity'],
            delta_y,
            D,
            config
        )

        total_loss += losses['total'].item() * batch_size
        total_outcome_loss += losses['outcome'].item() * batch_size
        total_propensity_loss += losses['propensity'].item() * batch_size
        n_samples += batch_size

    return {
        'loss': total_loss / n_samples,
        'outcome_loss': total_outcome_loss / n_samples,
        'propensity_loss': total_propensity_loss / n_samples
    }


def train_model(
    model: nn.Module,
    train_loader: Callable,
    val_loader: Callable,
    config: Config
) -> Tuple[nn.Module, TrainingHistory, float]:
    """
    Main training loop.

    Returns:
        Tuple of (trained model, training history, training time in seconds)
    """
    import time

    device = get_device(config)
    model = model.to(device)

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Early stopping
    early_stopper = None
    if config.training.early_stopping.enabled:
        early_stopper = EarlyStopping(
            config.training.early_stopping.patience,
            config.training.early_stopping.min_delta
        )

    # Training history
    history = TrainingHistory(
        train_loss=[],
        train_outcome_loss=[],
        train_propensity_loss=[],
        val_loss=[],
        val_outcome_loss=[],
        val_propensity_loss=[],
        lr=[]
    )

    log_message("Starting training...")
    start_time = time.time()

    for epoch in range(1, config.training.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, device)

        # Validate
        val_metrics = validate_epoch(model, val_loader, config, device)

        # Record history
        history.train_loss.append(train_metrics['loss'])
        history.train_outcome_loss.append(train_metrics['outcome_loss'])
        history.train_propensity_loss.append(train_metrics['propensity_loss'])
        history.val_loss.append(val_metrics['loss'])
        history.val_outcome_loss.append(val_metrics['outcome_loss'])
        history.val_propensity_loss.append(val_metrics['propensity_loss'])

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history.lr.append(current_lr)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Print progress
        if config.monitoring.verbose and epoch % config.monitoring.print_every == 0:
            log_message(
                f"Epoch {epoch:3d}/{config.training.epochs} | "
                f"Train: {train_metrics['loss']:.4f} (O:{train_metrics['outcome_loss']:.4f} P:{train_metrics['propensity_loss']:.4f}) | "
                f"Val: {val_metrics['loss']:.4f} (O:{val_metrics['outcome_loss']:.4f} P:{val_metrics['propensity_loss']:.4f}) | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping check
        if early_stopper is not None:
            if early_stopper.check(val_metrics['loss'], epoch, model):
                log_message(f"Early stopping at epoch {epoch} (best: {early_stopper.best_epoch}, loss: {early_stopper.best_loss:.4f})")
                break

        # Garbage collection
        if epoch % config.gc_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Restore best weights
    if early_stopper is not None and config.training.early_stopping.restore_best_weights:
        early_stopper.restore_best(model)
        log_message(f"Restored best weights from epoch {early_stopper.best_epoch}")

    training_time = time.time() - start_time
    log_message(f"Training complete in {training_time:.1f} seconds")

    return model, history, training_time
