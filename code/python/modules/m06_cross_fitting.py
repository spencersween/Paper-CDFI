"""
m06_cross_fitting.py - Cross-Fitting for Double Machine Learning

Implements K-fold cross-fitting with efficient out-of-fold predictions.
Single forward pass per fold produces all (n_units, n_gt) predictions.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import gc

from .m00_config import Config, get_device
from .m01_utils import log_message
from .m02_data_loader import UnitData, GTInfo, create_sample_masks
from .m03_covariate_selector import CovariateInfo
from .m04_nn_architecture import create_model, MultiTaskDiDNet
from .m05_nn_training import DiDDataset, train_model


@dataclass
class CrossFitResults:
    """Results from cross-fitting."""
    # Out-of-fold predictions: (n_units, n_gt)
    outcome_preds: np.ndarray   # mu_0 predictions
    propensity_preds: np.ndarray  # propensity scores (after sigmoid)

    # Data references
    unit_data: UnitData
    gt_info: GTInfo
    covariate_info: CovariateInfo

    # Sample masks
    sample_mask: np.ndarray
    treatment: np.ndarray

    # Fold info
    fold_assignments: np.ndarray
    n_folds: int


def assign_folds(
    unit_data: UnitData,
    config: Config
) -> np.ndarray:
    """
    Assign units to folds based on cluster stratification.

    Returns:
        fold_assignments: (n_units,) array with fold index (0 to n_folds-1)
    """
    np.random.seed(config.cross_fitting.seed)

    clusters = unit_data.clusters
    unique_clusters = np.unique(clusters)
    n_folds = config.cross_fitting.n_folds

    # Assign clusters to folds
    cluster_folds = np.random.choice(n_folds, size=len(unique_clusters))
    cluster_to_fold = {c: f for c, f in zip(unique_clusters, cluster_folds)}

    # Map to units
    fold_assignments = np.array([cluster_to_fold[c] for c in clusters], dtype=np.int64)

    # Report
    fold_counts = pd.Series(fold_assignments).value_counts().sort_index()
    log_message(f"Fold assignments: " + ", ".join([f"Fold {i}: {c}" for i, c in fold_counts.items()]))

    return fold_assignments


def compute_oof_predictions(
    model: MultiTaskDiDNet,
    X_full: np.ndarray,
    val_indices: np.ndarray,
    config: Config,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute out-of-fold predictions for validation units.

    Single forward pass produces (n_val, n_gt) predictions for all heads.
    """
    model.eval()

    X_val = X_full[val_indices]
    n_val = len(val_indices)
    batch_size = config.training.batch_size

    outcome_preds_list = []
    propensity_preds_list = []

    with torch.no_grad():
        for start_idx in range(0, n_val, batch_size):
            end_idx = min(start_idx + batch_size, n_val)
            X_batch = torch.tensor(
                X_val[start_idx:end_idx],
                dtype=torch.float32,
                device=device
            )

            outcome_pred, propensity_pred = model.predict(X_batch)

            outcome_preds_list.append(outcome_pred.cpu().numpy())
            propensity_preds_list.append(propensity_pred.cpu().numpy())

    outcome_preds = np.concatenate(outcome_preds_list, axis=0)
    propensity_preds = np.concatenate(propensity_preds_list, axis=0)

    return outcome_preds, propensity_preds


def run_cross_fitting(
    unit_data: UnitData,
    gt_info: GTInfo,
    covariate_info: CovariateInfo,
    config: Config
) -> CrossFitResults:
    """
    Run K-fold cross-fitting.

    For each fold:
    1. Train on K-1 folds
    2. Predict on held-out fold (single forward pass → all (g,t) predictions)

    Returns:
        CrossFitResults with (n_units, n_gt) out-of-fold predictions
    """
    device = get_device(config)
    n_folds = config.cross_fitting.n_folds
    n_units = unit_data.n_units
    n_gt = gt_info.n_gt

    log_message(f"\nStarting {n_folds}-fold cross-fitting")
    log_message(f"  Device: {device}")
    log_message(f"  Units: {n_units}, (g,t) pairs: {n_gt}")

    # Assign folds
    fold_assignments = assign_folds(unit_data, config)

    # Create sample masks
    sample_mask, treatment = create_sample_masks(unit_data, gt_info, config)

    # Initialize OOF prediction storage
    oof_outcome = np.full((n_units, n_gt), np.nan, dtype=np.float32)
    oof_propensity = np.full((n_units, n_gt), np.nan, dtype=np.float32)

    # Cross-fitting loop
    for fold in range(n_folds):
        log_message(f"\n{'='*60}")
        log_message(f"FOLD {fold + 1}/{n_folds}")
        log_message(f"{'='*60}")

        # Split indices
        val_mask = (fold_assignments == fold)
        train_mask = ~val_mask
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        log_message(f"  Train: {len(train_indices)} units, Val: {len(val_indices)} units")

        # Create datasets
        train_dataset = DiDDataset(
            X_full=unit_data.X_full,
            Y=unit_data.Y,
            sample_mask=sample_mask,
            treatment=treatment,
            t_indices=gt_info.t_indices,
            base_t_indices=gt_info.base_t_indices,
            unit_indices=train_indices
        )

        # Split training into train/val for early stopping
        n_train = len(train_indices)
        n_internal_val = int(n_train * config.training.validation_split)
        np.random.seed(config.cross_fitting.seed + fold)
        perm = np.random.permutation(n_train)

        internal_train_indices = train_indices[perm[n_internal_val:]]
        internal_val_indices = train_indices[perm[:n_internal_val]]

        internal_train_dataset = DiDDataset(
            X_full=unit_data.X_full,
            Y=unit_data.Y,
            sample_mask=sample_mask,
            treatment=treatment,
            t_indices=gt_info.t_indices,
            base_t_indices=gt_info.base_t_indices,
            unit_indices=internal_train_indices
        )

        internal_val_dataset = DiDDataset(
            X_full=unit_data.X_full,
            Y=unit_data.Y,
            sample_mask=sample_mask,
            treatment=treatment,
            t_indices=gt_info.t_indices,
            base_t_indices=gt_info.base_t_indices,
            unit_indices=internal_val_indices
        )

        # Create model
        model = create_model(covariate_info, gt_info, config, device)

        # Train
        model, history = train_model(
            model,
            internal_train_dataset,
            internal_val_dataset,
            config,
            device
        )

        # Compute OOF predictions for this fold's validation set
        log_message("Computing out-of-fold predictions...")
        outcome_preds, propensity_preds = compute_oof_predictions(
            model, unit_data.X_full, val_indices, config, device
        )

        # Store predictions
        oof_outcome[val_indices, :] = outcome_preds
        oof_propensity[val_indices, :] = propensity_preds

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clamp propensity scores
    oof_propensity = np.clip(
        oof_propensity,
        config.propensity.min_ps,
        config.propensity.max_ps
    )

    # Report coverage
    n_filled = np.sum(~np.isnan(oof_outcome))
    fill_rate = 100 * n_filled / oof_outcome.size
    log_message(f"\nCross-fitting complete")
    log_message(f"  OOF predictions: {n_filled:,} filled ({fill_rate:.1f}%)")

    return CrossFitResults(
        outcome_preds=oof_outcome,
        propensity_preds=oof_propensity,
        unit_data=unit_data,
        gt_info=gt_info,
        covariate_info=covariate_info,
        sample_mask=sample_mask,
        treatment=treatment,
        fold_assignments=fold_assignments,
        n_folds=n_folds
    )


def validate_cross_fitting(cf_results: CrossFitResults) -> bool:
    """Validate cross-fitting results."""
    outcome = cf_results.outcome_preds
    propensity = cf_results.propensity_preds

    n_units, n_gt = outcome.shape

    # Check fill rates
    outcome_filled = np.sum(~np.isnan(outcome))
    propensity_filled = np.sum(~np.isnan(propensity))

    log_message(f"\nCross-fitting validation:")
    log_message(f"  Outcome predictions: {outcome_filled:,} / {outcome.size:,}")
    log_message(f"  Propensity predictions: {propensity_filled:,} / {propensity.size:,}")

    # Propensity score range
    ps_min = np.nanmin(propensity)
    ps_max = np.nanmax(propensity)
    log_message(f"  Propensity range: [{ps_min:.4f}, {ps_max:.4f}]")

    # Outcome range
    out_min = np.nanmin(outcome)
    out_max = np.nanmax(outcome)
    log_message(f"  Outcome pred range: [{out_min:.4f}, {out_max:.4f}]")

    return True
