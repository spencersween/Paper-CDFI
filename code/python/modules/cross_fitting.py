"""
Cross-Fitting Module for DiD Estimation

Implements K-fold cross-fitting for double machine learning (DML).
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
import gc

from .utils import log_message, clamp
from .config import Config, get_device
from .data_loader import create_gt_sample
from .covariate_selector import CovariateInfo, prepare_gt_covariate_matrix, get_covariate_info
from .nn_architecture import create_model
from .nn_training import train_model


def assign_cluster_folds(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Assign cross-fitting folds to clusters.

    Returns DataFrame with cluster -> fold mapping.
    """
    np.random.seed(config.cross_fitting.seed)

    cluster_var = config.cross_fitting.stratify_by
    clusters = data[cluster_var].unique()
    n_clusters = len(clusters)
    n_folds = config.cross_fitting.n_folds

    log_message(f"Assigning {n_clusters} clusters to {n_folds} folds")

    # Random assignment
    fold_assignment = np.random.choice(range(1, n_folds + 1), size=n_clusters)

    cluster_folds = pd.DataFrame({
        cluster_var: clusters,
        'fold': fold_assignment
    })

    # Summary
    fold_counts = cluster_folds['fold'].value_counts().sort_index()
    log_message(f"Clusters per fold: " +
                ", ".join([f"Fold {k}: {v}" for k, v in fold_counts.items()]))

    return cluster_folds


def add_fold_column(data: pd.DataFrame, cluster_folds: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add fold assignments to data."""
    cluster_var = config.cross_fitting.stratify_by

    data = data.merge(cluster_folds, on=cluster_var, how='left')

    n_missing = data['fold'].isna().sum()
    if n_missing > 0:
        log_message(f"Warning: {n_missing} observations have missing fold assignments", level="WARNING")

    return data


def prepare_gt_training_data(
    data: pd.DataFrame,
    g: int,
    t: int,
    gt_index: int,
    covariate_info: CovariateInfo,
    config: Config
) -> Optional[Dict]:
    """Prepare training data for a single (g,t) pair."""
    sample = create_gt_sample(data, g, t, config)

    if len(sample) == 0:
        return None

    X = prepare_gt_covariate_matrix(sample, gt_index, covariate_info)

    return {
        'X': X,
        'delta_y': sample['delta_y'].values.astype(np.float32),
        'D': sample['D'].values.astype(np.float32),
        'fold': sample['fold'].values if 'fold' in sample.columns else None,
        'ids': sample[config.id_var].values,
        'n': len(sample),
        'gt_index': gt_index,
        'covariate_dim': X.shape[1]
    }


def prepare_combined_training_data(
    data: pd.DataFrame,
    gt_pairs: pd.DataFrame,
    covariate_info: CovariateInfo,
    train_folds: List[int],
    config: Config
) -> Dict:
    """Prepare full training dataset across all (g,t) pairs."""
    log_message(f"Preparing training data for folds: {train_folds}")

    # Filter to training folds
    train_data = data[data['fold'].isin(train_folds)].copy()

    gt_data_list = {}
    total_obs = 0

    for _, row in gt_pairs.iterrows():
        g, t = row['g'], row['t']
        gt_idx = row['gt_index']

        gt_data = prepare_gt_training_data(
            train_data, g, t, gt_idx, covariate_info, config
        )

        if gt_data is not None and gt_data['n'] > 0:
            gt_data_list[gt_idx] = gt_data
            total_obs += gt_data['n']

    log_message(f"Combined training data: {total_obs} total observations across {len(gt_data_list)} (g,t) pairs")

    return {
        'gt_data': gt_data_list,
        'total_obs': total_obs,
        'covariate_info': covariate_info
    }


def create_validation_split(combined_data: Dict, config: Config) -> Tuple[Dict, Dict]:
    """Create validation split from training data."""
    val_split = config.training.validation_split
    gt_data_list = combined_data['gt_data']

    train_gt_data = {}
    val_gt_data = {}

    for gt_idx, gt_data in gt_data_list.items():
        n = gt_data['n']
        n_val = max(1, int(n * val_split))

        indices = np.random.permutation(n)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_gt_data[gt_idx] = {
            'X': gt_data['X'][train_indices],
            'delta_y': gt_data['delta_y'][train_indices],
            'D': gt_data['D'][train_indices],
            'n': len(train_indices),
            'gt_index': gt_idx,
            'covariate_dim': gt_data['covariate_dim']
        }

        val_gt_data[gt_idx] = {
            'X': gt_data['X'][val_indices],
            'delta_y': gt_data['delta_y'][val_indices],
            'D': gt_data['D'][val_indices],
            'n': len(val_indices),
            'gt_index': gt_idx,
            'covariate_dim': gt_data['covariate_dim']
        }

    return (
        {'gt_data': train_gt_data, 'covariate_info': combined_data['covariate_info']},
        {'gt_data': val_gt_data, 'covariate_info': combined_data['covariate_info']}
    )


def create_gt_dataloader(
    combined_data: Dict,
    batch_size: int,
    shuffle: bool,
    device: torch.device
) -> Callable:
    """Create (g,t)-aware dataloader that returns batches."""
    gt_data_list = combined_data['gt_data']
    valid_gt_indices = list(gt_data_list.keys())

    def dataloader():
        # Shuffle order of (g,t) pairs if requested
        gt_order = np.random.permutation(valid_gt_indices) if shuffle else valid_gt_indices

        batches = []

        for gt_idx in gt_order:
            gt_data = gt_data_list[gt_idx]
            n = gt_data['n']

            # Shuffle within (g,t) if requested
            obs_order = np.random.permutation(n) if shuffle else np.arange(n)

            # Create batches
            n_batches = (n + batch_size - 1) // batch_size

            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, n)
                batch_indices = obs_order[start_idx:end_idx]

                X_batch = gt_data['X'][batch_indices]
                delta_y_batch = gt_data['delta_y'][batch_indices]
                D_batch = gt_data['D'][batch_indices]

                # Convert to tensors
                X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
                delta_y_tensor = torch.tensor(delta_y_batch, dtype=torch.float32, device=device)
                D_tensor = torch.tensor(D_batch, dtype=torch.float32, device=device)

                batches.append({
                    'X': X_tensor,
                    'delta_y': delta_y_tensor,
                    'D': D_tensor,
                    'gt_index': gt_idx,
                    'batch_size': len(batch_indices)
                })

        # Shuffle batches across (g,t) pairs
        if shuffle:
            np.random.shuffle(batches)

        return batches

    return dataloader


def create_dataloaders(train_data: Dict, val_data: Dict, config: Config) -> Dict:
    """Create train and validation dataloaders."""
    device = get_device(config)
    batch_size = config.training.batch_size

    train_loader = create_gt_dataloader(train_data, batch_size, shuffle=True, device=device)
    val_loader = create_gt_dataloader(val_data, batch_size, shuffle=False, device=device)

    return {
        'train': train_loader,
        'val': val_loader
    }


def compute_oof_predictions(
    model,
    val_data: pd.DataFrame,
    gt_pairs: pd.DataFrame,
    covariate_info: CovariateInfo,
    unit_ids: np.ndarray,
    id_to_row: Dict,
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute out-of-fold predictions for held-out data.

    Predictions are stored at the UNIT level (wide format):
    - Shape: (n_units, n_gt)
    - Each unit has one prediction per (g,t) pair they belong to
    """
    device = get_device(config)
    model.eval()

    n_units = len(unit_ids)
    n_gt = len(gt_pairs)

    outcome_preds = np.full((n_units, n_gt), np.nan)
    propensity_preds = np.full((n_units, n_gt), np.nan)

    with torch.no_grad():
        for _, row in gt_pairs.iterrows():
            g, t = row['g'], row['t']
            gt_idx = row['gt_index']

            # Create cross-sectional sample for this (g,t)
            sample = create_gt_sample(val_data, g, t, config)

            if len(sample) > 0:
                X = prepare_gt_covariate_matrix(sample, gt_idx, covariate_info)
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

                preds = model.predict_gt(X_tensor, gt_idx)

                # Map predictions back to unit-level storage
                sample_ids = sample[config.id_var].values
                for j, sid in enumerate(sample_ids):
                    if sid in id_to_row:
                        unit_row = id_to_row[sid]
                        outcome_preds[unit_row, gt_idx] = preds['outcome'][j, 0].cpu().numpy()
                        propensity_preds[unit_row, gt_idx] = preds['propensity'][j, 0].cpu().numpy()

    return outcome_preds, propensity_preds


def run_cross_fitting(
    data: pd.DataFrame,
    gt_pairs: pd.DataFrame,
    all_covariates: List[str],
    covariate_masks: np.ndarray,
    config: Config
) -> Dict:
    """
    Run full cross-fitting procedure.

    Returns dict with out-of-fold predictions at the UNIT level (wide format).
    - outcome: (n_units, n_gt) array
    - propensity: (n_units, n_gt) array
    """
    n_folds = config.cross_fitting.n_folds
    device = get_device(config)

    log_message(f"Starting {n_folds}-fold cross-fitting")

    # Get covariate info
    covariate_info = get_covariate_info(gt_pairs, data.columns.tolist(), config)

    # Assign folds
    cluster_folds = assign_cluster_folds(data, config)
    data = add_fold_column(data, cluster_folds, config)

    # Get unique units and create ID mapping
    unit_ids = data[config.id_var].unique()
    n_units = len(unit_ids)
    id_to_row = {uid: idx for idx, uid in enumerate(unit_ids)}

    # Initialize UNIT-level storage (wide format)
    n_gt = len(gt_pairs)
    oof_outcome = np.full((n_units, n_gt), np.nan)
    oof_propensity = np.full((n_units, n_gt), np.nan)

    # Track which units are in validation for each fold
    unit_folds = data.groupby(config.id_var)['fold'].first().to_dict()

    fold_models = []

    # Cross-fitting loop
    for fold in range(1, n_folds + 1):
        log_message(f"\n=== Fold {fold}/{n_folds} ===")

        # Get fold indices (observation level for training data prep)
        val_idx = data['fold'] == fold
        train_folds = [f for f in range(1, n_folds + 1) if f != fold]

        # Get validation unit IDs
        val_unit_ids = [uid for uid, f in unit_folds.items() if f == fold]
        val_unit_rows = [id_to_row[uid] for uid in val_unit_ids]

        # Prepare training data
        train_combined = prepare_combined_training_data(
            data, gt_pairs, covariate_info, train_folds, config
        )

        # Create validation split
        train_final, val_final = create_validation_split(train_combined, config)

        # Create dataloaders
        loaders = create_dataloaders(train_final, val_final, config)

        # Create model
        model = create_model(covariate_info, config)

        # Train
        model, history, training_time = train_model(model, loaders['train'], loaders['val'], config)
        fold_models.append(model)

        # Get out-of-fold predictions (unit level)
        log_message("Computing out-of-fold predictions...")

        val_data = data[val_idx].copy()
        outcome_preds, propensity_preds = compute_oof_predictions(
            model, val_data, gt_pairs, covariate_info, unit_ids, id_to_row, config
        )

        # Store predictions for validation units only
        for unit_row in val_unit_rows:
            oof_outcome[unit_row, :] = outcome_preds[unit_row, :]
            oof_propensity[unit_row, :] = propensity_preds[unit_row, :]

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clamp propensity scores
    oof_propensity = clamp(oof_propensity, config.propensity.min_ps, config.propensity.max_ps)

    log_message("Cross-fitting complete")

    return {
        'outcome': oof_outcome,
        'propensity': oof_propensity,
        'unit_ids': unit_ids,
        'id_to_row': id_to_row,
        'data': data,
        'gt_pairs': gt_pairs,
        'covariate_info': covariate_info,
        'fold_models': fold_models,
        'cluster_folds': cluster_folds
    }


def validate_cross_fitting(cf_results: Dict, config: Config) -> bool:
    """Validate cross-fitting results."""
    outcome = cf_results['outcome']
    propensity = cf_results['propensity']
    gt_pairs = cf_results['gt_pairs']
    unit_ids = cf_results['unit_ids']

    n_units, n_gt = outcome.shape

    # Count non-missing predictions per (g,t)
    n_filled_per_gt = np.sum(~np.isnan(outcome), axis=0)
    total_filled = n_filled_per_gt.sum()
    fill_rate = 100 * total_filled / outcome.size

    log_message(f"Prediction matrix: {n_units} units x {n_gt} (g,t) pairs")
    log_message(f"  Filled cells: {total_filled:,} ({fill_rate:.1f}%)")
    log_message(f"  Predictions per (g,t): min={n_filled_per_gt.min()}, max={n_filled_per_gt.max()}, "
                f"mean={n_filled_per_gt.mean():.0f}")

    # Validate propensity scores
    ps_min = np.nanmin(propensity)
    ps_max = np.nanmax(propensity)
    log_message(f"Propensity score range: [{ps_min:.4f}, {ps_max:.4f}]")

    # Check that we have predictions for each (g,t)
    n_empty_gt = (n_filled_per_gt == 0).sum()
    if n_empty_gt > 0:
        log_message(f"Warning: {n_empty_gt} (g,t) pairs have no predictions", level="WARNING")

    # Validate outcome predictions
    outcome_min = np.nanmin(outcome)
    outcome_max = np.nanmax(outcome)
    log_message(f"Outcome prediction range: [{outcome_min:.4f}, {outcome_max:.4f}]")

    return True
