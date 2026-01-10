"""
Covariate Selection Module for DiD Estimation

Implements dynamic covariate selection rules based on (g,t) pair.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .utils import log_message
from .config import Config


@dataclass
class CovariateInfo:
    """Information about covariates for all (g,t) pairs."""
    covariates_by_gt: Dict[int, List[str]]  # gt_index -> list of covariate names
    dims_by_gt: Dict[int, int]  # gt_index -> covariate dimension
    all_covariates: List[str]  # Union of all covariates
    n_gt: int


def get_covariates_for_gt(
    g: int,
    t: int,
    available_vars: List[str],
    config: Config
) -> List[str]:
    """
    Get covariates for a specific (g,t) pair.

    Rules:
    - Pre-treatment (t < g): X_ + V_ up to t-1
    - Post-treatment (t >= g): X_ + V_ up to g-1
    """
    covariates = []

    # Always include all time-invariant covariates
    x_cols = [c for c in available_vars if c.startswith(config.time_invariant_prefix)]
    covariates.extend(x_cols)

    # Determine cutoff year for time-varying covariates
    cutoff_year = t - 1 if t < g else g - 1

    # Include time-varying covariates up to cutoff
    v_cols = [c for c in available_vars if c.startswith(config.time_varying_prefix)]

    for col in v_cols:
        # Extract year from column name (e.g., V_var_1995 -> 1995)
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            try:
                year = int(parts[1])
                if year <= cutoff_year:
                    covariates.append(col)
            except ValueError:
                # Column doesn't end with year, include it
                covariates.append(col)
        else:
            # No year suffix, include it
            covariates.append(col)

    return sorted(covariates)


def get_covariate_info(
    gt_pairs: pd.DataFrame,
    available_vars: List[str],
    config: Config
) -> CovariateInfo:
    """
    Get covariate information for all (g,t) pairs.

    Returns CovariateInfo with per-(g,t) covariate lists and dimensions.
    """
    covariates_by_gt = {}
    dims_by_gt = {}
    all_covariates_set = set()

    for _, row in gt_pairs.iterrows():
        gt_idx = row['gt_index']
        g, t = row['g'], row['t']

        covs = get_covariates_for_gt(g, t, available_vars, config)
        covariates_by_gt[gt_idx] = covs
        dims_by_gt[gt_idx] = len(covs)
        all_covariates_set.update(covs)

    all_covariates = sorted(all_covariates_set)

    dims_array = np.array(list(dims_by_gt.values()))
    log_message(f"Covariate dimensions per (g,t): min={dims_array.min()}, "
                f"max={dims_array.max()}, mean={dims_array.mean():.1f}")

    return CovariateInfo(
        covariates_by_gt=covariates_by_gt,
        dims_by_gt=dims_by_gt,
        all_covariates=all_covariates,
        n_gt=len(gt_pairs)
    )


def prepare_gt_covariate_matrix(
    sample: pd.DataFrame,
    gt_index: int,
    covariate_info: CovariateInfo
) -> np.ndarray:
    """
    Prepare covariate matrix for a specific (g,t) pair.

    Returns numpy array with only the relevant covariates for this (g,t).
    """
    covariates = covariate_info.covariates_by_gt[gt_index]

    # Extract covariate columns
    X = sample[covariates].values.astype(np.float32)

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    return X


def create_covariate_masks(
    gt_pairs: pd.DataFrame,
    all_covariates: List[str],
    config: Config
) -> np.ndarray:
    """
    Create covariate masks for backward compatibility.

    Returns boolean matrix (n_gt x n_covariates) indicating which covariates
    are used for each (g,t) pair.
    """
    n_gt = len(gt_pairs)
    n_cov = len(all_covariates)
    cov_to_idx = {c: i for i, c in enumerate(all_covariates)}

    masks = np.zeros((n_gt, n_cov), dtype=bool)

    for _, row in gt_pairs.iterrows():
        gt_idx = row['gt_index']
        g, t = row['g'], row['t']

        covs = get_covariates_for_gt(g, t, all_covariates, config)
        for c in covs:
            if c in cov_to_idx:
                masks[gt_idx, cov_to_idx[c]] = True

    log_message(f"Covariate masks created: {n_gt} (g,t) pairs x {n_cov} covariates")
    covs_per_gt = masks.sum(axis=1)
    log_message(f"Covariates per (g,t): min={covs_per_gt.min()}, "
                f"max={covs_per_gt.max()}, mean={covs_per_gt.mean():.1f}")

    return masks


def validate_covariate_rules(config: Config) -> bool:
    """Validate covariate selection rules are properly configured."""
    log_message("Validating covariate selection rules...")

    # Check prefixes are defined
    if not config.time_invariant_prefix:
        raise ValueError("time_invariant_prefix must be defined")
    if not config.time_varying_prefix:
        raise ValueError("time_varying_prefix must be defined")

    log_message("Covariate selection rules validated successfully")
    return True
