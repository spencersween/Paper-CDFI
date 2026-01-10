"""
Nuisance Parameter Estimation Module

Extracts and organizes nuisance parameters from cross-fitting results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from .utils import log_message
from .config import Config
from .data_loader import create_gt_sample


@dataclass
class NuisanceEstimates:
    """Nuisance estimates for a specific (g,t) pair."""
    mu_0: np.ndarray
    ps: np.ndarray
    delta_y: np.ndarray
    D: np.ndarray
    ids: np.ndarray
    n: int
    n_treated: int
    n_control: int
    g: int
    t: int
    gt_index: int
    is_pre: bool
    event_time: int


def get_nuisance_gt(cf_results: Dict, g: int, t: int, config: Config) -> NuisanceEstimates:
    """Get nuisance estimates for a specific (g,t) pair."""
    gt_pairs = cf_results['gt_pairs']
    data = cf_results['data']
    id_to_row = cf_results['id_to_row']

    # Find gt_index
    gt_row = gt_pairs[(gt_pairs['g'] == g) & (gt_pairs['t'] == t)]
    if len(gt_row) == 0:
        raise ValueError(f"(g={g}, t={t}) pair not found in gt_pairs")

    gt_idx = gt_row['gt_index'].values[0]
    is_pre = gt_row['is_pre'].values[0]
    event_time = gt_row['event_time'].values[0]

    # Create estimation sample (cross-sectional at time t)
    sample = create_gt_sample(data, g, t, config)
    sample_ids = sample[config.id_var].values

    # Map sample IDs to unit-level row indices
    unit_rows = np.array([id_to_row.get(sid, -1) for sid in sample_ids])
    valid_rows = unit_rows >= 0

    if not valid_rows.all():
        n_missing = (~valid_rows).sum()
        log_message(f"Warning: {n_missing} sample units not found in id_to_row", level="WARNING")

    # Extract nuisance estimates from unit-level storage
    mu_0 = np.full(len(sample_ids), np.nan)
    ps = np.full(len(sample_ids), np.nan)
    mu_0[valid_rows] = cf_results['outcome'][unit_rows[valid_rows], gt_idx]
    ps[valid_rows] = cf_results['propensity'][unit_rows[valid_rows], gt_idx]

    return NuisanceEstimates(
        mu_0=mu_0,
        ps=ps,
        delta_y=sample['delta_y'].values,
        D=sample['D'].values,
        ids=sample_ids,
        n=len(sample_ids),
        n_treated=int((sample['D'] == 1).sum()),
        n_control=int((sample['D'] == 0).sum()),
        g=g,
        t=t,
        gt_index=gt_idx,
        is_pre=is_pre,
        event_time=event_time
    )


def organize_nuisance_estimates(cf_results: Dict, config: Config) -> Dict[str, NuisanceEstimates]:
    """Organize all nuisance estimates by (g,t) pair."""
    gt_pairs = cf_results['gt_pairs']
    nuisance_dict = {}

    for _, row in gt_pairs.iterrows():
        g, t = row['g'], row['t']
        key = f"g{g}_t{t}"
        nuisance_dict[key] = get_nuisance_gt(cf_results, g, t, config)

    return nuisance_dict


def diagnose_nuisance(cf_results: Dict, config: Config) -> Dict:
    """Run diagnostic checks on nuisance parameter estimates."""
    log_message("Running nuisance parameter diagnostics...")

    nuisance_dict = organize_nuisance_estimates(cf_results, config)

    # Compute summary statistics
    mu0_means = [n.mu_0.mean() for n in nuisance_dict.values() if not np.isnan(n.mu_0).all()]
    ps_means = [n.ps.mean() for n in nuisance_dict.values() if not np.isnan(n.ps).all()]
    ps_mins = [np.nanmin(n.ps) for n in nuisance_dict.values()]
    ps_maxs = [np.nanmax(n.ps) for n in nuisance_dict.values()]

    print("\n" + "=" * 60)
    print("NUISANCE PARAMETER DIAGNOSTICS")
    print("=" * 60 + "\n")

    print("Outcome Regression (mu_0):")
    print(f"  Mean across (g,t): {np.mean(mu0_means):.4f} (SD: {np.std(mu0_means):.4f})")
    print(f"  Range: [{min(mu0_means):.4f}, {max(mu0_means):.4f}]")

    print("\nPropensity Scores:")
    print(f"  Mean across (g,t): {np.mean(ps_means):.4f} (SD: {np.std(ps_means):.4f})")
    print(f"  Range: [{min(ps_mins):.4f}, {max(ps_maxs):.4f}]")

    return {
        'nuisance_dict': nuisance_dict,
        'mu0_means': mu0_means,
        'ps_means': ps_means
    }
