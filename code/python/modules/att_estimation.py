"""
ATT Estimation Module

Implements doubly-robust ATT(g,t) estimation following Callaway & Sant'Anna (2021).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .utils import log_message, clamp
from .config import Config
from .nuisance_estimation import get_nuisance_gt, NuisanceEstimates


def compute_att_gt(nuisance: NuisanceEstimates, config: Config) -> Dict:
    """
    Compute doubly-robust ATT(g,t) estimate.

    Uses the influence function approach from CS2021:
    ATT(g,t) = E[w1 * (DeltaY - mu_0)] - E[w0 * (DeltaY - mu_0)]
    """
    delta_y = nuisance.delta_y
    mu_0 = nuisance.mu_0
    ps = nuisance.ps
    D = nuisance.D
    n = nuisance.n

    # Handle missing values
    valid_idx = ~np.isnan(delta_y) & ~np.isnan(mu_0) & ~np.isnan(ps)
    if valid_idx.sum() < n * 0.5:
        log_message(f"Warning: More than 50% missing values for (g={nuisance.g}, t={nuisance.t})", level="WARNING")

    delta_y = delta_y[valid_idx]
    mu_0 = mu_0[valid_idx]
    ps = ps[valid_idx]
    D = D[valid_idx]
    n_valid = valid_idx.sum()

    # Clamp propensity scores
    ps = clamp(ps, config.propensity.min_ps, config.propensity.max_ps)

    # Probability of being in treated group
    p_g = D.mean()

    if p_g == 0 or p_g == 1:
        log_message(f"Warning: Degenerate treatment probability for (g={nuisance.g}, t={nuisance.t}): p_g = {p_g:.4f}", level="WARNING")
        return {
            'att': np.nan,
            'se': np.nan,
            'influence_function': np.full(n, np.nan),
            'n_valid': n_valid,
            'p_g': p_g
        }

    # Residuals
    residual = delta_y - mu_0

    # Weights
    w1 = D / p_g
    w0 = (1 - D) * ps / ((1 - ps) * p_g)

    # Normalize control weights
    n_control = (1 - D).sum()
    if n_control > 0:
        w0_sum = w0.sum()
        if w0_sum > 0:
            w0 = w0 * n_control / w0_sum

    # ATT estimate (doubly-robust)
    att = (w1 * residual).mean() - (w0 * residual).mean()

    # Influence function
    inf_func_valid = (D / p_g) * (residual - att) - ((1 - D) * ps / ((1 - ps) * p_g)) * residual

    # Expand to full sample
    inf_func = np.full(n, np.nan)
    inf_func[valid_idx] = inf_func_valid

    # Standard error
    se = np.sqrt((inf_func_valid ** 2).mean() / n_valid)

    return {
        'att': att,
        'se': se,
        'influence_function': inf_func,
        'n_valid': n_valid,
        'n_treated': int(D.sum()),
        'n_control': int(n_control),
        'p_g': p_g,
        'g': nuisance.g,
        't': nuisance.t,
        'is_pre': nuisance.is_pre,
        'event_time': nuisance.event_time
    }


def compute_all_att(cf_results: Dict, config: Config) -> Dict:
    """Compute all ATT(g,t) estimates."""
    log_message("Computing ATT(g,t) estimates...")

    gt_pairs = cf_results['gt_pairs']
    n_gt = len(gt_pairs)

    att_results = []
    influence_functions = []

    for _, row in gt_pairs.iterrows():
        g, t = row['g'], row['t']

        nuisance = get_nuisance_gt(cf_results, g, t, config)
        att_result = compute_att_gt(nuisance, config)

        att_results.append({
            'g': g,
            't': t,
            'gt_index': row['gt_index'],
            'is_pre': row['is_pre'],
            'event_time': row['event_time'],
            'att': att_result['att'],
            'se': att_result['se'],
            'n_valid': att_result['n_valid'],
            'n_treated': att_result['n_treated'],
            'n_control': att_result['n_control'],
            'p_g': att_result['p_g']
        })

        influence_functions.append(att_result['influence_function'])

    att_df = pd.DataFrame(att_results)

    # Add confidence intervals
    alpha = config.inference.alpha
    z = stats.norm.ppf(1 - alpha / 2)
    att_df['ci_lower'] = att_df['att'] - z * att_df['se']
    att_df['ci_upper'] = att_df['att'] + z * att_df['se']
    att_df['t_stat'] = att_df['att'] / att_df['se']
    att_df['p_value'] = 2 * stats.norm.sf(np.abs(att_df['t_stat']))

    log_message(f"Computed {len(att_df)} ATT(g,t) estimates")
    log_message(f"  Pre-treatment: {att_df['is_pre'].sum()} (mean ATT = {att_df[att_df['is_pre']]['att'].mean():.4f})")
    log_message(f"  Post-treatment: {(~att_df['is_pre']).sum()} (mean ATT = {att_df[~att_df['is_pre']]['att'].mean():.4f})")

    return {
        'att': att_df,
        'influence_functions': influence_functions,
        'data': cf_results['data'],
        'config': config
    }


def print_att_summary(att_results: Dict):
    """Print ATT summary."""
    att_df = att_results['att']

    print("\n" + "=" * 60)
    print("ATT(g,t) ESTIMATION SUMMARY")
    print("=" * 60 + "\n")

    print(f"Total (g,t) pairs: {len(att_df)}")
    print(f"Valid estimates: {att_df['att'].notna().sum()}")

    # Pre-treatment
    pre = att_df[att_df['is_pre']]
    if len(pre) > 0:
        print("\nPre-treatment periods (placebo test):")
        print(f"  Mean ATT: {pre['att'].mean():.4f} (SE: {pre['att'].std() / np.sqrt(pre['att'].notna().sum()):.4f})")
        print(f"  % significant at 5%: {100 * (pre['p_value'] < 0.05).mean():.1f}%")

    # Post-treatment
    post = att_df[~att_df['is_pre']]
    if len(post) > 0:
        print("\nPost-treatment periods:")
        print(f"  Mean ATT: {post['att'].mean():.4f} (SE: {post['att'].std() / np.sqrt(post['att'].notna().sum()):.4f})")
        print(f"  % significant at 5%: {100 * (post['p_value'] < 0.05).mean():.1f}%")

    print()
