"""
m07_att_estimation.py - ATT Estimation

Computes doubly-robust ATT(g,t) estimates with unit-level influence functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats

from .m00_config import Config
from .m01_utils import log_message, clamp
from .m02_data_loader import UnitData, GTInfo, compute_outcome_diffs
from .m06_cross_fitting import CrossFitResults


@dataclass
class ATTResults:
    """ATT estimation results."""
    # ATT estimates DataFrame
    att_df: pd.DataFrame

    # Unit-level influence functions: list of (n_units,) arrays, one per (g,t)
    influence_functions: List[np.ndarray]

    # References
    unit_data: UnitData
    gt_info: GTInfo
    cf_results: CrossFitResults


def compute_att_gt(
    gt_idx: int,
    g: int,
    t: int,
    is_pre: bool,
    event_time: int,
    outcome_preds: np.ndarray,  # (n_units,)
    propensity_preds: np.ndarray,  # (n_units,)
    delta_Y: np.ndarray,  # (n_units,)
    treatment: np.ndarray,  # (n_units,)
    sample_mask: np.ndarray,  # (n_units,)
    config: Config
) -> Dict:
    """
    Compute doubly-robust ATT(g,t) estimate.

    ATT(g,t) = E[w1 * (ΔY - μ₀)] - E[w0 * (ΔY - μ₀)]

    where:
        w1 = D / P(D=1)  (treated weight)
        w0 = (1-D) * ps / ((1-ps) * P(D=1))  (control weight)

    Returns influence function at UNIT level (n_units,).
    """
    n_units = len(outcome_preds)

    # Get in-sample units
    in_sample = sample_mask & ~np.isnan(delta_Y) & ~np.isnan(treatment)
    in_sample = in_sample & ~np.isnan(outcome_preds) & ~np.isnan(propensity_preds)

    n_in_sample = in_sample.sum()

    if n_in_sample == 0:
        return {
            'att': np.nan,
            'se': np.nan,
            'influence_function': np.zeros(n_units),
            'n_treated': 0,
            'n_control': 0,
            'p_g': np.nan
        }

    # Extract in-sample data
    D = treatment[in_sample]
    delta_y = delta_Y[in_sample]
    mu_0 = outcome_preds[in_sample]
    ps = propensity_preds[in_sample]

    # Clamp propensity scores
    ps = clamp(ps, config.propensity.min_ps, config.propensity.max_ps)

    # Probability of being treated in sample
    p_g = D.mean()
    n_treated = int(D.sum())
    n_control = int((1 - D).sum())

    if p_g == 0 or p_g == 1:
        return {
            'att': np.nan,
            'se': np.nan,
            'influence_function': np.zeros(n_units),
            'n_treated': n_treated,
            'n_control': n_control,
            'p_g': p_g
        }

    # Residuals
    residual = delta_y - mu_0

    # Doubly-robust ATT following CS2021:
    # ATT = E[ w * (ΔY - μ₀) ]
    # where w = (D - ps) / (p_g * (1 - ps))
    #
    # For treated (D=1): w = (1-ps) / (p_g*(1-ps)) = 1/p_g
    # For control (D=0): w = -ps / (p_g*(1-ps))

    w = (D - ps) / (p_g * (1 - ps))

    # ATT estimate (doubly-robust)
    att = np.mean(w * residual)

    # Influence function for CS2021 doubly-robust estimator:
    # IF_i = w_i * (ΔY_i - μ₀(X_i) - ATT)
    inf_func_in_sample = w * (residual - att)

    # Expand to unit level (zeros for out-of-sample units)
    inf_func = np.zeros(n_units)
    inf_func[in_sample] = inf_func_in_sample

    # Standard error (analytical, from IF)
    se = np.sqrt((inf_func_in_sample ** 2).mean() / n_in_sample)

    return {
        'att': att,
        'se': se,
        'influence_function': inf_func,
        'n_treated': n_treated,
        'n_control': n_control,
        'n_in_sample': n_in_sample,
        'p_g': p_g
    }


def compute_all_att(cf_results: CrossFitResults, config: Config) -> ATTResults:
    """
    Compute all ATT(g,t) estimates with unit-level influence functions.
    """
    log_message("Computing ATT(g,t) estimates...")

    unit_data = cf_results.unit_data
    gt_info = cf_results.gt_info
    outcome_preds = cf_results.outcome_preds
    propensity_preds = cf_results.propensity_preds
    sample_mask = cf_results.sample_mask
    treatment = cf_results.treatment

    n_units = unit_data.n_units
    n_gt = gt_info.n_gt

    # Compute outcome differences
    delta_Y = compute_outcome_diffs(
        unit_data.Y,
        gt_info.t_indices,
        gt_info.base_t_indices
    )

    att_list = []
    influence_functions = []

    for _, row in gt_info.df.iterrows():
        gt_idx = row['gt_index']
        g = row['g']
        t = row['t']
        is_pre = row['is_pre']
        event_time = row['event_time']

        result = compute_att_gt(
            gt_idx=gt_idx,
            g=g,
            t=t,
            is_pre=is_pre,
            event_time=event_time,
            outcome_preds=outcome_preds[:, gt_idx],
            propensity_preds=propensity_preds[:, gt_idx],
            delta_Y=delta_Y[:, gt_idx],
            treatment=treatment[:, gt_idx],
            sample_mask=sample_mask[:, gt_idx],
            config=config
        )

        att_list.append({
            'g': g,
            't': t,
            'gt_index': gt_idx,
            'is_pre': is_pre,
            'event_time': event_time,
            'att': result['att'],
            'se': result['se'],
            'n_treated': result['n_treated'],
            'n_control': result['n_control'],
            'p_g': result['p_g']
        })

        influence_functions.append(result['influence_function'])

    att_df = pd.DataFrame(att_list)

    # Add confidence intervals and p-values
    alpha = config.inference.alpha
    z = stats.norm.ppf(1 - alpha / 2)
    att_df['ci_lower'] = att_df['att'] - z * att_df['se']
    att_df['ci_upper'] = att_df['att'] + z * att_df['se']
    att_df['t_stat'] = att_df['att'] / att_df['se']
    att_df['p_value'] = 2 * stats.norm.sf(np.abs(att_df['t_stat']))

    # Report summary
    n_valid = att_df['att'].notna().sum()
    log_message(f"  Computed {n_valid} valid ATT estimates")

    pre = att_df[att_df['is_pre']]
    post = att_df[~att_df['is_pre']]
    if len(pre) > 0:
        log_message(f"  Pre-treatment: {len(pre)} (mean ATT = {pre['att'].mean():.4f})")
    if len(post) > 0:
        log_message(f"  Post-treatment: {len(post)} (mean ATT = {post['att'].mean():.4f})")

    return ATTResults(
        att_df=att_df,
        influence_functions=influence_functions,
        unit_data=unit_data,
        gt_info=gt_info,
        cf_results=cf_results
    )


def print_att_summary(att_results: ATTResults):
    """Print ATT summary."""
    att_df = att_results.att_df

    print("\n" + "=" * 60)
    print("ATT(g,t) ESTIMATION SUMMARY")
    print("=" * 60 + "\n")

    print(f"Total (g,t) pairs: {len(att_df)}")
    print(f"Valid estimates: {att_df['att'].notna().sum()}")

    pre = att_df[att_df['is_pre']]
    if len(pre) > 0:
        print("\nPre-treatment (placebo):")
        print(f"  Mean ATT: {pre['att'].mean():.4f}")
        print(f"  % significant at 5%: {100 * (pre['p_value'] < 0.05).mean():.1f}%")

    post = att_df[~att_df['is_pre']]
    if len(post) > 0:
        print("\nPost-treatment:")
        print(f"  Mean ATT: {post['att'].mean():.4f}")
        print(f"  % significant at 5%: {100 * (post['p_value'] < 0.05).mean():.1f}%")

    print()
