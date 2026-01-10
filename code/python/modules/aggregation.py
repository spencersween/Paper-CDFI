"""
Event Study Aggregation Module

Aggregates ATT(g,t) estimates to event study format and other summaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .utils import log_message
from .config import Config
from .inference import compute_simple_att


def aggregate_event_study(att_results: Dict, config: Config) -> Dict:
    """Aggregate ATT(g,t) to event study format."""
    log_message("Aggregating to event study...")

    att_df = att_results['att']
    boot_dist = att_results['bootstrap']['bootstrap_dist']

    # Event study window
    min_e = -config.event_study.pre_periods
    max_e = config.event_study.post_periods

    # Valid event times
    event_times = sorted(att_df['event_time'].unique())
    event_times = [e for e in event_times if min_e <= e <= max_e]

    n_events = len(event_times)
    n_boot = boot_dist.shape[1]

    # Storage
    es_results = []
    es_boot = np.full((n_events, n_boot), np.nan)

    for i, e in enumerate(event_times):
        idx = att_df['event_time'] == e
        n_groups = idx.sum()

        if n_groups == 0:
            continue

        # Weights
        if config.event_study.weight_by_group_size:
            weights = att_df.loc[idx, 'n_treated'].values
        else:
            weights = np.ones(n_groups)
        weights = weights / np.nansum(weights)

        # Weighted average ATT
        att_e = np.nansum(weights * att_df.loc[idx, 'att'].values)

        # Bootstrap distribution
        boot_e = np.nansum(weights[:, np.newaxis] * boot_dist[idx.values, :], axis=0)
        es_boot[i, :] = boot_e

        es_results.append({
            'event_time': e,
            'att': att_e,
            'n_groups': n_groups
        })

    es_df = pd.DataFrame(es_results)

    # Compute SEs and CIs
    es_df['se'] = np.nanstd(es_boot, axis=1)[:len(es_df)]

    alpha = config.inference.alpha
    es_df['ci_lower'] = np.nanpercentile(es_boot, 100 * alpha / 2, axis=1)[:len(es_df)]
    es_df['ci_upper'] = np.nanpercentile(es_boot, 100 * (1 - alpha / 2), axis=1)[:len(es_df)]

    # Uniform bands
    if config.inference.uniform_bands:
        att_vals = es_df['att'].values
        se_vals = es_df['se'].values
        t_stats = np.abs(es_boot[:len(es_df), :] - att_vals[:, np.newaxis]) / se_vals[:, np.newaxis]
        t_stats[~np.isfinite(t_stats)] = np.nan
        sup_t = np.nanmax(t_stats, axis=0)
        c_alpha = np.nanpercentile(sup_t, 100 * (1 - alpha))
        es_df['uniform_lower'] = att_vals - c_alpha * se_vals
        es_df['uniform_upper'] = att_vals + c_alpha * se_vals

    # Normalize to reference period
    ref_period = config.event_study.reference_period
    ref_idx = es_df['event_time'] == ref_period
    if ref_idx.any():
        ref_att = es_df.loc[ref_idx, 'att'].values[0]
        es_df['att_normalized'] = es_df['att'] - ref_att
        log_message(f"Normalized to event time {ref_period} (ATT = {ref_att:.4f})")
    else:
        es_df['att_normalized'] = es_df['att']

    log_message(f"Event study: {len(es_df)} event times ({es_df['event_time'].min()} to {es_df['event_time'].max()})")

    return {
        'event_study': es_df,
        'bootstrap_dist': es_boot[:len(es_df), :]
    }


def aggregate_by_group(att_results: Dict, config: Config) -> pd.DataFrame:
    """Aggregate by treatment group (cohort-specific effects)."""
    att_df = att_results['att']
    boot_dist = att_results['bootstrap']['bootstrap_dist']

    groups = sorted(att_df['g'].unique())
    n_boot = boot_dist.shape[1]

    group_results = []

    for g in groups:
        # Post-treatment ATTs for this group
        idx = (att_df['g'] == g) & (~att_df['is_pre'])
        n_periods = idx.sum()

        if n_periods == 0:
            continue

        att_g = att_df.loc[idx, 'att'].mean()
        boot_g = np.nanmean(boot_dist[idx.values, :], axis=0)
        se_g = np.nanstd(boot_g)

        alpha = config.inference.alpha
        ci_lower, ci_upper = np.nanpercentile(boot_g, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        group_results.append({
            'g': g,
            'att': att_g,
            'se': se_g,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_periods': n_periods
        })

    return pd.DataFrame(group_results)


def aggregate_by_time(att_results: Dict, config: Config) -> pd.DataFrame:
    """Aggregate by calendar time."""
    att_df = att_results['att']
    boot_dist = att_results['bootstrap']['bootstrap_dist']

    # Post-treatment only
    post_df = att_df[~att_df['is_pre']]
    times = sorted(post_df['t'].unique())
    n_boot = boot_dist.shape[1]

    time_results = []

    for t in times:
        idx = (att_df['t'] == t) & (~att_df['is_pre'])
        n_groups = idx.sum()

        if n_groups == 0:
            continue

        weights = att_df.loc[idx, 'n_treated'].values
        weights = weights / np.nansum(weights)

        att_t = np.nansum(weights * att_df.loc[idx, 'att'].values)
        boot_t = np.nansum(weights[:, np.newaxis] * boot_dist[idx.values, :], axis=0)
        se_t = np.nanstd(boot_t)

        alpha = config.inference.alpha
        ci_lower, ci_upper = np.nanpercentile(boot_t, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        time_results.append({
            't': t,
            'att': att_t,
            'se': se_t,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_groups': n_groups
        })

    return pd.DataFrame(time_results)


def aggregate_all(att_results: Dict, config: Config) -> Dict:
    """Create full aggregation summary."""
    log_message("Computing all aggregations...")

    return {
        'event_study': aggregate_event_study(att_results, config),
        'by_group': aggregate_by_group(att_results, config),
        'by_time': aggregate_by_time(att_results, config),
        'simple_att': compute_simple_att(att_results, config)
    }


def print_aggregation_summary(agg_results: Dict):
    """Print aggregation summary."""
    print("\n" + "=" * 60)
    print("AGGREGATION SUMMARY")
    print("=" * 60 + "\n")

    # Simple ATT
    sa = agg_results['simple_att']
    print("Simple ATT (weighted average post-treatment):")
    print(f"  ATT = {sa['att']:.4f} (SE = {sa['se']:.4f})")
    print(f"  95% CI: [{sa['ci_lower']:.4f}, {sa['ci_upper']:.4f}]")
    print(f"  p-value: {sa['p_value']:.4f}\n")

    # Event Study
    es = agg_results['event_study']['event_study']
    print("Event Study:")
    print(f"  Event times: {es['event_time'].min()} to {es['event_time'].max()}")

    pre = es[es['event_time'] < 0]
    post = es[es['event_time'] >= 0]

    if len(pre) > 0:
        print(f"  Pre-treatment mean ATT: {pre['att'].mean():.4f}")
    if len(post) > 0:
        print(f"  Post-treatment mean ATT: {post['att'].mean():.4f}")

    print()
