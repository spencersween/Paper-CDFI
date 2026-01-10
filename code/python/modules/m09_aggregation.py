"""
m09_aggregation.py - Event Study Aggregation

Aggregates ATT(g,t) estimates to event study format with proper
influence function aggregation for uniform inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

from .m00_config import Config
from .m01_utils import log_message
from .m07_att_estimation import ATTResults


def aggregate_event_study(att_results: ATTResults, config: Config) -> Dict:
    """
    Aggregate ATT(g,t) to event study format.

    For each event time e:
    1. Compute weighted average ATT(e) across groups
    2. Aggregate influence functions with same weights
    3. Compute clustered bootstrap from aggregated IFs
    """
    log_message("Aggregating to event study...")

    att_df = att_results.att_df
    influence_functions = att_results.influence_functions
    unit_data = att_results.unit_data
    boot_dist_gt = att_results.bootstrap_results['bootstrap_dist']

    n_units = unit_data.n_units
    n_bootstrap = boot_dist_gt.shape[1]

    # Event study window
    min_e = -config.event_study.pre_periods
    max_e = config.event_study.post_periods

    # Get valid event times
    event_times = sorted(att_df['event_time'].unique())
    event_times = [e for e in event_times if min_e <= e <= max_e]

    # Cluster information for bootstrap
    clusters = unit_data.clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    cluster_indices = np.array([cluster_map[c] for c in clusters], dtype=np.int64)

    es_results = []
    es_boot = np.zeros((len(event_times), n_bootstrap))
    agg_influence_functions = []

    for i, e in enumerate(event_times):
        mask = att_df['event_time'] == e
        idx_positions = np.where(mask)[0]
        n_groups = mask.sum()

        if n_groups == 0:
            es_results.append({
                'event_time': e,
                'att': np.nan,
                'se': np.nan,
                'n_groups': 0
            })
            agg_influence_functions.append(np.zeros(n_units))
            continue

        # Weights (by group size or equal)
        if config.event_study.weight_by_group_size:
            weights = att_df.loc[mask, 'n_treated'].values.astype(float)
        else:
            weights = np.ones(n_groups)

        weights = weights / np.nansum(weights)

        # Weighted average ATT
        att_vals = att_df.loc[mask, 'att'].values
        att_e = np.nansum(weights * att_vals)

        # Aggregate influence functions
        agg_if = np.zeros(n_units)
        for w, idx in zip(weights, idx_positions):
            if idx < len(influence_functions):
                agg_if += w * influence_functions[idx]
        agg_influence_functions.append(agg_if)

        # Aggregate bootstrap distribution
        boot_e = np.nansum(weights[:, np.newaxis] * boot_dist_gt[idx_positions, :], axis=0)
        es_boot[i, :] = boot_e

        es_results.append({
            'event_time': e,
            'att': att_e,
            'n_groups': n_groups,
            'total_treated': att_df.loc[mask, 'n_treated'].sum()
        })

    es_df = pd.DataFrame(es_results)

    # Compute SEs from bootstrap
    es_df['se'] = np.nanstd(es_boot, axis=1)

    # Pointwise CIs
    alpha = config.inference.alpha
    es_df['ci_lower'] = np.nanpercentile(es_boot, 100 * alpha / 2, axis=1)
    es_df['ci_upper'] = np.nanpercentile(es_boot, 100 * (1 - alpha / 2), axis=1)

    # P-values
    es_df['t_stat'] = es_df['att'] / es_df['se']
    es_df['p_value'] = 2 * stats.norm.sf(np.abs(es_df['t_stat']))

    # Uniform confidence bands
    uniform_bands = None
    if config.inference.uniform_bands:
        att_vals = es_df['att'].values
        se_vals = es_df['se'].values

        with np.errstate(divide='ignore', invalid='ignore'):
            t_stats = np.abs(es_boot - att_vals[:, np.newaxis]) / se_vals[:, np.newaxis]
            t_stats[~np.isfinite(t_stats)] = np.nan

        sup_t = np.nanmax(t_stats, axis=0)
        c_alpha = np.nanpercentile(sup_t, 100 * (1 - alpha))

        es_df['uniform_lower'] = att_vals - c_alpha * se_vals
        es_df['uniform_upper'] = att_vals + c_alpha * se_vals

        uniform_bands = {
            'uniform_lower': es_df['uniform_lower'].values,
            'uniform_upper': es_df['uniform_upper'].values,
            'sup_t_critical': c_alpha
        }

        log_message(f"  Sup-t critical value: {c_alpha:.3f}")

    # Normalize to reference period
    ref_period = config.event_study.reference_period
    ref_idx = es_df['event_time'] == ref_period
    if ref_idx.any():
        ref_att = es_df.loc[ref_idx, 'att'].values[0]
        es_df['att_normalized'] = es_df['att'] - ref_att
    else:
        es_df['att_normalized'] = es_df['att']

    log_message(f"  Event times: {es_df['event_time'].min()} to {es_df['event_time'].max()}")

    return {
        'event_study': es_df,
        'bootstrap_dist': es_boot,
        'aggregated_influence_functions': agg_influence_functions,
        'uniform_bands': uniform_bands,
        'n_clusters': n_clusters
    }


def aggregate_by_group(att_results: ATTResults, config: Config) -> pd.DataFrame:
    """Aggregate by treatment group (cohort-specific effects)."""
    att_df = att_results.att_df
    boot_dist = att_results.bootstrap_results['bootstrap_dist']

    groups = sorted(att_df['g'].unique())
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
        ci_lower = np.nanpercentile(boot_g, 100 * alpha / 2)
        ci_upper = np.nanpercentile(boot_g, 100 * (1 - alpha / 2))

        group_results.append({
            'g': g,
            'att': att_g,
            'se': se_g,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_periods': n_periods
        })

    return pd.DataFrame(group_results)


def aggregate_by_time(att_results: ATTResults, config: Config) -> pd.DataFrame:
    """Aggregate by calendar time."""
    att_df = att_results.att_df
    boot_dist = att_results.bootstrap_results['bootstrap_dist']

    # Post-treatment only
    post_df = att_df[~att_df['is_pre']]
    times = sorted(post_df['t'].unique())

    time_results = []

    for t in times:
        idx = (att_df['t'] == t) & (~att_df['is_pre'])
        n_groups = idx.sum()

        if n_groups == 0:
            continue

        weights = att_df.loc[idx, 'n_treated'].values.astype(float)
        weights = weights / np.nansum(weights)

        att_t = np.nansum(weights * att_df.loc[idx, 'att'].values)
        boot_t = np.nansum(weights[:, np.newaxis] * boot_dist[idx.values, :], axis=0)
        se_t = np.nanstd(boot_t)

        alpha = config.inference.alpha
        ci_lower = np.nanpercentile(boot_t, 100 * alpha / 2)
        ci_upper = np.nanpercentile(boot_t, 100 * (1 - alpha / 2))

        time_results.append({
            't': t,
            'att': att_t,
            'se': se_t,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_groups': n_groups
        })

    return pd.DataFrame(time_results)


def aggregate_all(att_results: ATTResults, config: Config) -> Dict:
    """Create full aggregation summary."""
    log_message("Computing all aggregations...")

    from .m08_inference import compute_simple_att

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
