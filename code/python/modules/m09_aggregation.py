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
    Aggregate ATT(g,t) to event study format following CS2021.

    For each event time e:
    1. Compute weighted average ATT(e) across groups
    2. Aggregate influence functions with same weights
    3. Run NEW clustered bootstrap on aggregated IFs (not aggregate (g,t) bootstrap!)

    This is critical: we must bootstrap the aggregated influence functions,
    not just aggregate the bootstrap distributions from (g,t).
    """
    log_message("Aggregating to event study...")

    att_df = att_results.att_df
    influence_functions = att_results.influence_functions
    unit_data = att_results.unit_data

    n_units = unit_data.n_units
    n_bootstrap = config.inference.n_bootstrap

    # Event study window
    min_e = -config.event_study.pre_periods
    max_e = config.event_study.post_periods

    # Get valid event times
    event_times = sorted(att_df['event_time'].unique())
    event_times = [e for e in event_times if min_e <= e <= max_e]
    n_event_times = len(event_times)

    # Cluster information for bootstrap
    clusters = unit_data.clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    cluster_indices = np.array([cluster_map[c] for c in clusters], dtype=np.int64)

    log_message(f"  Event times: {min(event_times)} to {max(event_times)}")
    log_message(f"  Clusters: {n_clusters}")

    # Step 1: Compute ATT(e) and aggregate influence functions for each event time
    es_results = []
    agg_influence_functions = []

    for e in event_times:
        mask = att_df['event_time'] == e
        idx_positions = np.where(mask)[0]
        n_groups = mask.sum()

        if n_groups == 0:
            es_results.append({
                'event_time': e,
                'att': np.nan,
                'n_groups': 0,
                'total_treated': 0
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

        # Aggregate influence functions with same weights
        # IF_e(i) = Σ_g w_g * IF_{g,t(g,e)}(i)
        agg_if = np.zeros(n_units)
        for w, idx in zip(weights, idx_positions):
            if idx < len(influence_functions):
                agg_if += w * influence_functions[idx]
        agg_influence_functions.append(agg_if)

        es_results.append({
            'event_time': e,
            'att': att_e,
            'n_groups': n_groups,
            'total_treated': att_df.loc[mask, 'n_treated'].sum()
        })

    es_df = pd.DataFrame(es_results)

    # Step 2: Run clustered multiplier bootstrap on aggregated influence functions
    # This follows CS2021 exactly: for each event time, bootstrap the aggregated IF
    log_message(f"  Running clustered bootstrap ({n_bootstrap} reps) on aggregated IFs...")

    # Generate multiplier weights at cluster level
    np.random.seed(config.inference.seed)
    if config.inference.multiplier_dist == "normal":
        xi = np.random.randn(n_clusters, n_bootstrap)
    else:  # rademacher
        xi = np.random.choice([-1, 1], size=(n_clusters, n_bootstrap))

    # Bootstrap distribution for event study
    es_boot = np.zeros((n_event_times, n_bootstrap))

    for i, (e, agg_if) in enumerate(zip(event_times, agg_influence_functions)):
        att_e = es_df.loc[es_df['event_time'] == e, 'att'].values[0]

        if np.isnan(att_e) or np.all(agg_if == 0):
            es_boot[i, :] = np.nan
            continue

        # Aggregate influence functions by cluster
        cluster_inf = np.zeros(n_clusters)
        for c in range(n_clusters):
            cluster_mask = cluster_indices == c
            cluster_inf[c] = np.nansum(agg_if[cluster_mask])

        # Bootstrap: ATT_e^* = ATT_e + (1/n) * Σ_c ξ_c * (Σ_{i∈c} IF_e(i))
        for b in range(n_bootstrap):
            es_boot[i, b] = att_e + np.sum(xi[:, b] * cluster_inf) / n_units

    # Compute SEs from bootstrap
    es_df['se'] = np.nanstd(es_boot, axis=1)

    # Pointwise CIs
    alpha = config.inference.alpha
    es_df['ci_lower'] = np.nanpercentile(es_boot, 100 * alpha / 2, axis=1)
    es_df['ci_upper'] = np.nanpercentile(es_boot, 100 * (1 - alpha / 2), axis=1)

    # P-values
    es_df['t_stat'] = es_df['att'] / es_df['se']
    es_df['p_value'] = 2 * stats.norm.sf(np.abs(es_df['t_stat']))

    # Step 3: Uniform confidence bands using sup-t method
    uniform_bands = None
    if config.inference.uniform_bands:
        att_vals = es_df['att'].values
        se_vals = es_df['se'].values

        # Compute t-statistics for each bootstrap sample
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stats = np.abs(es_boot - att_vals[:, np.newaxis]) / se_vals[:, np.newaxis]
            t_stats[~np.isfinite(t_stats)] = np.nan

        # Sup-t for each bootstrap sample (max across event times)
        sup_t = np.nanmax(t_stats, axis=0)

        # Critical value: (1-alpha) quantile of sup-t distribution
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

    return {
        'event_study': es_df,
        'bootstrap_dist': es_boot,
        'aggregated_influence_functions': agg_influence_functions,
        'uniform_bands': uniform_bands,
        'n_clusters': n_clusters
    }


def aggregate_by_group(att_results: ATTResults, config: Config) -> pd.DataFrame:
    """
    Aggregate by treatment group (cohort-specific effects).

    For each group g, compute weighted average of post-treatment ATT(g,t),
    then bootstrap the aggregated influence functions.
    """
    att_df = att_results.att_df
    influence_functions = att_results.influence_functions
    unit_data = att_results.unit_data

    n_units = unit_data.n_units
    n_bootstrap = config.inference.n_bootstrap

    # Cluster info for bootstrap
    clusters = unit_data.clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    cluster_indices = np.array([cluster_map[c] for c in clusters], dtype=np.int64)

    # Generate multiplier weights
    np.random.seed(config.inference.seed + 1)  # Different seed from event study
    if config.inference.multiplier_dist == "normal":
        xi = np.random.randn(n_clusters, n_bootstrap)
    else:
        xi = np.random.choice([-1, 1], size=(n_clusters, n_bootstrap))

    groups = sorted(att_df['g'].unique())
    group_results = []

    for g in groups:
        # Post-treatment ATTs for this group
        mask = (att_df['g'] == g) & (~att_df['is_pre'])
        idx_positions = np.where(mask)[0]
        n_periods = mask.sum()

        if n_periods == 0:
            continue

        # Equal weights across periods
        weights = np.ones(n_periods) / n_periods

        # Weighted average ATT
        att_vals = att_df.loc[mask, 'att'].values
        att_g = np.nansum(weights * att_vals)

        # Aggregate influence functions
        agg_if = np.zeros(n_units)
        for w, idx in zip(weights, idx_positions):
            if idx < len(influence_functions):
                agg_if += w * influence_functions[idx]

        # Clustered bootstrap
        cluster_inf = np.zeros(n_clusters)
        for c in range(n_clusters):
            cluster_mask = cluster_indices == c
            cluster_inf[c] = np.nansum(agg_if[cluster_mask])

        boot_g = np.array([att_g + np.sum(xi[:, b] * cluster_inf) / n_units
                          for b in range(n_bootstrap)])
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
    """
    Aggregate by calendar time.

    For each calendar time t, compute weighted average of ATT(g,t) across groups
    that are treated by time t, then bootstrap the aggregated influence functions.
    """
    att_df = att_results.att_df
    influence_functions = att_results.influence_functions
    unit_data = att_results.unit_data

    n_units = unit_data.n_units
    n_bootstrap = config.inference.n_bootstrap

    # Cluster info for bootstrap
    clusters = unit_data.clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    cluster_indices = np.array([cluster_map[c] for c in clusters], dtype=np.int64)

    # Generate multiplier weights
    np.random.seed(config.inference.seed + 2)  # Different seed
    if config.inference.multiplier_dist == "normal":
        xi = np.random.randn(n_clusters, n_bootstrap)
    else:
        xi = np.random.choice([-1, 1], size=(n_clusters, n_bootstrap))

    # Post-treatment only
    post_df = att_df[~att_df['is_pre']]
    times = sorted(post_df['t'].unique())

    time_results = []

    for t in times:
        mask = (att_df['t'] == t) & (~att_df['is_pre'])
        idx_positions = np.where(mask)[0]
        n_groups = mask.sum()

        if n_groups == 0:
            continue

        # Weight by number of treated units
        weights = att_df.loc[mask, 'n_treated'].values.astype(float)
        weights = weights / np.nansum(weights)

        # Weighted average ATT
        att_t = np.nansum(weights * att_df.loc[mask, 'att'].values)

        # Aggregate influence functions
        agg_if = np.zeros(n_units)
        for w, idx in zip(weights, idx_positions):
            if idx < len(influence_functions):
                agg_if += w * influence_functions[idx]

        # Clustered bootstrap
        cluster_inf = np.zeros(n_clusters)
        for c in range(n_clusters):
            cluster_mask = cluster_indices == c
            cluster_inf[c] = np.nansum(agg_if[cluster_mask])

        boot_t = np.array([att_t + np.sum(xi[:, b] * cluster_inf) / n_units
                          for b in range(n_bootstrap)])
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
