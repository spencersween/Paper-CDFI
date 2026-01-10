"""
Event Study Aggregation Module

Aggregates ATT(g,t) estimates to event study format and other summaries.
Implements proper influence function aggregation for uniform inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .utils import log_message
from .config import Config
from .inference import compute_simple_att


def aggregate_influence_functions(
    att_results: Dict,
    event_times: List[int],
    weights_by_event: Dict[int, np.ndarray],
    indices_by_event: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Aggregate influence functions to event study level.

    For each event time e, the aggregated IF is:
        IF_e = sum_j(w_j * IF_j) for all (g,t) pairs with event_time = e

    Returns:
        Aggregated influence functions: (n_events, n_units) array
    """
    influence_functions = att_results['influence_functions']
    n_units = influence_functions[0].shape[0] if len(influence_functions) > 0 else 0
    n_events = len(event_times)

    agg_if = np.zeros((n_events, n_units))

    for i, e in enumerate(event_times):
        if e not in weights_by_event:
            continue
        weights = weights_by_event[e]
        indices = indices_by_event[e]

        # Weighted sum of influence functions
        for w, idx in zip(weights, indices):
            if idx < len(influence_functions):
                agg_if[i, :] += w * influence_functions[idx]

    return agg_if


def compute_clustered_se_and_bootstrap(
    att_vals: np.ndarray,
    agg_influence_functions: np.ndarray,
    cluster_indices: np.ndarray,
    n_clusters: int,
    n_bootstrap: int,
    multiplier_dist: str = "normal",
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute clustered SEs and bootstrap distribution from aggregated influence functions.

    Args:
        agg_influence_functions: (n_events, n_units) array of aggregated IFs

    Returns:
        (se, bootstrap_dist) tuple
    """
    if seed is not None:
        np.random.seed(seed)

    n_events = agg_influence_functions.shape[0]
    n_units = agg_influence_functions.shape[1]

    # Aggregate influence functions by cluster
    cluster_if = np.zeros((n_events, n_clusters))
    for c in range(n_clusters):
        mask = cluster_indices == c
        cluster_if[:, c] = np.nansum(agg_influence_functions[:, mask], axis=1)

    # Clustered variance: Var(ATT_e) = (1/N^2) * sum_c(IF_ec^2)
    # But for multiplier bootstrap, we generate xi_c and compute:
    # ATT_e^* = ATT_e + (1/N) * sum_c(xi_c * IF_ec)

    # Generate multiplier weights
    if multiplier_dist == "normal":
        xi = np.random.randn(n_clusters, n_bootstrap)
    elif multiplier_dist == "rademacher":
        xi = np.random.choice([-1, 1], size=(n_clusters, n_bootstrap))
    else:
        xi = np.random.randn(n_clusters, n_bootstrap)

    # Bootstrap distribution
    boot_dist = np.zeros((n_events, n_bootstrap))
    for e in range(n_events):
        for b in range(n_bootstrap):
            boot_dist[e, b] = att_vals[e] + np.sum(xi[:, b] * cluster_if[e, :]) / n_units

    # Standard errors from bootstrap
    se = np.nanstd(boot_dist, axis=1)

    return se, boot_dist


def compute_uniform_bands(
    att_vals: np.ndarray,
    se_vals: np.ndarray,
    boot_dist: np.ndarray,
    alpha: float
) -> Dict:
    """
    Compute uniform confidence bands using sup-t method.

    The sup-t method finds critical value c such that:
        P(max_e |t_e| <= c) = 1 - alpha

    where t_e = (ATT_e^* - ATT_e) / SE_e
    """
    n_events = len(att_vals)

    # T-statistics for each bootstrap draw
    t_stats = np.abs(boot_dist - att_vals[:, np.newaxis]) / se_vals[:, np.newaxis]
    t_stats[~np.isfinite(t_stats)] = np.nan

    # Supremum t-statistic for each bootstrap draw
    sup_t = np.nanmax(t_stats, axis=0)

    # Critical value: (1-alpha) quantile of sup-t distribution
    c_alpha = np.nanpercentile(sup_t, 100 * (1 - alpha))

    log_message(f"  Uniform bands: sup-t critical value = {c_alpha:.3f} (alpha={alpha})")

    return {
        'uniform_lower': att_vals - c_alpha * se_vals,
        'uniform_upper': att_vals + c_alpha * se_vals,
        'sup_t_critical': c_alpha,
        'sup_t_distribution': sup_t
    }


def aggregate_event_study(att_results: Dict, config: Config) -> Dict:
    """
    Aggregate ATT(g,t) to event study format with proper influence function inference.

    Steps:
    1. Compute weighted average ATT(e) for each event time
    2. Aggregate influence functions using same weights
    3. Compute clustered bootstrap from aggregated IFs
    4. Compute pointwise CIs and uniform bands
    """
    log_message("Aggregating to event study with influence function inference...")

    att_df = att_results['att']
    data = att_results['data']

    # Event study window
    min_e = -config.event_study.pre_periods
    max_e = config.event_study.post_periods

    # Valid event times
    event_times = sorted(att_df['event_time'].unique())
    event_times = [e for e in event_times if min_e <= e <= max_e]

    n_events = len(event_times)

    # Storage for weights and indices (needed for IF aggregation)
    weights_by_event = {}
    indices_by_event = {}
    es_results = []

    for i, e in enumerate(event_times):
        mask = att_df['event_time'] == e
        idx_positions = np.where(mask)[0]
        n_groups = mask.sum()

        if n_groups == 0:
            continue

        # Weights (by group size or equal)
        if config.event_study.weight_by_group_size:
            weights = att_df.loc[mask, 'n_treated'].values.astype(float)
        else:
            weights = np.ones(n_groups)
        weights = weights / np.nansum(weights)

        # Store for IF aggregation
        weights_by_event[e] = weights
        indices_by_event[e] = idx_positions

        # Weighted average ATT
        att_e = np.nansum(weights * att_df.loc[mask, 'att'].values)

        es_results.append({
            'event_time': e,
            'att': att_e,
            'n_groups': n_groups,
            'total_treated': att_df.loc[mask, 'n_treated'].sum()
        })

    es_df = pd.DataFrame(es_results)
    att_vals = es_df['att'].values

    # Aggregate influence functions
    log_message("  Aggregating influence functions...")
    agg_if = aggregate_influence_functions(
        att_results, event_times, weights_by_event, indices_by_event
    )

    # Get cluster information at UNIT level
    # Each unit belongs to one cluster
    cluster_var = config.cluster_var
    unit_ids = att_results['unit_ids']
    id_to_row = att_results['id_to_row']

    # Get cluster for each unit (use first observation per unit)
    unit_clusters = data.groupby(config.id_var)[cluster_var].first()
    clusters = unit_clusters.unique()
    n_clusters = len(clusters)
    cluster_map = pd.Series(range(len(clusters)), index=clusters)

    # Map unit IDs to cluster indices
    cluster_indices = np.array([cluster_map[unit_clusters[uid]] for uid in unit_ids])

    log_message(f"  Computing clustered bootstrap ({config.inference.n_bootstrap} reps, {n_clusters} clusters)...")

    # Compute SEs and bootstrap from aggregated IFs
    se_vals, es_boot = compute_clustered_se_and_bootstrap(
        att_vals,
        agg_if,
        cluster_indices,
        n_clusters,
        config.inference.n_bootstrap,
        config.inference.multiplier_dist,
        config.inference.seed
    )

    es_df['se'] = se_vals

    # Pointwise CIs (percentile method)
    alpha = config.inference.alpha
    es_df['ci_lower'] = np.nanpercentile(es_boot, 100 * alpha / 2, axis=1)
    es_df['ci_upper'] = np.nanpercentile(es_boot, 100 * (1 - alpha / 2), axis=1)

    # P-values
    es_df['t_stat'] = att_vals / se_vals
    es_df['p_value'] = 2 * stats.norm.sf(np.abs(es_df['t_stat']))

    # Uniform confidence bands
    uniform_results = None
    if config.inference.uniform_bands:
        uniform_results = compute_uniform_bands(att_vals, se_vals, es_boot, alpha)
        es_df['uniform_lower'] = uniform_results['uniform_lower']
        es_df['uniform_upper'] = uniform_results['uniform_upper']

    # Normalize to reference period
    ref_period = config.event_study.reference_period
    ref_idx = es_df['event_time'] == ref_period
    if ref_idx.any():
        ref_att = es_df.loc[ref_idx, 'att'].values[0]
        es_df['att_normalized'] = es_df['att'] - ref_att
        log_message(f"  Normalized to event time {ref_period} (ATT = {ref_att:.4f})")
    else:
        es_df['att_normalized'] = es_df['att']

    log_message(f"  Event study complete: {len(es_df)} event times (e={es_df['event_time'].min()} to {es_df['event_time'].max()})")

    return {
        'event_study': es_df,
        'bootstrap_dist': es_boot,
        'aggregated_influence_functions': agg_if,
        'uniform_bands': uniform_results,
        'n_clusters': n_clusters
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
