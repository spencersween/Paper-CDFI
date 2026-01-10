"""
m08_inference.py - Inference via Clustered Bootstrap

Implements clustered multiplier bootstrap for ATT(g,t) estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

from .m00_config import Config
from .m01_utils import log_message
from .m07_att_estimation import ATTResults


def generate_multiplier_weights(
    n_clusters: int,
    n_bootstrap: int,
    distribution: str = "normal",
    seed: int = None
) -> np.ndarray:
    """Generate multiplier weights for bootstrap."""
    if seed is not None:
        np.random.seed(seed)

    if distribution == "normal":
        return np.random.randn(n_clusters, n_bootstrap)
    elif distribution == "rademacher":
        return np.random.choice([-1, 1], size=(n_clusters, n_bootstrap))
    else:
        return np.random.randn(n_clusters, n_bootstrap)


def clustered_bootstrap(att_results: ATTResults, config: Config) -> Dict:
    """
    Clustered multiplier bootstrap for ATT(g,t) estimates.

    Uses unit-level influence functions, aggregated by cluster.
    """
    log_message(f"Running clustered bootstrap ({config.inference.n_bootstrap} replications)...")

    unit_data = att_results.unit_data
    att_df = att_results.att_df
    influence_functions = att_results.influence_functions

    n_units = unit_data.n_units
    n_gt = len(att_df)
    n_bootstrap = config.inference.n_bootstrap

    # Get cluster information
    clusters = unit_data.clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    cluster_indices = np.array([cluster_map[c] for c in clusters], dtype=np.int64)

    log_message(f"  Units: {n_units}, Clusters: {n_clusters}")

    # Generate multiplier weights
    xi = generate_multiplier_weights(
        n_clusters,
        n_bootstrap,
        config.inference.multiplier_dist,
        config.inference.seed
    )

    # Bootstrap distribution
    boot_dist = np.full((n_gt, n_bootstrap), np.nan)

    for i in range(n_gt):
        inf_func = influence_functions[i]

        if np.all(inf_func == 0) or np.all(np.isnan(inf_func)):
            continue

        # Aggregate influence functions by cluster
        cluster_inf = np.zeros(n_clusters)
        for c in range(n_clusters):
            mask = cluster_indices == c
            cluster_inf[c] = np.nansum(inf_func[mask])

        # Bootstrap replications
        att_i = att_df.iloc[i]['att']
        if np.isnan(att_i):
            continue

        for b in range(n_bootstrap):
            boot_dist[i, b] = att_i + np.sum(xi[:, b] * cluster_inf) / n_units

    # Compute bootstrap SEs
    boot_se = np.nanstd(boot_dist, axis=1)

    # Pointwise CIs (percentile method)
    alpha = config.inference.alpha
    ci_lower = np.nanpercentile(boot_dist, 100 * alpha / 2, axis=1)
    ci_upper = np.nanpercentile(boot_dist, 100 * (1 - alpha / 2), axis=1)

    # Uniform bands (sup-t method)
    uniform_bands = None
    if config.inference.uniform_bands:
        uniform_bands = compute_uniform_bands(
            att_df['att'].values,
            boot_dist,
            boot_se,
            alpha
        )

    log_message("Bootstrap complete")

    return {
        'bootstrap_dist': boot_dist,
        'se': boot_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'uniform_bands': uniform_bands,
        'n_bootstrap': n_bootstrap,
        'n_clusters': n_clusters
    }


def compute_uniform_bands(
    att: np.ndarray,
    boot_dist: np.ndarray,
    se: np.ndarray,
    alpha: float
) -> Dict:
    """Compute uniform confidence bands using sup-t method."""
    # T-statistics for each bootstrap sample
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = np.abs(boot_dist - att[:, np.newaxis]) / se[:, np.newaxis]
        t_stats[~np.isfinite(t_stats)] = np.nan

    # Sup-t for each bootstrap sample
    sup_t = np.nanmax(t_stats, axis=0)

    # Critical value: (1-alpha) quantile of sup-t distribution
    c_alpha = np.nanpercentile(sup_t, 100 * (1 - alpha))

    log_message(f"  Sup-t critical value: {c_alpha:.3f}")

    return {
        'uniform_lower': att - c_alpha * se,
        'uniform_upper': att + c_alpha * se,
        'sup_t_critical': c_alpha
    }


def add_bootstrap_inference(att_results: ATTResults, config: Config) -> ATTResults:
    """Add bootstrap inference to ATT results."""
    boot_results = clustered_bootstrap(att_results, config)

    # Update ATT DataFrame
    att_df = att_results.att_df.copy()
    att_df['se_boot'] = boot_results['se']
    att_df['ci_lower_boot'] = boot_results['ci_lower']
    att_df['ci_upper_boot'] = boot_results['ci_upper']

    if boot_results['uniform_bands'] is not None:
        att_df['uniform_lower'] = boot_results['uniform_bands']['uniform_lower']
        att_df['uniform_upper'] = boot_results['uniform_bands']['uniform_upper']

    att_df['t_stat_boot'] = att_df['att'] / att_df['se_boot']
    att_df['p_value_boot'] = 2 * stats.norm.sf(np.abs(att_df['t_stat_boot']))

    # Update results
    att_results.att_df = att_df
    att_results.bootstrap_results = boot_results

    return att_results


def test_parallel_trends(att_results: ATTResults, config: Config) -> Dict:
    """Test whether pre-treatment ATT estimates are jointly zero."""
    log_message("Testing parallel trends...")

    att_df = att_results.att_df
    boot_dist = att_results.bootstrap_results['bootstrap_dist']

    # Pre-treatment indices
    pre_mask = att_df['is_pre'].values
    n_pre = pre_mask.sum()

    if n_pre == 0:
        log_message("  No pre-treatment periods found")
        return {'test_stat': np.nan, 'p_value': np.nan, 'n_pre': 0}

    att_pre = att_df.loc[pre_mask, 'att'].values
    boot_pre = boot_dist[pre_mask, :]

    # Mean pre-treatment ATT
    mean_att_pre = np.nanmean(att_pre)
    se_mean_pre = np.nanstd(np.nanmean(boot_pre, axis=0))

    if se_mean_pre == 0:
        test_stat = np.nan
        p_value = np.nan
    else:
        test_stat = np.abs(mean_att_pre / se_mean_pre)
        p_value = 2 * stats.norm.sf(test_stat)

    log_message(f"  Pre-treatment periods: {n_pre}")
    log_message(f"  Mean pre-treatment ATT: {mean_att_pre:.4f} (SE: {se_mean_pre:.4f})")
    log_message(f"  Test statistic: {test_stat:.3f}, p-value: {p_value:.4f}")

    return {
        'test_stat': test_stat,
        'p_value': p_value,
        'mean_att_pre': mean_att_pre,
        'se_mean_pre': se_mean_pre,
        'n_pre': n_pre,
        'reject': p_value < config.inference.alpha if not np.isnan(p_value) else np.nan
    }


def compute_simple_att(att_results: ATTResults, config: Config) -> Dict:
    """Compute simple ATT (weighted average post-treatment effect)."""
    att_df = att_results.att_df
    boot_dist = att_results.bootstrap_results['bootstrap_dist']

    # Post-treatment indices
    post_mask = ~att_df['is_pre'].values

    if not post_mask.any():
        return {'att': np.nan, 'se': np.nan}

    # Weights (by number of treated units)
    weights = att_df.loc[post_mask, 'n_treated'].values.astype(float)
    weights = weights / np.nansum(weights)

    # Weighted average ATT
    att_simple = np.nansum(weights * att_df.loc[post_mask, 'att'].values)

    # Bootstrap SE
    boot_simple = np.nansum(weights[:, np.newaxis] * boot_dist[post_mask, :], axis=0)
    se_simple = np.nanstd(boot_simple)

    alpha = config.inference.alpha
    ci_lower = np.nanpercentile(boot_simple, 100 * alpha / 2)
    ci_upper = np.nanpercentile(boot_simple, 100 * (1 - alpha / 2))

    log_message(f"Simple ATT: {att_simple:.4f} (SE: {se_simple:.4f})")
    log_message(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        'att': att_simple,
        'se': se_simple,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': att_simple / se_simple if se_simple > 0 else np.nan,
        'p_value': 2 * stats.norm.sf(np.abs(att_simple / se_simple)) if se_simple > 0 else np.nan,
        'n_post': post_mask.sum()
    }
