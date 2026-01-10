"""
Inference Module for DiD Estimation

Implements clustered multiplier bootstrap for inference following CS2021.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .utils import log_message
from .config import Config


def generate_multiplier_weights(
    n_clusters: int,
    n_bootstrap: int,
    distribution: str = "normal",
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate multiplier weights for bootstrap."""
    if seed is not None:
        np.random.seed(seed)

    if distribution == "normal":
        xi = np.random.randn(n_clusters, n_bootstrap)
    elif distribution == "rademacher":
        xi = np.random.choice([-1, 1], size=(n_clusters, n_bootstrap))
    else:
        raise ValueError(f"Unknown multiplier distribution: {distribution}")

    return xi


def clustered_bootstrap(att_results: Dict, config: Config) -> Dict:
    """
    Clustered multiplier bootstrap for ATT(g,t) estimates.

    Computes bootstrap distribution using cluster-level multiplier weights.
    """
    log_message(f"Running clustered multiplier bootstrap ({config.inference.n_bootstrap} replications)...")

    data = att_results['data']
    att_df = att_results['att']
    influence_functions = att_results['influence_functions']

    n_gt = len(att_df)
    n_bootstrap = config.inference.n_bootstrap
    cluster_var = config.cluster_var

    # Get cluster information
    clusters = data[cluster_var].unique()
    n_clusters = len(clusters)
    cluster_map = pd.Series(range(len(clusters)), index=clusters)
    cluster_indices = data[cluster_var].map(cluster_map).values

    log_message(f"  Clusters: {n_clusters}")

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

        if np.all(np.isnan(inf_func)):
            continue

        # Aggregate by cluster
        cluster_inf = np.zeros(n_clusters)
        for c in range(n_clusters):
            mask = cluster_indices == c
            cluster_inf[c] = np.nansum(inf_func[mask])

        # Bootstrap
        att_i = att_df.iloc[i]['att']
        for b in range(n_bootstrap):
            boot_dist[i, b] = att_i + np.sum(xi[:, b] * cluster_inf) / len(data)

    # Compute SEs
    boot_se = np.nanstd(boot_dist, axis=1)

    # Pointwise CIs
    alpha = config.inference.alpha
    ci_lower = np.nanpercentile(boot_dist, 100 * alpha / 2, axis=1)
    ci_upper = np.nanpercentile(boot_dist, 100 * (1 - alpha / 2), axis=1)

    # Uniform bands
    uniform_bands = None
    if config.inference.uniform_bands:
        uniform_bands = compute_uniform_bands(att_df['att'].values, boot_dist, boot_se, alpha)

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
    n_gt = len(att)
    n_boot = boot_dist.shape[1]

    # T-statistics for each bootstrap sample
    t_stats = np.abs(boot_dist - att[:, np.newaxis]) / se[:, np.newaxis]
    t_stats[~np.isfinite(t_stats)] = np.nan

    # Sup-t for each bootstrap sample
    sup_t = np.nanmax(t_stats, axis=0)

    # Critical value
    c_alpha = np.nanpercentile(sup_t, 100 * (1 - alpha))

    log_message(f"  Sup-t critical value (alpha={alpha:.2f}): {c_alpha:.3f}")

    return {
        'uniform_lower': att - c_alpha * se,
        'uniform_upper': att + c_alpha * se,
        'sup_t_critical': c_alpha
    }


def add_bootstrap_inference(att_results: Dict, config: Config) -> Dict:
    """Add bootstrap inference to ATT results."""
    boot_results = clustered_bootstrap(att_results, config)

    # Update ATT dataframe
    att_df = att_results['att'].copy()
    att_df['se_boot'] = boot_results['se']
    att_df['ci_lower_boot'] = boot_results['ci_lower']
    att_df['ci_upper_boot'] = boot_results['ci_upper']

    if boot_results['uniform_bands'] is not None:
        att_df['uniform_lower'] = boot_results['uniform_bands']['uniform_lower']
        att_df['uniform_upper'] = boot_results['uniform_bands']['uniform_upper']

    att_df['t_stat_boot'] = att_df['att'] / att_df['se_boot']
    att_df['p_value_boot'] = 2 * stats.norm.sf(np.abs(att_df['t_stat_boot']))

    att_results['att'] = att_df
    att_results['bootstrap'] = boot_results

    return att_results


def test_parallel_trends(att_results: Dict, config: Config) -> Dict:
    """Test whether pre-treatment ATT estimates are jointly zero."""
    log_message("Testing parallel trends (pre-treatment effects)...")

    att_df = att_results['att']
    boot_dist = att_results['bootstrap']['bootstrap_dist']

    # Pre-treatment indices
    pre_idx = att_df['is_pre'].values
    n_pre = pre_idx.sum()

    if n_pre == 0:
        log_message("Warning: No pre-treatment periods found", level="WARNING")
        return {'test_stat': np.nan, 'p_value': np.nan, 'n_pre': 0, 'reject': np.nan}

    att_pre = att_df.loc[pre_idx, 'att'].values
    boot_pre = boot_dist[pre_idx, :]

    # Mean pre-treatment ATT
    mean_att_pre = np.nanmean(att_pre)
    se_mean_pre = np.nanstd(np.nanmean(boot_pre, axis=0))

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
        'reject': p_value < config.inference.alpha
    }


def compute_simple_att(att_results: Dict, config: Config) -> Dict:
    """Compute simple ATT (weighted average post-treatment effect)."""
    att_df = att_results['att']
    boot_dist = att_results['bootstrap']['bootstrap_dist']

    # Post-treatment indices
    post_idx = ~att_df['is_pre'].values

    if not post_idx.any():
        return {'att': np.nan, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

    # Weighted average
    weights = att_df.loc[post_idx, 'n_treated'].values
    weights = weights / np.nansum(weights)

    att_simple = np.nansum(weights * att_df.loc[post_idx, 'att'].values)

    # Bootstrap SE
    boot_simple = np.nansum(weights[:, np.newaxis] * boot_dist[post_idx, :], axis=0)
    se_simple = np.nanstd(boot_simple)

    alpha = config.inference.alpha
    ci_lower, ci_upper = np.nanpercentile(boot_simple, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    log_message(f"Simple ATT (weighted avg post-treatment): {att_simple:.4f} (SE: {se_simple:.4f})")
    log_message(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        'att': att_simple,
        'se': se_simple,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': att_simple / se_simple,
        'p_value': 2 * stats.norm.sf(np.abs(att_simple / se_simple)),
        'n_post': post_idx.sum()
    }
