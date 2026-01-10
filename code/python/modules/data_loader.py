"""
Data Loading Module for DiD Estimation

Efficient data loading and preprocessing using pandas.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .utils import log_message, file_size_human, validate_columns, is_valid_gt_pair, get_treatment_groups, get_analysis_periods
from .config import Config


@dataclass
class PanelMetadata:
    """Metadata about the panel dataset."""
    n_obs: int
    n_units: int
    n_periods: int
    n_clusters: int
    time_periods: np.ndarray
    all_groups: np.ndarray
    treatment_groups: np.ndarray
    n_never_treated: int
    n_treatment_groups: int
    group_sizes: pd.DataFrame
    x_cols: List[str]
    v_cols: List[str]
    wy_cols: List[str]
    y_cols: List[str]
    n_x_covariates: int
    n_v_covariates: int
    n_wy_covariates: int


def load_panel_data(
    file_path: str,
    config: Config,
    sample_n: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, PanelMetadata]:
    """
    Load panel data from CSV file.

    Args:
        file_path: Path to CSV file
        config: Configuration object
        sample_n: Number of rows to sample (None = full data)
        columns: Specific columns to load (None = all)

    Returns:
        Tuple of (data DataFrame, metadata)
    """
    log_message(f"Loading data from: {file_path}", config=config)

    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    log_message(f"File size: {file_size_human(file_path)}", config=config)

    # Load data
    if sample_n is not None:
        log_message(f"Loading sample of {sample_n} rows", config=config)
        data = pd.read_csv(file_path, nrows=sample_n, usecols=columns)
    else:
        data = pd.read_csv(file_path, usecols=columns)

    log_message(f"Loaded {len(data)} rows x {len(data.columns)} columns", config=config)

    # Validate required columns
    required_cols = [
        config.id_var,
        config.time_var,
        config.group_var,
        config.cluster_var,
        config.outcome_var
    ]
    validate_columns(data, required_cols)

    # Ensure proper types
    data[config.id_var] = data[config.id_var].astype(int)
    data[config.time_var] = data[config.time_var].astype(int)
    data[config.group_var] = data[config.group_var].astype(int)

    # Filter to analysis period
    original_n = len(data)
    data = data[
        (data[config.time_var] >= config.analysis_start) &
        (data[config.time_var] <= config.analysis_end)
    ].copy()
    log_message(f"Filtered to analysis period: {original_n} -> {len(data)} rows", config=config)

    # Compute metadata
    metadata = compute_panel_metadata(data, config)

    # Sort by id and time
    data = data.sort_values([config.id_var, config.time_var]).reset_index(drop=True)

    return data, metadata


def compute_panel_metadata(data: pd.DataFrame, config: Config) -> PanelMetadata:
    """Compute panel metadata."""
    # Basic dimensions
    n_obs = len(data)
    n_units = data[config.id_var].nunique()
    n_periods = data[config.time_var].nunique()
    n_clusters = data[config.cluster_var].nunique()

    # Time periods
    time_periods = np.sort(data[config.time_var].unique())

    # Treatment groups
    all_groups = np.sort(data[config.group_var].unique())
    treatment_groups = all_groups[all_groups > 0]
    n_never_treated = data[data[config.group_var] == 0][config.id_var].nunique()

    # Group sizes
    group_sizes = data.groupby(config.group_var)[config.id_var].nunique().reset_index()
    group_sizes.columns = ['group', 'n_units']

    # Identify covariate columns
    all_cols = data.columns.tolist()
    x_cols = [c for c in all_cols if c.startswith(config.time_invariant_prefix)]
    v_cols = [c for c in all_cols if c.startswith(config.time_varying_prefix)]
    wy_cols = [c for c in all_cols if c.startswith(config.baseline_outcome_prefix)]
    y_cols = [c for c in all_cols if c.startswith('y_')]

    return PanelMetadata(
        n_obs=n_obs,
        n_units=n_units,
        n_periods=n_periods,
        n_clusters=n_clusters,
        time_periods=time_periods,
        all_groups=all_groups,
        treatment_groups=treatment_groups,
        n_never_treated=n_never_treated,
        n_treatment_groups=len(treatment_groups),
        group_sizes=group_sizes,
        x_cols=x_cols,
        v_cols=v_cols,
        wy_cols=wy_cols,
        y_cols=y_cols,
        n_x_covariates=len(x_cols),
        n_v_covariates=len(v_cols),
        n_wy_covariates=len(wy_cols)
    )


def get_gt_pairs(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Get all valid (g,t) pairs for estimation.

    Returns DataFrame with columns: g, t, is_pre, event_time, base_period, n_treated, n_control, gt_index
    """
    # Get treatment groups and periods
    groups = get_treatment_groups(data, config)
    periods = get_analysis_periods(config)

    # Generate all combinations
    gt_pairs = []
    for g in groups:
        for t in periods:
            if is_valid_gt_pair(g, t, config):
                gt_pairs.append({'g': g, 't': t})

    gt_df = pd.DataFrame(gt_pairs)

    # Add metadata
    gt_df['is_pre'] = gt_df['t'] < gt_df['g']
    gt_df['event_time'] = gt_df['t'] - gt_df['g']
    gt_df['base_period'] = np.where(gt_df['t'] >= gt_df['g'], gt_df['g'] - 1, gt_df['t'] - 1)

    # Count treated and control units
    n_treated_list = []
    n_control_list = []

    for _, row in gt_df.iterrows():
        g, t = row['g'], row['t']

        # Treated: units in group g
        n_treated = data[data[config.group_var] == g][config.id_var].nunique()

        # Control: not-yet-treated at time t
        n_control = data[
            (data[config.group_var] == 0) | (data[config.group_var] > t)
        ][config.id_var].nunique()

        n_treated_list.append(n_treated)
        n_control_list.append(n_control)

    gt_df['n_treated'] = n_treated_list
    gt_df['n_control'] = n_control_list
    gt_df['gt_index'] = range(len(gt_df))

    log_message(f"Generated {len(gt_df)} valid (g,t) pairs")
    log_message(f"  Pre-treatment pairs: {gt_df['is_pre'].sum()}")
    log_message(f"  Post-treatment pairs: {(~gt_df['is_pre']).sum()}")

    return gt_df


def create_gt_sample(
    data: pd.DataFrame,
    g: int,
    t: int,
    config: Config
) -> pd.DataFrame:
    """
    Create estimation sample for specific (g,t) pair.

    Includes treated units (group = g) and not-yet-treated controls.
    Computes outcome difference delta_y.
    """
    # Determine base period
    base_period = g - 1 if t >= g else t - 1

    # Get IDs of treated units
    treated_ids = data[data[config.group_var] == g][config.id_var].unique()

    # Get IDs of not-yet-treated units
    control_ids = data[
        (data[config.group_var] == 0) | (data[config.group_var] > t)
    ][config.id_var].unique()

    all_ids = np.concatenate([treated_ids, control_ids])

    # Get data for current period t
    sample_t = data[
        (data[config.id_var].isin(all_ids)) &
        (data[config.time_var] == t)
    ].copy()

    # Get data for base period
    sample_base = data[
        (data[config.id_var].isin(all_ids)) &
        (data[config.time_var] == base_period)
    ][[config.id_var, config.outcome_var]].copy()
    sample_base.columns = [config.id_var, 'y_base']

    # Merge to compute outcome difference
    sample = sample_t.merge(sample_base, on=config.id_var, how='left')

    # Compute outcome difference
    sample['delta_y'] = sample[config.outcome_var] - sample['y_base']

    # Add treatment indicator
    sample['D'] = (sample[config.group_var] == g).astype(int)

    # Drop missing
    sample = sample.dropna(subset=['delta_y'])

    return sample


def summarize_data(data: pd.DataFrame, metadata: PanelMetadata, config: Config):
    """Print data summary."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60 + "\n")

    print("Panel Dimensions:")
    print(f"  Observations: {metadata.n_obs:,}")
    print(f"  Units (ZIPs): {metadata.n_units:,}")
    print(f"  Time periods: {metadata.n_periods} ({metadata.time_periods.min()} - {metadata.time_periods.max()})")
    print(f"  Clusters: {metadata.n_clusters:,}")

    print("\nTreatment Groups:")
    print(f"  Never-treated units: {metadata.n_never_treated:,}")
    print(f"  Treatment cohorts: {metadata.n_treatment_groups} ({metadata.treatment_groups.min()} - {metadata.treatment_groups.max()})")

    print("\nCovariates:")
    print(f"  Time-invariant (X_): {metadata.n_x_covariates}")
    print(f"  Time-varying (V_): {metadata.n_v_covariates}")
    print(f"  Baseline outcomes (Wy_): {metadata.n_wy_covariates}")

    print("\nOutcome Variable:")
    outcome_stats = data[config.outcome_var].describe()
    n_missing = data[config.outcome_var].isna().sum()
    print(f"  {config.outcome_var}: mean={outcome_stats['mean']:.4f}, sd={outcome_stats['std']:.4f}, "
          f"range=[{outcome_stats['min']:.4f}, {outcome_stats['max']:.4f}], missing={n_missing}")
    print()
