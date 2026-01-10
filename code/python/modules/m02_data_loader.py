"""
m02_data_loader.py - Data Loading and Preparation

Loads panel data and creates the wide unit-level structure needed for efficient estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .m00_config import Config
from .m01_utils import log_message


@dataclass
class UnitData:
    """Wide unit-level data structure for efficient estimation."""
    # Unit identifiers and groups: (n_units,)
    unit_ids: np.ndarray
    groups: np.ndarray          # Treatment cohort (0 = never-treated)
    clusters: np.ndarray        # Cluster for bootstrap

    # Outcomes at all times: (n_units, n_times)
    Y: np.ndarray
    time_index: Dict[int, int]  # time -> column index in Y

    # Covariates: (n_units, n_max_covariates)
    X_full: np.ndarray
    covariate_names: List[str]

    # Metadata
    n_units: int
    n_times: int
    times: List[int]
    treatment_groups: List[int]  # Unique treatment cohorts (excluding never-treated)


@dataclass
class GTInfo:
    """Information about all (g,t) pairs."""
    # DataFrame with columns: g, t, gt_index, is_pre, event_time, base_period
    df: pd.DataFrame

    # Mappings
    n_gt: int
    gt_to_idx: Dict[Tuple[int, int], int]

    # For each (g,t): which base period determines covariates
    # base_period[gt_idx] = the period up to which covariates are used
    base_periods: np.ndarray

    # Unique base periods and their indices
    unique_base_periods: List[int]
    base_period_to_idx: Dict[int, int]
    gt_to_base_idx: np.ndarray  # (n_gt,) mapping gt_idx -> base_period_idx

    # For outcome differencing: (n_gt,) arrays
    t_indices: np.ndarray       # Time index for Y(t)
    base_t_indices: np.ndarray  # Time index for Y(base) - either t-1 or g-1


def load_panel_data(
    path: str,
    config: Config,
    sample_n: Optional[int] = None
) -> pd.DataFrame:
    """Load panel data from CSV."""
    log_message(f"Loading data from {path}")

    # Load with data types
    df = pd.read_csv(path, low_memory=False)

    if sample_n is not None:
        # Sample unique units
        unique_ids = df[config.id_var].unique()
        if sample_n < len(unique_ids):
            np.random.seed(config.seed)
            sampled_ids = np.random.choice(unique_ids, size=sample_n, replace=False)
            df = df[df[config.id_var].isin(sampled_ids)]
            log_message(f"Sampled {sample_n} units")

    # Filter to analysis period
    df = df[(df[config.time_var] >= config.analysis_start) &
            (df[config.time_var] <= config.analysis_end)]

    log_message(f"Loaded {len(df):,} observations, {df[config.id_var].nunique():,} units")

    return df


def create_unit_data(df: pd.DataFrame, config: Config) -> UnitData:
    """Create wide unit-level data structure from panel."""
    log_message("Creating unit-level data structure...")

    id_var = config.id_var
    time_var = config.time_var
    group_var = config.group_var
    cluster_var = config.cluster_var
    outcome_var = config.outcome_var

    # Get unique units and times
    unit_ids = df[id_var].unique()
    times = sorted(df[time_var].unique())
    n_units = len(unit_ids)
    n_times = len(times)

    # Create mappings
    unit_to_idx = {uid: i for i, uid in enumerate(unit_ids)}
    time_index = {t: i for i, t in enumerate(times)}

    # Extract unit-level info (from first observation per unit)
    unit_info = df.groupby(id_var).first().reset_index()
    unit_info = unit_info.set_index(id_var).loc[unit_ids].reset_index()

    groups = unit_info[group_var].values.astype(np.int64)
    clusters = unit_info[cluster_var].values

    # Treatment groups (excluding never-treated = 0)
    treatment_groups = sorted([g for g in np.unique(groups) if g != config.never_treated_code])

    # Create outcome matrix: (n_units, n_times)
    Y = np.full((n_units, n_times), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        uid = row[id_var]
        t = row[time_var]
        if uid in unit_to_idx and t in time_index:
            Y[unit_to_idx[uid], time_index[t]] = row[outcome_var]

    # Identify covariate columns
    time_invariant_cols = [c for c in df.columns if c.startswith(config.time_invariant_prefix)]
    time_varying_cols = [c for c in df.columns if c.startswith(config.time_varying_prefix)]
    all_covariate_cols = time_invariant_cols + time_varying_cols

    log_message(f"  Covariates: {len(time_invariant_cols)} time-invariant, {len(time_varying_cols)} time-varying")

    # Create covariate matrix: (n_units, n_covariates)
    # Use the first observation per unit (covariates should be constant within unit)
    X_full = unit_info[all_covariate_cols].values.astype(np.float32)

    # Handle missing values
    X_full = np.nan_to_num(X_full, nan=0.0)

    log_message(f"  Units: {n_units}, Times: {n_times}, Covariates: {X_full.shape[1]}")

    return UnitData(
        unit_ids=unit_ids,
        groups=groups,
        clusters=clusters,
        Y=Y,
        time_index=time_index,
        X_full=X_full,
        covariate_names=all_covariate_cols,
        n_units=n_units,
        n_times=n_times,
        times=times,
        treatment_groups=treatment_groups
    )


def create_gt_info(unit_data: UnitData, config: Config) -> GTInfo:
    """Create (g,t) pair information."""
    log_message("Creating (g,t) pair information...")

    times = unit_data.times
    treatment_groups = unit_data.treatment_groups
    time_index = unit_data.time_index

    gt_list = []
    gt_idx = 0

    for g in treatment_groups:
        for t in times:
            # Only include if t is in analysis period
            if t < config.analysis_start or t > config.analysis_end:
                continue

            # Determine if pre or post treatment
            is_pre = t < g
            event_time = t - g

            # Base period for covariates: t-1 for pre, g-1 for post
            base_period = (t - 1) if is_pre else (g - 1)

            # Base period for outcome differencing
            outcome_base = (t - 1) if is_pre else (g - 1)

            gt_list.append({
                'g': g,
                't': t,
                'gt_index': gt_idx,
                'is_pre': is_pre,
                'event_time': event_time,
                'base_period': base_period,
                'outcome_base': outcome_base
            })
            gt_idx += 1

    gt_df = pd.DataFrame(gt_list)
    n_gt = len(gt_df)

    # Create mappings
    gt_to_idx = {(row['g'], row['t']): row['gt_index'] for _, row in gt_df.iterrows()}

    # Base periods array
    base_periods = gt_df['base_period'].values

    # Unique base periods
    unique_base_periods = sorted(gt_df['base_period'].unique())
    base_period_to_idx = {bp: i for i, bp in enumerate(unique_base_periods)}

    # Map each (g,t) to its base period index
    gt_to_base_idx = np.array([base_period_to_idx[bp] for bp in base_periods], dtype=np.int64)

    # Time indices for outcome differencing
    t_indices = np.array([time_index.get(row['t'], -1) for _, row in gt_df.iterrows()], dtype=np.int64)
    base_t_indices = np.array([time_index.get(row['outcome_base'], -1) for _, row in gt_df.iterrows()], dtype=np.int64)

    log_message(f"  (g,t) pairs: {n_gt} ({gt_df['is_pre'].sum()} pre, {(~gt_df['is_pre']).sum()} post)")
    log_message(f"  Unique base periods: {len(unique_base_periods)}")

    return GTInfo(
        df=gt_df,
        n_gt=n_gt,
        gt_to_idx=gt_to_idx,
        base_periods=base_periods,
        unique_base_periods=unique_base_periods,
        base_period_to_idx=base_period_to_idx,
        gt_to_base_idx=gt_to_base_idx,
        t_indices=t_indices,
        base_t_indices=base_t_indices
    )


def create_sample_masks(unit_data: UnitData, gt_info: GTInfo, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample and treatment masks for each (g,t) pair.

    For each (g,t):
    - Sample includes: units in group g + units not-yet-treated at t
    - Treatment D=1 if group=g, D=0 if control

    Returns:
        sample_mask: (n_units, n_gt) bool array - is unit in this (g,t) sample?
        treatment: (n_units, n_gt) float array - 1 if treated, 0 if control, NaN if not in sample
    """
    log_message("Creating sample and treatment masks...")

    n_units = unit_data.n_units
    n_gt = gt_info.n_gt
    groups = unit_data.groups
    never_treated = config.never_treated_code

    sample_mask = np.zeros((n_units, n_gt), dtype=bool)
    treatment = np.full((n_units, n_gt), np.nan, dtype=np.float32)

    for _, row in gt_info.df.iterrows():
        g = row['g']
        t = row['t']
        gt_idx = row['gt_index']

        # Treated: units in group g
        treated_mask = (groups == g)

        # Control: units not-yet-treated at t (group > t) or never-treated (group = 0)
        # IMPORTANT: Exclude the treated group from controls to avoid overlap
        control_mask = ((groups > t) | (groups == never_treated)) & ~treated_mask

        # Sample is treated + control
        in_sample = treated_mask | control_mask

        sample_mask[:, gt_idx] = in_sample
        treatment[in_sample & treated_mask, gt_idx] = 1.0
        treatment[in_sample & control_mask, gt_idx] = 0.0

    n_in_sample = sample_mask.sum()
    log_message(f"  Sample coverage: {n_in_sample:,} (unit, g, t) combinations")
    log_message(f"  Average units per (g,t): {n_in_sample / n_gt:.0f}")

    return sample_mask, treatment


def compute_outcome_diffs(
    Y: np.ndarray,
    t_indices: np.ndarray,
    base_t_indices: np.ndarray
) -> np.ndarray:
    """
    Compute outcome differences for all (g,t) pairs.

    Args:
        Y: (n_units, n_times) outcome matrix
        t_indices: (n_gt,) time index for Y(t)
        base_t_indices: (n_gt,) time index for Y(base)

    Returns:
        delta_Y: (n_units, n_gt) outcome differences
    """
    n_units = Y.shape[0]
    n_gt = len(t_indices)

    # Handle invalid indices
    valid_t = (t_indices >= 0) & (t_indices < Y.shape[1])
    valid_base = (base_t_indices >= 0) & (base_t_indices < Y.shape[1])
    valid = valid_t & valid_base

    delta_Y = np.full((n_units, n_gt), np.nan, dtype=np.float32)

    for gt_idx in range(n_gt):
        if valid[gt_idx]:
            delta_Y[:, gt_idx] = Y[:, t_indices[gt_idx]] - Y[:, base_t_indices[gt_idx]]

    return delta_Y
