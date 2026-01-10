"""
m03_covariate_selector.py - Covariate Selection

Defines covariate masks for each base period.
Pre-treatment (t < g): use covariates through t-1
Post-treatment (t >= g): use covariates through g-1

Since post-treatment periods within a group all use the same covariates,
we only need one mask per unique base period.
"""

import numpy as np
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .m00_config import Config
from .m01_utils import log_message
from .m02_data_loader import UnitData, GTInfo


@dataclass
class CovariateInfo:
    """Covariate mask information by base period."""
    # All covariate names
    all_covariates: List[str]
    n_covariates: int

    # Masks by base period: {base_period: bool array of shape (n_covariates,)}
    masks_by_base: Dict[int, np.ndarray]

    # Dimensions by base period
    dims_by_base: Dict[int, int]

    # Time-invariant vs time-varying
    time_invariant_cols: List[str]
    time_varying_cols: List[str]
    time_varying_base: List[str]  # Base names without year suffix


def extract_year_from_covariate(col_name: str, prefix: str) -> int:
    """Extract year from time-varying covariate name like V_totpop_1995."""
    # Pattern: prefix + base + _year
    pattern = rf"^{re.escape(prefix)}(.+)_(\d{{4}})$"
    match = re.match(pattern, col_name)
    if match:
        return int(match.group(2))
    return -1


def create_covariate_info(
    unit_data: UnitData,
    gt_info: GTInfo,
    config: Config
) -> CovariateInfo:
    """Create covariate masks for each unique base period."""
    log_message("Creating covariate masks by base period...")

    covariate_names = unit_data.covariate_names
    n_covariates = len(covariate_names)

    # Separate time-invariant and time-varying
    time_invariant_cols = [c for c in covariate_names if c.startswith(config.time_invariant_prefix)]
    time_varying_cols = [c for c in covariate_names if c.startswith(config.time_varying_prefix)]

    # Extract base names and years for time-varying covariates
    tv_info = []
    time_varying_base = set()
    for col in time_varying_cols:
        year = extract_year_from_covariate(col, config.time_varying_prefix)
        if year > 0:
            # Extract base name
            base_name = col.rsplit('_', 1)[0]
            time_varying_base.add(base_name)
            tv_info.append((col, year))

    time_varying_base = sorted(list(time_varying_base))

    # Create masks for each unique base period
    unique_base_periods = gt_info.unique_base_periods
    masks_by_base = {}
    dims_by_base = {}

    for base_period in unique_base_periods:
        mask = np.zeros(n_covariates, dtype=bool)

        for i, col in enumerate(covariate_names):
            if col.startswith(config.time_invariant_prefix):
                # Always include time-invariant
                mask[i] = True
            elif col.startswith(config.time_varying_prefix):
                # Include if year <= base_period
                year = extract_year_from_covariate(col, config.time_varying_prefix)
                if year > 0 and year <= base_period:
                    mask[i] = True

        masks_by_base[base_period] = mask
        dims_by_base[base_period] = mask.sum()

    log_message(f"  Base periods: {len(unique_base_periods)}")
    log_message(f"  Covariate dims range: {min(dims_by_base.values())} - {max(dims_by_base.values())}")

    return CovariateInfo(
        all_covariates=covariate_names,
        n_covariates=n_covariates,
        masks_by_base=masks_by_base,
        dims_by_base=dims_by_base,
        time_invariant_cols=time_invariant_cols,
        time_varying_cols=time_varying_cols,
        time_varying_base=time_varying_base
    )
