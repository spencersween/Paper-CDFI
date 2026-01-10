"""
Utility Functions for DiD Estimation
"""

import os
import time
from datetime import datetime
from typing import Optional, List
import numpy as np


def log_message(message: str, level: str = "INFO", config=None):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def file_size_human(path: str) -> str:
    """Return human-readable file size."""
    if not os.path.exists(path):
        return "File not found"

    size_bytes = os.path.getsize(path)

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} PB"


def clamp(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clamp values to range [min_val, max_val]."""
    return np.clip(x, min_val, max_val)


def is_valid_gt_pair(g: int, t: int, config) -> bool:
    """Check if (g, t) pair is valid for estimation."""
    # Must be within analysis period
    if t < config.analysis_start or t > config.analysis_end:
        return False

    # Treatment must start within analysis period
    if g < config.analysis_start or g > config.analysis_end:
        return False

    # Need at least one pre-treatment period for base
    if t < config.analysis_start + 1:
        return False

    return True


def get_treatment_groups(data, config) -> np.ndarray:
    """Get unique treatment groups (excluding never-treated)."""
    groups = data[config.group_var].unique()
    groups = groups[groups > 0]
    return np.sort(groups)


def get_analysis_periods(config) -> np.ndarray:
    """Get analysis time periods."""
    return np.arange(config.analysis_start, config.analysis_end + 1)


def validate_columns(data, required_cols: List[str]):
    """Validate that required columns exist in data."""
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


class Timer:
    """Simple timer for tracking execution time."""

    def __init__(self):
        self.start_time = time.time()
        self.laps = {}

    def lap(self, name: str):
        """Record a lap time."""
        self.laps[name] = time.time() - self.start_time

    def elapsed(self) -> float:
        """Get total elapsed time in seconds."""
        return time.time() - self.start_time

    def report(self):
        """Print timing report."""
        print("\nTiming Report:")
        print("-" * 40)

        prev_time = 0
        for name, lap_time in self.laps.items():
            duration = lap_time - prev_time
            print(f"  {name}: {duration:.1f}s")
            prev_time = lap_time

        print("-" * 40)
        print(f"  Total: {self.elapsed():.1f}s")
        print()
