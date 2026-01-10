"""
m01_utils.py - Utility Functions

Common utility functions used across the estimation pipeline.
"""

import time
import numpy as np
from typing import Optional
from datetime import datetime


class Timer:
    """Simple timer for tracking execution time."""

    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def reset(self):
        self.start_time = time.time()


def log_message(msg: str, level: str = "INFO"):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}")


def clamp(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clamp array values to [min_val, max_val]."""
    return np.clip(x, min_val, max_val)


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Safe division, returning fill where b is zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill
    return result
