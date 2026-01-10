"""
m00_config.py - Configuration Module

Provides configuration dataclasses and creation functions for the DiD estimation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
import random
import numpy as np


@dataclass
class ArchitectureConfig:
    """Neural network architecture configuration."""
    input_projection_dim: int = 128
    shared_layers: List[int] = field(default_factory=lambda: [256, 128])
    outcome_head_layers: List[int] = field(default_factory=lambda: [64])
    propensity_head_layers: List[int] = field(default_factory=lambda: [64])
    activation: str = "relu"
    dropout: float = 0.1
    layer_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 512
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine"  # "none", "cosine", "step", "plateau"
    T_max: int = 100
    eta_min: float = 1e-6


@dataclass
class CrossFittingConfig:
    """Cross-fitting configuration."""
    n_folds: int = 2
    stratify_by: str = "cluster_county"
    seed: int = 42


@dataclass
class PropensityConfig:
    """Propensity score configuration."""
    min_ps: float = 0.001
    max_ps: float = 0.999


@dataclass
class InferenceConfig:
    """Inference configuration."""
    n_bootstrap: int = 1000
    alpha: float = 0.05
    uniform_bands: bool = True
    multiplier_dist: str = "normal"
    seed: int = 123


@dataclass
class EventStudyConfig:
    """Event study configuration."""
    pre_periods: int = 10
    post_periods: int = 10
    reference_period: int = -1
    weight_by_group_size: bool = True


@dataclass
class LossConfig:
    """Loss function configuration."""
    outcome_weight: float = 1.0
    propensity_weight: float = 1.0


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    verbose: bool = True
    print_every: int = 10


@dataclass
class Config:
    """Main configuration."""
    # Variable names
    outcome: str = "sfr_pc"
    id_var: str = "id"
    time_var: str = "time"
    group_var: str = "group"
    cluster_var: str = "cluster_county"

    # Prefixes
    time_invariant_prefix: str = "X_"
    time_varying_prefix: str = "V_"

    # Analysis period
    analysis_start: int = 1996
    analysis_end: int = 2014

    # Never-treated group code
    never_treated_code: int = 0

    # Sub-configs
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cross_fitting: CrossFittingConfig = field(default_factory=CrossFittingConfig)
    propensity: PropensityConfig = field(default_factory=PropensityConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    event_study: EventStudyConfig = field(default_factory=EventStudyConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Computational
    device: str = "auto"
    seed: int = 42

    @property
    def outcome_var(self) -> str:
        return f"y_{self.outcome}"


def create_config(**kwargs) -> Config:
    """Create configuration with optional overrides."""
    config = Config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if isinstance(value, dict) and hasattr(attr, '__dataclass_fields__'):
                # Update nested dataclass
                for k, v in value.items():
                    if hasattr(attr, k):
                        setattr(attr, k, v)
            else:
                setattr(config, key, value)

    return config


def get_device(config: Config) -> torch.device:
    """Get torch device."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config(config: Config):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"\nOutcome: y_{config.outcome}")
    print(f"Analysis period: {config.analysis_start}-{config.analysis_end}")
    print(f"Cluster variable: {config.cluster_var}")

    print(f"\nArchitecture:")
    print(f"  Projection dim: {config.architecture.input_projection_dim}")
    print(f"  Shared layers: {config.architecture.shared_layers}")
    print(f"  Dropout: {config.architecture.dropout}")

    print(f"\nTraining:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Early stopping patience: {config.training.early_stopping_patience}")

    print(f"\nCross-fitting: {config.cross_fitting.n_folds} folds")
    print(f"Bootstrap: {config.inference.n_bootstrap} replications")
    print(f"Event study window: [-{config.event_study.pre_periods}, +{config.event_study.post_periods}]")
    print()
