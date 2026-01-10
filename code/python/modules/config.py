"""
Configuration Module for DiD Estimation

Provides default configuration and helper functions for the estimation pipeline.
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
    residual_connections: bool = False


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    restore_best_weights: bool = True


@dataclass
class GradientClippingConfig:
    """Gradient clipping configuration."""
    enabled: bool = True
    max_norm: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 256
    validation_split: float = 0.2
    shuffle: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    gradient_clipping: GradientClippingConfig = field(default_factory=GradientClippingConfig)


@dataclass
class AdamWConfig:
    """AdamW optimizer configuration."""
    lr: float = 0.001
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "none"  # "none", "step", "cosine", "reduce_on_plateau"
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100
    eta_min: float = 1e-6
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 1e-6


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
    multiplier_dist: str = "normal"
    uniform_bands: bool = True
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
    print_every: int = 5
    plot_loss: bool = False
    save_checkpoints: bool = False
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    log_file: Optional[str] = None
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""
    # Variable names
    outcome: str = "sfr_pc"
    outcome_var: str = field(init=False)
    id_var: str = "id"
    time_var: str = "time"
    group_var: str = "group"
    cluster_var: str = "cluster_county"

    # Prefixes
    time_invariant_prefix: str = "X_"
    time_varying_prefix: str = "V_"
    baseline_outcome_prefix: str = "Wy_"

    # Analysis period
    analysis_start: int = 1996
    analysis_end: int = 2014

    # Random seed
    seed: int = 42

    # Device
    device: str = "auto"

    # Garbage collection frequency
    gc_every: int = 50

    # Sub-configurations
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: str = "adamw"
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {"adamw": AdamWConfig()})
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cross_fitting: CrossFittingConfig = field(default_factory=CrossFittingConfig)
    propensity: PropensityConfig = field(default_factory=PropensityConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    event_study: EventStudyConfig = field(default_factory=EventStudyConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def __post_init__(self):
        self.outcome_var = f"y_{self.outcome}"


def create_config(**kwargs) -> Config:
    """Create configuration with optional overrides."""
    config = Config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            if isinstance(value, dict) and hasattr(config, key):
                # Handle nested config updates
                existing = getattr(config, key)
                if hasattr(existing, '__dataclass_fields__'):
                    for k, v in value.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    setattr(config, key, value)
            else:
                setattr(config, key, value)

    # Update outcome_var if outcome changed
    config.outcome_var = f"y_{config.outcome}"

    return config


def get_device(config: Config) -> torch.device:
    """Get torch device based on configuration."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
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
    print("\nCallaway & Sant'Anna DiD Configuration")
    print("=" * 40)
    print(f"\nOutcome: {config.outcome_var}")
    print(f"Analysis period: {config.analysis_start}-{config.analysis_end}")
    print(f"Cluster variable: {config.cluster_var}")

    print(f"\nNeural Network:")
    print(f"  Input projection dim: {config.architecture.input_projection_dim}")
    print(f"  Shared layers: {' -> '.join(map(str, config.architecture.shared_layers))}")
    print(f"  Activation: {config.architecture.activation}")
    print(f"  Dropout: {config.architecture.dropout:.2f}")
    print(f"  Layer norm: {config.architecture.layer_norm}")

    print(f"\nTraining:")
    print(f"  Optimizer: {config.optimizer} (lr={config.optimizer_params['adamw'].lr if isinstance(config.optimizer_params.get('adamw'), AdamWConfig) else config.optimizer_params.get('adamw', {}).get('lr', 0.001)})")
    print(f"  Epochs: {config.training.epochs}, Batch size: {config.training.batch_size}")
    print(f"  Early stopping: {config.training.early_stopping.enabled} (patience={config.training.early_stopping.patience})")

    print(f"\nCross-fitting:")
    print(f"  Folds: {config.cross_fitting.n_folds} (by {config.cross_fitting.stratify_by})")

    print(f"\nInference:")
    print(f"  Bootstrap samples: {config.inference.n_bootstrap}")
    print(f"  Alpha: {config.inference.alpha:.3f}")
    print(f"  Propensity clamp: [{config.propensity.min_ps}, {config.propensity.max_ps}]")

    print(f"\nEvent Study:")
    print(f"  Window: [{-config.event_study.pre_periods}, +{config.event_study.post_periods}]")
    print(f"  Reference period: {config.event_study.reference_period}")
    print()
