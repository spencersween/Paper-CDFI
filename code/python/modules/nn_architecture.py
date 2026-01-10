"""
Neural Network Architecture Module for DiD Estimation

Implements multi-task neural network for joint nuisance parameter estimation.
Architecture: Per-(g,t) input projections -> Shared encoder -> Task-specific heads
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .utils import log_message
from .config import Config, get_device


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "selu": nn.SELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def build_head(
    dims: List[int],
    activation: str,
    dropout: float,
    final_activation: Optional[str] = None
) -> nn.Sequential:
    """Build a task-specific head."""
    layers = []

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]

        layers.append(nn.Linear(in_dim, out_dim))

        # Not the last layer
        if i < len(dims) - 2:
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout / 2))

    # Final activation
    if final_activation is not None:
        layers.append(get_activation(final_activation))

    return nn.Sequential(*layers)


class MultiTaskDiDNet(nn.Module):
    """
    Multi-task DiD Neural Network with Per-(g,t) Input Projections.

    Architecture:
    - Per-(g,t) input projection: Maps from (g,t)-specific covariate dim to common dim
    - Shared encoder layers: Processes the common representation
    - Outcome regression heads: One per (g,t) pair for E[DeltaY | X, D=0]
    - Propensity score heads: One per (g,t) pair for P(G=g | X)
    """

    def __init__(self, covariate_dims: Dict[int, int], n_gt_pairs: int, config: Config):
        super().__init__()

        self.covariate_dims = covariate_dims
        self.n_gt_pairs = n_gt_pairs
        self.config = config
        arch = config.architecture

        # Common dimension for all (g,t) pairs after input projection
        self.common_dim = arch.input_projection_dim

        # =========================================================================
        # PER-(g,t) INPUT PROJECTIONS
        # =========================================================================
        self.input_projections = nn.ModuleDict()

        for gt_idx in range(n_gt_pairs):
            input_dim = covariate_dims[gt_idx]
            proj = nn.Sequential(
                nn.Linear(input_dim, self.common_dim),
                get_activation(arch.activation)
            )
            self.input_projections[str(gt_idx)] = proj

        # =========================================================================
        # SHARED ENCODER
        # =========================================================================
        shared_dims = [self.common_dim] + list(arch.shared_layers)
        shared_layers = []

        for i in range(len(shared_dims) - 1):
            in_dim = shared_dims[i]
            out_dim = shared_dims[i + 1]

            shared_layers.append(nn.Linear(in_dim, out_dim))

            if arch.layer_norm:
                shared_layers.append(nn.LayerNorm(out_dim))

            shared_layers.append(get_activation(arch.activation))

            if arch.dropout > 0:
                shared_layers.append(nn.Dropout(arch.dropout))

        self.shared_encoder = nn.Sequential(*shared_layers)
        shared_out_dim = arch.shared_layers[-1]

        # =========================================================================
        # OUTCOME REGRESSION HEADS
        # =========================================================================
        outcome_dims = [shared_out_dim] + list(arch.outcome_head_layers) + [1]
        self.outcome_heads = nn.ModuleDict()

        for gt_idx in range(n_gt_pairs):
            head = build_head(outcome_dims, arch.activation, arch.dropout, final_activation=None)
            self.outcome_heads[str(gt_idx)] = head

        # =========================================================================
        # PROPENSITY SCORE HEADS
        # =========================================================================
        propensity_dims = [shared_out_dim] + list(arch.propensity_head_layers) + [1]
        self.propensity_heads = nn.ModuleDict()

        for gt_idx in range(n_gt_pairs):
            head = build_head(propensity_dims, arch.activation, arch.dropout, final_activation="sigmoid")
            self.propensity_heads[str(gt_idx)] = head

        # Log model info
        n_params = sum(p.numel() for p in self.parameters())
        log_message("MultiTaskDiDNet initialized:")
        log_message(f"  Input dims per (g,t): min={min(covariate_dims.values())}, max={max(covariate_dims.values())}")
        log_message(f"  Common projection dim: {self.common_dim}")
        log_message(f"  Shared layers: {' -> '.join(map(str, arch.shared_layers))}")
        log_message(f"  (g,t) pairs: {n_gt_pairs}")
        log_message(f"  Total parameters: {n_params:,}")

    def forward_gt(self, x: torch.Tensor, gt_index: int) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific (g,t) pair during TRAINING (keeps gradients).

        Args:
            x: Tensor (batch_size, covariate_dim_for_gt)
            gt_index: Which (g,t) to use (0-indexed)

        Returns:
            Dict with 'outcome' and 'propensity' tensors
        """
        gt_key = str(gt_index)

        # Apply (g,t)-specific input projection
        h = self.input_projections[gt_key](x)

        # Shared encoder
        h = self.shared_encoder(h)

        # Task-specific heads
        outcome = self.outcome_heads[gt_key](h)
        propensity = self.propensity_heads[gt_key](h)

        return {
            'outcome': outcome,
            'propensity': propensity
        }

    @torch.no_grad()
    def predict_gt(self, x: torch.Tensor, gt_index: int) -> Dict[str, torch.Tensor]:
        """
        Predict for a specific (g,t) pair (for INFERENCE - disables gradients).

        Args:
            x: Tensor (batch_size, covariate_dim_for_gt)
            gt_index: Which (g,t) to use (0-indexed)

        Returns:
            Dict with 'outcome' and 'propensity' tensors
        """
        self.eval()
        return self.forward_gt(x, gt_index)

    def get_input_dim(self, gt_index: int) -> int:
        """Get expected input dimension for a specific (g,t)."""
        return self.covariate_dims[gt_index]


def initialize_weights(model: nn.Module, method: str = "kaiming"):
    """Initialize model weights."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif method == "normal":
                nn.init.normal_(module.weight, mean=0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    return model


def create_model(covariate_info, config: Config) -> MultiTaskDiDNet:
    """Create model from configuration."""
    covariate_dims = covariate_info.dims_by_gt
    n_gt_pairs = covariate_info.n_gt

    # Create model
    model = MultiTaskDiDNet(covariate_dims, n_gt_pairs, config)

    # Initialize weights
    model = initialize_weights(model, method="kaiming")

    # Move to device
    device = get_device(config)
    model = model.to(device)

    log_message(f"Model created on device: {device}")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: MultiTaskDiDNet):
    """Print model summary."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60 + "\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of (g,t) pairs: {model.n_gt_pairs}")
    print(f"Common projection dim: {model.common_dim}")
    print(f"Covariate dims: min={min(model.covariate_dims.values())}, max={max(model.covariate_dims.values())}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Estimate memory
    memory_mb = total_params * 4 / (1024**2)  # float32
    print(f"Estimated memory: {memory_mb:.1f} MB")
    print()
