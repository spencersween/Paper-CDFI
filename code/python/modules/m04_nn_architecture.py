"""
m04_nn_architecture.py - Neural Network Architecture

Multi-task network with:
- Per-base-period input projections (shared across (g,t) with same base period)
- Shared encoder
- Per-(g,t) outcome and propensity heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .m00_config import Config
from .m02_data_loader import GTInfo
from .m03_covariate_selector import CovariateInfo


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh()
    }
    return activations.get(name, nn.ReLU())


class InputProjection(nn.Module):
    """Projects variable-size covariate input to fixed dimension."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class SharedEncoder(nn.Module):
    """Shared encoder applied to all (g,t) paths."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm: bool = True
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TaskHead(nn.Module):
    """Task-specific head for outcome or propensity."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiTaskDiDNet(nn.Module):
    """
    Multi-task neural network for DiD nuisance estimation.

    Architecture:
    1. Per-base-period input projections (shared across (g,t) with same base period)
    2. Shared encoder (same weights for all paths)
    3. Per-(g,t) outcome and propensity heads

    Forward pass:
    1. For each base period, project covariates
    2. Gather projections for each (g,t) based on its base period
    3. Pass through shared encoder
    4. Apply per-(g,t) heads
    """

    def __init__(
        self,
        covariate_info: CovariateInfo,
        gt_info: GTInfo,
        config: Config,
        device: torch.device
    ):
        super().__init__()

        self.n_gt = gt_info.n_gt
        self.device = device

        arch = config.architecture
        proj_dim = arch.input_projection_dim
        self.proj_dim = proj_dim

        # Store base period info
        self.unique_base_periods = gt_info.unique_base_periods
        self.n_base_periods = len(self.unique_base_periods)

        # Register gt_to_base_idx as buffer (moves with model to device)
        self.register_buffer(
            'gt_to_base_idx',
            torch.tensor(gt_info.gt_to_base_idx, dtype=torch.long)
        )

        # Create covariate masks as buffers (one per base period)
        for i, bp in enumerate(self.unique_base_periods):
            mask = covariate_info.masks_by_base[bp]
            self.register_buffer(f'cov_mask_{i}', torch.tensor(mask, dtype=torch.bool))

        # Per-base-period input projections
        self.input_projections = nn.ModuleList()
        for bp in self.unique_base_periods:
            input_dim = covariate_info.dims_by_base[bp]
            self.input_projections.append(
                InputProjection(input_dim, proj_dim, arch.dropout)
            )

        # Shared encoder
        self.shared_encoder = SharedEncoder(
            input_dim=proj_dim,
            hidden_dims=arch.shared_layers,
            dropout=arch.dropout,
            activation=arch.activation,
            layer_norm=arch.layer_norm
        )

        encoder_out_dim = self.shared_encoder.output_dim

        # Per-(g,t) outcome heads
        self.outcome_heads = nn.ModuleList([
            TaskHead(
                encoder_out_dim,
                arch.outcome_head_layers,
                output_dim=1,
                dropout=arch.dropout,
                activation=arch.activation
            )
            for _ in range(self.n_gt)
        ])

        # Per-(g,t) propensity heads
        self.propensity_heads = nn.ModuleList([
            TaskHead(
                encoder_out_dim,
                arch.propensity_head_layers,
                output_dim=1,
                dropout=arch.dropout,
                activation=arch.activation
            )
            for _ in range(self.n_gt)
        ])

        self.to(device)

    def forward(self, X_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            X_full: (batch, n_covariates) full covariate matrix

        Returns:
            outcome_preds: (batch, n_gt) outcome predictions (mu_0)
            propensity_preds: (batch, n_gt) propensity logits
        """
        batch_size = X_full.shape[0]

        # Step 1: Compute projections for each base period
        # projections[i] = (batch, proj_dim) for base period i
        base_projections = []
        for i in range(self.n_base_periods):
            mask = getattr(self, f'cov_mask_{i}')
            x_masked = X_full[:, mask]
            proj = self.input_projections[i](x_masked)
            base_projections.append(proj)

        # Stack: (batch, n_base_periods, proj_dim)
        base_projections = torch.stack(base_projections, dim=1)

        # Step 2: Gather projections for each (g,t) based on its base period
        # gathered: (batch, n_gt, proj_dim)
        gathered = base_projections[:, self.gt_to_base_idx, :]

        # Step 3: Apply shared encoder to all (g,t) paths
        # Reshape to (batch * n_gt, proj_dim)
        flat = gathered.view(-1, self.proj_dim)
        encoded = self.shared_encoder(flat)
        # Reshape back to (batch, n_gt, encoder_out_dim)
        encoded = encoded.view(batch_size, self.n_gt, -1)

        # Step 4: Apply per-(g,t) heads
        outcome_preds = []
        propensity_preds = []

        for gt_idx in range(self.n_gt):
            h = encoded[:, gt_idx, :]  # (batch, encoder_out_dim)
            outcome_preds.append(self.outcome_heads[gt_idx](h))
            propensity_preds.append(self.propensity_heads[gt_idx](h))

        # Stack: (batch, n_gt)
        outcome_preds = torch.cat(outcome_preds, dim=1)
        propensity_preds = torch.cat(propensity_preds, dim=1)

        return outcome_preds, propensity_preds

    def predict(self, X_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with sigmoid applied to propensity."""
        self.eval()
        with torch.no_grad():
            outcome_preds, propensity_logits = self.forward(X_full)
            propensity_preds = torch.sigmoid(propensity_logits)
        return outcome_preds, propensity_preds


def create_model(
    covariate_info: CovariateInfo,
    gt_info: GTInfo,
    config: Config,
    device: torch.device
) -> MultiTaskDiDNet:
    """Create the multi-task DiD network."""
    return MultiTaskDiDNet(covariate_info, gt_info, config, device)
