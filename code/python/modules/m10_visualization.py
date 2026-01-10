"""
m10_visualization.py - Visualization

Creates publication-ready event study plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


def plot_event_study(
    es_results: Dict,
    title: str = "Event Study",
    subtitle: Optional[str] = None,
    show_uniform_bands: bool = True,
    show_pointwise_ci: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[Dict] = None
) -> plt.Figure:
    """
    Create publication-ready event study plot.

    Args:
        es_results: Event study results from aggregate_event_study
        title: Plot title
        subtitle: Optional subtitle
        show_uniform_bands: Show uniform confidence bands
        show_pointwise_ci: Show pointwise confidence intervals
        figsize: Figure size
        colors: Custom color scheme

    Returns:
        matplotlib Figure
    """
    es_df = es_results['event_study']

    # Default colors
    if colors is None:
        colors = {
            'point': '#2C3E50',
            'pointwise_ci': '#3498DB',
            'uniform_band': '#E74C3C',
            'zero_line': '#7F8C8D',
            'reference_line': '#95A5A6'
        }

    fig, ax = plt.subplots(figsize=figsize)

    event_times = es_df['event_time'].values
    att = es_df['att'].values

    # Zero reference line
    ax.axhline(y=0, color=colors['zero_line'], linestyle='-', linewidth=1, alpha=0.7)

    # Vertical line at treatment
    ax.axvline(x=-0.5, color=colors['reference_line'], linestyle='--', linewidth=1, alpha=0.5)

    # Uniform confidence bands
    if show_uniform_bands and 'uniform_lower' in es_df.columns:
        uniform_lower = es_df['uniform_lower'].values
        uniform_upper = es_df['uniform_upper'].values
        ax.fill_between(
            event_times,
            uniform_lower,
            uniform_upper,
            alpha=0.15,
            color=colors['uniform_band'],
            label='95% Uniform CI'
        )

    # Pointwise confidence intervals
    if show_pointwise_ci:
        ci_lower = es_df['ci_lower'].values
        ci_upper = es_df['ci_upper'].values
        ax.fill_between(
            event_times,
            ci_lower,
            ci_upper,
            alpha=0.25,
            color=colors['pointwise_ci'],
            label='95% Pointwise CI'
        )

    # Point estimates
    ax.plot(
        event_times,
        att,
        'o-',
        color=colors['point'],
        linewidth=2,
        markersize=6,
        label='ATT'
    )

    # Labels
    ax.set_xlabel('Event Time (Periods Relative to Treatment)', fontsize=12)
    ax.set_ylabel('Average Treatment Effect', fontsize=12)

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='best', framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate(
        'Pre-treatment',
        xy=(event_times.min() + 1, ax.get_ylim()[1] * 0.9),
        fontsize=10,
        color='gray'
    )
    ax.annotate(
        'Post-treatment',
        xy=(1, ax.get_ylim()[1] * 0.9),
        fontsize=10,
        color='gray'
    )

    plt.tight_layout()

    return fig


def plot_group_effects(
    group_df: pd.DataFrame,
    title: str = "Treatment Effects by Cohort",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot treatment effects by cohort."""
    fig, ax = plt.subplots(figsize=figsize)

    groups = group_df['g'].values
    att = group_df['att'].values
    ci_lower = group_df['ci_lower'].values
    ci_upper = group_df['ci_upper'].values

    # Error bars
    yerr = np.array([att - ci_lower, ci_upper - att])

    ax.errorbar(
        groups,
        att,
        yerr=yerr,
        fmt='o',
        capsize=4,
        capthick=2,
        color='#2C3E50'
    )

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    ax.set_xlabel('Treatment Cohort', fontsize=12)
    ax.set_ylabel('Average Treatment Effect', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def save_event_study(
    fig: plt.Figure,
    filename: str,
    output_dir: str,
    formats: Tuple[str, ...] = ('png', 'pdf')
):
    """Save event study plot in multiple formats."""
    import os

    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")


def plot_training_history(history, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Total loss
    axes[0].plot(history.train_loss, label='Train')
    axes[0].plot(history.val_loss, label='Validation')
    axes[0].axvline(x=history.best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()

    # Outcome loss
    axes[1].plot(history.train_outcome_loss, label='Outcome')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Outcome Loss (MSE)')

    # Propensity loss
    axes[2].plot(history.train_propensity_loss, label='Propensity')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Propensity Loss (BCE)')

    plt.tight_layout()

    return fig
