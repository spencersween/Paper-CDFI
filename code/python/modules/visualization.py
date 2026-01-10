"""
Visualization Module for DiD Estimation

Creates publication-ready event study plots using matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import os

from .utils import log_message, ensure_dir
from .config import Config


def plot_event_study(
    es_results: Dict,
    title: str = "Event Study: Effect of CDFI Lending on Entrepreneurship",
    subtitle: Optional[str] = None,
    show_uniform_bands: bool = True,
    show_pointwise_ci: bool = True,
    normalized: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[Dict] = None
) -> plt.Figure:
    """Create publication-ready event study plot."""
    if colors is None:
        colors = {
            'point': '#2C3E50',
            'line': '#2C3E50',
            'ci': '#3498DB',
            'uniform': '#E74C3C',
            'reference': '#7F8C8D'
        }

    es = es_results['event_study']

    # Choose ATT column
    if normalized and 'att_normalized' in es.columns:
        att_col = 'att_normalized'
        y_label = 'ATT (Normalized)'
    else:
        att_col = 'att'
        y_label = 'ATT'

    fig, ax = plt.subplots(figsize=figsize)

    # Reference lines
    ax.axvline(x=-0.5, linestyle='--', color=colors['reference'], linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, linestyle='-', color=colors['reference'], linewidth=0.5, alpha=0.7)

    # Uniform bands
    if show_uniform_bands and 'uniform_lower' in es.columns:
        ax.fill_between(
            es['event_time'],
            es['uniform_lower'],
            es['uniform_upper'],
            color=colors['uniform'],
            alpha=0.15,
            label='Uniform 95% CI'
        )

    # Pointwise CI
    if show_pointwise_ci:
        ax.fill_between(
            es['event_time'],
            es['ci_lower'],
            es['ci_upper'],
            color=colors['ci'],
            alpha=0.25,
            label='Pointwise 95% CI'
        )

    # Line and points
    ax.plot(es['event_time'], es[att_col], color=colors['line'], linewidth=0.8)
    ax.scatter(es['event_time'], es[att_col], color=colors['point'], s=50, zorder=5)

    # Labels
    ax.set_xlabel('Periods Relative to Treatment', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center',
                fontsize=11, color='gray')

    # X-axis ticks
    x_ticks = np.arange(es['event_time'].min(), es['event_time'].max() + 1, 2)
    ax.set_xticks(x_ticks)

    # Legend
    ax.legend(loc='best', frameon=True, fontsize=9)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    return fig


def save_event_study(
    fig: plt.Figure,
    filename: str = "event_study",
    output_dir: str = "outputs/figures",
    dpi: int = 300
):
    """Save event study plot."""
    ensure_dir(output_dir)

    # PNG
    png_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    log_message(f"Saved: {png_path}")

    # PDF
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    log_message(f"Saved: {pdf_path}")

    return {'png': png_path, 'pdf': pdf_path}


def plot_att_by_group(group_results: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot ATT by treatment group."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(y=0, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)

    ax.errorbar(
        range(len(group_results)),
        group_results['att'],
        yerr=[group_results['att'] - group_results['ci_lower'],
              group_results['ci_upper'] - group_results['att']],
        fmt='o',
        color='#2C3E50',
        ecolor='#3498DB',
        capsize=3,
        markersize=8
    )

    ax.set_xticks(range(len(group_results)))
    ax.set_xticklabels(group_results['g'].astype(int), rotation=45, ha='right')
    ax.set_xlabel('Treatment Year (Cohort)', fontsize=11)
    ax.set_ylabel('ATT', fontsize=11)
    ax.set_title('ATT by Treatment Cohort', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_att_by_time(time_results: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot ATT over calendar time."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(y=0, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)

    ax.fill_between(
        time_results['t'],
        time_results['ci_lower'],
        time_results['ci_upper'],
        color='#3498DB',
        alpha=0.25
    )
    ax.plot(time_results['t'], time_results['att'], color='#2C3E50', linewidth=0.8)
    ax.scatter(time_results['t'], time_results['att'], color='#2C3E50', s=50)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('ATT', fontsize=11)
    ax.set_title('ATT Over Calendar Time', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def generate_all_figures(
    att_results: Dict,
    agg_results: Dict,
    output_dir: str,
    config: Config
) -> Dict:
    """Generate all output figures."""
    log_message("Generating output figures...")

    ensure_dir(output_dir)

    # Main event study
    es_fig = plot_event_study(
        agg_results['event_study'],
        title=f"Effect of CDFI Lending on {config.outcome_var}",
        subtitle="Callaway & Sant'Anna (2021) DiD with Neural Network Nuisance Estimation"
    )
    save_event_study(es_fig, "event_study", output_dir)

    # Normalized
    es_fig_norm = plot_event_study(
        agg_results['event_study'],
        title=f"Effect of CDFI Lending on {config.outcome_var} (Normalized)",
        normalized=True
    )
    save_event_study(es_fig_norm, "event_study_normalized", output_dir)

    # By group
    group_fig = plot_att_by_group(agg_results['by_group'])
    group_fig.savefig(os.path.join(output_dir, "att_by_group.png"), dpi=300, bbox_inches='tight')

    # By time
    time_fig = plot_att_by_time(agg_results['by_time'])
    time_fig.savefig(os.path.join(output_dir, "att_by_time.png"), dpi=300, bbox_inches='tight')

    log_message(f"Figures saved to: {output_dir}")

    plt.close('all')

    return {
        'event_study': es_fig,
        'event_study_normalized': es_fig_norm,
        'by_group': group_fig,
        'by_time': time_fig
    }
