# CDFI Lending and Entrepreneurship

Implementation of Callaway & Sant'Anna (2021) difference-in-differences estimator with neural network nuisance parameter estimation.

## Overview

This project estimates the causal effect of Community Development Financial Institution (CDFI) lending on local entrepreneurship using a doubly-robust difference-in-differences approach with staggered treatment adoption.

**Key Features:**
- Callaway & Sant'Anna (2021) group-time ATT estimation
- Neural network nuisance parameter estimation (outcome regression + propensity scores)
- Multi-task architecture with shared encoder and per-(g,t) heads
- Clustered multiplier bootstrap inference with uniform confidence bands
- Event study aggregation

## Implementation Status

| Language | Status | Description |
|----------|--------|-------------|
| **Python** | **Active** | Full implementation using PyTorch |
| R | Under Development | Parallel implementation using `torch` for R |

## Project Structure

```
.
├── code/
│   ├── python/
│   │   ├── modules/              # Core modules (m00-m10)
│   │   └── notebooks/            # Jupyter notebooks
│   ├── r/                        # R implementation (under development)
│   │   ├── R/                    # R modules (00-10)
│   │   └── scripts/              # Runnable scripts
│   └── stata/                    # Data cleaning pipeline
├── data/                         # NOT IN REPO - see Data Setup below
│   ├── raw/                      # Source data
│   ├── intermediate/             # Processing outputs
│   └── analysis/                 # Final analysis dataset
├── outputs/
│   └── figures/                  # Event study plots
├── CLAUDE.md                     # Technical specification
└── README.md
```

## Quick Start (Python)

### Dependencies

```bash
pip install torch numpy pandas scipy matplotlib
```

### Running the Estimation

```python
# In Jupyter notebook or Python script
import sys
sys.path.insert(0, "code/python")

from modules import (
    create_config, set_seed, get_device,
    load_panel_data, create_unit_data, create_gt_info,
    create_covariate_info, run_cross_fitting,
    compute_all_att, add_bootstrap_inference,
    aggregate_all, plot_event_study
)

# Configure
config = create_config(
    outcome="sfr_pc",
    analysis_start=1996,
    analysis_end=2014
)

# Load and prepare data
panel_df = load_panel_data("data/analysis/final_analysis_dataset.csv", config)
unit_data = create_unit_data(panel_df, config)
gt_info = create_gt_info(unit_data, config)
covariate_info = create_covariate_info(unit_data, gt_info, config)

# Cross-fitting and estimation
cf_results = run_cross_fitting(unit_data, gt_info, covariate_info, config)
att_results = compute_all_att(cf_results, config)
att_results = add_bootstrap_inference(att_results, config)

# Aggregation and visualization
agg_results = aggregate_all(att_results, config)
fig = plot_event_study(agg_results['event_study'])
```

Or use the notebook: `code/python/notebooks/estimation_runner.ipynb`

## Data Setup

Data files are stored separately (Google Drive) due to size (~2.6 GB). To set up:

1. Download the `data/` folder from the shared drive
2. Place it in the project root: `data/analysis/final_analysis_dataset.csv`

## Key Configuration Options

All hyperparameters are configurable via `create_config()`:

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Data** | `outcome` | "sfr_pc" | Outcome variable (without y_ prefix) |
| | `analysis_start` | 1996 | First analysis period |
| | `analysis_end` | 2014 | Last analysis period |
| **Architecture** | `input_projection_dim` | 128 | Per-base-period projection dimension |
| | `shared_layers` | [256, 128] | Shared encoder layers |
| | `dropout` | 0.2 | Dropout probability |
| **Training** | `epochs` | 100 | Maximum training epochs |
| | `batch_size` | 512 | Batch size |
| | `early_stopping_patience` | 15 | Epochs without improvement |
| **Cross-fitting** | `n_folds` | 2 | K-fold cross-fitting |
| **Inference** | `n_bootstrap` | 1000 | Bootstrap replications |
| | `alpha` | 0.05 | Significance level |
| **Event Study** | `pre_periods` | 10 | Pre-treatment periods |
| | `post_periods` | 10 | Post-treatment periods |

## Methodology

### Doubly-Robust ATT Estimation

For each (g,t) pair:

```
ATT(g,t) = E[ w * (ΔY - μ₀(X)) ]

where:
  w = (D - ps(X)) / (p_g * (1 - ps(X)))
  ΔY = Y(t) - Y(base)
  μ₀(X) = E[ΔY | X, D=0]  (outcome regression)
  ps(X) = P(D=1 | X)      (propensity score)
  p_g = P(D=1)            (treatment probability in sample)
```

### Control Group

- **Treated**: Units first treated in period g
- **Control**: Units not-yet-treated at time t (group > t) OR never-treated, excluding the treated group

### Inference

Clustered multiplier bootstrap following CS2021:
1. Compute unit-level influence functions
2. Aggregate to cluster level
3. Generate multiplier weights at cluster level
4. Bootstrap: `ATT* = ATT + (1/n) * Σ_c ξ_c * IF_c`

For event study aggregation:
1. Aggregate influence functions with same weights as ATT aggregation
2. Run **new** clustered bootstrap on aggregated IFs
3. Compute uniform bands via sup-t method

## Outputs

- `outputs/figures/event_study.png` - Event study plot with confidence bands
- `outputs/estimation_results.pkl` - Full results (Python pickle)

## Reference

Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
