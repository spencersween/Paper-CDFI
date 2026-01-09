# CDFI Lending and Entrepreneurship: Callaway & Sant'Anna DiD Estimation

Implementation of Callaway & Sant'Anna (2021) difference-in-differences estimator with neural network nuisance parameter estimation using R `torch`.

## Project Structure

```
.
├── code/
│   ├── r/
│   │   ├── R/                    # R modules (00-10)
│   │   └── scripts/              # Runnable scripts
│   └── stata/                    # Data cleaning pipeline
├── data/                         # NOT IN REPO - see Data Setup below
│   ├── raw/                      # Source data
│   ├── intermediate/             # Processing outputs
│   └── analysis/                 # Final analysis dataset
├── outputs/
│   └── figures/                  # Event study plots
├── markdown/                     # Documentation
├── CLAUDE.md                     # AI assistant instructions
└── README.md
```

## Data Setup

Data files are stored separately (Google Drive) due to size (~5.4 GB). To set up:

1. Download the `data/` folder from [your Google Drive link]
2. Place it in the project root so you have `data/analysis/final_analysis_dataset.csv`

### Required Data Files

- `data/analysis/final_analysis_dataset.csv` (2.4 GB) - Main analysis dataset
- `data/analysis/final_analysis_dataset.dta` - Stata format

## R Dependencies

```r
install.packages(c("data.table", "torch", "ggplot2", "R6"))

# Install torch backend (run once)
torch::install_torch()
```

## Running the Estimation

### Quick Test (small sample)
```r
source("code/r/scripts/test_pipeline.R")
```

### Full Estimation
```r
source("code/r/scripts/run_estimation.R")
results <- run_estimation(outcome = "sfr_pc", sample_n = NULL)
```

### Minimal Architecture (for testing on full data)
```r
source("code/r/scripts/run_minimal.R")
```

## Key Configuration Options

Edit in the script or pass to `create_config()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `architecture$input_projection_dim` | 128 | Input projection dimension |
| `architecture$shared_layers` | c(256, 128) | Shared encoder layers |
| `training$epochs` | 100 | Maximum training epochs |
| `training$batch_size` | 256 | Batch size |
| `cross_fitting$n_folds` | 2 | K-fold cross-fitting |
| `inference$n_bootstrap` | 1000 | Bootstrap replications |

## Outputs

- `outputs/figures/event_study.png` - Main event study plot
- `outputs/figures/event_study.pdf` - PDF version
- `outputs/estimation_results.rds` - Full results object

## Methodology

- **Treatment Effects**: ATT(g,t) for each group-time pair
- **Control Group**: Not-yet-treated units
- **Nuisance Estimation**: Multi-task neural network with per-(g,t) input projections
- **Inference**: Clustered multiplier bootstrap with uniform confidence bands

## Reference

Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
