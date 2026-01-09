# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# ⚠️ CRITICAL SYSTEM CONSTRAINTS

## Large Dataset Warning
The dataset at `data/analysis/final_analysis_dataset.csv` is **2.6 GB** (verified 2026-01-09).

**YOU MUST NEVER:**
- Use the `view` tool on this file
- Attempt to read the entire file with `read.csv()` or similar
- Load the full dataset into memory in Claude Code environment
- Display or print the complete dataset

**YOU MUST ALWAYS:**
1. Check file size first: `ls -lh [filename]`
2. For files > 100 MB, use sampling:
```bash
# Bash inspection
head -n 100 file.csv
tail -n 100 file.csv
wc -l file.csv
```
3. For R analysis, use `data.table::fread()` with `nrows`:
```r
library(data.table)
df_sample <- fread("data/analysis/final_analysis_dataset.csv",
                   nrows = 1000,
                   header = TRUE)
str(df_sample)
summary(df_sample)
```

This constraint applies to ALL phases of the project.

---

## Working Style

- **Work modularly**: Break tasks into discrete, testable components. Complete one module before moving to the next.
- **Ask questions**: When requirements are ambiguous or multiple approaches exist, ask before implementing.
- **Check in after major tasks**: After completing a significant piece of work (e.g., a new function, module, or fixing a complex issue), pause and summarize what was done before proceeding.

---

## Project Structure

```
Paper -- CDFI -- Sween 2026/
├── code/
│   ├── stata/
│   │   └── 0. clean.do              # Data pipeline (COMPLETE)
│   └── r/                            # Estimation code (TO IMPLEMENT)
├── data/
│   ├── raw/                          # Source data (DO NOT MODIFY)
│   │   ├── cdfi_transactions/        # CDFI TLR data (FY2003-2021)
│   │   ├── entrepreneurship/         # Startup Cartography Project
│   │   ├── crosswalks/               # Geographic mappings
│   │   ├── covariates/               # ACS, banks, land cover
│   │   └── labormarket/              # County Business Patterns
│   ├── intermediate/                 # Pipeline outputs (~400 MB total)
│   └── analysis/                     # Final datasets (2.6 GB CSV, 1.2 GB DTA)
├── outputs/
│   ├── figures/
│   └── tables/
└── markdown/                         # Documentation
```

---

## Common Commands

### Data Pipeline (Stata)
```stata
# From project root in Stata
cd "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"
do "code/stata/0. clean.do"
```
Required packages: `gtools`, `ftools`

### R Analysis (to be implemented)
```r
# Install dependencies
install.packages(c("torch", "data.table", "ggplot2"))

# Load analysis dataset (ALWAYS sample for exploration)
library(data.table)
df <- fread("data/analysis/final_analysis_dataset.csv", nrows = 1000)
```

---

## Panel Structure

| Field | Variable | Description |
|-------|----------|-------------|
| Unit | `id` | ZIP code |
| Time | `time` | Year (1988-2016) |
| Treatment cohort | `group` | Year of first CDFI activity (0 if never-treated) |
| Clustering | `cluster_county` | Primary clustering variable |
| | `cluster_zcta`, `cluster_state` | Alternative clustering options |

**Analysis period**: 1996-2014 (when CDFI program active)

---

## Variable Naming Conventions

| Prefix | Type | Example |
|--------|------|---------|
| `y_` | Outcome variables | `y_sfr_pc` (startups per 1k pop) |
| `X_` | Time-invariant covariates | `X_totpop08_12`, `X_aland10` |
| `V_` | Time-varying covariates | `V_lenders_pc`, `V_totpop` |
| `Wy_` | Year-specific baselines | `Wy_y_sfr_pc_2005` |
| `G_` | Treatment timing | `G_final`, `G_county`, `G_intensity` |
| `i_` | Indicator variables | `i_treat`, `i_treat_post` |

### Key Outcome Variables
- `y_sfr` / `y_sfr_pc`: Startup formation rate (count / per 1k pop)
- `y_eqi`: Entrepreneurial quality index
- `y_growth` / `y_growth_pc`: Business growth (count / per 1k pop)
- `y_logwage`: Log average wage
- `y_empop`: Employment-to-population ratio (%)

### Key Treatment Variables
- `i_treat`: Ever-treated indicator (=1 if `G_final` > 0)
- `i_treat_post`: Post-treatment indicator (=1 if `time` >= `G_final`)
- `G_final`: Treatment year (earliest of county or ZIP intensity)

---

# Methodological Framework

## Project Overview

Implementation of the Callaway & Sant'Anna (2021) difference-in-differences estimator using a custom neural network architecture within a double machine learning (DML) framework. Uses R `torch` to estimate group-time average treatment effects on the treated (ATT(g,t)) with not-yet-treated units as controls.

### Current State

1. **Data Pipeline (Complete)** - Stata code in `code/stata/0. clean.do` constructs `data/analysis/final_analysis_dataset.dta` from raw CDFI transaction records, entrepreneurship data (Startup Cartography Project), and covariates.

2. **Estimation Pipeline (Specification Phase)** - R torch-based DiD estimator with neural network nuisance parameter estimation. Not yet implemented.

## Estimation Strategy

**Treatment Effect Estimand**: ATT(g,t) for each group `g` and time period `t`

**Control Group**: Not-yet-treated units

**Inference**: Semi-parametric doubly-robust influence functions with clustered (county) multiplier bootstrap

---

## Covariate Selection Rules

### Pre-Treatment Periods (t < g)
- Include: All `X_` covariates (time-invariant)
- Include: Subset of `V_` covariates corresponding to periods up to `t-1`
- **Trend Differences**: Pairwise period trend differences (short-term)

### Treatment Periods (t ≥ g)
- Include: All `X_` covariates (time-invariant)
- Include: All `V_` covariates corresponding to periods up to `g-1`
- **Trend Differences**: (g-1)-anchored trend differences (long-term)

---

## Nuisance Parameters

For each (g,t) pair, estimate:

1. **Conditional Mean of Untreated Potential Outcome Trend**:
   - E[Y(t) - Y(t') | X, V, D=0] for control units

2. **Propensity Score**:
   - P(G = g | X, V, G = g or G = Not-Yet-Treated)

---

## Neural Network Architecture

### Requirements

**Multi-Task Learning Framework**:
- Simultaneously learn all group-and-time-specific nuisance parameters
- Multi-headed architecture for different (g,t) combinations
- Masking strategy for handling different covariate subsets across (g,t) pairs

**Cross-Fitting**:
- K-fold cross-fitting within DML framework
- Single unified training process (not separate for each ATT(g,t))

### Configurable Parameters

**Optimization**: L-BFGS or AdamW with configurable learning rate, weight decay, schedulers

**Training**: Epochs, batch size, early stopping (patience, min delta)

**Architecture**: Hidden dimensions, activation functions, dropout rates, batch normalization

**Monitoring**: Live performance printing, loss curve plotting

---

## Statistical Inference

### Event Study Aggregation
- **Pre-Treatment Periods**: 10 periods before treatment
- **Post-Treatment Periods**: 10 periods after treatment
- Pool ATT(g,t) estimates across groups, weight by group size

### Bootstrap Inference
- Clustered multiplier bootstrap following Callaway & Sant'Anna (2021)
- Point estimates, standard errors, pointwise CIs, uniform confidence bands

---

## Deliverables

1. **Estimation Pipeline**: Complete R torch-based implementation
2. **Results**: ATT(g,t) estimates, event study aggregates, uniform confidence bands
3. **Visualization**: Publication-ready ggplot2 event study graph (-10 to +10)
4. **Diagnostics**: Training/validation loss curves, convergence diagnostics, pre-trend tests

---

## Configuration Template

```r
config <- list(
  # Data
  outcome_prefix = "y_",
  time_invariant_prefix = "X_",
  time_varying_prefix = "V_",

  # Neural Network
  architecture = list(
    hidden_dims = c(256, 128, 64),
    activation = "relu",
    dropout = 0.2,
    batch_norm = TRUE
  ),

  # Optimization
  optimizer = "adamw",  # or "lbfgs"
  learning_rate = 0.001,
  weight_decay = 0.01,
  scheduler = "cosine",

  # Training
  epochs = 100,
  batch_size = 256,
  early_stopping = list(
    patience = 10,
    min_delta = 1e-4
  ),

  # Cross-fitting
  n_folds = 5,

  # Inference
  n_bootstrap = 1000,
  alpha = 0.05,

  # Event Study
  pre_periods = 10,
  post_periods = 10,

  # Monitoring
  verbose = TRUE,
  plot_loss = TRUE,
  print_every = 10
)
```

---

## Key Implementation Challenges

1. **Dynamic Covariate Sets**: Different (g,t) pairs require different covariate subsets
2. **Multi-Task Architecture**: Joint estimation of all nuisance parameters
3. **Cross-Fitting**: Proper sample splitting within torch workflow
4. **Memory Management**: Efficient handling of large panel datasets
5. **Gradient Flow**: Ensuring stable training across multiple heads

---

## Technical Stack

**Primary Framework**: R with `torch` package

**Required Packages**: `torch`, `data.table`, `ggplot2`

**Computational Requirements**: GPU recommended, sufficient memory for multi-task learning

---

## References

Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
