#!/usr/bin/env Rscript
#' =============================================================================
#' Minimal Architecture Full Dataset Run
#'
#' 2 hidden nodes in every layer, full dataset
#' =============================================================================

cat("\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nMINIMAL ARCHITECTURE FULL DATASET RUN\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

# Track total time
total_start <- Sys.time()

# =============================================================================
# SETUP
# =============================================================================

project_root <- "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"
setwd(project_root)

cat("Loading modules...\n")

# Source all modules
source("code/r/R/00_config.R")
source("code/r/R/utils.R")
source("code/r/R/01_data_loader.R")
source("code/r/R/02_covariate_selector.R")
source("code/r/R/03_cross_fitting.R")
source("code/r/R/04_nn_architecture.R")
source("code/r/R/05_nn_training.R")
source("code/r/R/06_nuisance_estimation.R")
source("code/r/R/07_att_estimation.R")
source("code/r/R/08_inference.R")
source("code/r/R/09_aggregation.R")
source("code/r/R/10_visualization.R")

# Check required packages
check_packages(c("data.table", "torch", "ggplot2", "R6"))

library(data.table)
library(torch)
library(ggplot2)

cat("Modules loaded.\n\n")

# =============================================================================
# CONFIGURATION - MINIMAL ARCHITECTURE (2 hidden nodes everywhere)
# =============================================================================

cat("Creating minimal architecture configuration...\n")

config <- create_config(
  outcome = "sfr_pc",

  # MINIMAL architecture - 2 hidden nodes in EVERY layer
  architecture = list(
    input_projection_dim = 2L,     # 2 nodes
    shared_layers = c(2L),          # 2 nodes
    outcome_head_layers = c(2L),    # 2 nodes
    propensity_head_layers = c(2L), # 2 nodes
    activation = "relu",
    dropout = 0.0,                  # No dropout for tiny network
    layer_norm = FALSE,             # No layer norm for tiny network
    residual_connections = FALSE
  ),

  # Training settings
  training = list(
    epochs = 50L,
    batch_size = 512L,
    validation_split = 0.2,
    shuffle = TRUE,
    early_stopping = list(
      enabled = TRUE,
      patience = 10L,
      min_delta = 1e-4,
      monitor = "val_loss",
      restore_best_weights = TRUE
    ),
    gradient_clipping = list(
      enabled = TRUE,
      max_norm = 1.0
    )
  ),

  # 2-fold cross-fitting
  cross_fitting = list(
    n_folds = 2L,
    stratify_by = "cluster_county",
    seed = 42L
  ),

  # Bootstrap - reduced for speed
  inference = list(
    n_bootstrap = 500L,
    alpha = 0.05,
    multiplier_dist = "normal",
    uniform_bands = TRUE,
    seed = 123L
  ),

  # Monitoring
  monitoring = list(
    verbose = TRUE,
    print_every = 5L,
    plot_loss = FALSE,
    save_checkpoints = FALSE,
    checkpoint_dir = "checkpoints",
    checkpoint_every = 10L,
    log_file = NULL,
    log_level = "INFO"
  )
)

print(config)

# Set random seeds
set_seed(config$seed)

# =============================================================================
# PHASE 1: DATA LOADING
# =============================================================================

phase1_start <- Sys.time()
cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("PHASE 1: DATA LOADING (FULL DATASET)\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

data_path <- file.path(project_root, "data/analysis/final_analysis_dataset.csv")

# Load FULL dataset (no sample_n)
panel <- load_panel_data(data_path, config, sample_n = NULL)
data <- panel$data
metadata <- panel$metadata

summarize_data(data, metadata, config)
phase1_time <- as.numeric(difftime(Sys.time(), phase1_start, units = "secs"))
cat(sprintf("\nPhase 1 time: %.1f seconds\n", phase1_time))

# =============================================================================
# PHASE 2: (g,t) PAIRS AND COVARIATES
# =============================================================================

phase2_start <- Sys.time()
cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("PHASE 2: COVARIATE SETUP\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Get valid (g,t) pairs
gt_pairs <- get_gt_pairs(data, config)

# Get covariate information
all_vars <- names(data)
covariate_info <- get_covariate_info(gt_pairs, all_vars, config)
all_covariates <- covariate_info$all_covariates
cat(sprintf("Total unique covariates (union): %d\n", length(all_covariates)))

# Create covariate masks (for backward compatibility)
covariate_masks <- create_covariate_masks(gt_pairs, all_covariates, config)

# Validate covariate selection rules
validate_covariate_rules(config)

phase2_time <- as.numeric(difftime(Sys.time(), phase2_start, units = "secs"))
cat(sprintf("\nPhase 2 time: %.1f seconds\n", phase2_time))

# =============================================================================
# PHASE 3: CROSS-FITTING
# =============================================================================

phase3_start <- Sys.time()
cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("PHASE 3: CROSS-FITTING\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

cf_results <- run_cross_fitting(
  data,
  gt_pairs,
  all_covariates,
  covariate_masks,
  config
)

# Validate cross-fitting results
validate_cross_fitting(cf_results, config)

phase3_time <- as.numeric(difftime(Sys.time(), phase3_start, units = "secs"))
cat(sprintf("\nPhase 3 time: %.1f seconds\n", phase3_time))

# =============================================================================
# PHASE 4: ATT ESTIMATION
# =============================================================================

phase4_start <- Sys.time()
cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("PHASE 4: ATT ESTIMATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Diagnose nuisance parameters
nuisance_diag <- diagnose_nuisance(cf_results, config)

# Compute all ATT(g,t) estimates
att_results <- compute_all_att(cf_results, config)

print_att_summary(att_results)

phase4_time <- as.numeric(difftime(Sys.time(), phase4_start, units = "secs"))
cat(sprintf("\nPhase 4 time: %.1f seconds\n", phase4_time))

# =============================================================================
# PHASE 5: INFERENCE
# =============================================================================

phase5_start <- Sys.time()
cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("PHASE 5: INFERENCE\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Add bootstrap inference
att_results <- add_bootstrap_inference(att_results, config)

# Test parallel trends
pt_test <- test_parallel_trends(att_results, config)

# Simple ATT
simple_att <- compute_simple_att(att_results, config)

phase5_time <- as.numeric(difftime(Sys.time(), phase5_start, units = "secs"))
cat(sprintf("\nPhase 5 time: %.1f seconds\n", phase5_time))

# =============================================================================
# PHASE 6: AGGREGATION
# =============================================================================

phase6_start <- Sys.time()
cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("PHASE 6: AGGREGATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

agg_results <- aggregate_all(att_results, config)

print_aggregation_summary(agg_results)

phase6_time <- as.numeric(difftime(Sys.time(), phase6_start, units = "secs"))
cat(sprintf("\nPhase 6 time: %.1f seconds\n", phase6_time))

# =============================================================================
# PHASE 7: VISUALIZATION
# =============================================================================

phase7_start <- Sys.time()
cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("PHASE 7: VISUALIZATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

output_dir <- file.path(project_root, "outputs/figures")
figures <- generate_all_figures(att_results, agg_results, output_dir, config)

phase7_time <- as.numeric(difftime(Sys.time(), phase7_start, units = "secs"))
cat(sprintf("\nPhase 7 time: %.1f seconds\n", phase7_time))

# =============================================================================
# COMPLETE
# =============================================================================

total_time <- as.numeric(difftime(Sys.time(), total_start, units = "secs"))

cat("\n" |> paste0(paste(rep("=", 60), collapse = ""), "\n"))
cat("ESTIMATION COMPLETE\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

cat("TIMING SUMMARY:\n")
cat(sprintf("  Phase 1 (Data Loading):    %8.1f seconds\n", phase1_time))
cat(sprintf("  Phase 2 (Covariate Setup): %8.1f seconds\n", phase2_time))
cat(sprintf("  Phase 3 (Cross-Fitting):   %8.1f seconds\n", phase3_time))
cat(sprintf("  Phase 4 (ATT Estimation):  %8.1f seconds\n", phase4_time))
cat(sprintf("  Phase 5 (Inference):       %8.1f seconds\n", phase5_time))
cat(sprintf("  Phase 6 (Aggregation):     %8.1f seconds\n", phase6_time))
cat(sprintf("  Phase 7 (Visualization):   %8.1f seconds\n", phase7_time))
cat(sprintf("  ----------------------------------------\n"))
cat(sprintf("  TOTAL:                     %8.1f seconds (%.1f minutes)\n", total_time, total_time/60))

cat("\nFigures saved to:\n")
cat(sprintf("  %s\n", output_dir))

# Save results
results <- list(
  config = config,
  metadata = metadata,
  gt_pairs = gt_pairs,
  covariate_info = covariate_info,
  cf_results = cf_results,
  att_results = att_results,
  agg_results = agg_results,
  parallel_trends_test = pt_test,
  simple_att = simple_att,
  figures = figures,
  nuisance_diagnostics = nuisance_diag,
  timing = list(
    phase1 = phase1_time,
    phase2 = phase2_time,
    phase3 = phase3_time,
    phase4 = phase4_time,
    phase5 = phase5_time,
    phase6 = phase6_time,
    phase7 = phase7_time,
    total = total_time
  )
)

output_path <- file.path(project_root, "outputs/minimal_results.rds")
saveRDS(results, output_path)
cat(sprintf("\nResults saved to: %s\n", output_path))

cat("\nDone!\n")
