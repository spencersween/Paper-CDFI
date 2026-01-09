#!/usr/bin/env Rscript
#' =============================================================================
#' Callaway & Sant'Anna (2021) DiD Estimation Pipeline
#'
#' Main entry point for running the full estimation procedure.
#' =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Set working directory to project root
project_root <- "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"
setwd(project_root)

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
check_packages(c(
  "data.table",
  "torch",
  "ggplot2",
  "R6",
  "coro"
))

# Load packages
library(data.table)
library(torch)
library(ggplot2)


# =============================================================================
# CONFIGURATION
# =============================================================================

#' Run full estimation pipeline
#'
#' @param outcome Character. Outcome variable (without y_ prefix)
#' @param sample_n Integer or NULL. Sample size for testing (NULL = full data)
#' @param ... Additional config overrides
#' @return List with all results
run_estimation <- function(outcome = "sfr_pc", sample_n = NULL, ...) {

  timer <- Timer$new()

  # Create configuration
  config <- create_config(
    outcome = outcome,
    ...
  )

  print(config)

  # Set random seeds
  set_seed(config$seed)

  # =============================================================================
  # PHASE 1: DATA LOADING
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("PHASE 1: DATA LOADING")
  log_message(paste(rep("=", 60), collapse = ""))

  data_path <- file.path(project_root, "data/analysis/final_analysis_dataset.csv")

  panel <- load_panel_data(data_path, config, sample_n = sample_n)
  data <- panel$data
  metadata <- panel$metadata

  summarize_data(data, metadata, config)
  timer$lap("data_loading")

  # =============================================================================
  # PHASE 2: (g,t) PAIRS AND COVARIATES
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("PHASE 2: COVARIATE SETUP")
  log_message(paste(rep("=", 60), collapse = ""))

  # Get valid (g,t) pairs
  gt_pairs <- get_gt_pairs(data, config)

  # Get covariate information for all (g,t) pairs
  # This includes per-(g,t) covariate dimensions for the input projection architecture
  all_vars <- names(data)
  covariate_info <- get_covariate_info(gt_pairs, all_vars, config)
  all_covariates <- covariate_info$all_covariates
  log_message(sprintf("Total unique covariates (union): %d", length(all_covariates)))

  # Create covariate masks (for backward compatibility and validation)
  covariate_masks <- create_covariate_masks(gt_pairs, all_covariates, config)

  # Validate covariate selection rules
  validate_covariate_rules(config)

  timer$lap("covariate_setup")

  # =============================================================================
  # PHASE 3: CROSS-FITTING
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("PHASE 3: CROSS-FITTING")
  log_message(paste(rep("=", 60), collapse = ""))

  cf_results <- run_cross_fitting(
    data,
    gt_pairs,
    all_covariates,
    covariate_masks,
    config
  )

  # Validate cross-fitting results
  validate_cross_fitting(cf_results, config)

  timer$lap("cross_fitting")

  # =============================================================================
  # PHASE 4: ATT ESTIMATION
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("PHASE 4: ATT ESTIMATION")
  log_message(paste(rep("=", 60), collapse = ""))

  # Diagnose nuisance parameters
  nuisance_diag <- diagnose_nuisance(cf_results, config)

  # Compute all ATT(g,t) estimates
  att_results <- compute_all_att(cf_results, config)

  print_att_summary(att_results)

  timer$lap("att_estimation")

  # =============================================================================
  # PHASE 5: INFERENCE
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("PHASE 5: INFERENCE")
  log_message(paste(rep("=", 60), collapse = ""))

  # Add bootstrap inference
  att_results <- add_bootstrap_inference(att_results, config)

  # Test parallel trends
  pt_test <- test_parallel_trends(att_results, config)

  # Simple ATT
  simple_att <- compute_simple_att(att_results, config)

  timer$lap("inference")

  # =============================================================================
  # PHASE 6: AGGREGATION
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("PHASE 6: AGGREGATION")
  log_message(paste(rep("=", 60), collapse = ""))

  agg_results <- aggregate_all(att_results, config)

  print_aggregation_summary(agg_results)

  timer$lap("aggregation")

  # =============================================================================
  # PHASE 7: VISUALIZATION
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("PHASE 7: VISUALIZATION")
  log_message(paste(rep("=", 60), collapse = ""))

  output_dir <- file.path(project_root, "outputs/figures")
  figures <- generate_all_figures(att_results, agg_results, output_dir, config)

  timer$lap("visualization")

  # =============================================================================
  # COMPLETE
  # =============================================================================

  log_message("\n" %+% paste(rep("=", 60), collapse = "") %+% "\n")
  log_message("ESTIMATION COMPLETE")
  log_message(paste(rep("=", 60), collapse = ""))

  timer$report()

  # Return all results
  list(
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
    nuisance_diagnostics = nuisance_diag
  )
}


# String concatenation helper
`%+%` <- function(a, b) paste0(a, b)


# =============================================================================
# RUN
# =============================================================================

if (sys.nframe() == 0) {
  # Running as script

  # Parse command line arguments (optional)
  args <- commandArgs(trailingOnly = TRUE)

  sample_n <- NULL
  if (length(args) > 0) {
    sample_n <- as.integer(args[1])
    cat(sprintf("Running with sample size: %d\n", sample_n))
  }

  # Run estimation
  results <- run_estimation(
    outcome = "sfr_pc",
    sample_n = sample_n
  )

  # Save results
  output_path <- file.path(project_root, "outputs/estimation_results.rds")
  saveRDS(results, output_path)
  cat(sprintf("\nResults saved to: %s\n", output_path))
}
