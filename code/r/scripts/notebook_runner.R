# =============================================================================
# CDFI DiD Estimation - Notebook Runner
# =============================================================================
# Copy this script into your notebook and run cell by cell.
# Adjust paths and configurations as needed for your environment.
# =============================================================================

# =============================================================================
# CELL 1: SETUP - Paths and Dependencies
# =============================================================================

# Set your project root (adjust for your environment)
project_root <- "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"  # <-- CHANGE THIS
setwd(project_root)

# Install packages if needed (uncomment if first run)
# install.packages(c("data.table", "torch", "ggplot2", "R6"))
# torch::install_torch()  # Run once to install torch backend

# Load packages
library(data.table)
library(torch)
library(ggplot2)

# Source all modules
cat("Loading modules...\n")
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

cat("Setup complete.\n")

# =============================================================================
# CELL 2: CONFIGURATION
# =============================================================================
# Set QUICK_TEST = TRUE for fast testing with minimal settings
# Set QUICK_TEST = FALSE for full estimation

QUICK_TEST <- TRUE  # <-- CHANGE THIS FOR FULL RUN

if (QUICK_TEST) {
  cat("\n*** QUICK TEST MODE - using minimal settings ***\n\n")
  sample_n <- 10000        # Small sample
  n_folds <- 2L            # Minimum folds
  n_bootstrap <- 100L      # Fewer bootstrap reps
  epochs <- 5L             # Few epochs
  batch_size <- 512L       # Reasonable batch
} else {
  cat("\n*** FULL ESTIMATION MODE ***\n\n")
  sample_n <- NULL         # Full data
  n_folds <- 2L            # 2-fold cross-fitting
  n_bootstrap <- 1000L     # Full bootstrap
  epochs <- 100L           # Full training
  batch_size <- 512L       # Batch size
}

config <- create_config(

  # =========================================================================
  # OUTCOME VARIABLE
  # =========================================================================
  outcome = "sfr_pc",  # Without y_ prefix

  # =========================================================================
  # NEURAL NETWORK ARCHITECTURE
  # =========================================================================
  architecture = list(
    input_projection_dim = if (QUICK_TEST) 16L else 128L,
    shared_layers = if (QUICK_TEST) c(32L) else c(256L, 128L),
    outcome_head_layers = if (QUICK_TEST) c(16L) else c(64L),
    propensity_head_layers = if (QUICK_TEST) c(16L) else c(64L),
    activation = "relu",
    dropout = 0.2,
    layer_norm = TRUE,
    residual_connections = FALSE
  ),

  # =========================================================================
  # OPTIMIZER
  # =========================================================================
  optimizer = "adamw",
  optimizer_params = list(
    adamw = list(
      lr = 0.001,
      weight_decay = 0.01,
      betas = c(0.9, 0.999),
      eps = 1e-8
    ),
    lbfgs = list(
      lr = 1.0,
      max_iter = 20L,
      history_size = 100L,
      line_search_fn = "strong_wolfe"
    )
  ),

  # =========================================================================
  # LEARNING RATE SCHEDULER
  # =========================================================================
  scheduler = list(
    type = "cosine",
    T_max = epochs,
    eta_min = 1e-6,
    step_size = 30L,
    gamma = 0.1,
    patience = 10L,
    factor = 0.5,
    min_lr = 1e-6
  ),

  # =========================================================================
  # TRAINING
  # =========================================================================
  training = list(
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.1,
    shuffle = TRUE,

    early_stopping = list(
      enabled = TRUE,
      patience = if (QUICK_TEST) 3L else 15L,
      min_delta = 1e-4,
      monitor = "val_loss",
      restore_best_weights = TRUE
    ),

    gradient_clipping = list(
      enabled = TRUE,
      max_norm = 1.0
    )
  ),

  # =========================================================================
  # LOSS FUNCTION
  # =========================================================================
  loss = list(
    outcome_weight = 1.0,
    propensity_weight = 1.0,
    l1_penalty = 0.0,
    l2_penalty = 0.0
  ),

  # =========================================================================
  # CROSS-FITTING (DML)
  # =========================================================================
  cross_fitting = list(
    n_folds = n_folds,
    stratify_by = "cluster_county",
    seed = 42L
  ),

  # =========================================================================
  # PROPENSITY SCORE
  # =========================================================================
  propensity = list(
    min_ps = 0.001,
    max_ps = 0.999,
    trim = FALSE,
    trim_threshold = 0.01
  ),

  # =========================================================================
  # INFERENCE
  # =========================================================================
  inference = list(
    n_bootstrap = n_bootstrap,
    alpha = 0.05,
    uniform_bands = TRUE,
    pointwise_ci = TRUE,
    multiplier_dist = "normal",
    seed = 123L
  ),

  # =========================================================================
  # EVENT STUDY
  # =========================================================================
  event_study = list(
    pre_periods = 10L,
    post_periods = 10L,
    reference_period = -1L,
    weight_by_group_size = TRUE,
    drop_first_period = TRUE,
    drop_last_period = TRUE
  ),

  # =========================================================================
  # MONITORING & OUTPUT
  # =========================================================================
  monitoring = list(
    verbose = TRUE,
    print_every = if (QUICK_TEST) 1L else 10L,
    plot_loss = TRUE,
    save_checkpoints = FALSE,
    checkpoint_dir = "checkpoints",
    checkpoint_every = 10L,
    log_file = NULL,
    log_level = "INFO"
  ),

  # =========================================================================
  # COMPUTATIONAL
  # =========================================================================
  device = "cpu",
  seed = 42L,
  gc_every = 10L
)

# Print key settings
cat(sprintf("Configuration:\n"))
cat(sprintf("  Sample size: %s\n", if (is.null(sample_n)) "FULL" else format(sample_n, big.mark = ",")))
cat(sprintf("  Cross-fitting folds: %d\n", n_folds))
cat(sprintf("  Bootstrap replications: %d\n", n_bootstrap))
cat(sprintf("  Max epochs: %d\n", epochs))
cat(sprintf("  Batch size: %d\n", batch_size))

# Set random seed
set_seed(config$seed)

# =============================================================================
# CELL 3: LOAD DATA
# =============================================================================

data_path <- file.path(project_root, "data/analysis/final_analysis_dataset.csv")

# Check file exists
if (!file.exists(data_path)) {
  stop(sprintf("Data file not found: %s\nPlease download from Google Drive.", data_path))
}

cat("\nLoading data...\n")
t1 <- Sys.time()

panel <- load_panel_data(data_path, config, sample_n = sample_n)
data <- panel$data
metadata <- panel$metadata

t2 <- Sys.time()
cat(sprintf("Data loaded in %.1f seconds\n", as.numeric(difftime(t2, t1, units = "secs"))))

# Print summary
summarize_data(data, metadata, config)

# =============================================================================
# CELL 4: SETUP (g,t) PAIRS AND COVARIATES
# =============================================================================

cat("\nSetting up (g,t) pairs and covariates...\n")
t1 <- Sys.time()

# Get valid (g,t) pairs
gt_pairs <- get_gt_pairs(data, config)

# Get covariate information (per-(g,t) dimensions)
covariate_info <- get_covariate_info(gt_pairs, names(data), config)

# Create covariate masks (for validation)
covariate_masks <- create_covariate_masks(gt_pairs, covariate_info$all_covariates, config)

# Validate
validate_covariate_rules(config)

t2 <- Sys.time()
cat(sprintf("Setup complete in %.1f seconds\n", as.numeric(difftime(t2, t1, units = "secs"))))

cat(sprintf("\nSummary:\n"))
cat(sprintf("  (g,t) pairs: %d (%d pre, %d post)\n",
            nrow(gt_pairs), sum(gt_pairs$is_pre), sum(!gt_pairs$is_pre)))
cat(sprintf("  Covariates: %d (dims per (g,t): %d-%d)\n",
            length(covariate_info$all_covariates),
            min(covariate_info$dims_by_gt),
            max(covariate_info$dims_by_gt)))

# =============================================================================
# CELL 5: CROSS-FITTING (MAIN TRAINING)
# =============================================================================
# This is the computationally intensive step

cat("\n", rep("=", 60), "\n")
cat("CROSS-FITTING\n")
cat(rep("=", 60), "\n\n")

t1 <- Sys.time()

cf_results <- tryCatch({
  run_cross_fitting(
    data,
    gt_pairs,
    covariate_info$all_covariates,
    covariate_masks,
    config
  )
}, error = function(e) {
  cat(sprintf("\nERROR in cross-fitting: %s\n", e$message))
  cat("Stack trace:\n")
  print(sys.calls())
  stop(e)
})

t2 <- Sys.time()
training_time <- as.numeric(difftime(t2, t1, units = "secs"))
cat(sprintf("\nCross-fitting complete in %.1f seconds (%.1f minutes)\n",
            training_time, training_time/60))

# Validate results
validate_cross_fitting(cf_results, config)

# =============================================================================
# CELL 6: ATT ESTIMATION
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("ATT ESTIMATION\n")
cat(rep("=", 60), "\n\n")

t1 <- Sys.time()

# Diagnose nuisance parameters
nuisance_diag <- diagnose_nuisance(cf_results, config)

# Compute all ATT(g,t) estimates
att_results <- compute_all_att(cf_results, config)

# Print summary
print_att_summary(att_results)

t2 <- Sys.time()
cat(sprintf("\nATT estimation complete in %.1f seconds\n",
            as.numeric(difftime(t2, t1, units = "secs"))))

# =============================================================================
# CELL 7: INFERENCE (BOOTSTRAP)
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("INFERENCE\n")
cat(rep("=", 60), "\n\n")

t1 <- Sys.time()

# Add bootstrap inference
att_results <- add_bootstrap_inference(att_results, config)

# Test parallel trends
pt_test <- test_parallel_trends(att_results, config)

# Compute simple ATT
simple_att <- compute_simple_att(att_results, config)

t2 <- Sys.time()
cat(sprintf("\nInference complete in %.1f seconds\n",
            as.numeric(difftime(t2, t1, units = "secs"))))

# Print results
cat("\n--- Parallel Trends Test ---\n")
cat(sprintf("Mean pre-treatment ATT: %.4f (SE: %.4f)\n",
            pt_test$mean_att_pre, pt_test$se_mean_pre))
cat(sprintf("Test statistic: %.3f, p-value: %.4f\n",
            pt_test$test_stat, pt_test$p_value))
cat(sprintf("Reject parallel trends at 5%%: %s\n",
            ifelse(pt_test$reject, "YES", "NO")))

cat("\n--- Simple ATT (Weighted Average) ---\n")
cat(sprintf("ATT: %.4f (SE: %.4f)\n", simple_att$att, simple_att$se))
cat(sprintf("95%% CI: [%.4f, %.4f]\n", simple_att$ci_lower, simple_att$ci_upper))
cat(sprintf("p-value: %.4f\n", simple_att$p_value))

# =============================================================================
# CELL 8: AGGREGATION
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("AGGREGATION\n")
cat(rep("=", 60), "\n\n")

agg_results <- aggregate_all(att_results, config)

# Print summary
print_aggregation_summary(agg_results)

# View event study table
cat("\n--- Event Study Estimates ---\n")
es_cols <- c("event_time", "att", "se", "ci_lower", "ci_upper")
if ("uniform_lower" %in% names(agg_results$event_study$event_study)) {
  es_cols <- c(es_cols, "uniform_lower", "uniform_upper")
}
print(agg_results$event_study$event_study[, ..es_cols])

# =============================================================================
# CELL 9: VISUALIZATION
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("VISUALIZATION\n")
cat(rep("=", 60), "\n\n")

# Create output directory
output_dir <- file.path(project_root, "outputs/figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Generate event study plot
es_plot <- tryCatch({
  plot_event_study(
    agg_results$event_study,
    title = "Effect of CDFI Lending on Startup Formation Rate",
    subtitle = "Callaway & Sant'Anna (2021) DiD with Neural Network Nuisance Estimation",
    show_uniform_bands = TRUE,
    show_pointwise_ci = TRUE
  )
}, error = function(e) {
  cat(sprintf("Warning: Could not create plot: %s\n", e$message))
  NULL
})

if (!is.null(es_plot)) {
  # Display plot
  print(es_plot)

  # Save plot
  ggsave(
    file.path(output_dir, "event_study.png"),
    es_plot,
    width = 10, height = 6, dpi = 300
  )
  ggsave(
    file.path(output_dir, "event_study.pdf"),
    es_plot,
    width = 10, height = 6
  )

  cat(sprintf("Plots saved to: %s\n", output_dir))
}

# =============================================================================
# CELL 10: SAVE RESULTS
# =============================================================================

# Compile all results
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
  nuisance_diagnostics = nuisance_diag,
  training_time = training_time
)

# Save as RDS
output_path <- file.path(project_root, "outputs/estimation_results.rds")
saveRDS(results, output_path)
cat(sprintf("Results saved to: %s\n", output_path))

# =============================================================================
# CELL 11: QUICK RESULTS SUMMARY
# =============================================================================

cat("\n")
cat(rep("=", 60), "\n")
cat("ESTIMATION COMPLETE\n")
cat(rep("=", 60), "\n\n")

cat("KEY RESULTS:\n\n")

cat("1. Simple ATT (weighted average post-treatment):\n")
cat(sprintf("   ATT = %.4f, SE = %.4f, p = %.4f\n",
            simple_att$att, simple_att$se, simple_att$p_value))
cat(sprintf("   95%% CI: [%.4f, %.4f]\n\n",
            simple_att$ci_lower, simple_att$ci_upper))

cat("2. Parallel Trends:\n")
cat(sprintf("   Pre-treatment ATT = %.4f (should be ~0)\n", pt_test$mean_att_pre))
cat(sprintf("   p-value = %.4f (want > 0.05)\n\n", pt_test$p_value))

cat("3. Event Study:\n")
es <- agg_results$event_study$event_study
cat(sprintf("   Event times: %d to %d\n", min(es$event_time), max(es$event_time)))
cat(sprintf("   Pre-treatment mean: %.4f\n", mean(es[event_time < 0, att])))
cat(sprintf("   Post-treatment mean: %.4f\n", mean(es[event_time >= 0, att])))
if (!is.null(agg_results$event_study$uniform_bands)) {
  cat(sprintf("   Sup-t critical value: %.3f\n", agg_results$event_study$uniform_bands$sup_t_critical))
}

cat("\n")
cat(sprintf("Training time: %.1f minutes\n", training_time/60))
cat(sprintf("Figures saved to: %s\n", output_dir))
cat(sprintf("Results saved to: %s\n", output_path))

if (QUICK_TEST) {
  cat("\n*** This was a QUICK TEST run. Set QUICK_TEST = FALSE for full estimation. ***\n")
}
