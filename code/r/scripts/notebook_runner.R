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
project_root <- "/path/to/your/project"  # <-- CHANGE THIS
setwd(project_root)

# Install packages if needed (uncomment if first run)
# install.packages(c("data.table", "torch", "ggplot2", "R6"))
# torch::install_torch()  # Run once to install torch backend

# Load packages
library(data.table)
library(torch)
library(ggplot2)

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

cat("Setup complete.\n")

# =============================================================================
# CELL 2: CONFIGURATION
# =============================================================================
# Modify these settings based on your needs

config <- create_config(

  # =========================================================================
  # OUTCOME VARIABLE
  # =========================================================================
  outcome = "sfr_pc",  # Without y_ prefix

  # =========================================================================
  # NEURAL NETWORK ARCHITECTURE
  # =========================================================================
  architecture = list(
    input_projection_dim = 128L,         # Dimension after per-(g,t) input projection
    shared_layers = c(256L, 128L),       # Shared encoder layers
    outcome_head_layers = c(64L, 32L),   # Outcome regression head layers
    propensity_head_layers = c(64L, 32L),# Propensity score head layers
    activation = "relu",                  # "relu", "leaky_relu", "elu", "gelu"
    dropout = 0.2,                        # Dropout rate [0, 1)
    layer_norm = TRUE,                    # Use LayerNorm (better for variable inputs)
    residual_connections = FALSE          # Skip connections
  ),

  # =========================================================================
  # OPTIMIZER
  # =========================================================================
  optimizer = "adamw",  # "adamw" or "lbfgs"
  optimizer_params = list(
    adamw = list(
      lr = 0.001,                         # Learning rate
      weight_decay = 0.01,                # L2 regularization
      betas = c(0.9, 0.999),              # Adam betas
      eps = 1e-8                          # Numerical stability
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
    type = "cosine",                      # "none", "step", "cosine", "reduce_on_plateau"
    T_max = 100L,                         # Cosine: period length
    eta_min = 1e-6,                       # Cosine: minimum LR
    step_size = 30L,                      # Step: epochs between decay
    gamma = 0.1,                          # Step: decay factor
    patience = 10L,                       # Plateau: epochs before decay
    factor = 0.5,                         # Plateau: decay factor
    min_lr = 1e-6                         # Plateau: minimum LR
  ),

  # =========================================================================
  # TRAINING
  # =========================================================================
  training = list(
    epochs = 100L,                        # Maximum epochs
    batch_size = 512L,                    # Batch size
    validation_split = 0.2,               # Internal validation split
    shuffle = TRUE,                       # Shuffle training data

    early_stopping = list(
      enabled = TRUE,
      patience = 15L,                     # Epochs to wait for improvement
      min_delta = 1e-4,                   # Minimum improvement threshold
      monitor = "val_loss",               # Metric to monitor
      restore_best_weights = TRUE         # Restore best model at end
    ),

    gradient_clipping = list(
      enabled = TRUE,
      max_norm = 1.0                      # Max gradient norm
    )
  ),

  # =========================================================================
  # LOSS FUNCTION
  # =========================================================================
  loss = list(
    outcome_weight = 1.0,                 # Weight for outcome regression loss
    propensity_weight = 1.0,              # Weight for propensity score loss
    l1_penalty = 0.0,                     # Additional L1 regularization
    l2_penalty = 0.0                      # Additional L2 regularization
  ),

  # =========================================================================
  # CROSS-FITTING (DML)
  # =========================================================================
  cross_fitting = list(
    n_folds = 2L,                         # K for K-fold (2 is fastest, 5 is standard)
    stratify_by = "cluster_county",       # Split along cluster variable
    seed = 42L                            # Reproducibility
  ),

  # =========================================================================
  # PROPENSITY SCORE
  # =========================================================================
  propensity = list(
    min_ps = 0.001,                       # Lower clamp for propensity scores
    max_ps = 0.999,                       # Upper clamp for propensity scores
    trim = FALSE,                         # Drop extreme propensity observations
    trim_threshold = 0.01                 # Trimming threshold
  ),

  # =========================================================================
  # INFERENCE
  # =========================================================================
  inference = list(
    n_bootstrap = 1000L,                  # Bootstrap replications
    alpha = 0.05,                         # Significance level (95% CI)
    uniform_bands = TRUE,                 # Compute uniform confidence bands
    pointwise_ci = TRUE,                  # Compute pointwise CIs
    multiplier_dist = "normal",           # "normal" or "rademacher"
    seed = 123L                           # Bootstrap seed
  ),

  # =========================================================================
  # EVENT STUDY
  # =========================================================================
  event_study = list(
    pre_periods = 10L,                    # Periods before treatment
    post_periods = 10L,                   # Periods after treatment
    reference_period = -1L,               # Reference period for normalization
    weight_by_group_size = TRUE,          # Weight by number of treated units
    drop_first_period = TRUE,             # Drop first event time
    drop_last_period = TRUE               # Drop last event time
  ),

  # =========================================================================
  # MONITORING & OUTPUT
  # =========================================================================
  monitoring = list(
    verbose = TRUE,                       # Print progress
    print_every = 10L,                    # Print every N epochs
    plot_loss = TRUE,                     # Plot loss curves
    save_checkpoints = FALSE,             # Save model checkpoints
    checkpoint_dir = "checkpoints",       # Checkpoint directory
    checkpoint_every = 10L,               # Save every N epochs
    log_file = NULL,                      # Log file path (NULL = no file)
    log_level = "INFO"                    # "DEBUG", "INFO", "WARNING", "ERROR"
  ),

  # =========================================================================
  # COMPUTATIONAL
  # =========================================================================
  device = "auto",                        # "cpu", "cuda", "mps", or "auto"
  seed = 42L,                             # Global random seed
  gc_every = 10L                          # Garbage collection frequency
)

# Print configuration
print(config)

# Set random seed
set_seed(config$seed)

# =============================================================================
# CELL 3: LOAD DATA
# =============================================================================

data_path <- file.path(project_root, "data/analysis/final_analysis_dataset.csv")

# Load data (set sample_n for testing, NULL for full data)
sample_n <- NULL  # Set to e.g. 50000 for testing, NULL for full dataset

cat("Loading data...\n")
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

cat("Setting up (g,t) pairs and covariates...\n")
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

cat("\n" , rep("=", 60), "\n")
cat("CROSS-FITTING\n")
cat(rep("=", 60), "\n\n")

t1 <- Sys.time()

cf_results <- run_cross_fitting(
  data,
  gt_pairs,
  covariate_info$all_covariates,
  covariate_masks,
  config
)

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
print(agg_results$event_study$event_study[, .(event_time, att, se, ci_lower, ci_upper, uniform_lower, uniform_upper)])

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
es_plot <- plot_event_study(
  agg_results$event_study,
  title = "Effect of CDFI Lending on Startup Formation Rate",
  subtitle = "Callaway & Sant'Anna (2021) DiD with Neural Network Nuisance Estimation",
  show_uniform_bands = TRUE,
  show_pointwise_ci = TRUE
)

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
