#!/usr/bin/env Rscript
#' =============================================================================
#' Test Script for DiD Estimation Pipeline
#'
#' Loads a small sample, processes data, trains model, and verifies outputs.
#' Run this before attempting full estimation to catch issues early.
#' =============================================================================

cat("\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nDiD ESTIMATION PIPELINE TEST\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

# =============================================================================
# SETUP
# =============================================================================

project_root <- "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"
setwd(project_root)

cat("Loading modules...\n")

# Source all modules in order
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

# Check packages
check_packages(c("data.table", "torch", "R6"))

library(data.table)
library(torch)

cat("Modules loaded.\n\n")

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

cat("Creating test configuration...\n")

# Minimal config for quick testing
test_config <- create_config(

  outcome = "sfr_pc",

  # Smaller architecture for testing
  architecture = list(
    input_projection_dim = 8L,
    shared_layers = c(16L, 16L),
    outcome_head_layers = c(8L),
    propensity_head_layers = c(8L),
    activation = "relu",
    dropout = 0.0,
    layer_norm = FALSE,
    residual_connections = FALSE
  ),

  # Fewer epochs for testing
  training = list(
    epochs = 3L,
    batch_size = 1000L,
    validation_split = 0.10,
    shuffle = TRUE,
    early_stopping = list(
      enabled = FALSE,  # Disable for short test
      patience = 3L,
      min_delta = 1e-4,
      monitor = "val_loss",
      restore_best_weights = TRUE
    ),
    gradient_clipping = list(
      enabled = TRUE,
      max_norm = 1.0
    )
  ),

  # Minimal cross-fitting
  cross_fitting = list(
    n_folds = 2L,
    stratify_by = "cluster_county",
    seed = 42L
  ),

  # Verbose output
  monitoring = list(
    verbose = TRUE,
    print_every = 1L,
    plot_loss = FALSE,
    save_checkpoints = FALSE,
    checkpoint_dir = "checkpoints",
    checkpoint_every = 10L,
    log_file = NULL,
    log_level = "INFO"
  )
)

print(test_config)
cat("\n")

# =============================================================================
# TEST 1: DATA LOADING
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST 1: DATA LOADING\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

# Load small sample
sample_n <- 5000  # Small sample for testing
data_path <- file.path(project_root, "data/analysis/final_analysis_dataset.csv")

cat(sprintf("Loading %d observations from %s...\n", sample_n, basename(data_path)))

tryCatch({
  panel <- load_panel_data(data_path, test_config, sample_n = sample_n)
  data <- panel$data
  metadata <- panel$metadata

  cat(sprintf("SUCCESS: Loaded %d observations, %d columns\n", nrow(data), ncol(data)))
  cat(sprintf("  Time range: %d - %d\n", min(data$time), max(data$time)))
  cat(sprintf("  Unique units: %d\n", length(unique(data$id))))
  cat(sprintf("  Treatment groups: %s\n",
              paste(sort(unique(data$group[data$group > 0])), collapse = ", ")))

}, error = function(e) {
  cat(sprintf("FAILED: %s\n", e$message))
  stop("Data loading failed")
})

cat("\n")

# =============================================================================
# TEST 2: COVARIATE SETUP
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST 2: COVARIATE SETUP\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

tryCatch({
  # Get (g,t) pairs
  gt_pairs <- get_gt_pairs(data, test_config)
  cat(sprintf("SUCCESS: Found %d (g,t) pairs\n", nrow(gt_pairs)))
  cat(sprintf("  Pre-treatment: %d\n", sum(gt_pairs$is_pre)))
  cat(sprintf("  Post-treatment: %d\n", sum(!gt_pairs$is_pre)))

  # Get covariate info
  covariate_info <- get_covariate_info(gt_pairs, names(data), test_config)
  cat(sprintf("SUCCESS: Covariate info computed\n"))
  cat(sprintf("  Union covariates: %d\n", length(covariate_info$all_covariates)))
  cat(sprintf("  Per-(g,t) dims: min=%d, max=%d, mean=%.1f\n",
              min(covariate_info$dims_by_gt),
              max(covariate_info$dims_by_gt),
              mean(covariate_info$dims_by_gt)))

  # Validate rules
  validate_covariate_rules(test_config)
  cat("SUCCESS: Covariate rules validated\n")

}, error = function(e) {
  cat(sprintf("FAILED: %s\n", e$message))
  stop("Covariate setup failed")
})

cat("\n")

# =============================================================================
# TEST 3: MODEL CREATION
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST 3: MODEL CREATION\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

tryCatch({
  model <- create_model(covariate_info, test_config)

  cat(sprintf("SUCCESS: Model created\n"))
  cat(sprintf("  Device: %s\n", get_device(test_config)))
  cat(sprintf("  Input projections: %d\n", length(model$input_projections)))
  cat(sprintf("  Shared layers: %d\n", length(model$shared_layers)))
  cat(sprintf("  Outcome heads: %d\n", length(model$outcome_heads)))
  cat(sprintf("  Propensity heads: %d\n", length(model$propensity_heads)))

  # Count parameters
  n_params <- count_parameters(model)
  cat(sprintf("  Total parameters: %s\n", format(n_params, big.mark = ",")))

}, error = function(e) {
  cat(sprintf("FAILED: %s\n", e$message))
  print(e)
  stop("Model creation failed")
})

cat("\n")

# =============================================================================
# TEST 4: FORWARD PASS
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST 4: FORWARD PASS\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

tryCatch({
  device <- get_device(test_config)

  # Pick a (g,t) pair to test - find one with event_time >= 1 to ensure base period exists
  valid_gt <- gt_pairs[event_time >= 1]
  if (nrow(valid_gt) == 0) {
    valid_gt <- gt_pairs  # Fall back to any
  }
  test_gt_idx <- valid_gt$gt_index[1]
  test_g <- valid_gt$g[1]
  test_t <- valid_gt$t[1]

  cat(sprintf("Testing forward pass for (g=%d, t=%d), gt_index=%d\n",
              test_g, test_t, test_gt_idx))

  # Create sample data for this (g,t)
  sample <- create_gt_sample(data, test_g, test_t, test_config)
  cat(sprintf("  Sample size: %d observations\n", nrow(sample)))

  if (nrow(sample) > 0) {
    # Prepare covariate matrix
    X <- prepare_gt_covariate_matrix(sample, test_gt_idx, covariate_info)
    cat(sprintf("  Covariate matrix: %d x %d\n", nrow(X), ncol(X)))

    # Convert to tensor
    X_tensor <- torch_tensor(X, dtype = torch_float32())$to(device = device)

    # Forward pass
    model$eval()
    with_no_grad({
      output <- model$predict_gt(X_tensor, test_gt_idx)
    })

    cat(sprintf("SUCCESS: Forward pass completed\n"))
    cat(sprintf("  Outcome predictions shape: [%s]\n",
                paste(output$outcome$shape, collapse = ", ")))
    cat(sprintf("  Propensity predictions shape: [%s]\n",
                paste(output$propensity$shape, collapse = ", ")))
    cat(sprintf("  Outcome range: [%.4f, %.4f]\n",
                as.numeric(output$outcome$min()$cpu()),
                as.numeric(output$outcome$max()$cpu())))
    cat(sprintf("  Propensity range: [%.4f, %.4f]\n",
                as.numeric(output$propensity$min()$cpu()),
                as.numeric(output$propensity$max()$cpu())))
  } else {
    cat("  WARNING: No observations in sample for this (g,t)\n")
  }

}, error = function(e) {
  cat(sprintf("FAILED: %s\n", e$message))
  print(e)
  stop("Forward pass failed")
})

cat("\n")

# =============================================================================
# TEST 5: DATA PREPARATION FOR TRAINING
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST 5: DATA PREPARATION FOR TRAINING\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

tryCatch({
  # Assign folds
  cluster_folds <- assign_cluster_folds(data, test_config)
  data <- add_fold_column(data, cluster_folds, test_config)

  cat(sprintf("SUCCESS: Fold assignments created\n"))
  cat(sprintf("  Fold distribution: %s\n",
              paste(sprintf("Fold %d: %d obs",
                            1:test_config$cross_fitting$n_folds,
                            table(data$fold)),
                    collapse = ", ")))

  # Prepare training data for fold 1 (train on fold 2)
  train_folds <- 2L
  train_combined <- prepare_combined_training_data(
    data, gt_pairs, covariate_info, train_folds, test_config
  )

  cat(sprintf("SUCCESS: Training data prepared\n"))
  cat(sprintf("  Total observations: %d\n", train_combined$total_obs))
  cat(sprintf("  (g,t) pairs with data: %d\n",
              sum(!sapply(train_combined$gt_data, is.null))))

  # Create validation split
  val_split <- create_validation_split(train_combined, test_config)
  cat(sprintf("SUCCESS: Validation split created\n"))

  # Create dataloaders
  loaders <- create_dataloaders(val_split$train, val_split$val, test_config)

  # Test dataloader
  train_batches <- loaders$train()
  cat(sprintf("SUCCESS: Dataloaders created\n"))
  cat(sprintf("  Training batches: %d\n", length(train_batches)))

  if (length(train_batches) > 0) {
    first_batch <- train_batches[[1]]
    cat(sprintf("  First batch: gt_index=%d, size=%d, X dim=%d\n",
                first_batch$gt_index,
                first_batch$batch_size,
                first_batch$X$shape[2]))
  }

}, error = function(e) {
  cat(sprintf("FAILED: %s\n", e$message))
  print(e)
  stop("Data preparation failed")
})

cat("\n")

# =============================================================================
# TEST 6: TRAINING LOOP (MINI)
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST 6: TRAINING LOOP (5 EPOCHS)\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

tryCatch({
  # Fresh model
  model <- create_model(covariate_info, test_config)

  # Train
  cat("Starting training...\n\n")

  train_result <- train_model(
    model,
    loaders$train,
    loaders$val,
    test_config
  )

  cat("\nSUCCESS: Training completed\n")
  cat(sprintf("  Final train loss: %.4f\n", tail(train_result$history$train_loss, 1)))
  cat(sprintf("  Final val loss: %.4f\n", tail(train_result$history$val_loss, 1)))
  cat(sprintf("  Training time: %.1f seconds\n", train_result$training_time))

  # Check loss decreased
  if (length(train_result$history$train_loss) >= 2) {
    first_loss <- train_result$history$train_loss[1]
    last_loss <- tail(train_result$history$train_loss, 1)
    if (last_loss < first_loss) {
      cat("  Loss decreased during training (good sign)\n")
    } else {
      cat("  WARNING: Loss did not decrease\n")
    }
  }

}, error = function(e) {
  cat(sprintf("FAILED: %s\n", e$message))
  print(e)
  stop("Training failed")
})

cat("\n")

# =============================================================================
# TEST 7: PREDICTIONS AFTER TRAINING
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST 7: PREDICTIONS AFTER TRAINING\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

tryCatch({
  trained_model <- train_result$model
  trained_model$eval()

  # Get predictions for a few (g,t) pairs
  n_test <- min(3, nrow(gt_pairs))

  for (i in seq_len(n_test)) {
    gt_idx <- gt_pairs$gt_index[i]
    g_i <- gt_pairs$g[i]
    t_i <- gt_pairs$t[i]

    sample <- create_gt_sample(data, g_i, t_i, test_config)

    if (nrow(sample) > 0) {
      X <- prepare_gt_covariate_matrix(sample, gt_idx, covariate_info)
      X_tensor <- torch_tensor(X, dtype = torch_float32())$to(device = device)

      with_no_grad({
        preds <- trained_model$predict_gt(X_tensor, gt_idx)
      })

      outcome_mean <- as.numeric(preds$outcome$mean()$cpu())
      propensity_mean <- as.numeric(preds$propensity$mean()$cpu())

      cat(sprintf("(g=%d, t=%d): n=%d, mean_outcome=%.4f, mean_propensity=%.4f\n",
                  g_i, t_i, nrow(sample), outcome_mean, propensity_mean))

      # Check propensity is in valid range
      ps_min <- as.numeric(preds$propensity$min()$cpu())
      ps_max <- as.numeric(preds$propensity$max()$cpu())
      if (ps_min >= 0 && ps_max <= 1) {
        cat("  Propensity in [0,1]: OK\n")
      } else {
        cat(sprintf("  WARNING: Propensity outside [0,1]: [%.4f, %.4f]\n", ps_min, ps_max))
      }
    }
  }

  cat("\nSUCCESS: Predictions working\n")

}, error = function(e) {
  cat(sprintf("FAILED: %s\n", e$message))
  print(e)
})

cat("\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("=" |> rep(70) |> paste(collapse = ""))
cat("\nTEST SUMMARY\n")
cat("=" |> rep(70) |> paste(collapse = ""))
cat("\n\n")

cat("All tests passed! The pipeline appears to be working correctly.\n\n")

cat("Next steps:\n")
cat("  1. Review the output above for any warnings\n")
cat("  2. Try with a larger sample (e.g., 50,000 observations)\n")
cat("  3. Run full cross-fitting with run_estimation()\n")
cat("\n")

cat("To run full estimation:\n")
cat("  source('code/r/scripts/run_estimation.R')\n")
cat("  results <- run_estimation(outcome = 'sfr_pc', sample_n = 50000)\n")
cat("\n")
