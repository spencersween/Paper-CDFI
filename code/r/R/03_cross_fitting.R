#' Cross-Fitting Module for DiD Estimation
#'
#' Implements K-fold cross-fitting for double machine learning (DML).
#' Splits data by cluster to ensure proper out-of-sample predictions.

#' Assign cross-fitting folds to clusters
#'
#' Assigns each cluster to a fold for cross-fitting. This ensures that
#' all observations within a cluster are in the same fold.
#'
#' @param data data.table. Panel data
#' @param config Configuration object
#' @return data.table with fold assignments (one row per cluster)
assign_cluster_folds <- function(data, config) {

  set.seed(config$cross_fitting$seed)

  # Get unique clusters
  cluster_var <- config$cross_fitting$stratify_by
  clusters <- unique(data[[cluster_var]])
  n_clusters <- length(clusters)
  n_folds <- config$cross_fitting$n_folds

  log_message(sprintf("Assigning %d clusters to %d folds", n_clusters, n_folds))

  # Random assignment to folds
  fold_assignment <- sample(rep(1:n_folds, length.out = n_clusters))

  cluster_folds <- data.table::data.table(
    cluster = clusters,
    fold = fold_assignment
  )
  data.table::setnames(cluster_folds, "cluster", cluster_var)

  # Summary
  fold_counts <- table(cluster_folds$fold)
  log_message(sprintf("Clusters per fold: %s",
                      paste(sprintf("Fold %s: %d", names(fold_counts), as.integer(fold_counts)),
                            collapse = ", ")))

  cluster_folds
}


#' Add fold assignments to data
#'
#' @param data data.table. Panel data
#' @param cluster_folds data.table. Cluster-fold mapping
#' @param config Configuration object
#' @return data.table with fold column added
add_fold_column <- function(data, cluster_folds, config) {

  cluster_var <- config$cross_fitting$stratify_by

  # Merge fold assignments
  data <- merge(
    data,
    cluster_folds,
    by = cluster_var,
    all.x = TRUE
  )

  # Check for missing folds
  n_missing <- sum(is.na(data$fold))
  if (n_missing > 0) {
    warning(sprintf("%d observations have missing fold assignments", n_missing))
  }

  data
}


#' Get training and validation indices for a fold
#'
#' @param data data.table with fold column
#' @param fold Integer. Fold to hold out for validation
#' @param config Configuration object
#' @return List with train_idx and val_idx
get_fold_indices <- function(data, fold, config) {

  val_idx <- which(data$fold == fold)
  train_idx <- which(data$fold != fold)

  list(
    train_idx = train_idx,
    val_idx = val_idx,
    n_train = length(train_idx),
    n_val = length(val_idx)
  )
}


#' Prepare training data for a single (g,t) pair
#'
#' Creates the training dataset for outcome regression and propensity estimation.
#' Returns (g,t)-specific covariate matrix (not a union matrix with masking).
#'
#' @param data data.table. Full panel data with fold column
#' @param g Integer. Treatment group
#' @param t Integer. Time period
#' @param gt_index Integer. Index of this (g,t) pair
#' @param covariate_info List. From get_covariate_info()
#' @param config Configuration object
#' @return List with X, delta_y, D for this (g,t)
prepare_gt_training_data <- function(data, g, t, gt_index, covariate_info, config) {

  # Get estimation sample
  sample <- create_gt_sample(data, g, t, config)

  if (nrow(sample) == 0) {
    return(NULL)
  }

  # Prepare (g,t)-specific covariate matrix (only relevant covariates)
  X <- prepare_gt_covariate_matrix(sample, gt_index, covariate_info)

  # Get cluster IDs for cluster-based validation split
  cluster_var <- config$cross_fitting$stratify_by
  cluster_ids <- if (cluster_var %in% names(sample)) sample[[cluster_var]] else NULL

  list(
    X = X,
    delta_y = sample$delta_y,
    D = sample$D,
    fold = sample$fold,
    ids = sample[[config$id_var]],
    cluster_ids = cluster_ids,
    n = nrow(sample),
    gt_index = gt_index,
    covariate_dim = ncol(X)
  )
}


#' Prepare full training dataset across all (g,t) pairs
#'
#' Combines data from all (g,t) pairs. Since each (g,t) has different
#' covariate dimensions, we store them separately by (g,t) index.
#'
#' @param data data.table. Full panel data with fold column
#' @param gt_pairs data.table. Valid (g,t) pairs
#' @param covariate_info List. From get_covariate_info()
#' @param train_folds Integer vector. Folds to include in training
#' @param config Configuration object
#' @return List with per-(g,t) training data
prepare_combined_training_data <- function(data, gt_pairs, covariate_info,
                                            train_folds, config) {

  log_message(sprintf("Preparing training data for folds: %s",
                      paste(train_folds, collapse = ", ")))

  # Filter data to training folds
  train_data <- data[fold %in% train_folds]

  # Collect data separately by (g,t) pair
  gt_data_list <- vector("list", nrow(gt_pairs))
  total_obs <- 0

  for (i in seq_len(nrow(gt_pairs))) {
    g_i <- gt_pairs$g[i]
    t_i <- gt_pairs$t[i]
    gt_idx <- gt_pairs$gt_index[i]

    gt_data <- prepare_gt_training_data(
      train_data, g_i, t_i, gt_idx, covariate_info, config
    )

    if (!is.null(gt_data) && gt_data$n > 0) {
      gt_data_list[[gt_idx]] <- gt_data
      total_obs <- total_obs + gt_data$n
    }
  }

  log_message(sprintf("Combined training data: %d total observations across %d (g,t) pairs",
                      total_obs, sum(!sapply(gt_data_list, is.null))))

  list(
    gt_data = gt_data_list,
    total_obs = total_obs,
    covariate_info = covariate_info
  )
}


#' Create data loaders for training
#'
#' Creates torch dataloaders that handle (g,t)-specific data correctly.
#'
#' @param train_data List. From prepare_combined_training_data
#' @param val_data List. From prepare_combined_training_data
#' @param config Configuration object
#' @return List with train and val dataloaders
create_dataloaders <- function(train_data, val_data, config) {

  device <- get_device(config)
  batch_size <- config$training$batch_size

  # Create custom dataset that returns (X, delta_y, D, gt_index) tuples
  # We'll batch by (g,t) pair for efficiency
  train_loader <- create_gt_dataloader(train_data, batch_size, shuffle = TRUE, device)
  val_loader <- create_gt_dataloader(val_data, batch_size, shuffle = FALSE, device)

  list(
    train = train_loader,
    val = val_loader
  )
}


#' Create (g,t)-aware dataloader
#'
#' Returns batches organized by (g,t) pair for efficient training.
#'
#' @param combined_data List from prepare_combined_training_data
#' @param batch_size Integer. Batch size
#' @param shuffle Logical. Whether to shuffle
#' @param device Character. torch device
#' @return Generator function that yields batches
create_gt_dataloader <- function(combined_data, batch_size, shuffle, device) {

  gt_data_list <- combined_data$gt_data
  valid_gt_indices <- which(!sapply(gt_data_list, is.null))

  # Create a generator that cycles through (g,t) pairs
  function() {
    # Shuffle order of (g,t) pairs if requested
    gt_order <- if (shuffle) sample(valid_gt_indices) else valid_gt_indices

    batches <- list()

    for (gt_idx in gt_order) {
      gt_data <- gt_data_list[[gt_idx]]
      n <- gt_data$n

      # Shuffle within (g,t) if requested
      obs_order <- if (shuffle) sample(n) else seq_len(n)

      # Create batches for this (g,t)
      n_batches <- ceiling(n / batch_size)

      for (b in seq_len(n_batches)) {
        start_idx <- (b - 1) * batch_size + 1
        end_idx <- min(b * batch_size, n)
        batch_indices <- obs_order[start_idx:end_idx]

        X_batch <- gt_data$X[batch_indices, , drop = FALSE]
        delta_y_batch <- gt_data$delta_y[batch_indices]
        D_batch <- gt_data$D[batch_indices]

        # Convert to tensors
        X_tensor <- torch::torch_tensor(X_batch, dtype = torch::torch_float32())$to(device = device)
        delta_y_tensor <- torch::torch_tensor(delta_y_batch, dtype = torch::torch_float32())$to(device = device)
        D_tensor <- torch::torch_tensor(D_batch, dtype = torch::torch_float32())$to(device = device)

        batches[[length(batches) + 1]] <- list(
          X = X_tensor,
          delta_y = delta_y_tensor,
          D = D_tensor,
          gt_index = gt_idx,
          batch_size = length(batch_indices)
        )
      }
    }

    # Optionally shuffle the order of batches across (g,t) pairs
    if (shuffle) {
      batches <- batches[sample(length(batches))]
    }

    batches
  }
}


#' Run full cross-fitting procedure
#'
#' Performs K-fold cross-fitting for nuisance parameter estimation.
#'
#' @param data data.table. Full panel data
#' @param gt_pairs data.table. Valid (g,t) pairs
#' @param all_covariates Character vector. All covariate names (union)
#' @param covariate_masks List. From create_covariate_masks() (for backward compat)
#' @param config Configuration object
#' @return List with out-of-fold predictions for all observations
run_cross_fitting <- function(data, gt_pairs, all_covariates,
                               covariate_masks, config) {

  n_folds <- config$cross_fitting$n_folds
  device <- get_device(config)

  log_message(sprintf("Starting %d-fold cross-fitting", n_folds))

  # Get covariate info (per-(g,t) dimensions and names)
  covariate_info <- get_covariate_info(gt_pairs, names(data), config)

  # Assign folds to clusters
  cluster_folds <- assign_cluster_folds(data, config)
  data <- add_fold_column(data, cluster_folds, config)

  # Initialize storage for out-of-fold predictions
  # We need predictions for each (g,t) pair for each observation
  n_obs <- nrow(data)
  n_gt <- nrow(gt_pairs)

  oof_outcome <- matrix(NA_real_, nrow = n_obs, ncol = n_gt)
  oof_propensity <- matrix(NA_real_, nrow = n_obs, ncol = n_gt)

  # Store trained models
  fold_models <- list()

  # Cross-fitting loop
  for (fold in seq_len(n_folds)) {

    log_message(sprintf("\n=== Fold %d/%d ===", fold, n_folds))

    # Get fold indices
    fold_idx <- get_fold_indices(data, fold, config)
    train_folds <- setdiff(1:n_folds, fold)

    # Prepare training data (per-(g,t) structure)
    train_combined <- prepare_combined_training_data(
      data, gt_pairs, covariate_info, train_folds, config
    )

    # Create internal validation split for early stopping BY CLUSTER
    # (not by unit, to avoid leakage within clusters)
    val_combined <- create_validation_split(train_combined, data, train_folds, config)
    train_final <- val_combined$train
    val_final <- val_combined$val

    # Create data loaders
    loaders <- create_dataloaders(train_final, val_final, config)

    # Create model with per-(g,t) input projections
    model <- create_model(covariate_info, config)

    # Train model
    train_result <- train_model(model, loaders$train, loaders$val, config)
    fold_models[[fold]] <- train_result$model

    # Get out-of-fold predictions
    log_message("Computing out-of-fold predictions...")

    oof_preds <- compute_oof_predictions(
      train_result$model,
      data[fold_idx$val_idx, ],
      gt_pairs,
      covariate_info,
      config
    )

    # Store predictions
    oof_outcome[fold_idx$val_idx, ] <- oof_preds$outcome
    oof_propensity[fold_idx$val_idx, ] <- oof_preds$propensity

    # Clean up
    gc(verbose = FALSE)
  }

  # Clamp propensity scores
  oof_propensity <- clamp(oof_propensity, config$propensity$min_ps, config$propensity$max_ps)

  log_message("Cross-fitting complete")

  list(
    outcome = oof_outcome,
    propensity = oof_propensity,
    data = data,
    gt_pairs = gt_pairs,
    covariate_info = covariate_info,
    fold_models = fold_models,
    cluster_folds = cluster_folds
  )
}


#' Create validation split from training data BY CLUSTER
#'
#' Splits training data by cluster ID to avoid leakage within clusters.
#' This is critical for proper inference - observations from the same cluster
#' should not appear in both training and validation.
#'
#' @param combined_data List from prepare_combined_training_data
#' @param data data.table. Full data with cluster information
#' @param train_folds Integer vector. Folds used for training
#' @param config Configuration object
#' @return List with train and val data
create_validation_split <- function(combined_data, data, train_folds, config) {

  val_split <- config$training$validation_split
  cluster_var <- config$cross_fitting$stratify_by
  gt_data_list <- combined_data$gt_data

  # Get unique clusters in training folds
  train_data <- data[fold %in% train_folds]
  train_clusters <- unique(train_data[[cluster_var]])
  n_train_clusters <- length(train_clusters)

  # Randomly select clusters for internal validation
  # Random each fold - no fixed seed
  n_val_clusters <- max(1, floor(n_train_clusters * val_split))
  perm_clusters <- sample(n_train_clusters)
  val_cluster_set <- train_clusters[perm_clusters[seq_len(n_val_clusters)]]

  log_message(sprintf("  Internal split: %d/%d clusters for train/val",
                      n_train_clusters - n_val_clusters, n_val_clusters))

  train_gt_data <- vector("list", length(gt_data_list))
  val_gt_data <- vector("list", length(gt_data_list))

  for (gt_idx in seq_along(gt_data_list)) {
    gt_data <- gt_data_list[[gt_idx]]

    if (is.null(gt_data)) next

    # Get cluster IDs for each observation in this (g,t) sample
    # We need to look up the cluster based on the unit IDs stored in gt_data
    # Note: gt_data should have ids stored from prepare_gt_training_data
    if (!"cluster_ids" %in% names(gt_data)) {
      # Fallback: if cluster IDs weren't stored, use random split (less ideal)
      n <- gt_data$n
      n_val <- max(1, floor(n * val_split))
      val_indices <- sample(n, n_val)
      train_indices <- setdiff(seq_len(n), val_indices)
    } else {
      # Use cluster-based split
      cluster_ids <- gt_data$cluster_ids
      val_mask <- cluster_ids %in% val_cluster_set
      train_indices <- which(!val_mask)
      val_indices <- which(val_mask)

      # Handle edge cases
      if (length(train_indices) == 0) train_indices <- seq_len(gt_data$n)
      if (length(val_indices) == 0) val_indices <- sample(gt_data$n, 1)
    }

    # Training split
    train_gt_data[[gt_idx]] <- list(
      X = gt_data$X[train_indices, , drop = FALSE],
      delta_y = gt_data$delta_y[train_indices],
      D = gt_data$D[train_indices],
      n = length(train_indices),
      gt_index = gt_idx,
      covariate_dim = gt_data$covariate_dim
    )

    # Validation split
    val_gt_data[[gt_idx]] <- list(
      X = gt_data$X[val_indices, , drop = FALSE],
      delta_y = gt_data$delta_y[val_indices],
      D = gt_data$D[val_indices],
      n = length(val_indices),
      gt_index = gt_idx,
      covariate_dim = gt_data$covariate_dim
    )
  }

  list(
    train = list(
      gt_data = train_gt_data,
      covariate_info = combined_data$covariate_info
    ),
    val = list(
      gt_data = val_gt_data,
      covariate_info = combined_data$covariate_info
    )
  )
}


#' Compute out-of-fold predictions for held-out data
#'
#' @param model Trained torch model with per-(g,t) input projections
#' @param val_data data.table. Validation data
#' @param gt_pairs data.table. (g,t) pairs
#' @param covariate_info List from get_covariate_info
#' @param config Configuration object
#' @return List with outcome and propensity matrices
compute_oof_predictions <- function(model, val_data, gt_pairs, covariate_info, config) {

  device <- get_device(config)
  model$eval()

  n_val <- nrow(val_data)
  n_gt <- nrow(gt_pairs)

  outcome_preds <- matrix(NA_real_, nrow = n_val, ncol = n_gt)
  propensity_preds <- matrix(NA_real_, nrow = n_val, ncol = n_gt)

  torch::with_no_grad({
    for (i in seq_len(n_gt)) {
      g_i <- gt_pairs$g[i]
      t_i <- gt_pairs$t[i]
      gt_idx <- gt_pairs$gt_index[i]

      # Get sample for this (g,t)
      sample_data <- create_gt_sample(val_data, g_i, t_i, config)

      if (nrow(sample_data) > 0) {
        # Prepare (g,t)-specific covariate matrix
        X <- prepare_gt_covariate_matrix(sample_data, gt_idx, covariate_info)

        # Convert to tensor
        X_tensor <- torch::torch_tensor(X, dtype = torch::torch_float32())$to(device = device)

        # Get predictions using (g,t)-specific head
        preds <- model$predict_gt(X_tensor, gt_idx)

        # Map back to validation indices
        val_ids <- val_data[[config$id_var]]
        sample_ids <- sample_data[[config$id_var]]

        for (j in seq_along(sample_ids)) {
          val_row <- which(val_ids == sample_ids[j])
          if (length(val_row) > 0) {
            outcome_preds[val_row[1], i] <- as.numeric(preds$outcome[j, 1]$cpu())
            propensity_preds[val_row[1], i] <- as.numeric(preds$propensity[j, 1]$cpu())
          }
        }
      }
    }
  })

  list(
    outcome = outcome_preds,
    propensity = propensity_preds
  )
}


#' Validate cross-fitting results
#'
#' @param cf_results List from run_cross_fitting
#' @param config Configuration object
#' @return Logical. TRUE if valid
validate_cross_fitting <- function(cf_results, config) {

  outcome <- cf_results$outcome
  propensity <- cf_results$propensity

  # Check for missing predictions
  n_missing_outcome <- sum(is.na(outcome))
  n_missing_propensity <- sum(is.na(propensity))

  if (n_missing_outcome > 0) {
    log_message(sprintf("Warning: %d missing outcome predictions (%.1f%%)",
                        n_missing_outcome, 100 * n_missing_outcome / length(outcome)),
                level = "WARNING")
  }

  if (n_missing_propensity > 0) {
    log_message(sprintf("Warning: %d missing propensity predictions (%.1f%%)",
                        n_missing_propensity, 100 * n_missing_propensity / length(propensity)),
                level = "WARNING")
  }

  # Check propensity score range
  ps_min <- min(propensity, na.rm = TRUE)
  ps_max <- max(propensity, na.rm = TRUE)
  log_message(sprintf("Propensity score range: [%.4f, %.4f]", ps_min, ps_max))

  if (ps_min < config$propensity$min_ps || ps_max > config$propensity$max_ps) {
    log_message("Propensity scores outside expected range - clamping applied",
                level = "WARNING")
  }

  TRUE
}
