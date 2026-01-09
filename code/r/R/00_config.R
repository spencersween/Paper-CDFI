#' Configuration Module for Callaway & Sant'Anna DiD Estimator
#'
#' Creates and validates configuration objects for the estimation pipeline.

#' Create configuration object for DiD estimation
#'
#' @param outcome Character. Outcome variable name (without y_ prefix)
#' @param ... Override any default parameters
#' @return List. Complete configuration object
create_config <- function(outcome = "sfr_pc", ...) {


  config <- list(
    # =========================================================================
    # DATA STRUCTURE
    # =========================================================================

    # Variable naming conventions
    outcome = outcome,
    outcome_var = paste0("y_", outcome),
    time_invariant_prefix = "X_",
    time_varying_prefix = "V_",
    baseline_outcome_prefix = "Wy_",

    # Panel identifiers
    id_var = "id",
    time_var = "time",
    group_var = "group",
    cluster_var = "cluster_county",
    treat_var = "i_treat",
    treat_post_var = "i_treat_post",

    # Analysis period
    panel_start = 1988L,
    panel_end = 2014L,
    analysis_start = 1996L,
    analysis_end = 2014L,

    # Time-varying covariate base variables (without year suffix)
    time_varying_base = c("V_totpop", "V_lenders_pc", "V_areadev"),

    # =========================================================================
    # NEURAL NETWORK ARCHITECTURE
    # =========================================================================

    architecture = list(
      # Per-(g,t) input projection dimension
      # Each (g,t) pair's covariates are projected to this common dimension
      input_projection_dim = 128L,

      # Shared encoder layers (after input projection)
      shared_layers = c(256L, 128L),

      # Task-specific head layers
      outcome_head_layers = c(64L, 32L),
      propensity_head_layers = c(64L, 32L),

      # Activation function: "relu", "leaky_relu", "elu", "gelu"
      activation = "relu",

      # Regularization
      dropout = 0.2,

      # Use LayerNorm instead of BatchNorm (better for variable input dimensions)
      layer_norm = TRUE,

      # Skip connections (residual)
      residual_connections = FALSE
    ),

    # =========================================================================
    # OPTIMIZATION
    # =========================================================================

    # Optimizer: "adamw" or "lbfgs"
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

    # Learning rate scheduler
    scheduler = list(
      # Type: "none", "step", "cosine", "reduce_on_plateau"
      type = "cosine",

      # Cosine annealing parameters
      T_max = 100L,
      eta_min = 1e-6,

      # Step scheduler parameters
      step_size = 30L,
      gamma = 0.1,

      # ReduceOnPlateau parameters
      patience = 10L,
      factor = 0.5,
      min_lr = 1e-6
    ),

    # =========================================================================
    # TRAINING
    # =========================================================================

    training = list(
      epochs = 100L,
      batch_size = 512L,
      validation_split = 0.2,
      shuffle = TRUE,

      # Early stopping
      early_stopping = list(
        enabled = TRUE,
        patience = 15L,
        min_delta = 1e-4,
        monitor = "val_loss",
        restore_best_weights = TRUE
      ),

      # Gradient clipping
      gradient_clipping = list(
        enabled = TRUE,
        max_norm = 1.0
      )
    ),

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================

    loss = list(
      # Task weights
      outcome_weight = 1.0,
      propensity_weight = 1.0,

      # L1/L2 regularization (additional to optimizer weight_decay)
      l1_penalty = 0.0,
      l2_penalty = 0.0
    ),

    # =========================================================================
    # CROSS-FITTING
    # =========================================================================

    cross_fitting = list(
      n_folds = 2L,
      stratify_by = "cluster_county",  # Split by cluster
      seed = 42L
    ),

    # =========================================================================
    # PROPENSITY SCORE
    # =========================================================================

    propensity = list(
      # Clamp propensity scores to avoid extreme weights
      min_ps = 0.001,
      max_ps = 0.999,

      # Trimming (drop observations with extreme propensity)
      trim = FALSE,
      trim_threshold = 0.01
    ),

    # =========================================================================
    # INFERENCE
    # =========================================================================

    inference = list(
      n_bootstrap = 1000L,
      alpha = 0.05,

      # Confidence band types
      uniform_bands = TRUE,
      pointwise_ci = TRUE,

      # Bootstrap method
      multiplier_dist = "normal",  # "normal" or "rademacher"

      seed = 42L
    ),

    # =========================================================================
    # EVENT STUDY
    # =========================================================================

    event_study = list(
      pre_periods = 10L,
      post_periods = 10L,
      reference_period = -1L,  # Normalize to t = -1
      weight_by_group_size = TRUE,

      # Drop endpoints
      drop_first_period = TRUE,
      drop_last_period = TRUE
    ),

    # =========================================================================
    # MONITORING & OUTPUT
    # =========================================================================

    monitoring = list(
      verbose = TRUE,
      print_every = 10L,
      plot_loss = TRUE,

      # Checkpoints
      save_checkpoints = FALSE,
      checkpoint_dir = "checkpoints",
      checkpoint_every = 10L,

      # Logging
      log_file = NULL,
      log_level = "INFO"
    ),

    # =========================================================================
    # COMPUTATIONAL
    # =========================================================================

    # Device: "cpu", "cuda", "mps", or "auto"
    device = "auto",

    # Random seed for reproducibility
    seed = 42L,

    # Parallel processing
    n_cores = max(1L, parallel::detectCores() - 1L),

    # Memory management
    gc_every = 10L  # Run garbage collection every N batches
  )

  # Override with user-supplied arguments
  user_args <- list(...)
  for (name in names(user_args)) {
    if (name %in% names(config)) {
      if (is.list(config[[name]]) && is.list(user_args[[name]])) {
        # Merge nested lists
        config[[name]] <- modifyList(config[[name]], user_args[[name]])
      } else {
        config[[name]] <- user_args[[name]]
      }
    } else {
      warning(sprintf("Unknown config parameter: %s", name))
    }
  }

  # Update outcome_var if outcome was changed
  config$outcome_var <- paste0("y_", config$outcome)

  # Validate configuration
  validate_config(config)

  class(config) <- c("did_config", "list")
  config
}


#' Validate configuration object
#'
#' @param config List. Configuration object
#' @return Invisible TRUE if valid, otherwise throws error
validate_config <- function(config) {


  # Check required fields
 required_fields <- c("outcome", "id_var", "time_var", "group_var", "cluster_var")
  for (field in required_fields) {
    if (is.null(config[[field]]) || config[[field]] == "") {
      stop(sprintf("Required config field '%s' is missing or empty", field))
    }
  }

  # Validate analysis period
  if (config$analysis_start < config$panel_start) {
    stop("analysis_start cannot be before panel_start")
  }
  if (config$analysis_end > config$panel_end) {
    stop("analysis_end cannot be after panel_end")
  }

  # Validate architecture
  if (config$architecture$input_projection_dim < 1) {
    stop("input_projection_dim must be >= 1")
  }
  if (length(config$architecture$shared_layers) == 0) {
    stop("At least one shared layer is required")
  }
  if (config$architecture$dropout < 0 || config$architecture$dropout >= 1) {
    stop("Dropout must be in [0, 1)")
  }

  # Validate optimizer
  if (!config$optimizer %in% c("adamw", "lbfgs")) {
    stop("Optimizer must be 'adamw' or 'lbfgs'")
  }

  # Validate training
  if (config$training$epochs < 1) {
    stop("epochs must be >= 1")
  }
  if (config$training$batch_size < 1) {
    stop("batch_size must be >= 1")
  }

  # Validate cross-fitting
  if (config$cross_fitting$n_folds < 2) {
    stop("n_folds must be >= 2")
  }

  # Validate propensity bounds
  if (config$propensity$min_ps <= 0 || config$propensity$max_ps >= 1) {
    stop("Propensity bounds must be in (0, 1)")
  }
  if (config$propensity$min_ps >= config$propensity$max_ps) {
    stop("min_ps must be < max_ps")
  }

  # Validate inference
  if (config$inference$n_bootstrap < 100) {
    warning("n_bootstrap < 100 may lead to unstable inference")
  }
  if (config$inference$alpha <= 0 || config$inference$alpha >= 1) {
    stop("alpha must be in (0, 1)")
  }

  # Validate event study
  if (config$event_study$pre_periods < 0 || config$event_study$post_periods < 0) {
    stop("pre_periods and post_periods must be >= 0")
  }

  invisible(TRUE)
}


#' Print configuration object
#'
#' @param x did_config object
#' @param ... Additional arguments (ignored)
#' @export
print.did_config <- function(x, ...) {
  cat("Callaway & Sant'Anna DiD Configuration\n")
  cat("=======================================\n\n")

cat(sprintf("Outcome: %s\n", x$outcome_var))
  cat(sprintf("Analysis period: %d-%d\n", x$analysis_start, x$analysis_end))
  cat(sprintf("Cluster variable: %s\n\n", x$cluster_var))

  cat("Neural Network:\n")
  cat(sprintf("  Input projection dim: %d\n", x$architecture$input_projection_dim))
  cat(sprintf("  Shared layers: %s\n", paste(x$architecture$shared_layers, collapse = " -> ")))
  cat(sprintf("  Activation: %s\n", x$architecture$activation))
  cat(sprintf("  Dropout: %.2f\n", x$architecture$dropout))
  cat(sprintf("  Layer norm: %s\n\n", x$architecture$layer_norm))

  cat("Training:\n")
  cat(sprintf("  Optimizer: %s (lr=%.4f)\n", x$optimizer,
              x$optimizer_params[[x$optimizer]]$lr))
  cat(sprintf("  Epochs: %d, Batch size: %d\n", x$training$epochs, x$training$batch_size))
  cat(sprintf("  Early stopping: %s (patience=%d)\n\n",
              x$training$early_stopping$enabled, x$training$early_stopping$patience))

  cat("Cross-fitting:\n")
  cat(sprintf("  Folds: %d (by %s)\n\n", x$cross_fitting$n_folds, x$cross_fitting$stratify_by))

  cat("Inference:\n")
  cat(sprintf("  Bootstrap samples: %d\n", x$inference$n_bootstrap))
  cat(sprintf("  Alpha: %.3f\n", x$inference$alpha))
  cat(sprintf("  Propensity clamp: [%.3f, %.3f]\n\n", x$propensity$min_ps, x$propensity$max_ps))

  cat("Event Study:\n")
  cat(sprintf("  Window: [-%d, +%d]\n", x$event_study$pre_periods, x$event_study$post_periods))
  cat(sprintf("  Reference period: %d\n", x$event_study$reference_period))

  invisible(x)
}


#' Get device for torch operations
#'
#' @param config did_config object
#' @return Character. Device name ("cpu", "cuda", or "mps")
get_device <- function(config) {
  if (config$device == "auto") {
    if (torch::cuda_is_available()) {
      return("cuda")
    } else if (torch::backends_mps_is_available()) {
      return("mps")
    } else {
      return("cpu")
    }
  }
  config$device
}


#' Set random seeds for reproducibility
#'
#' @param seed Integer. Random seed
set_seed <- function(seed) {
  set.seed(seed)
  torch::torch_manual_seed(seed)
  if (torch::cuda_is_available()) {
    torch::cuda_manual_seed_all(seed)
  }
}
