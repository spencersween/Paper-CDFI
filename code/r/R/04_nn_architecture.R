#' Neural Network Architecture Module for DiD Estimation
#'
#' Implements multi-task neural network for joint nuisance parameter estimation.
#' Architecture: Per-(g,t) input projections -> Shared encoder -> Task-specific heads
#'
#' Key design: Each (g,t) pair has its own input projection layer that maps from
#' its specific covariate dimension to a common hidden dimension. This correctly
#' handles the different conditioning sets required by the DiD methodology.

#' Multi-task DiD Neural Network with Per-(g,t) Input Projections
#'
#' Architecture:
#' - Per-(g,t) input projection: Maps from (g,t)-specific covariate dim to common dim
#' - Shared encoder layers: Processes the common representation
#' - Outcome regression heads: One per (g,t) pair for E[DeltaY | X, D=0]
#' - Propensity score heads: One per (g,t) pair for P(G=g | X)
#'
#' @param covariate_dims Named integer vector. Covariate dimension for each (g,t) pair
#' @param n_gt_pairs Integer. Number of (g,t) pairs
#' @param config Configuration object
#' @return torch::nn_module
MultiTaskDiDNet <- torch::nn_module(
  classname = "MultiTaskDiDNet",

  initialize = function(covariate_dims, n_gt_pairs, config) {

    self$covariate_dims <- covariate_dims
    self$n_gt_pairs <- n_gt_pairs
    self$config <- config
    arch <- config$architecture

    # Common dimension for all (g,t) pairs after input projection
    self$common_dim <- arch$input_projection_dim

    # =========================================================================
    # PER-(g,t) INPUT PROJECTIONS
    # =========================================================================
    # Each (g,t) pair has its own linear layer mapping from its specific
    # covariate dimension to the common dimension

    self$input_projections <- torch::nn_module_list()

    for (gt_idx in seq_len(n_gt_pairs)) {
      input_dim <- covariate_dims[gt_idx]
      proj <- torch::nn_sequential(
        torch::nn_linear(input_dim, self$common_dim),
        get_activation(arch$activation)
      )
      self$input_projections$append(proj)
    }

    # =========================================================================
    # SHARED ENCODER
    # =========================================================================
    # Takes the common_dim representation and processes through shared layers

    shared_dims <- c(self$common_dim, arch$shared_layers)
    self$shared_layers <- torch::nn_module_list()

    for (i in seq_len(length(shared_dims) - 1)) {
      in_dim <- shared_dims[i]
      out_dim <- shared_dims[i + 1]

      # Linear layer
      self$shared_layers$append(torch::nn_linear(in_dim, out_dim))

      # Layer normalization (better than BatchNorm for variable inputs)
      if (arch$layer_norm) {
        self$shared_layers$append(torch::nn_layer_norm(out_dim))
      }

      # Activation
      self$shared_layers$append(get_activation(arch$activation))

      # Dropout
      if (arch$dropout > 0) {
        self$shared_layers$append(torch::nn_dropout(arch$dropout))
      }
    }

    # Output dimension of shared encoder
    shared_out_dim <- tail(arch$shared_layers, 1)

    # =========================================================================
    # OUTCOME REGRESSION HEADS
    # =========================================================================

    outcome_dims <- c(shared_out_dim, arch$outcome_head_layers, 1L)
    self$outcome_heads <- torch::nn_module_list()

    for (gt in seq_len(n_gt_pairs)) {
      head <- build_head(outcome_dims, arch, final_activation = NULL)
      self$outcome_heads$append(head)
    }

    # =========================================================================
    # PROPENSITY SCORE HEADS
    # =========================================================================

    propensity_dims <- c(shared_out_dim, arch$propensity_head_layers, 1L)
    self$propensity_heads <- torch::nn_module_list()

    for (gt in seq_len(n_gt_pairs)) {
      head <- build_head(propensity_dims, arch, final_activation = "sigmoid")
      self$propensity_heads$append(head)
    }

    # Log model info
    n_params <- count_parameters(self)
    log_message(sprintf("MultiTaskDiDNet initialized:"))
    log_message(sprintf("  Input dims per (g,t): min=%d, max=%d",
                        min(covariate_dims), max(covariate_dims)))
    log_message(sprintf("  Common projection dim: %d", self$common_dim))
    log_message(sprintf("  Shared layers: %s", paste(arch$shared_layers, collapse = " -> ")))
    log_message(sprintf("  (g,t) pairs: %d", n_gt_pairs))
    log_message(sprintf("  Total parameters: %s", format(n_params, big.mark = ",")))
  },

  forward = function(x_list, gt_indices) {
    # x_list: List of tensors, one per observation, with (g,t)-specific dimensions
    #         OR a single tensor if all observations in batch are same (g,t)
    # gt_indices: Integer vector indicating which (g,t) pair each observation belongs to

    batch_size <- length(gt_indices)
    device <- if (is.list(x_list)) x_list[[1]]$device else x_list$device

    # Initialize output tensors
    outcome_out <- torch::torch_zeros(batch_size, 1, device = device)
    propensity_out <- torch::torch_zeros(batch_size, 1, device = device)

    # Get unique (g,t) indices in this batch
    unique_gt <- unique(as.integer(gt_indices))

    for (gt_idx in unique_gt) {
      # Find observations for this (g,t)
      obs_mask <- which(gt_indices == gt_idx)

      if (length(obs_mask) == 0) next

      # Get the input tensor for these observations
      if (is.list(x_list)) {
        # x_list contains per-observation tensors
        x_gt <- torch::torch_stack(x_list[obs_mask])
      } else {
        # x_list is a single tensor (homogeneous batch)
        x_gt <- x_list[obs_mask, , drop = FALSE]
      }

      # Apply (g,t)-specific input projection
      h <- self$input_projections[[gt_idx]](x_gt)

      # Shared encoder forward pass
      n_layers <- length(self$shared_layers)
      for (i in seq_len(n_layers)) {
        h <- self$shared_layers[[i]](h)
      }

      # Task-specific heads
      outcome_gt <- self$outcome_heads[[gt_idx]](h)
      propensity_gt <- self$propensity_heads[[gt_idx]](h)

      # Store in output tensors
      outcome_out[obs_mask, ] <- outcome_gt
      propensity_out[obs_mask, ] <- propensity_gt
    }

    list(
      outcome = outcome_out,
      propensity = propensity_out
    )
  },

  # Forward pass for a specific (g,t) pair during TRAINING (keeps gradients)
  forward_gt = function(x, gt_index) {
    # x: Tensor (batch_size, covariate_dim_for_gt)
    # gt_index: Integer, which (g,t) to use (1-indexed)
    # NOTE: This method keeps gradients enabled for training

    # Apply (g,t)-specific input projection
    h <- self$input_projections[[gt_index]](x)

    # Shared encoder
    n_layers <- length(self$shared_layers)
    for (i in seq_len(n_layers)) {
      h <- self$shared_layers[[i]](h)
    }

    # Specific heads
    outcome <- self$outcome_heads[[gt_index]](h)
    propensity <- self$propensity_heads[[gt_index]](h)

    list(
      outcome = outcome,
      propensity = propensity
    )
  },

  # Predict for a specific (g,t) pair (for INFERENCE - disables gradients)
  predict_gt = function(x, gt_index) {
    # x: Tensor (batch_size, covariate_dim_for_gt)
    # gt_index: Integer, which (g,t) to use (1-indexed)

    self$eval()

    torch::with_no_grad({
      # Apply (g,t)-specific input projection
      h <- self$input_projections[[gt_index]](x)

      # Shared encoder
      n_layers <- length(self$shared_layers)
      for (i in seq_len(n_layers)) {
        h <- self$shared_layers[[i]](h)
      }

      # Specific heads
      outcome <- self$outcome_heads[[gt_index]](h)
      propensity <- self$propensity_heads[[gt_index]](h)

      list(
        outcome = outcome,
        propensity = propensity
      )
    })
  },

  # Get expected input dimension for a specific (g,t)
  get_input_dim = function(gt_index) {
    self$covariate_dims[gt_index]
  }
)


#' Build a task-specific head
#'
#' @param dims Integer vector. Layer dimensions
#' @param arch Architecture config
#' @param final_activation Character or NULL. Activation for final layer
#' @return torch::nn_sequential
build_head <- function(dims, arch, final_activation = NULL) {

  layers <- list()

  for (i in seq_len(length(dims) - 1)) {
    in_dim <- dims[i]
    out_dim <- dims[i + 1]

    # Linear layer
    layers[[length(layers) + 1]] <- torch::nn_linear(in_dim, out_dim)

    # Not the last layer
    if (i < length(dims) - 1) {
      # Activation
      layers[[length(layers) + 1]] <- get_activation(arch$activation)

      # Dropout (lighter in heads)
      if (arch$dropout > 0) {
        layers[[length(layers) + 1]] <- torch::nn_dropout(arch$dropout / 2)
      }
    }
  }

  # Final activation
  if (!is.null(final_activation)) {
    layers[[length(layers) + 1]] <- get_activation(final_activation)
  }

  do.call(torch::nn_sequential, layers)
}


#' Simpler single-task network for testing
#'
#' Estimates nuisance parameters for a single (g,t) pair.
#' Useful for debugging and validation.
#'
#' @param input_dim Integer. Covariate dimension
#' @param config Configuration object
#' @return torch::nn_module
SingleTaskDiDNet <- torch::nn_module(
  classname = "SingleTaskDiDNet",

  initialize = function(input_dim, config) {

    self$input_dim <- input_dim
    arch <- config$architecture

    # Input projection to common dim
    self$input_proj <- torch::nn_sequential(
      torch::nn_linear(input_dim, arch$input_projection_dim),
      get_activation(arch$activation)
    )

    # Shared layers
    shared_dims <- c(arch$input_projection_dim, arch$shared_layers)
    layers <- list()

    for (i in seq_len(length(shared_dims) - 1)) {
      layers[[length(layers) + 1]] <- torch::nn_linear(shared_dims[i], shared_dims[i + 1])
      if (arch$layer_norm) {
        layers[[length(layers) + 1]] <- torch::nn_layer_norm(shared_dims[i + 1])
      }
      layers[[length(layers) + 1]] <- get_activation(arch$activation)
      if (arch$dropout > 0) {
        layers[[length(layers) + 1]] <- torch::nn_dropout(arch$dropout)
      }
    }

    self$shared <- do.call(torch::nn_sequential, layers)

    # Outcome head
    shared_out <- tail(arch$shared_layers, 1)
    outcome_dims <- c(shared_out, arch$outcome_head_layers, 1L)
    self$outcome_head <- build_head(outcome_dims, arch, final_activation = NULL)

    # Propensity head
    propensity_dims <- c(shared_out, arch$propensity_head_layers, 1L)
    self$propensity_head <- build_head(propensity_dims, arch, final_activation = "sigmoid")
  },

  forward = function(x) {
    h <- self$input_proj(x)
    h <- self$shared(h)
    list(
      outcome = self$outcome_head(h),
      propensity = self$propensity_head(h)
    )
  }
)


#' Initialize model weights
#'
#' @param model torch nn_module
#' @param method Character. "xavier", "kaiming", or "normal"
initialize_weights <- function(model, method = "kaiming") {

  for (module in model$modules) {
    if (inherits(module, "nn_linear")) {
      if (method == "xavier") {
        torch::nn_init_xavier_uniform_(module$weight)
      } else if (method == "kaiming") {
        torch::nn_init_kaiming_uniform_(module$weight, a = 0, mode = "fan_in",
                                        nonlinearity = "relu")
      } else if (method == "normal") {
        torch::nn_init_normal_(module$weight, mean = 0, std = 0.02)
      }
      if (!is.null(module$bias)) {
        torch::nn_init_zeros_(module$bias)
      }
    } else if (inherits(module, "nn_layer_norm")) {
      torch::nn_init_ones_(module$weight)
      torch::nn_init_zeros_(module$bias)
    }
  }

  invisible(model)
}


#' Create model from configuration
#'
#' @param covariate_info List from get_covariate_info()
#' @param config Configuration object
#' @return torch nn_module on appropriate device
create_model <- function(covariate_info, config) {

  covariate_dims <- covariate_info$dims_by_gt
  n_gt_pairs <- covariate_info$n_gt

  # Create model
  model <- MultiTaskDiDNet(covariate_dims, n_gt_pairs, config)

  # Initialize weights
  model <- initialize_weights(model, method = "kaiming")

  # Move to device
  device <- get_device(config)
  model <- model$to(device = device)

  log_message(sprintf("Model created on device: %s", device))

  model
}


#' Model summary
#'
#' @param model torch nn_module
print_model_summary <- function(model) {

  cat("\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\nMODEL SUMMARY\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\n\n")

  # Count parameters
  total_params <- 0
  trainable_params <- 0

  for (name in names(model$parameters)) {
    param <- model$parameters[[name]]
    n <- prod(param$shape)
    total_params <- total_params + n
    if (param$requires_grad) {
      trainable_params <- trainable_params + n
    }
  }

  cat(sprintf("Number of (g,t) pairs: %d\n", model$n_gt_pairs))
  cat(sprintf("Common projection dim: %d\n", model$common_dim))
  cat(sprintf("Covariate dims: min=%d, max=%d\n",
              min(model$covariate_dims), max(model$covariate_dims)))
  cat(sprintf("Total parameters: %s\n", format(total_params, big.mark = ",")))
  cat(sprintf("Trainable parameters: %s\n", format(trainable_params, big.mark = ",")))

  # Estimate memory
  memory_mb <- total_params * 4 / (1024^2)  # float32
  cat(sprintf("Estimated memory: %.1f MB\n", memory_mb))

  cat("\n")
}


#' Get activation function
#'
#' @param name Character. Activation name
#' @return torch activation module
get_activation <- function(name) {
  switch(name,
    "relu" = torch::nn_relu(),
    "leaky_relu" = torch::nn_leaky_relu(negative_slope = 0.01),
    "elu" = torch::nn_elu(),
    "gelu" = torch::nn_gelu(),
    "sigmoid" = torch::nn_sigmoid(),
    "tanh" = torch::nn_tanh(),
    "selu" = torch::nn_selu(),
    stop(sprintf("Unknown activation: %s", name))
  )
}


#' Count model parameters
#'
#' @param model torch nn_module
#' @return Integer. Total number of parameters
count_parameters <- function(model) {
  total <- 0
  for (param in model$parameters) {
    total <- total + prod(param$shape)
  }
  total
}
