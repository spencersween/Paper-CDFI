#' Neural Network Training Module for DiD Estimation
#'
#' Training loop, loss functions, and optimization for multi-task DiD network.
#' Supports per-(g,t) input projection architecture.

#' Compute multi-task loss
#'
#' Combines outcome regression loss (MSE) and propensity score loss (BCE).
#' Outcome loss is computed only on control units (D=0).
#'
#' @param outcome_pred Tensor. Predicted outcome regression values
#' @param propensity_pred Tensor. Predicted propensity scores
#' @param outcome_target Tensor. True outcome differences
#' @param treatment_target Tensor. True treatment indicators (0/1)
#' @param config Configuration object
#' @return List with total loss and component losses
compute_loss <- function(outcome_pred, propensity_pred,
                         outcome_target, treatment_target, config) {

  # Outcome regression: MSE loss (only on control units for training)
  # We train to predict E[DeltaY | X, D=0]
  control_mask <- treatment_target == 0

  if (any(as.logical(control_mask$cpu()))) {
    outcome_loss <- torch::nnf_mse_loss(
      outcome_pred[control_mask, , drop = FALSE],
      outcome_target[control_mask]$unsqueeze(2)
    )
  } else {
    outcome_loss <- torch::torch_tensor(0, device = outcome_pred$device)
  }

  # Propensity score: Binary cross-entropy loss (on all units)
  propensity_loss <- torch::nnf_binary_cross_entropy(
    propensity_pred,
    treatment_target$unsqueeze(2),
    reduction = "mean"
  )

  # Weighted combination
  total_loss <- (config$loss$outcome_weight * outcome_loss +
                 config$loss$propensity_weight * propensity_loss)

  list(
    total = total_loss,
    outcome = outcome_loss,
    propensity = propensity_loss
  )
}


#' Create optimizer
#'
#' @param model torch nn_module
#' @param config Configuration object
#' @return torch optimizer
create_optimizer <- function(model, config) {

  params <- model$parameters

  if (config$optimizer == "adamw") {
    opt_config <- config$optimizer_params$adamw
    optimizer <- torch::optim_adamw(
      params,
      lr = opt_config$lr,
      weight_decay = opt_config$weight_decay,
      betas = opt_config$betas,
      eps = opt_config$eps
    )
  } else if (config$optimizer == "lbfgs") {
    opt_config <- config$optimizer_params$lbfgs
    optimizer <- torch::optim_lbfgs(
      params,
      lr = opt_config$lr,
      max_iter = opt_config$max_iter,
      history_size = opt_config$history_size,
      line_search_fn = opt_config$line_search_fn
    )
  } else {
    stop(sprintf("Unknown optimizer: %s", config$optimizer))
  }

  optimizer
}


#' Create learning rate scheduler
#'
#' @param optimizer torch optimizer
#' @param config Configuration object
#' @return torch lr_scheduler or NULL
create_scheduler <- function(optimizer, config) {

  sched <- config$scheduler

  if (sched$type == "none") {
    return(NULL)
  } else if (sched$type == "step") {
    return(torch::lr_step(optimizer, step_size = sched$step_size, gamma = sched$gamma))
  } else if (sched$type == "cosine") {
    return(torch::lr_cosine_annealing(optimizer, T_max = sched$T_max, eta_min = sched$eta_min))
  } else if (sched$type == "reduce_on_plateau") {
    return(torch::lr_reduce_on_plateau(
      optimizer,
      mode = "min",
      factor = sched$factor,
      patience = sched$patience,
      min_lr = sched$min_lr
    ))
  }

  NULL
}


#' Training step for one epoch
#'
#' Iterates over batches organized by (g,t) pair. Each batch contains
#' observations from a single (g,t) with the appropriate covariate dimension.
#'
#' @param model torch nn_module
#' @param dataloader_fn Function. Generator that returns list of batches
#' @param optimizer torch optimizer
#' @param config Configuration object
#' @param device Character. Device name
#' @return List with epoch metrics
train_epoch <- function(model, dataloader_fn, optimizer, config, device) {

  model$train()

  total_loss <- 0
  total_outcome_loss <- 0
  total_propensity_loss <- 0
  n_batches <- 0
  n_samples <- 0

  # Get batches from generator
  batches <- dataloader_fn()

  for (batch in batches) {
    # Batch contains: X, delta_y, D, gt_index, batch_size
    # X is already on device and has (g,t)-specific dimension
    x <- batch$X
    delta_y <- batch$delta_y
    D <- batch$D
    gt_idx <- batch$gt_index

    # Forward pass with (g,t)-specific input
    optimizer$zero_grad()

    # Use forward_gt for training (keeps gradients enabled)
    output <- model$forward_gt(x, gt_idx)

    # Compute loss
    losses <- compute_loss(
      output$outcome,
      output$propensity,
      delta_y,
      D,
      config
    )

    # Backward pass
    losses$total$backward()

    # Gradient clipping
    if (config$training$gradient_clipping$enabled) {
      torch::nn_utils_clip_grad_norm_(
        model$parameters,
        config$training$gradient_clipping$max_norm
      )
    }

    # Optimizer step
    optimizer$step()

    # Accumulate metrics
    batch_size <- batch$batch_size
    total_loss <- total_loss + as.numeric(losses$total$item()) * batch_size
    total_outcome_loss <- total_outcome_loss + as.numeric(losses$outcome$item()) * batch_size
    total_propensity_loss <- total_propensity_loss + as.numeric(losses$propensity$item()) * batch_size
    n_batches <- n_batches + 1
    n_samples <- n_samples + batch_size

    # Garbage collection periodically
    if (n_batches %% config$gc_every == 0) {
      gc(verbose = FALSE)
    }
  }

  # Return sample-weighted averages
  list(
    loss = total_loss / n_samples,
    outcome_loss = total_outcome_loss / n_samples,
    propensity_loss = total_propensity_loss / n_samples
  )
}


#' Validation step
#'
#' @param model torch nn_module
#' @param dataloader_fn Function. Generator that returns list of batches
#' @param config Configuration object
#' @param device Character. Device name
#' @return List with validation metrics
validate_epoch <- function(model, dataloader_fn, config, device) {

  model$eval()

  total_loss <- 0
  total_outcome_loss <- 0
  total_propensity_loss <- 0
  n_samples <- 0

  torch::with_no_grad({
    batches <- dataloader_fn()

    for (batch in batches) {
      x <- batch$X
      delta_y <- batch$delta_y
      D <- batch$D
      gt_idx <- batch$gt_index

      output <- model$predict_gt(x, gt_idx)

      losses <- compute_loss(
        output$outcome,
        output$propensity,
        delta_y,
        D,
        config
      )

      batch_size <- batch$batch_size
      total_loss <- total_loss + as.numeric(losses$total$item()) * batch_size
      total_outcome_loss <- total_outcome_loss + as.numeric(losses$outcome$item()) * batch_size
      total_propensity_loss <- total_propensity_loss + as.numeric(losses$propensity$item()) * batch_size
      n_samples <- n_samples + batch_size
    }
  })

  list(
    loss = total_loss / n_samples,
    outcome_loss = total_outcome_loss / n_samples,
    propensity_loss = total_propensity_loss / n_samples
  )
}


#' Early stopping tracker
#'
#' @param patience Integer. Number of epochs to wait
#' @param min_delta Numeric. Minimum improvement
#' @return R6 class
EarlyStopping <- R6::R6Class(
  "EarlyStopping",

  public = list(
    patience = NULL,
    min_delta = NULL,
    counter = 0,
    best_loss = Inf,
    best_epoch = 0,
    best_state = NULL,

    initialize = function(patience, min_delta) {
      self$patience <- patience
      self$min_delta <- min_delta
    },

    check = function(val_loss, epoch, model) {
      if (val_loss < self$best_loss - self$min_delta) {
        self$best_loss <- val_loss
        self$best_epoch <- epoch
        self$best_state <- lapply(model$state_dict(), function(x) x$clone())
        self$counter <- 0
        return(FALSE)  # Don't stop
      } else {
        self$counter <- self$counter + 1
        if (self$counter >= self$patience) {
          return(TRUE)  # Stop
        }
        return(FALSE)
      }
    },

    restore_best = function(model) {
      if (!is.null(self$best_state)) {
        model$load_state_dict(self$best_state)
      }
    }
  )
)


#' Train model
#'
#' Main training loop with early stopping and learning rate scheduling.
#' Uses (g,t)-aware batching for efficient training with per-(g,t) input projections.
#'
#' @param model torch nn_module
#' @param train_loader Function. Generator for training batches
#' @param val_loader Function. Generator for validation batches
#' @param config Configuration object
#' @return List with trained model and history
train_model <- function(model, train_loader, val_loader, config) {

  device <- get_device(config)
  model <- model$to(device = device)

  # Create optimizer and scheduler
  optimizer <- create_optimizer(model, config)
  scheduler <- create_scheduler(optimizer, config)

  # Early stopping
  early_stopper <- NULL
  if (config$training$early_stopping$enabled) {
    early_stopper <- EarlyStopping$new(
      config$training$early_stopping$patience,
      config$training$early_stopping$min_delta
    )
  }

  # Training history
  history <- list(
    train_loss = numeric(),
    train_outcome_loss = numeric(),
    train_propensity_loss = numeric(),
    val_loss = numeric(),
    val_outcome_loss = numeric(),
    val_propensity_loss = numeric(),
    lr = numeric()
  )

  # Training loop
  log_message("Starting training...")
  timer <- Timer$new()

  for (epoch in seq_len(config$training$epochs)) {

    # Train
    train_metrics <- train_epoch(model, train_loader, optimizer, config, device)

    # Validate
    val_metrics <- validate_epoch(model, val_loader, config, device)

    # Record history
    history$train_loss <- c(history$train_loss, train_metrics$loss)
    history$train_outcome_loss <- c(history$train_outcome_loss, train_metrics$outcome_loss)
    history$train_propensity_loss <- c(history$train_propensity_loss, train_metrics$propensity_loss)
    history$val_loss <- c(history$val_loss, val_metrics$loss)
    history$val_outcome_loss <- c(history$val_outcome_loss, val_metrics$outcome_loss)
    history$val_propensity_loss <- c(history$val_propensity_loss, val_metrics$propensity_loss)

    # Get current learning rate
    current_lr <- optimizer$param_groups[[1]]$lr
    history$lr <- c(history$lr, current_lr)

    # Update scheduler
    if (!is.null(scheduler)) {
      if (inherits(scheduler, "lr_reduce_on_plateau")) {
        scheduler$step(val_metrics$loss)
      } else {
        scheduler$step()
      }
    }

    # Print progress
    if (config$monitoring$verbose && epoch %% config$monitoring$print_every == 0) {
      log_message(sprintf(
        "Epoch %3d/%d | Train: %.4f (O:%.4f P:%.4f) | Val: %.4f (O:%.4f P:%.4f) | LR: %.2e",
        epoch, config$training$epochs,
        train_metrics$loss, train_metrics$outcome_loss, train_metrics$propensity_loss,
        val_metrics$loss, val_metrics$outcome_loss, val_metrics$propensity_loss,
        current_lr
      ))
    }

    # Early stopping check
    if (!is.null(early_stopper)) {
      if (early_stopper$check(val_metrics$loss, epoch, model)) {
        log_message(sprintf("Early stopping at epoch %d (best: %d, loss: %.4f)",
                            epoch, early_stopper$best_epoch, early_stopper$best_loss))
        break
      }
    }
  }

  # Restore best weights
  if (!is.null(early_stopper) && config$training$early_stopping$restore_best_weights) {
    early_stopper$restore_best(model)
    log_message(sprintf("Restored best weights from epoch %d", early_stopper$best_epoch))
  }

  elapsed <- timer$elapsed()
  log_message(sprintf("Training complete in %.1f seconds", elapsed))

  list(
    model = model,
    history = history,
    best_epoch = if (!is.null(early_stopper)) early_stopper$best_epoch else config$training$epochs,
    training_time = elapsed
  )
}


#' Plot training history
#'
#' @param history List from train_model
#' @return ggplot object
plot_training_history <- function(history) {

  df <- data.frame(
    epoch = rep(seq_along(history$train_loss), 2),
    loss = c(history$train_loss, history$val_loss),
    type = rep(c("Train", "Validation"), each = length(history$train_loss))
  )

  ggplot2::ggplot(df, ggplot2::aes(x = epoch, y = loss, color = type)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::labs(
      title = "Training History",
      x = "Epoch",
      y = "Loss",
      color = ""
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "bottom")
}
