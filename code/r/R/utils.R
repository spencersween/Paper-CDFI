#' Utility Functions for DiD Estimation Pipeline
#'
#' Helper functions used across all modules.

# =============================================================================
# LOGGING
# =============================================================================

#' Log message with timestamp
#'
#' @param msg Character. Message to log
#' @param level Character. Log level: "INFO", "WARNING", "ERROR", "DEBUG"
#' @param config Optional config object for log file
log_message <- function(msg, level = "INFO", config = NULL) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  formatted <- sprintf("[%s] %s: %s", timestamp, level, msg)

  if (!is.null(config) && !is.null(config$monitoring$log_file)) {
    cat(formatted, "\n", file = config$monitoring$log_file, append = TRUE)
  }

  if (is.null(config) || config$monitoring$verbose) {
    message(formatted)
  }
}


#' Progress bar for long operations
#'
#' @param current Integer. Current iteration
#' @param total Integer. Total iterations
#' @param prefix Character. Prefix text
#' @param width Integer. Bar width in characters
progress_bar <- function(current, total, prefix = "", width = 50) {
  pct <- current / total
  filled <- round(pct * width)
  bar <- paste0(
    strrep("=", filled),
    strrep(" ", width - filled)
  )
  cat(sprintf("\r%s [%s] %3.0f%% (%d/%d)",
              prefix, bar, pct * 100, current, total))
  if (current == total) cat("\n")
  flush.console()
}


# =============================================================================
# DATA HELPERS
# =============================================================================

#' Safe division avoiding Inf/NaN
#'
#' @param x Numeric. Numerator
#' @param y Numeric. Denominator
#' @param default Numeric. Value to return when y is zero
#' @return Numeric vector
safe_divide <- function(x, y, default = 0) {
  result <- x / y
  result[!is.finite(result)] <- default
  result
}


#' Winsorize values at given quantiles
#'
#' @param x Numeric vector
#' @param lower Numeric. Lower quantile (e.g., 0.01)
#' @param upper Numeric. Upper quantile (e.g., 0.99)
#' @return Numeric vector
winsorize <- function(x, lower = 0.01, upper = 0.99) {
  q <- quantile(x, c(lower, upper), na.rm = TRUE)
  pmin(pmax(x, q[1]), q[2])
}


#' Standardize variable (z-score)
#'
#' @param x Numeric vector
#' @param na.rm Logical. Remove NA values
#' @return List with standardized values, mean, and sd
standardize <- function(x, na.rm = TRUE) {
  m <- mean(x, na.rm = na.rm)
  s <- sd(x, na.rm = na.rm)
  if (s == 0) s <- 1  # Avoid division by zero
  list(
    z = (x - m) / s,
    mean = m,
    sd = s
  )
}


#' Unstandardize variable
#'
#' @param z Numeric vector of z-scores
#' @param mean Original mean
#' @param sd Original sd
#' @return Numeric vector on original scale
unstandardize <- function(z, mean, sd) {
  z * sd + mean
}


#' Clamp values to range
#'
#' @param x Numeric vector
#' @param lower Numeric. Lower bound
#' @param upper Numeric. Upper bound
#' @return Numeric vector
clamp <- function(x, lower, upper) {
  pmin(pmax(x, lower), upper)
}


# =============================================================================
# PANEL DATA HELPERS
# =============================================================================

#' Get unique treatment groups from panel data
#'
#' @param data data.table with group variable
#' @param config Configuration object
#' @return Integer vector of treatment cohorts (excluding never-treated)
get_treatment_groups <- function(data, config) {
  groups <- unique(data[[config$group_var]])
  groups <- groups[groups > 0]  # Exclude never-treated (group = 0)
  sort(groups)
}


#' Get time periods in analysis window
#'
#' @param config Configuration object
#' @return Integer vector of time periods
get_analysis_periods <- function(config) {
  seq.int(config$analysis_start, config$analysis_end)
}


#' Check if (g, t) pair is valid for estimation
#'
#' @param g Integer. Treatment group
#' @param t Integer. Time period
#' @param config Configuration object
#' @return Logical
is_valid_gt_pair <- function(g, t, config) {
  # Group must be in analysis period
  if (g < config$analysis_start || g > config$analysis_end) {
    return(FALSE)
  }

  # Time must be in analysis period
  if (t < config$analysis_start || t > config$analysis_end) {
    return(FALSE)
  }

  # Need at least one pre-treatment period for baseline
  # For post-treatment: need g-1 as baseline
  # For pre-treatment: need t-1 as baseline
  if (t >= g) {
    # Post-treatment: need g-1 in panel
    if (g - 1 < config$panel_start) {
      return(FALSE)
    }
  } else {
    # Pre-treatment: need t-1 in panel
    if (t - 1 < config$panel_start) {
      return(FALSE)
    }
  }

  TRUE
}


#' Compute event time (time relative to treatment)
#'
#' @param t Integer. Calendar time
#' @param g Integer. Treatment group (cohort year)
#' @return Integer. Event time (t - g)
event_time <- function(t, g) {
  t - g
}


# =============================================================================
# TORCH HELPERS
# =============================================================================

#' Convert R vector/matrix to torch tensor
#'
#' @param x Numeric vector or matrix
#' @param device Character. Device to place tensor on
#' @param dtype torch dtype
#' @return torch_tensor
to_tensor <- function(x, device = "cpu", dtype = torch::torch_float32()) {
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  tensor <- torch::torch_tensor(x, dtype = dtype)
  tensor$to(device = device)
}


#' Convert torch tensor to R vector/matrix
#'
#' @param tensor torch_tensor
#' @return Numeric vector or matrix
from_tensor <- function(tensor) {
  as.array(tensor$cpu())
}


#' Get activation function by name
#'
#' @param name Character. Activation name
#' @return torch nn_module
get_activation <- function(name) {
  switch(name,
    "relu" = torch::nn_relu(),
    "leaky_relu" = torch::nn_leaky_relu(),
    "elu" = torch::nn_elu(),
    "gelu" = torch::nn_gelu(),
    "tanh" = torch::nn_tanh(),
    "sigmoid" = torch::nn_sigmoid(),
    stop(sprintf("Unknown activation function: %s", name))
  )
}


#' Count trainable parameters in a model
#'
#' @param model torch nn_module
#' @return Integer
count_parameters <- function(model) {
  params <- model$parameters
  sum(sapply(params, function(p) prod(p$shape)))
}


#' Memory usage of tensor in MB
#'
#' @param tensor torch_tensor
#' @return Numeric
tensor_memory_mb <- function(tensor) {
  bytes <- prod(tensor$shape) * 4  # Assuming float32
  bytes / (1024^2)
}


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

#' Compute weighted mean
#'
#' @param x Numeric vector
#' @param w Numeric vector of weights
#' @param na.rm Logical. Remove NA values
#' @return Numeric
weighted_mean <- function(x, w, na.rm = TRUE) {
  if (na.rm) {
    idx <- !is.na(x) & !is.na(w)
    x <- x[idx]
    w <- w[idx]
  }
  sum(x * w) / sum(w)
}


#' Compute weighted variance
#'
#' @param x Numeric vector
#' @param w Numeric vector of weights
#' @param na.rm Logical. Remove NA values
#' @return Numeric
weighted_var <- function(x, w, na.rm = TRUE) {
  if (na.rm) {
    idx <- !is.na(x) & !is.na(w)
    x <- x[idx]
    w <- w[idx]
  }
  m <- weighted_mean(x, w, na.rm = FALSE)
  sum(w * (x - m)^2) / sum(w)
}


#' Compute cluster-robust standard error
#'
#' @param influence_values Numeric vector of influence function values
#' @param cluster_ids Vector of cluster identifiers
#' @return Numeric. Standard error
cluster_se <- function(influence_values, cluster_ids) {
  clusters <- unique(cluster_ids)
  n_clusters <- length(clusters)

  # Sum influence functions within clusters
  cluster_sums <- tapply(influence_values, cluster_ids, sum, na.rm = TRUE)

  # Variance of cluster sums
  var_cluster <- var(cluster_sums) * (n_clusters - 1) / n_clusters

  # SE
  sqrt(var_cluster / n_clusters)
}


# =============================================================================
# FILE I/O HELPERS
# =============================================================================

#' Check if file exists and is readable
#'
#' @param path Character. File path
#' @return Logical
file_exists_readable <- function(path) {
  file.exists(path) && file.access(path, mode = 4) == 0
}


#' Get file size in human-readable format
#'
#' @param path Character. File path
#' @return Character
file_size_human <- function(path) {
  size <- file.info(path)$size
  units <- c("B", "KB", "MB", "GB", "TB")
  i <- 1
  while (size >= 1024 && i < length(units)) {
    size <- size / 1024
    i <- i + 1
  }
  sprintf("%.1f %s", size, units[i])
}


#' Create directory if it doesn't exist
#'
#' @param path Character. Directory path
ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}


# =============================================================================
# TIMING HELPERS
# =============================================================================

#' Timer class for benchmarking
Timer <- R6::R6Class(
  "Timer",
  public = list(
    start_time = NULL,
    laps = NULL,

    initialize = function() {
      self$start_time <- Sys.time()
      self$laps <- list()
    },

    lap = function(name) {
      self$laps[[name]] <- Sys.time()
    },

    elapsed = function(from = NULL) {
      if (is.null(from)) {
        difftime(Sys.time(), self$start_time, units = "secs")
      } else {
        difftime(Sys.time(), self$laps[[from]], units = "secs")
      }
    },

    report = function() {
      cat("Timing Report:\n")
      prev <- self$start_time
      for (name in names(self$laps)) {
        elapsed <- difftime(self$laps[[name]], prev, units = "secs")
        cat(sprintf("  %s: %.2f seconds\n", name, elapsed))
        prev <- self$laps[[name]]
      }
      total <- difftime(Sys.time(), self$start_time, units = "secs")
      cat(sprintf("  Total: %.2f seconds\n", total))
    }
  )
)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

#' Assert condition with informative error
#'
#' @param condition Logical
#' @param msg Character. Error message if condition is FALSE
assert <- function(condition, msg) {
  if (!condition) {
    stop(msg, call. = FALSE)
  }
}


#' Check if required packages are installed
#'
#' @param packages Character vector of package names
#' @return Logical. TRUE if all installed
check_packages <- function(packages) {
  missing <- packages[!sapply(packages, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    stop(sprintf(
      "Missing required packages: %s\nInstall with: install.packages(c(%s))",
      paste(missing, collapse = ", "),
      paste(sprintf('"%s"', missing), collapse = ", ")
    ))
  }
  invisible(TRUE)
}


#' Validate data has required columns
#'
#' @param data data.table
#' @param required Character vector of required column names
validate_columns <- function(data, required) {
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing, collapse = ", ")))
  }
  invisible(TRUE)
}
