#' Data Loading Module for DiD Estimation
#'
#' Efficient data loading and preprocessing using data.table.

#' Load panel data from CSV file
#'
#' Uses data.table::fread for efficient loading. Validates required columns
#' and computes panel metadata.
#'
#' @param file_path Character. Path to CSV file
#' @param config Configuration object
#' @param sample_n Integer or NULL. Sample n rows for testing (NULL = full data)
#' @param columns Character vector or NULL. Specific columns to load (NULL = all)
#' @return List with data.table and metadata
load_panel_data <- function(file_path, config, sample_n = NULL, columns = NULL) {

  log_message(sprintf("Loading data from: %s", file_path), config = config)

  # Check file exists
  if (!file.exists(file_path)) {
    stop(sprintf("Data file not found: %s", file_path))
  }

  log_message(sprintf("File size: %s", file_size_human(file_path)), config = config)

  # Determine columns to load
  if (is.null(columns)) {
    # Read header to get all columns
    header <- data.table::fread(file_path, nrows = 0)
    columns <- names(header)
  }

  # Load data
  if (!is.null(sample_n)) {
    log_message(sprintf("Loading sample of %d rows", sample_n), config = config)
    data <- data.table::fread(
      file_path,
      nrows = sample_n,
      select = columns,
      showProgress = config$monitoring$verbose
    )
  } else {
    data <- data.table::fread(
      file_path,
      select = columns,
      showProgress = config$monitoring$verbose
    )
  }

  log_message(sprintf("Loaded %d rows x %d columns", nrow(data), ncol(data)),
              config = config)

  # Validate required columns
  required_cols <- c(
    config$id_var,
    config$time_var,
    config$group_var,
    config$cluster_var,
    config$outcome_var
  )
  validate_columns(data, required_cols)

  # Ensure proper types
  data[, (config$id_var) := as.integer(get(config$id_var))]
  data[, (config$time_var) := as.integer(get(config$time_var))]
  data[, (config$group_var) := as.integer(get(config$group_var))]

  # Filter to analysis period
  original_n <- nrow(data)
  data <- data[get(config$time_var) >= config$analysis_start &
               get(config$time_var) <= config$analysis_end]
  log_message(sprintf("Filtered to analysis period: %d -> %d rows",
                      original_n, nrow(data)), config = config)

  # Compute metadata
  metadata <- compute_panel_metadata(data, config)

  # Set keys for fast lookups
  data.table::setkeyv(data, c(config$id_var, config$time_var))

  list(
    data = data,
    metadata = metadata
  )
}


#' Compute panel metadata
#'
#' @param data data.table
#' @param config Configuration object
#' @return List with panel metadata
compute_panel_metadata <- function(data, config) {

  # Basic dimensions
  n_obs <- nrow(data)
  n_units <- data[, uniqueN(get(config$id_var))]
  n_periods <- data[, uniqueN(get(config$time_var))]
  n_clusters <- data[, uniqueN(get(config$cluster_var))]

  # Time periods
  time_periods <- sort(unique(data[[config$time_var]]))

  # Treatment groups
  all_groups <- sort(unique(data[[config$group_var]]))
  treatment_groups <- all_groups[all_groups > 0]
  n_never_treated <- data[get(config$group_var) == 0, uniqueN(get(config$id_var))]

  # Group sizes
  group_var <- config$group_var
  id_var <- config$id_var
  group_sizes <- data[, .(
    n_units = uniqueN(get(id_var))
  ), by = c(group_var)]
  data.table::setnames(group_sizes, group_var, "group")

  # Identify covariate columns
  all_cols <- names(data)
  x_cols <- grep(paste0("^", config$time_invariant_prefix), all_cols, value = TRUE)
  v_cols <- grep(paste0("^", config$time_varying_prefix), all_cols, value = TRUE)
  wy_cols <- grep(paste0("^", config$baseline_outcome_prefix), all_cols, value = TRUE)
  y_cols <- grep("^y_", all_cols, value = TRUE)

  list(
    n_obs = n_obs,
    n_units = n_units,
    n_periods = n_periods,
    n_clusters = n_clusters,
    time_periods = time_periods,
    all_groups = all_groups,
    treatment_groups = treatment_groups,
    n_never_treated = n_never_treated,
    n_treatment_groups = length(treatment_groups),
    group_sizes = group_sizes,
    x_cols = x_cols,
    v_cols = v_cols,
    wy_cols = wy_cols,
    y_cols = y_cols,
    n_x_covariates = length(x_cols),
    n_v_covariates = length(v_cols),
    n_wy_covariates = length(wy_cols)
  )
}


#' Get all valid (g,t) pairs for estimation
#'
#' Determines which group-time combinations are valid for ATT estimation
#' based on data availability and analysis period.
#'
#' @param data data.table. Panel data
#' @param config Configuration object
#' @return data.table with columns: g, t, is_pre, n_treated, n_control, base_period
get_gt_pairs <- function(data, config) {

  # Get treatment groups
  groups <- get_treatment_groups(data, config)
  periods <- get_analysis_periods(config)

  # Generate all combinations
  gt_pairs <- data.table::CJ(g = groups, t = periods)

  # Filter to valid pairs
  gt_pairs <- gt_pairs[mapply(is_valid_gt_pair, g, t,
                              MoreArgs = list(config = config))]

  # Add metadata for each pair
  gt_pairs[, `:=`(
    is_pre = t < g,
    event_time = t - g,
    base_period = ifelse(t >= g, g - 1L, t - 1L)
  )]

  # Count treated and control units for each (g,t) pair
  gt_pairs[, `:=`(
    n_treated = NA_integer_,
    n_control = NA_integer_
  )]

  for (i in seq_len(nrow(gt_pairs))) {
    g_i <- gt_pairs$g[i]
    t_i <- gt_pairs$t[i]

    # Treated: units in group g
    n_treated <- data[get(config$group_var) == g_i, uniqueN(get(config$id_var))]

    # Control: not-yet-treated at time t
    # This includes never-treated (group = 0) and groups that start after t
    n_control <- data[get(config$group_var) == 0 | get(config$group_var) > t_i,
                      uniqueN(get(config$id_var))]

    data.table::set(gt_pairs, i, "n_treated", n_treated)
    data.table::set(gt_pairs, i, "n_control", n_control)
  }

  # Add gt_index for reference
  gt_pairs[, gt_index := .I]

  log_message(sprintf("Generated %d valid (g,t) pairs", nrow(gt_pairs)))
  log_message(sprintf("  Pre-treatment pairs: %d", sum(gt_pairs$is_pre)))
  log_message(sprintf("  Post-treatment pairs: %d", sum(!gt_pairs$is_pre)))

  gt_pairs
}


#' Create estimation sample for specific (g,t) pair
#'
#' Extracts the relevant subset of data for estimating ATT(g,t).
#' Includes treated units (group = g) and not-yet-treated controls.
#'
#' @param data data.table. Full panel data
#' @param g Integer. Treatment group (cohort year)
#' @param t Integer. Time period
#' @param config Configuration object
#' @return data.table. Estimation sample with outcome differences computed
create_gt_sample <- function(data, g, t, config) {

  # Determine base period for outcome differencing
  base_period <- if (t >= g) g - 1L else t - 1L

  # Get IDs of treated units (group = g)
  treated_ids <- data[get(config$group_var) == g, unique(get(config$id_var))]

  # Get IDs of not-yet-treated units at time t
  # This includes never-treated (group = 0) and groups starting after t
  control_ids <- data[get(config$group_var) == 0 | get(config$group_var) > t,
                      unique(get(config$id_var))]

  # Get data for current period t
  sample_t <- data[get(config$id_var) %in% c(treated_ids, control_ids) &
                   get(config$time_var) == t]

  # Get data for base period
  sample_base <- data[get(config$id_var) %in% c(treated_ids, control_ids) &
                      get(config$time_var) == base_period]

  # Merge to compute outcome difference
  sample <- merge(
    sample_t,
    sample_base[, .(id_temp = get(config$id_var),
                    y_base = get(config$outcome_var))],
    by.x = config$id_var,
    by.y = "id_temp",
    all.x = TRUE
  )

  # Compute outcome difference
  sample[, delta_y := get(config$outcome_var) - y_base]

  # Add treatment indicator for this (g,t) comparison
  sample[, D := as.integer(get(config$group_var) == g)]

  # Drop observations with missing outcome difference
  sample <- sample[!is.na(delta_y)]

  sample
}


#' Get outcome variable for a specific time period
#'
#' Handles both current outcome (y_*) and year-specific baseline (Wy_*_YYYY).
#'
#' @param data data.table
#' @param outcome Base outcome name (e.g., "sfr_pc")
#' @param period Integer. Time period
#' @param config Configuration object
#' @return Numeric vector
get_outcome <- function(data, outcome, period, config) {
  col_name <- paste0("y_", outcome)
  if (col_name %in% names(data)) {
    return(data[[col_name]])
  }

  # Try year-specific column
  wy_col <- paste0("Wy_y_", outcome, "_", period)
  if (wy_col %in% names(data)) {
    return(data[[wy_col]])
  }

  stop(sprintf("Outcome column not found: %s or %s", col_name, wy_col))
}


#' Create balanced panel check
#'
#' Verifies panel is balanced (all units observed in all periods).
#'
#' @param data data.table
#' @param config Configuration object
#' @return Logical. TRUE if balanced
check_balanced_panel <- function(data, config) {
  id_var <- config$id_var
  obs_per_unit <- data[, .N, by = c(id_var)]
  n_periods <- data[, uniqueN(get(config$time_var))]

  is_balanced <- all(obs_per_unit$N == n_periods)

  if (!is_balanced) {
    min_obs <- min(obs_per_unit$N)
    max_obs <- max(obs_per_unit$N)
    log_message(sprintf("Panel is UNBALANCED: observations per unit range from %d to %d",
                        min_obs, max_obs), level = "WARNING")
  } else {
    log_message("Panel is balanced")
  }

  is_balanced
}


#' Summarize data for diagnostics
#'
#' @param data data.table
#' @param metadata Panel metadata from load_panel_data
#' @param config Configuration object
#' @return Invisible. Prints summary.
summarize_data <- function(data, metadata, config) {

  cat("\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\nDATA SUMMARY\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\n\n")

  cat("Panel Dimensions:\n")
  cat(sprintf("  Observations: %s\n", format(metadata$n_obs, big.mark = ",")))
  cat(sprintf("  Units (ZIPs): %s\n", format(metadata$n_units, big.mark = ",")))
  cat(sprintf("  Time periods: %d (%d - %d)\n",
              metadata$n_periods,
              min(metadata$time_periods),
              max(metadata$time_periods)))
  cat(sprintf("  Clusters: %s\n", format(metadata$n_clusters, big.mark = ",")))

  cat("\nTreatment Groups:\n")
  cat(sprintf("  Never-treated units: %s\n",
              format(metadata$n_never_treated, big.mark = ",")))
  cat(sprintf("  Treatment cohorts: %d (%d - %d)\n",
              metadata$n_treatment_groups,
              min(metadata$treatment_groups),
              max(metadata$treatment_groups)))

  cat("\nCovariates:\n")
  cat(sprintf("  Time-invariant (X_): %d\n", metadata$n_x_covariates))
  cat(sprintf("  Time-varying (V_): %d\n", metadata$n_v_covariates))
  cat(sprintf("  Baseline outcomes (Wy_): %d\n", metadata$n_wy_covariates))

  cat("\nOutcome Variable:\n")
  outcome_stats <- data[, .(
    mean = mean(get(config$outcome_var), na.rm = TRUE),
    sd = sd(get(config$outcome_var), na.rm = TRUE),
    min = min(get(config$outcome_var), na.rm = TRUE),
    max = max(get(config$outcome_var), na.rm = TRUE),
    n_missing = sum(is.na(get(config$outcome_var)))
  )]
  cat(sprintf("  %s: mean=%.4f, sd=%.4f, range=[%.4f, %.4f], missing=%d\n",
              config$outcome_var,
              outcome_stats$mean, outcome_stats$sd,
              outcome_stats$min, outcome_stats$max,
              outcome_stats$n_missing))

  cat("\n")
  invisible(NULL)
}
