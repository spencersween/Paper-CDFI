#' Covariate Selection Module for DiD Estimation
#'
#' Implements dynamic covariate selection rules following CLAUDE.md specification:
#' - Pre-treatment (t < g): X_ covariates + V_ up to t-1
#' - Post-treatment (t >= g): X_ covariates + V_ up to g-1

#' Select covariates for a specific (g,t) pair
#'
#' Following CLAUDE.md specification for covariate selection rules.
#'
#' @param g Integer. Treatment group year
#' @param t Integer. Time period
#' @param available_vars Character vector. All available variable names
#' @param config Configuration object
#' @return Character vector of selected covariate names
select_covariates_gt <- function(g, t, available_vars, config) {

  selected <- character()

  # 1. Always include all X_ (time-invariant) covariates
  x_vars <- grep(paste0("^", config$time_invariant_prefix), available_vars, value = TRUE)
  selected <- c(selected, x_vars)

  # 2. Include V_ (time-varying) covariates up to appropriate cutoff
  if (t < g) {
    # Pre-treatment: include V_ up to t-1
    cutoff_year <- t - 1L
  } else {
    # Post-treatment: include V_ up to g-1
    cutoff_year <- g - 1L
  }

  # Get year-specific V_ covariates
  v_vars <- select_time_varying_covariates(available_vars, cutoff_year, config)
  selected <- c(selected, v_vars)

  # Remove duplicates (shouldn't happen, but be safe)
  selected <- unique(selected)

  selected
}


#' Select time-varying covariates up to a cutoff year
#'
#' @param available_vars Character vector. All available variable names
#' @param cutoff_year Integer. Include covariates up to this year
#' @param config Configuration object
#' @return Character vector of selected V_ covariate names
select_time_varying_covariates <- function(available_vars, cutoff_year, config) {

  selected <- character()

  # Get all V_ prefixed variables
  v_vars <- grep(paste0("^", config$time_varying_prefix), available_vars, value = TRUE)

  # Parse year from variable names (format: V_varname_YYYY)
  for (v in v_vars) {
    # Extract year from end of variable name
    year_match <- regmatches(v, regexpr("_[0-9]{4}$", v))
    if (length(year_match) > 0) {
      year <- as.integer(substr(year_match, 2, 5))
      if (year <= cutoff_year && year >= config$panel_start) {
        selected <- c(selected, v)
      }
    } else {
      # V_ variable without year suffix (current period value)
      # Include the base V_ variables
      selected <- c(selected, v)
    }
  }

  selected
}


#' Create covariate mask matrix for all (g,t) pairs
#'
#' Creates a binary mask matrix indicating which covariates are active
#' for each (g,t) pair. Used for neural network input masking.
#'
#' @param gt_pairs data.table. Valid (g,t) pairs from get_gt_pairs()
#' @param all_covariates Character vector. All possible covariates (union)
#' @param config Configuration object
#' @return List with:
#'   - masks: Matrix (n_gt_pairs x n_covariates) of 0/1 masks
#'   - covariate_names: Character vector of covariate names (columns)
#'   - gt_to_mask: Integer vector mapping gt_index to mask row
create_covariate_masks <- function(gt_pairs, all_covariates, config) {

  n_gt <- nrow(gt_pairs)
  n_cov <- length(all_covariates)

  # Initialize mask matrix
  masks <- matrix(0L, nrow = n_gt, ncol = n_cov)
  colnames(masks) <- all_covariates

  # Create mask for each (g,t) pair
  for (i in seq_len(n_gt)) {
    g_i <- gt_pairs$g[i]
    t_i <- gt_pairs$t[i]

    # Get selected covariates for this (g,t)
    selected <- select_covariates_gt(g_i, t_i, all_covariates, config)

    # Set mask to 1 for selected covariates
    masks[i, all_covariates %in% selected] <- 1L
  }

  # Summary statistics
  covariates_per_gt <- rowSums(masks)
  log_message(sprintf("Covariate masks created: %d (g,t) pairs x %d covariates",
                      n_gt, n_cov))
  log_message(sprintf("Covariates per (g,t): min=%d, max=%d, mean=%.1f",
                      min(covariates_per_gt), max(covariates_per_gt),
                      mean(covariates_per_gt)))

  list(
    masks = masks,
    covariate_names = all_covariates,
    n_covariates = n_cov,
    covariates_per_gt = covariates_per_gt,
    gt_to_mask = gt_pairs$gt_index
  )
}


#' Get all unique covariates across all (g,t) pairs
#'
#' Computes the union of covariates needed for any (g,t) pair.
#' This determines the input dimension for the neural network.
#'
#' @param gt_pairs data.table. Valid (g,t) pairs
#' @param available_vars Character vector. All available variable names
#' @param config Configuration object
#' @return Character vector of all unique covariates
get_all_covariates <- function(gt_pairs, available_vars, config) {

  all_selected <- character()

  for (i in seq_len(nrow(gt_pairs))) {
    selected <- select_covariates_gt(
      gt_pairs$g[i],
      gt_pairs$t[i],
      available_vars,
      config
    )
    all_selected <- union(all_selected, selected)
  }

  # Sort for consistent ordering
  sort(all_selected)
}


#' Get covariate information for all (g,t) pairs
#'
#' Returns the specific covariates and dimensions for each (g,t) pair.
#' Used for per-(g,t) input projection architecture.
#'
#' @param gt_pairs data.table. Valid (g,t) pairs
#' @param available_vars Character vector. All available variable names
#' @param config Configuration object
#' @return List with:
#'   - covariates_by_gt: List mapping gt_index to character vector of covariate names
#'   - dims_by_gt: Named integer vector mapping gt_index to covariate dimension
#'   - all_covariates: Character vector of all unique covariates (union)
get_covariate_info <- function(gt_pairs, available_vars, config) {

  n_gt <- nrow(gt_pairs)
  covariates_by_gt <- vector("list", n_gt)
  dims_by_gt <- integer(n_gt)
  all_selected <- character()

  for (i in seq_len(n_gt)) {
    selected <- select_covariates_gt(
      gt_pairs$g[i],
      gt_pairs$t[i],
      available_vars,
      config
    )
    # Sort for consistent ordering within each (g,t)
    selected <- sort(selected)
    covariates_by_gt[[i]] <- selected
    dims_by_gt[i] <- length(selected)
    all_selected <- union(all_selected, selected)
  }

  names(covariates_by_gt) <- as.character(gt_pairs$gt_index)
  names(dims_by_gt) <- as.character(gt_pairs$gt_index)

  log_message(sprintf("Covariate dimensions per (g,t): min=%d, max=%d, mean=%.1f",
                      min(dims_by_gt), max(dims_by_gt), mean(dims_by_gt)))

  list(
    covariates_by_gt = covariates_by_gt,
    dims_by_gt = dims_by_gt,
    all_covariates = sort(all_selected),
    n_gt = n_gt
  )
}


#' Prepare (g,t)-specific covariate matrix
#'
#' Extracts only the covariates relevant for a specific (g,t) pair.
#' Returns a matrix with the exact dimension for that pair.
#'
#' @param sample data.table. Estimation sample
#' @param gt_index Integer. The (g,t) pair index
#' @param covariate_info List from get_covariate_info()
#' @return Matrix (n_obs x n_cov_gt) with only relevant covariates
prepare_gt_covariate_matrix <- function(sample, gt_index, covariate_info) {

  # Get covariates for this (g,t)
  cov_names <- covariate_info$covariates_by_gt[[as.character(gt_index)]]
  n_obs <- nrow(sample)
  n_cov <- length(cov_names)

  # Extract covariate matrix
  X <- matrix(NA_real_, nrow = n_obs, ncol = n_cov)
  colnames(X) <- cov_names

  for (j in seq_along(cov_names)) {
    cov <- cov_names[j]
    if (cov %in% names(sample)) {
      X[, j] <- sample[[cov]]
    } else {
      warning(sprintf("Covariate '%s' not found in sample", cov))
      X[, j] <- 0
    }
  }

  # Handle missing values
  X[is.na(X)] <- 0

  X
}


#' Prepare covariate matrix for a (g,t) sample
#'
#' Extracts covariates for a given estimation sample and applies masking.
#'
#' @param sample data.table. Estimation sample from create_gt_sample()
#' @param g Integer. Treatment group
#' @param t Integer. Time period
#' @param all_covariates Character vector. All covariate names
#' @param config Configuration object
#' @return Matrix (n_obs x n_covariates) with zeros for inactive covariates
prepare_covariate_matrix <- function(sample, g, t, all_covariates, config) {

  n_obs <- nrow(sample)
  n_cov <- length(all_covariates)

  # Initialize matrix with zeros
  X <- matrix(0, nrow = n_obs, ncol = n_cov)
  colnames(X) <- all_covariates

  # Get covariates for this (g,t)
  selected <- select_covariates_gt(g, t, all_covariates, config)
  selected <- intersect(selected, names(sample))  # Only use available columns

  # Fill in values for selected covariates
  for (cov in selected) {
    if (cov %in% names(sample)) {
      X[, cov] <- sample[[cov]]
    }
  }

  # Handle missing values (replace with 0 for now - masked anyway if truly missing)
  X[is.na(X)] <- 0

  X
}


#' Get covariate summary statistics
#'
#' @param data data.table. Panel data
#' @param config Configuration object
#' @return data.table with covariate statistics
summarize_covariates <- function(data, config) {

  all_vars <- names(data)

  # Time-invariant covariates
  x_vars <- grep(paste0("^", config$time_invariant_prefix), all_vars, value = TRUE)
  v_vars <- grep(paste0("^", config$time_varying_prefix), all_vars, value = TRUE)

  stats_list <- list()

  # X_ covariates
  for (v in x_vars) {
    vals <- data[[v]]
    stats_list[[v]] <- data.table(
      variable = v,
      type = "time_invariant",
      mean = mean(vals, na.rm = TRUE),
      sd = sd(vals, na.rm = TRUE),
      min = min(vals, na.rm = TRUE),
      max = max(vals, na.rm = TRUE),
      pct_missing = mean(is.na(vals)) * 100,
      pct_zero = mean(vals == 0, na.rm = TRUE) * 100
    )
  }

  # V_ covariates (sample - just current period values)
  v_base <- config$time_varying_base
  for (v in v_base) {
    if (v %in% all_vars) {
      vals <- data[[v]]
      stats_list[[v]] <- data.table(
        variable = v,
        type = "time_varying",
        mean = mean(vals, na.rm = TRUE),
        sd = sd(vals, na.rm = TRUE),
        min = min(vals, na.rm = TRUE),
        max = max(vals, na.rm = TRUE),
        pct_missing = mean(is.na(vals)) * 100,
        pct_zero = mean(vals == 0, na.rm = TRUE) * 100
      )
    }
  }

  rbindlist(stats_list)
}


#' Validate covariate selection rules
#'
#' Tests that covariate selection follows CLAUDE.md specification.
#'
#' @param config Configuration object
#' @return Logical. TRUE if all tests pass
validate_covariate_rules <- function(config) {

  log_message("Validating covariate selection rules...")

  # Create test variable names
  test_vars <- c(
    # Time-invariant
    paste0("X_var", 1:5),
    # Time-varying with years
    paste0("V_totpop_", 1990:2010),
    paste0("V_lenders_pc_", 1990:2010)
  )

  # Test case 1: Pre-treatment (g=2005, t=2000)
  # Should include V_ up to t-1 = 1999
  selected_pre <- select_covariates_gt(2005, 2000, test_vars, config)
  v_years_pre <- as.integer(gsub(".*_([0-9]{4})$", "\\1",
                                  grep("^V_", selected_pre, value = TRUE)))
  max_v_year_pre <- max(v_years_pre)
  assert(max_v_year_pre <= 1999,
         sprintf("Pre-treatment: max V_ year should be <= 1999, got %d", max_v_year_pre))

  # Test case 2: Post-treatment (g=2005, t=2008)
  # Should include V_ up to g-1 = 2004
  selected_post <- select_covariates_gt(2005, 2008, test_vars, config)
  v_years_post <- as.integer(gsub(".*_([0-9]{4})$", "\\1",
                                   grep("^V_", selected_post, value = TRUE)))
  max_v_year_post <- max(v_years_post)
  assert(max_v_year_post <= 2004,
         sprintf("Post-treatment: max V_ year should be <= 2004, got %d", max_v_year_post))

  # Test case 3: All X_ variables should always be included
  x_in_pre <- sum(grepl("^X_", selected_pre))
  x_in_post <- sum(grepl("^X_", selected_post))
  assert(x_in_pre == 5, "All X_ variables should be in pre-treatment selection")
  assert(x_in_post == 5, "All X_ variables should be in post-treatment selection")

  log_message("Covariate selection rules validated successfully")
  TRUE
}


#' Print covariate selection for a specific (g,t) pair
#'
#' Useful for debugging and verification.
#'
#' @param g Integer. Treatment group
#' @param t Integer. Time period
#' @param available_vars Character vector. All available variables
#' @param config Configuration object
print_covariate_selection <- function(g, t, available_vars, config) {

  cat(sprintf("\nCovariate Selection for (g=%d, t=%d)\n", g, t))
  cat(sprintf("Is pre-treatment: %s\n", t < g))
  cat(sprintf("Cutoff year for V_: %d\n", if (t < g) t - 1 else g - 1))
  cat("-" |> rep(50) |> paste(collapse = ""))
  cat("\n")

  selected <- select_covariates_gt(g, t, available_vars, config)

  x_selected <- grep("^X_", selected, value = TRUE)
  v_selected <- grep("^V_", selected, value = TRUE)

  cat(sprintf("X_ covariates: %d\n", length(x_selected)))
  cat(sprintf("V_ covariates: %d\n", length(v_selected)))
  cat(sprintf("Total: %d\n", length(selected)))

  if (length(v_selected) > 0) {
    # Show year range of V_ covariates
    v_years <- as.integer(gsub(".*_([0-9]{4})$", "\\1", v_selected))
    v_years <- v_years[!is.na(v_years)]
    if (length(v_years) > 0) {
      cat(sprintf("V_ year range: %d - %d\n", min(v_years), max(v_years)))
    }
  }

  invisible(selected)
}
