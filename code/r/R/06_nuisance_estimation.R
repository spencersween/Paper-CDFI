#' Nuisance Parameter Estimation Module
#'
#' Extracts and organizes nuisance parameters (outcome regression and
#' propensity scores) from cross-fitting results.
#'
#' Works with the per-(g,t) input projection architecture where each (g,t) pair
#' has its own covariate dimension and input projection layer.

#' Get nuisance estimates for a specific (g,t) pair
#'
#' @param cf_results List from run_cross_fitting
#' @param g Integer. Treatment group
#' @param t Integer. Time period
#' @param config Configuration object
#' @return List with mu_0, ps, delta_y, D and metadata for the (g,t) estimation sample
get_nuisance_gt <- function(cf_results, g, t, config) {

  gt_pairs <- cf_results$gt_pairs
  data <- cf_results$data


  # Find gt_index and metadata
  gt_row <- gt_pairs[gt_pairs$g == g & gt_pairs$t == t, ]
  if (nrow(gt_row) == 0) {
    stop(sprintf("(g=%d, t=%d) pair not found in gt_pairs", g, t))
  }
  gt_idx <- gt_row$gt_index
  is_pre <- gt_row$is_pre
  event_time <- gt_row$event_time

  # Create estimation sample to identify relevant observations
  sample <- create_gt_sample(data, g, t, config)
  sample_ids <- sample[[config$id_var]]

  # Get data indices for sample observations
  data_ids <- data[[config$id_var]]

  # Map sample observations to data rows
  data_rows <- match(sample_ids, data_ids)

  # Extract nuisance estimates (out-of-fold predictions)
  mu_0 <- cf_results$outcome[data_rows, gt_idx]
  ps <- cf_results$propensity[data_rows, gt_idx]

  list(
    mu_0 = mu_0,
    ps = ps,
    delta_y = sample$delta_y,
    D = sample$D,
    ids = sample_ids,
    n = length(sample_ids),
    n_treated = sum(sample$D == 1),
    n_control = sum(sample$D == 0),
    g = g,
    t = t,
    gt_index = gt_idx,
    is_pre = is_pre,
    event_time = event_time
  )
}


#' Organize all nuisance estimates by (g,t) pair
#'
#' @param cf_results List from run_cross_fitting
#' @param config Configuration object
#' @return List of nuisance estimates, one element per (g,t)
organize_nuisance_estimates <- function(cf_results, config) {

  gt_pairs <- cf_results$gt_pairs
  n_gt <- nrow(gt_pairs)

  nuisance_list <- vector("list", n_gt)

  for (i in seq_len(n_gt)) {
    g_i <- gt_pairs$g[i]
    t_i <- gt_pairs$t[i]

    nuisance_list[[i]] <- get_nuisance_gt(cf_results, g_i, t_i, config)
    nuisance_list[[i]]$g <- g_i
    nuisance_list[[i]]$t <- t_i
    nuisance_list[[i]]$gt_index <- gt_pairs$gt_index[i]
    nuisance_list[[i]]$is_pre <- gt_pairs$is_pre[i]
    nuisance_list[[i]]$event_time <- gt_pairs$event_time[i]
  }

  names(nuisance_list) <- paste0("g", gt_pairs$g, "_t", gt_pairs$t)

  nuisance_list
}


#' Summarize nuisance parameter quality
#'
#' @param nuisance_list List from organize_nuisance_estimates
#' @param config Configuration object
#' @return data.table with summary statistics
summarize_nuisance_quality <- function(nuisance_list, config) {

  summaries <- lapply(nuisance_list, function(nu) {
    data.table::data.table(
      g = nu$g,
      t = nu$t,
      is_pre = nu$is_pre,
      event_time = nu$event_time,
      n = nu$n,
      n_treated = nu$n_treated,
      n_control = nu$n_control,

      # Outcome regression
      mu0_mean = mean(nu$mu_0, na.rm = TRUE),
      mu0_sd = sd(nu$mu_0, na.rm = TRUE),
      mu0_min = min(nu$mu_0, na.rm = TRUE),
      mu0_max = max(nu$mu_0, na.rm = TRUE),
      mu0_missing = sum(is.na(nu$mu_0)),

      # Propensity score
      ps_mean = mean(nu$ps, na.rm = TRUE),
      ps_sd = sd(nu$ps, na.rm = TRUE),
      ps_min = min(nu$ps, na.rm = TRUE),
      ps_max = max(nu$ps, na.rm = TRUE),
      ps_missing = sum(is.na(nu$ps)),

      # Outcome
      delta_y_mean = mean(nu$delta_y, na.rm = TRUE),
      delta_y_sd = sd(nu$delta_y, na.rm = TRUE)
    )
  })

  summary_dt <- data.table::rbindlist(summaries)
  data.table::setorder(summary_dt, g, t)

  summary_dt
}


#' Check propensity score overlap
#'
#' Examines overlap between treated and control propensity score distributions.
#'
#' @param nuisance_list List from organize_nuisance_estimates
#' @param config Configuration object
#' @return data.table with overlap statistics
check_propensity_overlap <- function(nuisance_list, config) {

  overlap_stats <- lapply(nuisance_list, function(nu) {
    ps_treated <- nu$ps[nu$D == 1]
    ps_control <- nu$ps[nu$D == 0]

    # Overlap measures
    if (length(ps_treated) > 0 && length(ps_control) > 0) {
      # Range overlap
      treated_range <- range(ps_treated, na.rm = TRUE)
      control_range <- range(ps_control, na.rm = TRUE)

      overlap_min <- max(treated_range[1], control_range[1])
      overlap_max <- min(treated_range[2], control_range[2])

      overlap_width <- max(0, overlap_max - overlap_min)
      total_width <- max(treated_range[2], control_range[2]) -
                     min(treated_range[1], control_range[1])

      overlap_ratio <- if (total_width > 0) overlap_width / total_width else 0

      data.table::data.table(
        g = nu$g,
        t = nu$t,
        ps_treated_mean = mean(ps_treated, na.rm = TRUE),
        ps_treated_sd = sd(ps_treated, na.rm = TRUE),
        ps_control_mean = mean(ps_control, na.rm = TRUE),
        ps_control_sd = sd(ps_control, na.rm = TRUE),
        overlap_ratio = overlap_ratio,
        n_extreme_treated = sum(ps_treated > config$propensity$max_ps |
                                 ps_treated < config$propensity$min_ps, na.rm = TRUE),
        n_extreme_control = sum(ps_control > config$propensity$max_ps |
                                 ps_control < config$propensity$min_ps, na.rm = TRUE)
      )
    } else {
      data.table::data.table(
        g = nu$g,
        t = nu$t,
        ps_treated_mean = NA_real_,
        ps_treated_sd = NA_real_,
        ps_control_mean = NA_real_,
        ps_control_sd = NA_real_,
        overlap_ratio = NA_real_,
        n_extreme_treated = NA_integer_,
        n_extreme_control = NA_integer_
      )
    }
  })

  data.table::rbindlist(overlap_stats)
}


#' Diagnose nuisance estimation
#'
#' Runs diagnostic checks on nuisance parameter estimates.
#'
#' @param cf_results List from run_cross_fitting
#' @param config Configuration object
#' @return List with diagnostic results
diagnose_nuisance <- function(cf_results, config) {

  log_message("Running nuisance parameter diagnostics...")

  nuisance_list <- organize_nuisance_estimates(cf_results, config)

  # Summary statistics
  summary <- summarize_nuisance_quality(nuisance_list, config)

  # Overlap analysis
  overlap <- check_propensity_overlap(nuisance_list, config)

  # Print summary
  cat("\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\nNUISANCE PARAMETER DIAGNOSTICS\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\n\n")

  # Overall stats
  cat("Outcome Regression (mu_0):\n")
  cat(sprintf("  Mean across (g,t): %.4f (SD: %.4f)\n",
              mean(summary$mu0_mean), sd(summary$mu0_mean)))
  cat(sprintf("  Range: [%.4f, %.4f]\n",
              min(summary$mu0_min), max(summary$mu0_max)))
  cat(sprintf("  Total missing: %d\n", sum(summary$mu0_missing)))

  cat("\nPropensity Scores:\n")
  cat(sprintf("  Mean across (g,t): %.4f (SD: %.4f)\n",
              mean(summary$ps_mean), sd(summary$ps_mean)))
  cat(sprintf("  Range: [%.4f, %.4f]\n",
              min(summary$ps_min), max(summary$ps_max)))
  cat(sprintf("  Total missing: %d\n", sum(summary$ps_missing)))

  cat("\nOverlap:\n")
  cat(sprintf("  Mean overlap ratio: %.3f\n", mean(overlap$overlap_ratio, na.rm = TRUE)))
  poor_overlap <- sum(overlap$overlap_ratio < 0.5, na.rm = TRUE)
  if (poor_overlap > 0) {
    cat(sprintf("  WARNING: %d (g,t) pairs with overlap < 0.5\n", poor_overlap))
  }

  cat("\n")

  list(
    nuisance_list = nuisance_list,
    summary = summary,
    overlap = overlap
  )
}
