#' ATT Estimation Module
#'
#' Implements doubly-robust ATT(g,t) estimation following Callaway & Sant'Anna (2021).

#' Compute doubly-robust ATT(g,t) estimate
#'
#' Uses the influence function approach from CS2021:
#' ATT(g,t) = E[w1 * (DeltaY - mu_0)] - E[w0 * (DeltaY - mu_0)]
#'
#' @param nuisance List with mu_0, ps, delta_y, D from get_nuisance_gt
#' @param config Configuration object
#' @return List with ATT estimate, SE, and influence function
compute_att_gt <- function(nuisance, config) {

  # Extract components
  delta_y <- nuisance$delta_y
  mu_0 <- nuisance$mu_0
  ps <- nuisance$ps
  D <- nuisance$D
  n <- nuisance$n

  # Handle missing values
  valid_idx <- !is.na(delta_y) & !is.na(mu_0) & !is.na(ps)
  if (sum(valid_idx) < n * 0.5) {
    warning(sprintf("More than 50%% missing values for (g=%d, t=%d)",
                    nuisance$g, nuisance$t))
  }

  delta_y <- delta_y[valid_idx]
  mu_0 <- mu_0[valid_idx]
  ps <- ps[valid_idx]
  D <- D[valid_idx]
  n_valid <- sum(valid_idx)

  # Clamp propensity scores
  ps <- clamp(ps, config$propensity$min_ps, config$propensity$max_ps)

  # Probability of being in treated group
  p_g <- mean(D)

  if (p_g == 0 || p_g == 1) {
    warning(sprintf("Degenerate treatment probability for (g=%d, t=%d): p_g = %.4f",
                    nuisance$g, nuisance$t, p_g))
    return(list(
      att = NA_real_,
      se = NA_real_,
      influence_function = rep(NA_real_, n),
      n_valid = n_valid,
      p_g = p_g
    ))
  }

  # Residuals
  residual <- delta_y - mu_0

  # Weights
  # Treated: w1 = D / p_g
  # Control: w0 = (1-D) * ps / ((1-ps) * p_g)
  w1 <- D / p_g
  w0 <- (1 - D) * ps / ((1 - ps) * p_g)

  # Normalize control weights to sum to n_control
  n_control <- sum(1 - D)
  if (n_control > 0) {
    w0_sum <- sum(w0)
    if (w0_sum > 0) {
      w0 <- w0 * n_control / w0_sum
    }
  }

  # ATT estimate (doubly-robust)
  att <- mean(w1 * residual) - mean(w0 * residual)

  # Influence function for each observation
  # IF_i = (D_i/p_g) * (residual_i - ATT) - ((1-D_i)*ps_i/((1-ps_i)*p_g)) * residual_i
  inf_func_valid <- (D / p_g) * (residual - att) -
                    ((1 - D) * ps / ((1 - ps) * p_g)) * residual

  # Expand back to full sample
  inf_func <- rep(NA_real_, n)
  inf_func[valid_idx] <- inf_func_valid

  # Standard error (simple version - proper SE from bootstrap)
  se <- sqrt(mean(inf_func_valid^2) / n_valid)

  list(
    att = att,
    se = se,
    influence_function = inf_func,
    n_valid = n_valid,
    n_treated = sum(D),
    n_control = n_control,
    p_g = p_g,
    g = nuisance$g,
    t = nuisance$t,
    is_pre = nuisance$is_pre,
    event_time = nuisance$event_time
  )
}


#' Compute all ATT(g,t) estimates
#'
#' @param cf_results List from run_cross_fitting
#' @param config Configuration object
#' @return data.table with ATT estimates for all (g,t) pairs
compute_all_att <- function(cf_results, config) {

  log_message("Computing ATT(g,t) estimates...")

  gt_pairs <- cf_results$gt_pairs
  n_gt <- nrow(gt_pairs)

  # Storage for results
  att_results <- vector("list", n_gt)
  influence_functions <- vector("list", n_gt)

  for (i in seq_len(n_gt)) {
    g_i <- gt_pairs$g[i]
    t_i <- gt_pairs$t[i]

    # Get nuisance estimates
    nuisance <- get_nuisance_gt(cf_results, g_i, t_i, config)

    # Compute ATT
    att_result <- compute_att_gt(nuisance, config)

    # Store
    att_results[[i]] <- data.table::data.table(
      g = g_i,
      t = t_i,
      gt_index = gt_pairs$gt_index[i],
      is_pre = gt_pairs$is_pre[i],
      event_time = gt_pairs$event_time[i],
      att = att_result$att,
      se = att_result$se,
      n_valid = att_result$n_valid,
      n_treated = att_result$n_treated,
      n_control = att_result$n_control,
      p_g = att_result$p_g
    )

    influence_functions[[i]] <- att_result$influence_function
  }

  # Combine results
  att_dt <- data.table::rbindlist(att_results)

  # Add confidence intervals (preliminary - proper ones from bootstrap)
  alpha <- config$inference$alpha
  z <- qnorm(1 - alpha/2)
  att_dt[, `:=`(
    ci_lower = att - z * se,
    ci_upper = att + z * se,
    t_stat = att / se,
    p_value = 2 * pnorm(-abs(att / se))
  )]

  log_message(sprintf("Computed %d ATT(g,t) estimates", nrow(att_dt)))
  log_message(sprintf("  Pre-treatment: %d (mean ATT = %.4f)",
                      sum(att_dt$is_pre),
                      mean(att_dt[is_pre == TRUE, att], na.rm = TRUE)))
  log_message(sprintf("  Post-treatment: %d (mean ATT = %.4f)",
                      sum(!att_dt$is_pre),
                      mean(att_dt[is_pre == FALSE, att], na.rm = TRUE)))

  list(
    att = att_dt,
    influence_functions = influence_functions,
    data = cf_results$data,
    config = config
  )
}


#' Print ATT summary
#'
#' @param att_results List from compute_all_att
print_att_summary <- function(att_results) {

  att_dt <- att_results$att

  cat("\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\nATT(g,t) ESTIMATION SUMMARY\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\n\n")

  # Overall
  cat(sprintf("Total (g,t) pairs: %d\n", nrow(att_dt)))
  cat(sprintf("Valid estimates: %d\n", sum(!is.na(att_dt$att))))

  # Pre-treatment (should be ~0)
  pre <- att_dt[is_pre == TRUE]
  if (nrow(pre) > 0) {
    cat("\nPre-treatment periods (placebo test):\n")
    cat(sprintf("  Mean ATT: %.4f (SE: %.4f)\n",
                mean(pre$att, na.rm = TRUE),
                sd(pre$att, na.rm = TRUE) / sqrt(sum(!is.na(pre$att)))))
    cat(sprintf("  % significant at 5%%: %.1f%%\n",
                100 * mean(pre$p_value < 0.05, na.rm = TRUE)))
  }

  # Post-treatment
  post <- att_dt[is_pre == FALSE]
  if (nrow(post) > 0) {
    cat("\nPost-treatment periods:\n")
    cat(sprintf("  Mean ATT: %.4f (SE: %.4f)\n",
                mean(post$att, na.rm = TRUE),
                sd(post$att, na.rm = TRUE) / sqrt(sum(!is.na(post$att)))))
    cat(sprintf("  % significant at 5%%: %.1f%%\n",
                100 * mean(post$p_value < 0.05, na.rm = TRUE)))
  }

  cat("\n")
}


#' Get ATT for specific event times
#'
#' @param att_results List from compute_all_att
#' @param event_times Integer vector. Event times to extract
#' @return data.table with ATT estimates for specified event times
get_att_by_event_time <- function(att_results, event_times) {
  att_results$att[event_time %in% event_times]
}


#' Get ATT for specific treatment groups
#'
#' @param att_results List from compute_all_att
#' @param groups Integer vector. Treatment groups to extract
#' @return data.table with ATT estimates for specified groups
get_att_by_group <- function(att_results, groups) {
  att_results$att[g %in% groups]
}
