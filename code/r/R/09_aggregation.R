#' Event Study Aggregation Module
#'
#' Aggregates ATT(g,t) estimates to event study format and other summaries.

#' Aggregate ATT(g,t) to event study
#'
#' Pools ATT estimates across groups for each event time (relative period).
#'
#' @param att_results List from add_bootstrap_inference
#' @param config Configuration object
#' @return data.table with event study estimates
aggregate_event_study <- function(att_results, config) {

  log_message("Aggregating to event study...")

  att_dt <- att_results$att
  boot_dist <- att_results$bootstrap$bootstrap_dist

  # Filter to event study window
  min_e <- -config$event_study$pre_periods
  max_e <- config$event_study$post_periods

  # Get valid event times
  event_times <- sort(unique(att_dt$event_time))
  event_times <- event_times[event_times >= min_e & event_times <= max_e]

  n_events <- length(event_times)
  n_boot <- ncol(boot_dist)

  # Storage
  es_results <- data.table::data.table(
    event_time = event_times,
    att = NA_real_,
    se = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    uniform_lower = NA_real_,
    uniform_upper = NA_real_,
    n_groups = NA_integer_
  )

  # Bootstrap distributions for event study
  es_boot <- matrix(NA_real_, nrow = n_events, ncol = n_boot)

  for (i in seq_len(n_events)) {
    e <- event_times[i]

    # Get ATT estimates for this event time
    idx <- which(att_dt$event_time == e)
    n_groups <- length(idx)

    if (n_groups == 0) next

    # Weights (proportional to group size if configured)
    if (config$event_study$weight_by_group_size) {
      weights <- att_dt$n_treated[idx]
    } else {
      weights <- rep(1, n_groups)
    }
    weights <- weights / sum(weights, na.rm = TRUE)

    # Weighted average ATT
    att_e <- sum(weights * att_dt$att[idx], na.rm = TRUE)

    # Bootstrap distribution
    boot_e <- apply(boot_dist[idx, , drop = FALSE], 2, function(x) {
      sum(weights * x, na.rm = TRUE)
    })

    es_boot[i, ] <- boot_e

    # Store results
    es_results[i, att := att_e]
    es_results[i, n_groups := n_groups]
  }

  # Compute SEs and CIs from bootstrap
  es_results[, se := apply(es_boot, 1, sd, na.rm = TRUE)]

  alpha <- config$inference$alpha
  es_results[, ci_lower := apply(es_boot, 1, quantile, probs = alpha/2, na.rm = TRUE)]
  es_results[, ci_upper := apply(es_boot, 1, quantile, probs = 1 - alpha/2, na.rm = TRUE)]

  # Uniform confidence bands
  if (config$inference$uniform_bands) {
    uniform <- compute_es_uniform_bands(es_results$att, es_boot, es_results$se, alpha)
    es_results[, uniform_lower := uniform$uniform_lower]
    es_results[, uniform_upper := uniform$uniform_upper]
  }

  # Normalize to reference period
  ref_period <- config$event_study$reference_period
  ref_idx <- which(es_results$event_time == ref_period)
  if (length(ref_idx) > 0) {
    ref_att <- es_results$att[ref_idx]
    es_results[, att_normalized := att - ref_att]
    log_message(sprintf("Normalized to event time %d (ATT = %.4f)", ref_period, ref_att))
  } else {
    es_results[, att_normalized := att]
    log_message(sprintf("Reference period %d not found, no normalization", ref_period))
  }

  log_message(sprintf("Event study: %d event times (%d to %d)",
                      nrow(es_results), min(es_results$event_time), max(es_results$event_time)))

  list(
    event_study = es_results,
    bootstrap_dist = es_boot
  )
}


#' Compute uniform bands for event study
#'
#' @param att Numeric vector. Point estimates
#' @param boot_dist Matrix. Bootstrap distribution
#' @param se Numeric vector. Standard errors
#' @param alpha Numeric. Significance level
#' @return data.table with uniform bands
compute_es_uniform_bands <- function(att, boot_dist, se, alpha) {

  n_events <- length(att)
  n_boot <- ncol(boot_dist)

  # T-statistics for each bootstrap sample
  t_stats <- matrix(NA_real_, nrow = n_events, ncol = n_boot)
  for (b in seq_len(n_boot)) {
    t_stats[, b] <- abs(boot_dist[, b] - att) / se
  }

  t_stats[!is.finite(t_stats)] <- NA

  # Sup-t for each bootstrap sample
  sup_t <- apply(t_stats, 2, max, na.rm = TRUE)

  # Critical value
  c_alpha <- quantile(sup_t, 1 - alpha, na.rm = TRUE)

  data.table::data.table(
    uniform_lower = att - c_alpha * se,
    uniform_upper = att + c_alpha * se
  )
}


#' Aggregate by treatment group (cohort-specific effects)
#'
#' @param att_results List from add_bootstrap_inference
#' @param config Configuration object
#' @return data.table with group-specific estimates
aggregate_by_group <- function(att_results, config) {

  att_dt <- att_results$att
  boot_dist <- att_results$bootstrap$bootstrap_dist

  groups <- sort(unique(att_dt$g))
  n_groups <- length(groups)
  n_boot <- ncol(boot_dist)

  group_results <- data.table::data.table(
    g = groups,
    att = NA_real_,
    se = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    n_periods = NA_integer_
  )

  for (i in seq_len(n_groups)) {
    g <- groups[i]

    # Post-treatment ATTs for this group
    idx <- which(att_dt$g == g & att_dt$is_pre == FALSE)
    n_periods <- length(idx)

    if (n_periods == 0) next

    # Simple average
    att_g <- mean(att_dt$att[idx], na.rm = TRUE)

    # Bootstrap
    boot_g <- apply(boot_dist[idx, , drop = FALSE], 2, mean, na.rm = TRUE)
    se_g <- sd(boot_g, na.rm = TRUE)

    alpha <- config$inference$alpha
    ci <- quantile(boot_g, probs = c(alpha/2, 1 - alpha/2), na.rm = TRUE)

    group_results[i, `:=`(
      att = att_g,
      se = se_g,
      ci_lower = ci[1],
      ci_upper = ci[2],
      n_periods = n_periods
    )]
  }

  group_results
}


#' Aggregate by calendar time
#'
#' @param att_results List from add_bootstrap_inference
#' @param config Configuration object
#' @return data.table with calendar time estimates
aggregate_by_time <- function(att_results, config) {

  att_dt <- att_results$att
  boot_dist <- att_results$bootstrap$bootstrap_dist

  # Only post-treatment
  post_dt <- att_dt[is_pre == FALSE]
  times <- sort(unique(post_dt$t))
  n_times <- length(times)
  n_boot <- ncol(boot_dist)

  time_results <- data.table::data.table(
    t = times,
    att = NA_real_,
    se = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    n_groups = NA_integer_
  )

  for (i in seq_len(n_times)) {
    t_i <- times[i]

    idx <- which(att_dt$t == t_i & att_dt$is_pre == FALSE)
    n_groups <- length(idx)

    if (n_groups == 0) next

    # Weighted average
    weights <- att_dt$n_treated[idx]
    weights <- weights / sum(weights, na.rm = TRUE)

    att_t <- sum(weights * att_dt$att[idx], na.rm = TRUE)

    # Bootstrap
    boot_t <- apply(boot_dist[idx, , drop = FALSE], 2, function(x) {
      sum(weights * x, na.rm = TRUE)
    })
    se_t <- sd(boot_t, na.rm = TRUE)

    alpha <- config$inference$alpha
    ci <- quantile(boot_t, probs = c(alpha/2, 1 - alpha/2), na.rm = TRUE)

    time_results[i, `:=`(
      att = att_t,
      se = se_t,
      ci_lower = ci[1],
      ci_upper = ci[2],
      n_groups = n_groups
    )]
  }

  time_results
}


#' Create full aggregation summary
#'
#' @param att_results List from add_bootstrap_inference
#' @param config Configuration object
#' @return List with all aggregation results
aggregate_all <- function(att_results, config) {

  log_message("Computing all aggregations...")

  event_study <- aggregate_event_study(att_results, config)
  by_group <- aggregate_by_group(att_results, config)
  by_time <- aggregate_by_time(att_results, config)
  simple_att <- compute_simple_att(att_results, config)

  list(
    event_study = event_study,
    by_group = by_group,
    by_time = by_time,
    simple_att = simple_att
  )
}


#' Print aggregation summary
#'
#' @param agg_results List from aggregate_all
print_aggregation_summary <- function(agg_results) {

  cat("\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\nAGGREGATION SUMMARY\n")
  cat("=" |> rep(60) |> paste(collapse = ""))
  cat("\n\n")

  # Simple ATT
  sa <- agg_results$simple_att
  cat("Simple ATT (weighted average post-treatment):\n")
  cat(sprintf("  ATT = %.4f (SE = %.4f)\n", sa$att, sa$se))
  cat(sprintf("  95%% CI: [%.4f, %.4f]\n", sa$ci_lower, sa$ci_upper))
  cat(sprintf("  p-value: %.4f\n\n", sa$p_value))

  # Event Study
  es <- agg_results$event_study$event_study
  cat("Event Study:\n")
  cat(sprintf("  Event times: %d to %d\n",
              min(es$event_time), max(es$event_time)))

  # Pre-treatment
  pre <- es[event_time < 0]
  if (nrow(pre) > 0) {
    cat(sprintf("  Pre-treatment mean ATT: %.4f\n", mean(pre$att, na.rm = TRUE)))
  }

  # Post-treatment
  post <- es[event_time >= 0]
  if (nrow(post) > 0) {
    cat(sprintf("  Post-treatment mean ATT: %.4f\n", mean(post$att, na.rm = TRUE)))
  }

  cat("\n")
}
