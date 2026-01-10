#' Event Study Aggregation Module
#'
#' Aggregates ATT(g,t) estimates to event study format and other summaries.
#' Implements proper influence function aggregation for uniform inference.


#' Aggregate influence functions to event study level
#'
#' For each event time e, the aggregated IF is:
#'   IF_e = sum_j(w_j * IF_j) for all (g,t) pairs with event_time = e
#'
#' @param influence_functions List of influence function vectors
#' @param event_times Vector of event times to aggregate
#' @param weights_by_event List of weight vectors by event time
#' @param indices_by_event List of index vectors by event time
#' @param n_obs Number of observations
#' @return Matrix of aggregated influence functions (n_events x n_obs)
aggregate_influence_functions <- function(influence_functions, event_times,
                                          weights_by_event, indices_by_event, n_obs) {

  n_events <- length(event_times)
  agg_if <- matrix(0, nrow = n_events, ncol = n_obs)

  for (i in seq_len(n_events)) {
    e <- event_times[i]

    if (is.null(weights_by_event[[as.character(e)]])) next

    weights <- weights_by_event[[as.character(e)]]
    indices <- indices_by_event[[as.character(e)]]

    # Weighted sum of influence functions
    for (j in seq_along(weights)) {
      idx <- indices[j]
      if (idx <= length(influence_functions)) {
        agg_if[i, ] <- agg_if[i, ] + weights[j] * influence_functions[[idx]]
      }
    }
  }

  agg_if
}


#' Compute clustered SEs and bootstrap from aggregated influence functions
#'
#' @param att_vals Vector of point estimates
#' @param agg_if Matrix of aggregated influence functions (n_events x n_obs)
#' @param cluster_indices Vector mapping observations to cluster indices
#' @param n_clusters Number of clusters
#' @param n_bootstrap Number of bootstrap replications
#' @param multiplier_dist Distribution for multiplier weights ("normal" or "rademacher")
#' @param seed Random seed
#' @return List with se and bootstrap_dist
compute_clustered_es_bootstrap <- function(att_vals, agg_if, cluster_indices,
                                           n_clusters, n_bootstrap,
                                           multiplier_dist = "normal", seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  n_events <- nrow(agg_if)
  n_obs <- ncol(agg_if)

  # Aggregate influence functions by cluster
  cluster_if <- matrix(0, nrow = n_events, ncol = n_clusters)
  for (c in seq_len(n_clusters)) {
    mask <- cluster_indices == c
    cluster_if[, c] <- rowSums(agg_if[, mask, drop = FALSE], na.rm = TRUE)
  }

  # Generate multiplier weights
  if (multiplier_dist == "normal") {
    xi <- matrix(rnorm(n_clusters * n_bootstrap), nrow = n_clusters, ncol = n_bootstrap)
  } else if (multiplier_dist == "rademacher") {
    xi <- matrix(sample(c(-1, 1), n_clusters * n_bootstrap, replace = TRUE),
                 nrow = n_clusters, ncol = n_bootstrap)
  } else {
    xi <- matrix(rnorm(n_clusters * n_bootstrap), nrow = n_clusters, ncol = n_bootstrap)
  }

  # Bootstrap distribution
  # ATT_e^* = ATT_e + (1/N) * sum_c(xi_c * IF_ec)
  boot_dist <- matrix(NA_real_, nrow = n_events, ncol = n_bootstrap)

  for (e in seq_len(n_events)) {
    for (b in seq_len(n_bootstrap)) {
      boot_dist[e, b] <- att_vals[e] + sum(xi[, b] * cluster_if[e, ]) / n_obs
    }
  }

  # Standard errors from bootstrap
  se <- apply(boot_dist, 1, sd, na.rm = TRUE)

  list(se = se, bootstrap_dist = boot_dist)
}


#' Compute uniform bands for event study using sup-t method
#'
#' The sup-t method finds critical value c such that:
#'   P(max_e |t_e| <= c) = 1 - alpha
#' where t_e = (ATT_e^* - ATT_e) / SE_e
#'
#' @param att Numeric vector. Point estimates
#' @param boot_dist Matrix. Bootstrap distribution (n_events x n_bootstrap)
#' @param se Numeric vector. Standard errors
#' @param alpha Numeric. Significance level
#' @return List with uniform_lower, uniform_upper, sup_t_critical
compute_es_uniform_bands <- function(att, boot_dist, se, alpha) {

  n_events <- length(att)
  n_boot <- ncol(boot_dist)

  # T-statistics for each bootstrap sample
  t_stats <- abs(boot_dist - att) / se
  t_stats[!is.finite(t_stats)] <- NA

  # Sup-t for each bootstrap sample
  sup_t <- apply(t_stats, 2, max, na.rm = TRUE)

  # Critical value: (1-alpha) quantile of sup-t distribution
  c_alpha <- quantile(sup_t, 1 - alpha, na.rm = TRUE)

  log_message(sprintf("  Uniform bands: sup-t critical value = %.3f (alpha=%.2f)", c_alpha, alpha))

  list(
    uniform_lower = att - c_alpha * se,
    uniform_upper = att + c_alpha * se,
    sup_t_critical = c_alpha,
    sup_t_distribution = sup_t
  )
}


#' Aggregate ATT(g,t) to event study with proper influence function inference
#'
#' Steps:
#' 1. Compute weighted average ATT(e) for each event time
#' 2. Aggregate influence functions using same weights
#' 3. Compute clustered bootstrap from aggregated IFs
#' 4. Compute pointwise CIs and uniform bands
#'
#' @param att_results List from compute_all_att (must include influence_functions and data)
#' @param config Configuration object
#' @return List with event_study, bootstrap_dist, aggregated_influence_functions, uniform_bands
aggregate_event_study <- function(att_results, config) {

  log_message("Aggregating to event study with influence function inference...")

  att_dt <- att_results$att
  data <- att_results$data
  influence_functions <- att_results$influence_functions

  # Filter to event study window
  min_e <- -config$event_study$pre_periods
  max_e <- config$event_study$post_periods

  # Get valid event times
  event_times <- sort(unique(att_dt$event_time))
  event_times <- event_times[event_times >= min_e & event_times <= max_e]

  n_events <- length(event_times)
  n_obs <- nrow(data)

  # Storage for weights and indices (needed for IF aggregation)
  weights_by_event <- list()
  indices_by_event <- list()
  es_results <- data.table::data.table(
    event_time = integer(),
    att = numeric(),
    n_groups = integer(),
    total_treated = numeric()
  )

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

    # Store for IF aggregation
    weights_by_event[[as.character(e)]] <- weights
    indices_by_event[[as.character(e)]] <- idx

    # Weighted average ATT
    att_e <- sum(weights * att_dt$att[idx], na.rm = TRUE)

    es_results <- rbind(es_results, data.table::data.table(
      event_time = e,
      att = att_e,
      n_groups = n_groups,
      total_treated = sum(att_dt$n_treated[idx], na.rm = TRUE)
    ))
  }

  att_vals <- es_results$att

  # Aggregate influence functions
  log_message("  Aggregating influence functions...")
  agg_if <- aggregate_influence_functions(
    influence_functions, event_times, weights_by_event, indices_by_event, n_obs
  )

  # Get cluster information
  cluster_var <- config$cluster_var
  clusters <- unique(data[[cluster_var]])
  n_clusters <- length(clusters)
  cluster_map <- setNames(seq_along(clusters), clusters)
  cluster_indices <- cluster_map[as.character(data[[cluster_var]])]

  log_message(sprintf("  Computing clustered bootstrap (%d reps, %d clusters)...",
                      config$inference$n_bootstrap, n_clusters))

  # Compute SEs and bootstrap from aggregated IFs
  boot_results <- compute_clustered_es_bootstrap(
    att_vals,
    agg_if,
    cluster_indices,
    n_clusters,
    config$inference$n_bootstrap,
    config$inference$multiplier_dist,
    config$inference$seed
  )

  es_results[, se := boot_results$se]
  es_boot <- boot_results$bootstrap_dist

  # Pointwise CIs (percentile method)
  alpha <- config$inference$alpha
  es_results[, ci_lower := apply(es_boot, 1, quantile, probs = alpha/2, na.rm = TRUE)]
  es_results[, ci_upper := apply(es_boot, 1, quantile, probs = 1 - alpha/2, na.rm = TRUE)]

  # P-values
  es_results[, t_stat := att / se]
  es_results[, p_value := 2 * pnorm(-abs(t_stat))]

  # Uniform confidence bands
  uniform_results <- NULL
  if (config$inference$uniform_bands) {
    uniform_results <- compute_es_uniform_bands(att_vals, es_boot, boot_results$se, alpha)
    es_results[, uniform_lower := uniform_results$uniform_lower]
    es_results[, uniform_upper := uniform_results$uniform_upper]
  }

  # Normalize to reference period
  ref_period <- config$event_study$reference_period
  ref_idx <- which(es_results$event_time == ref_period)
  if (length(ref_idx) > 0) {
    ref_att <- es_results$att[ref_idx]
    es_results[, att_normalized := att - ref_att]
    log_message(sprintf("  Normalized to event time %d (ATT = %.4f)", ref_period, ref_att))
  } else {
    es_results[, att_normalized := att]
    log_message(sprintf("  Reference period %d not found, no normalization", ref_period))
  }

  log_message(sprintf("  Event study complete: %d event times (e=%d to %d)",
                      nrow(es_results), min(es_results$event_time), max(es_results$event_time)))

  list(
    event_study = es_results,
    bootstrap_dist = es_boot,
    aggregated_influence_functions = agg_if,
    uniform_bands = uniform_results,
    n_clusters = n_clusters
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
