#' Inference Module for DiD Estimation
#'
#' Implements clustered multiplier bootstrap for inference following
#' Callaway & Sant'Anna (2021).

#' Generate multiplier weights
#'
#' @param n_clusters Integer. Number of clusters
#' @param n_bootstrap Integer. Number of bootstrap replications
#' @param distribution Character. "normal" or "rademacher"
#' @param seed Integer. Random seed
#' @return Matrix (n_clusters x n_bootstrap) of multiplier weights
generate_multiplier_weights <- function(n_clusters, n_bootstrap,
                                         distribution = "normal", seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  if (distribution == "normal") {
    xi <- matrix(rnorm(n_clusters * n_bootstrap), n_clusters, n_bootstrap)
  } else if (distribution == "rademacher") {
    xi <- matrix(sample(c(-1, 1), n_clusters * n_bootstrap, replace = TRUE),
                 n_clusters, n_bootstrap)
  } else {
    stop(sprintf("Unknown multiplier distribution: %s", distribution))
  }

  xi
}


#' Clustered multiplier bootstrap for ATT(g,t) estimates
#'
#' Computes bootstrap distribution of ATT estimates using cluster-level
#' multiplier weights.
#'
#' @param att_results List from compute_all_att
#' @param config Configuration object
#' @return List with bootstrap results
clustered_bootstrap <- function(att_results, config) {

  log_message(sprintf("Running clustered multiplier bootstrap (%d replications)...",
                      config$inference$n_bootstrap))

  data <- att_results$data
  att_dt <- att_results$att
  influence_functions <- att_results$influence_functions

  n_gt <- nrow(att_dt)
  n_bootstrap <- config$inference$n_bootstrap
  cluster_var <- config$cluster_var

  # Get cluster information
  clusters <- unique(data[[cluster_var]])
  n_clusters <- length(clusters)
  cluster_map <- match(data[[cluster_var]], clusters)

  log_message(sprintf("  Clusters: %d", n_clusters))

  # Generate multiplier weights
  xi <- generate_multiplier_weights(
    n_clusters,
    n_bootstrap,
    config$inference$multiplier_dist,
    config$inference$seed
  )

  # Bootstrap distribution for each (g,t) pair
  boot_dist <- matrix(NA_real_, nrow = n_gt, ncol = n_bootstrap)

  for (i in seq_len(n_gt)) {
    inf_func <- influence_functions[[i]]

    # Handle missing values
    if (all(is.na(inf_func))) {
      boot_dist[i, ] <- NA_real_
      next
    }

    # Aggregate influence functions by cluster
    # For each cluster c, compute sum of influence functions
    cluster_inf <- tapply(inf_func, cluster_map, sum, na.rm = TRUE)

    # Some clusters may not have any observations in this (g,t) sample
    # Fill in zeros for those
    full_cluster_inf <- rep(0, n_clusters)
    cluster_ids_present <- as.integer(names(cluster_inf))
    full_cluster_inf[cluster_ids_present] <- cluster_inf

    # Bootstrap replication
    att_i <- att_dt$att[i]
    for (b in seq_len(n_bootstrap)) {
      boot_dist[i, b] <- att_i + sum(xi[, b] * full_cluster_inf) / nrow(data)
    }
  }

  # Compute standard errors from bootstrap
  boot_se <- apply(boot_dist, 1, sd, na.rm = TRUE)

  # Pointwise confidence intervals
  alpha <- config$inference$alpha
  pointwise_ci <- t(apply(boot_dist, 1, function(x) {
    quantile(x, probs = c(alpha/2, 1 - alpha/2), na.rm = TRUE)
  }))
  colnames(pointwise_ci) <- c("ci_lower", "ci_upper")

  # Uniform confidence bands (sup-t method)
  uniform_bands <- NULL
  if (config$inference$uniform_bands) {
    uniform_bands <- compute_uniform_bands(att_dt$att, boot_dist, boot_se, alpha)
  }

  log_message("Bootstrap complete")

  list(
    bootstrap_dist = boot_dist,
    se = boot_se,
    pointwise_ci = pointwise_ci,
    uniform_bands = uniform_bands,
    n_bootstrap = n_bootstrap,
    n_clusters = n_clusters
  )
}


#' Compute uniform confidence bands using sup-t method
#'
#' @param att Numeric vector. Point estimates
#' @param boot_dist Matrix. Bootstrap distribution (n_gt x n_bootstrap)
#' @param se Numeric vector. Standard errors
#' @param alpha Numeric. Significance level
#' @return data.table with uniform bands
compute_uniform_bands <- function(att, boot_dist, se, alpha) {

  n_gt <- length(att)
  n_boot <- ncol(boot_dist)

  # Compute t-statistics for each bootstrap sample
  t_stats <- matrix(NA_real_, nrow = n_gt, ncol = n_boot)
  for (b in seq_len(n_boot)) {
    t_stats[, b] <- abs(boot_dist[, b] - att) / se
  }

  # Replace Inf/NaN with NA
  t_stats[!is.finite(t_stats)] <- NA

  # Sup-t statistic for each bootstrap sample
  sup_t <- apply(t_stats, 2, max, na.rm = TRUE)

  # Critical value
  c_alpha <- quantile(sup_t, 1 - alpha, na.rm = TRUE)

  log_message(sprintf("  Sup-t critical value (alpha=%.2f): %.3f", alpha, c_alpha))

  # Uniform bands
  uniform_lower <- att - c_alpha * se
  uniform_upper <- att + c_alpha * se

  data.table::data.table(
    uniform_lower = uniform_lower,
    uniform_upper = uniform_upper,
    sup_t_critical = c_alpha
  )
}


#' Add bootstrap inference to ATT results
#'
#' @param att_results List from compute_all_att
#' @param config Configuration object
#' @return Updated att_results with bootstrap inference
add_bootstrap_inference <- function(att_results, config) {

  boot_results <- clustered_bootstrap(att_results, config)

  # Update ATT data.table
  att_dt <- att_results$att
  att_dt[, se_boot := boot_results$se]
  att_dt[, ci_lower_boot := boot_results$pointwise_ci[, "ci_lower"]]
  att_dt[, ci_upper_boot := boot_results$pointwise_ci[, "ci_upper"]]

  if (!is.null(boot_results$uniform_bands)) {
    att_dt[, uniform_lower := boot_results$uniform_bands$uniform_lower]
    att_dt[, uniform_upper := boot_results$uniform_bands$uniform_upper]
  }

  # Update t-statistics and p-values with bootstrap SE
  att_dt[, t_stat_boot := att / se_boot]
  att_dt[, p_value_boot := 2 * pnorm(-abs(t_stat_boot))]

  att_results$att <- att_dt
  att_results$bootstrap <- boot_results

  att_results
}


#' Hypothesis test for pre-treatment effects (parallel trends)
#'
#' Tests whether pre-treatment ATT estimates are jointly zero.
#'
#' @param att_results List from add_bootstrap_inference
#' @param config Configuration object
#' @return List with test results
test_parallel_trends <- function(att_results, config) {

  log_message("Testing parallel trends (pre-treatment effects)...")

  att_dt <- att_results$att
  boot_dist <- att_results$bootstrap$bootstrap_dist

  # Pre-treatment indices
  pre_idx <- which(att_dt$is_pre == TRUE)
  n_pre <- length(pre_idx)

  if (n_pre == 0) {
    warning("No pre-treatment periods found")
    return(list(
      test_stat = NA,
      p_value = NA,
      n_pre = 0,
      reject = NA
    ))
  }

  # Point estimates for pre-treatment periods
  att_pre <- att_dt$att[pre_idx]
  boot_pre <- boot_dist[pre_idx, , drop = FALSE]

  # Simple test: Wald test statistic
  # H0: mean of pre-treatment ATTs = 0
  mean_att_pre <- mean(att_pre, na.rm = TRUE)
  se_mean_pre <- sd(apply(boot_pre, 2, mean, na.rm = TRUE))

  test_stat <- abs(mean_att_pre / se_mean_pre)
  p_value <- 2 * pnorm(-test_stat)

  # Also compute max absolute pre-treatment ATT
  max_abs_pre <- max(abs(att_pre), na.rm = TRUE)
  se_max <- att_dt$se_boot[pre_idx][which.max(abs(att_pre))]

  log_message(sprintf("  Pre-treatment periods: %d", n_pre))
  log_message(sprintf("  Mean pre-treatment ATT: %.4f (SE: %.4f)", mean_att_pre, se_mean_pre))
  log_message(sprintf("  Test statistic: %.3f, p-value: %.4f", test_stat, p_value))

  list(
    test_stat = test_stat,
    p_value = p_value,
    mean_att_pre = mean_att_pre,
    se_mean_pre = se_mean_pre,
    max_abs_pre = max_abs_pre,
    n_pre = n_pre,
    reject = p_value < config$inference$alpha
  )
}


#' Compute simple ATT (average post-treatment effect)
#'
#' @param att_results List from add_bootstrap_inference
#' @param config Configuration object
#' @return List with simple ATT estimate and inference
compute_simple_att <- function(att_results, config) {

  att_dt <- att_results$att
  boot_dist <- att_results$bootstrap$bootstrap_dist

  # Post-treatment indices
  post_idx <- which(att_dt$is_pre == FALSE)

  if (length(post_idx) == 0) {
    return(list(att = NA, se = NA, ci_lower = NA, ci_upper = NA))
  }

  # Weighted average (by group size)
  weights <- att_dt$n_treated[post_idx]
  weights <- weights / sum(weights, na.rm = TRUE)

  att_simple <- sum(weights * att_dt$att[post_idx], na.rm = TRUE)

  # Bootstrap SE
  boot_simple <- apply(boot_dist[post_idx, , drop = FALSE], 2, function(x) {
    sum(weights * x, na.rm = TRUE)
  })

  se_simple <- sd(boot_simple, na.rm = TRUE)

  alpha <- config$inference$alpha
  ci <- quantile(boot_simple, probs = c(alpha/2, 1 - alpha/2), na.rm = TRUE)

  log_message(sprintf("Simple ATT (weighted avg post-treatment): %.4f (SE: %.4f)",
                      att_simple, se_simple))
  log_message(sprintf("  95%% CI: [%.4f, %.4f]", ci[1], ci[2]))

  list(
    att = att_simple,
    se = se_simple,
    ci_lower = ci[1],
    ci_upper = ci[2],
    t_stat = att_simple / se_simple,
    p_value = 2 * pnorm(-abs(att_simple / se_simple)),
    n_post = length(post_idx)
  )
}
