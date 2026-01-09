#' Visualization Module for DiD Estimation
#'
#' Creates publication-ready event study plots and diagnostics using ggplot2.

#' Create publication-ready event study plot
#'
#' @param es_results Event study results from aggregate_event_study
#' @param title Character. Plot title
#' @param subtitle Character. Plot subtitle (optional)
#' @param show_uniform_bands Logical. Show uniform confidence bands
#' @param show_pointwise_ci Logical. Show pointwise confidence intervals
#' @param normalized Logical. Use normalized ATT (relative to reference period)
#' @param colors List. Color scheme
#' @return ggplot object
plot_event_study <- function(
    es_results,
    title = "Event Study: Effect of CDFI Lending on Entrepreneurship",
    subtitle = NULL,
    show_uniform_bands = TRUE,
    show_pointwise_ci = TRUE,
    normalized = FALSE,
    colors = list(
      point = "#2C3E50",
      line = "#2C3E50",
      ci = "#3498DB",
      uniform = "#E74C3C",
      reference = "#7F8C8D",
      pre_treatment = "#95A5A6"
    )
) {

  es <- es_results$event_study

  # Choose ATT column
  if (normalized && "att_normalized" %in% names(es)) {
    es[, att_plot := att_normalized]
    y_label <- "ATT (Normalized)"
  } else {
    es[, att_plot := att]
    y_label <- "ATT"
  }

  # Base plot
  p <- ggplot2::ggplot(es, ggplot2::aes(x = event_time, y = att_plot))

  # Add vertical line at treatment (event_time = 0)
  p <- p + ggplot2::geom_vline(
    xintercept = -0.5,
    linetype = "dashed",
    color = colors$reference,
    linewidth = 0.5
  )

  # Add horizontal line at zero
  p <- p + ggplot2::geom_hline(
    yintercept = 0,
    linetype = "solid",
    color = colors$reference,
    linewidth = 0.5
  )

  # Uniform confidence bands (wider, behind pointwise)
  if (show_uniform_bands && "uniform_lower" %in% names(es)) {
    p <- p + ggplot2::geom_ribbon(
      ggplot2::aes(ymin = uniform_lower, ymax = uniform_upper),
      fill = colors$uniform,
      alpha = 0.15
    )
  }

  # Pointwise confidence intervals
  if (show_pointwise_ci) {
    p <- p + ggplot2::geom_ribbon(
      ggplot2::aes(ymin = ci_lower, ymax = ci_upper),
      fill = colors$ci,
      alpha = 0.25
    )
  }

  # Line connecting points
  p <- p + ggplot2::geom_line(
    color = colors$line,
    linewidth = 0.8
  )

  # Points
  p <- p + ggplot2::geom_point(
    color = colors$point,
    size = 2.5
  )

  # Labels
  p <- p + ggplot2::labs(
    title = title,
    subtitle = subtitle,
    x = "Periods Relative to Treatment",
    y = y_label
  )

  # Theme
  p <- p + ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 14),
      plot.subtitle = ggplot2::element_text(color = "gray40", size = 11),
      axis.title = ggplot2::element_text(size = 11),
      axis.text = ggplot2::element_text(size = 10),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(10, 15, 10, 10)
    )

  # Scale x-axis
  x_breaks <- seq(min(es$event_time), max(es$event_time), by = 2)
  p <- p + ggplot2::scale_x_continuous(breaks = x_breaks)

  p
}


#' Save event study plot
#'
#' @param plot ggplot object
#' @param filename Character. Output filename (without extension)
#' @param output_dir Character. Output directory
#' @param width Numeric. Plot width in inches
#' @param height Numeric. Plot height in inches
#' @param dpi Integer. Resolution
save_event_study <- function(plot, filename = "event_study",
                              output_dir = "outputs/figures",
                              width = 10, height = 6, dpi = 300) {

  ensure_dir(output_dir)

  # PNG
  png_path <- file.path(output_dir, paste0(filename, ".png"))
  ggplot2::ggsave(png_path, plot, width = width, height = height, dpi = dpi)
  log_message(sprintf("Saved: %s", png_path))

  # PDF
  pdf_path <- file.path(output_dir, paste0(filename, ".pdf"))
  ggplot2::ggsave(pdf_path, plot, width = width, height = height)
  log_message(sprintf("Saved: %s", pdf_path))

  invisible(list(png = png_path, pdf = pdf_path))
}


#' Plot training loss curves
#'
#' @param history Training history from train_model
#' @return ggplot object
plot_loss_curves <- function(history) {

  n_epochs <- length(history$train_loss)

  df <- data.frame(
    epoch = rep(1:n_epochs, 4),
    loss = c(
      history$train_outcome_loss,
      history$train_propensity_loss,
      history$val_outcome_loss,
      history$val_propensity_loss
    ),
    type = rep(c("Train Outcome", "Train Propensity",
                 "Val Outcome", "Val Propensity"), each = n_epochs)
  )

  ggplot2::ggplot(df, ggplot2::aes(x = epoch, y = loss, color = type)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::labs(
      title = "Training History",
      x = "Epoch",
      y = "Loss",
      color = ""
    ) +
    ggplot2::scale_color_manual(values = c(
      "Train Outcome" = "#3498DB",
      "Train Propensity" = "#2ECC71",
      "Val Outcome" = "#E74C3C",
      "Val Propensity" = "#F39C12"
    )) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "bottom",
      legend.text = ggplot2::element_text(size = 9)
    )
}


#' Plot propensity score distribution
#'
#' @param nuisance_list List from organize_nuisance_estimates
#' @param gt_index Integer. Which (g,t) pair to plot
#' @return ggplot object
plot_propensity_distribution <- function(nuisance_list, gt_index = 1) {

  nu <- nuisance_list[[gt_index]]

  df <- data.frame(
    ps = nu$ps,
    treated = factor(nu$D, levels = c(0, 1), labels = c("Control", "Treated"))
  )

  ggplot2::ggplot(df, ggplot2::aes(x = ps, fill = treated)) +
    ggplot2::geom_histogram(
      bins = 50,
      alpha = 0.6,
      position = "identity"
    ) +
    ggplot2::labs(
      title = sprintf("Propensity Score Distribution (g=%d, t=%d)", nu$g, nu$t),
      x = "Propensity Score",
      y = "Count",
      fill = ""
    ) +
    ggplot2::scale_fill_manual(values = c("Control" = "#3498DB", "Treated" = "#E74C3C")) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "bottom")
}


#' Plot ATT by treatment group
#'
#' @param group_results data.table from aggregate_by_group
#' @return ggplot object
plot_att_by_group <- function(group_results) {

  ggplot2::ggplot(group_results, ggplot2::aes(x = factor(g), y = att)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = ci_lower, ymax = ci_upper),
      width = 0.2,
      color = "#3498DB"
    ) +
    ggplot2::geom_point(size = 3, color = "#2C3E50") +
    ggplot2::labs(
      title = "ATT by Treatment Cohort",
      x = "Treatment Year (Cohort)",
      y = "ATT"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    )
}


#' Plot ATT over calendar time
#'
#' @param time_results data.table from aggregate_by_time
#' @return ggplot object
plot_att_by_time <- function(time_results) {

  ggplot2::ggplot(time_results, ggplot2::aes(x = t, y = att)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = ci_lower, ymax = ci_upper),
      fill = "#3498DB",
      alpha = 0.25
    ) +
    ggplot2::geom_line(color = "#2C3E50", linewidth = 0.8) +
    ggplot2::geom_point(size = 2.5, color = "#2C3E50") +
    ggplot2::labs(
      title = "ATT Over Calendar Time",
      x = "Year",
      y = "ATT"
    ) +
    ggplot2::theme_minimal()
}


#' Create diagnostic panel
#'
#' Combines multiple diagnostic plots.
#'
#' @param att_results List from add_bootstrap_inference
#' @param agg_results List from aggregate_all
#' @param training_history Training history (optional)
#' @return Combined plot (requires patchwork)
create_diagnostic_panel <- function(att_results, agg_results, training_history = NULL) {

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    warning("Install 'patchwork' package for combined diagnostic panel")
    return(NULL)
  }

  # Event study
  p1 <- plot_event_study(agg_results$event_study, title = "Event Study")

  # ATT by group
  p2 <- plot_att_by_group(agg_results$by_group)

  # ATT by time
  p3 <- plot_att_by_time(agg_results$by_time)

  # Combine
  if (!is.null(training_history)) {
    p4 <- plot_loss_curves(training_history)
    combined <- (p1 | p4) / (p2 | p3)
  } else {
    combined <- p1 / (p2 | p3)
  }

  combined + patchwork::plot_annotation(
    title = "DiD Estimation Diagnostics",
    theme = ggplot2::theme(plot.title = ggplot2::element_text(size = 16, face = "bold"))
  )
}


#' Generate all output figures
#'
#' @param att_results List from add_bootstrap_inference
#' @param agg_results List from aggregate_all
#' @param output_dir Character. Output directory
#' @param config Configuration object
generate_all_figures <- function(att_results, agg_results, output_dir, config) {

  log_message("Generating output figures...")

  ensure_dir(output_dir)

  # Main event study plot
  es_plot <- plot_event_study(
    agg_results$event_study,
    title = sprintf("Effect of CDFI Lending on %s", config$outcome_var),
    subtitle = "Callaway & Sant'Anna (2021) DiD with Neural Network Nuisance Estimation"
  )
  save_event_study(es_plot, "event_study", output_dir)

  # Normalized version
  es_plot_norm <- plot_event_study(
    agg_results$event_study,
    title = sprintf("Effect of CDFI Lending on %s (Normalized)", config$outcome_var),
    normalized = TRUE
  )
  save_event_study(es_plot_norm, "event_study_normalized", output_dir)

  # ATT by group
  group_plot <- plot_att_by_group(agg_results$by_group)
  ggplot2::ggsave(
    file.path(output_dir, "att_by_group.png"),
    group_plot,
    width = 10, height = 6, dpi = 300
  )

  # ATT by time
  time_plot <- plot_att_by_time(agg_results$by_time)
  ggplot2::ggsave(
    file.path(output_dir, "att_by_time.png"),
    time_plot,
    width = 10, height = 6, dpi = 300
  )

  log_message(sprintf("Figures saved to: %s", output_dir))

  invisible(list(
    event_study = es_plot,
    event_study_normalized = es_plot_norm,
    by_group = group_plot,
    by_time = time_plot
  ))
}
