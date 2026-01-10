"""
DiD Estimation Pipeline - Python/PyTorch Implementation

Callaway & Sant'Anna (2021) difference-in-differences with neural network nuisance estimation.
"""

from .config import (
    Config,
    create_config,
    get_device,
    set_seed,
    print_config,
    ArchitectureConfig,
    TrainingConfig,
    CrossFittingConfig,
    InferenceConfig,
    EventStudyConfig
)

from .utils import (
    log_message,
    ensure_dir,
    file_size_human,
    clamp,
    Timer
)

from .data_loader import (
    load_panel_data,
    get_gt_pairs,
    create_gt_sample,
    summarize_data,
    PanelMetadata
)

from .covariate_selector import (
    get_covariate_info,
    get_covariates_for_gt,
    prepare_gt_covariate_matrix,
    create_covariate_masks,
    validate_covariate_rules,
    CovariateInfo
)

from .nn_architecture import (
    MultiTaskDiDNet,
    create_model,
    count_parameters,
    print_model_summary
)

from .nn_training import (
    train_model,
    compute_loss,
    create_optimizer,
    TrainingHistory
)

from .cross_fitting import (
    run_cross_fitting,
    validate_cross_fitting,
    assign_cluster_folds,
    add_fold_column
)

from .nuisance_estimation import (
    get_nuisance_gt,
    organize_nuisance_estimates,
    diagnose_nuisance
)

from .att_estimation import (
    compute_att_gt,
    compute_all_att,
    print_att_summary
)

from .inference import (
    add_bootstrap_inference,
    test_parallel_trends,
    compute_simple_att,
    clustered_bootstrap
)

from .aggregation import (
    aggregate_all,
    aggregate_event_study,
    aggregate_by_group,
    aggregate_by_time,
    print_aggregation_summary
)

from .visualization import (
    plot_event_study,
    save_event_study,
    generate_all_figures,
    plot_att_by_group,
    plot_att_by_time
)

__version__ = "1.0.0"
__all__ = [
    # Config
    'Config', 'create_config', 'get_device', 'set_seed', 'print_config',
    # Utils
    'log_message', 'ensure_dir', 'Timer',
    # Data
    'load_panel_data', 'get_gt_pairs', 'create_gt_sample', 'summarize_data',
    # Covariates
    'get_covariate_info', 'prepare_gt_covariate_matrix', 'validate_covariate_rules',
    # Model
    'MultiTaskDiDNet', 'create_model',
    # Training
    'train_model', 'run_cross_fitting', 'validate_cross_fitting',
    # Estimation
    'compute_all_att', 'print_att_summary', 'diagnose_nuisance',
    # Inference
    'add_bootstrap_inference', 'test_parallel_trends', 'compute_simple_att',
    # Aggregation
    'aggregate_all', 'print_aggregation_summary',
    # Visualization
    'plot_event_study', 'save_event_study', 'generate_all_figures'
]
