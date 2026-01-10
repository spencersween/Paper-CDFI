"""
CDFI DiD Estimation Pipeline v2.0

Callaway & Sant'Anna (2021) Difference-in-Differences with Neural Network
Nuisance Parameter Estimation.

Redesigned for efficiency:
- Wide unit-level data structure
- Base-period grouped covariate projections
- Single forward pass for all (g,t) heads
- On-the-fly outcome differencing

Modules (numbered in order of use):
- m00_config: Configuration
- m01_utils: Utility functions
- m02_data_loader: Data loading and unit-level structure
- m03_covariate_selector: Covariate masks by base period
- m04_nn_architecture: Multi-task neural network
- m05_nn_training: Training loop
- m06_cross_fitting: K-fold cross-fitting
- m07_att_estimation: Doubly-robust ATT estimation
- m08_inference: Clustered bootstrap inference
- m09_aggregation: Event study aggregation
- m10_visualization: Plotting
"""

# Configuration
from .m00_config import (
    Config,
    create_config,
    get_device,
    set_seed,
    print_config
)

# Utilities
from .m01_utils import (
    Timer,
    log_message,
    clamp
)

# Data loading
from .m02_data_loader import (
    UnitData,
    GTInfo,
    load_panel_data,
    create_unit_data,
    create_gt_info,
    create_sample_masks,
    compute_outcome_diffs
)

# Covariate selection
from .m03_covariate_selector import (
    CovariateInfo,
    create_covariate_info
)

# Neural network
from .m04_nn_architecture import (
    MultiTaskDiDNet,
    create_model
)

# Training
from .m05_nn_training import (
    DiDDataset,
    train_model,
    TrainingHistory
)

# Cross-fitting
from .m06_cross_fitting import (
    CrossFitResults,
    run_cross_fitting,
    validate_cross_fitting
)

# ATT estimation
from .m07_att_estimation import (
    ATTResults,
    compute_all_att,
    print_att_summary
)

# Inference
from .m08_inference import (
    add_bootstrap_inference,
    test_parallel_trends,
    compute_simple_att
)

# Aggregation
from .m09_aggregation import (
    aggregate_event_study,
    aggregate_all,
    print_aggregation_summary
)

# Visualization
from .m10_visualization import (
    plot_event_study,
    save_event_study,
    plot_training_history
)

__version__ = "2.0.0"
__all__ = [
    # Config
    'Config', 'create_config', 'get_device', 'set_seed', 'print_config',
    # Utils
    'Timer', 'log_message', 'clamp',
    # Data
    'UnitData', 'GTInfo', 'load_panel_data', 'create_unit_data', 'create_gt_info',
    'create_sample_masks', 'compute_outcome_diffs',
    # Covariates
    'CovariateInfo', 'create_covariate_info',
    # Model
    'MultiTaskDiDNet', 'create_model',
    # Training
    'DiDDataset', 'train_model', 'TrainingHistory',
    # Cross-fitting
    'CrossFitResults', 'run_cross_fitting', 'validate_cross_fitting',
    # ATT
    'ATTResults', 'compute_all_att', 'print_att_summary',
    # Inference
    'add_bootstrap_inference', 'test_parallel_trends', 'compute_simple_att',
    # Aggregation
    'aggregate_event_study', 'aggregate_all', 'print_aggregation_summary',
    # Visualization
    'plot_event_study', 'save_event_study', 'plot_training_history'
]
