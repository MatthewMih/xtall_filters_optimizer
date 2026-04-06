"""JSON-driven quartz filter AC analysis and differentiable optimization."""

__version__ = "0.1.0"

from xtal_filters.engine import ACAnalysis
from xtal_filters.optimize import OptimizationConfig, run_optimization
from xtal_filters.target_io import generate_target_artifacts

__all__ = [
    "ACAnalysis",
    "OptimizationConfig",
    "generate_target_artifacts",
    "run_optimization",
]
