"""Convenience re-exports for the public API."""

__all__ = [
    # --- Core API ---
    "AssetsDistribution",
    "PortfolioFrontier",
    "PortfolioWrapper",
    "Constraints",
    "TransactionCosts",
    "InfeasibleOptimizationError",
    # --- Moment estimators ---
    "estimate_sample_moments",
    "estimate_moments",
    "shrink_mean_jorion",
    "shrink_mean_james_stein",
    "robust_mean_huber",
    "robust_mean_median_of_means",
    "shrink_covariance_ledoit_wolf",
    "shrink_covariance_oas",
    "shrink_covariance_nls",
    "factor_covariance_poet",
    "robust_covariance_tyler",
    "sparse_precision_glasso",
    # --- Bayesian ---
    "RobustBayesPosterior",
    "posterior_moments_black_litterman",
    "posterior_moments_niw",
    "posterior_moments_niw_with_uncertainty",
    # --- Probabilities ---
    "generate_uniform_probabilities",
    "generate_exp_decay_probabilities",
    "silverman_bandwidth",
    "generate_gaussian_kernel_probabilities",
    "compute_effective_number_scenarios",
    "compute_effective_number_scenarios_hhi",
    # --- Views ---
    "entropy_pooling",
    "FlexibleViewsProcessor",
    "BlackLittermanProcessor",
    # --- Optimisers (advanced) ---
    "MeanVariance",
    "MeanCVaR",
    "RelaxedRiskParity",
    "RobustOptimizer",
    # --- Discrete allocation ---
    "discretize_weights",
    "DiscreteAllocationInput",
    "DiscreteAllocationResult",
    # --- Ensembles ---
    "average_frontiers",
    "exposure_stack_frontiers",
    "risk_percentile_selections",
    "assemble_portfolio_ensemble",
    "make_portfolio_spec",
    "stack_portfolios",
    # --- Risk utilities ---
    "portfolio_variance",
    "portfolio_volatility",
    # --- Plotting ---
    "plot_frontiers",
    "plot_frontiers_grid",
    "plot_weights",
    "plot_frontier_report",
    "plot_robust_path",
    "plot_param_impact",
    "plot_robust_surface",
    "plot_assumptions_3d",
    # --- Stress testing ---
    "stress_test",
    "exp_decay_stress",
    "kernel_focus_stress",
    "entropy_pooling_stress",
    "linear_map",
    # --- Performance ---
    "scenario_pnl",
    "performance_report",
]

from .portfolioapi import AssetsDistribution, PortfolioFrontier, PortfolioWrapper, TransactionCosts
from .utils.constraints import Constraints
from .utils.functions import portfolio_variance, portfolio_volatility
from .moments import (
    estimate_sample_moments,
    estimate_moments,
    factor_covariance_poet,
    posterior_moments_black_litterman,
    posterior_moments_niw,
    posterior_moments_niw_with_uncertainty,
    robust_covariance_tyler,
    robust_mean_huber,
    robust_mean_median_of_means,
    shrink_covariance_nls,
    shrink_covariance_ledoit_wolf,
    shrink_covariance_oas,
    shrink_mean_james_stein,
    shrink_mean_jorion,
    sparse_precision_glasso,
)
from .bayesian import RobustBayesPosterior
from .optimization import InfeasibleOptimizationError, MeanCVaR, MeanVariance, RelaxedRiskParity, RobustOptimizer
from .probabilities import (
    compute_effective_number_scenarios,
    compute_effective_number_scenarios_hhi,
    generate_exp_decay_probabilities,
    generate_gaussian_kernel_probabilities,
    generate_uniform_probabilities,
    silverman_bandwidth,
)
from .views import BlackLittermanProcessor, FlexibleViewsProcessor, entropy_pooling
from .discrete_allocation import (
    DiscreteAllocationInput,
    DiscreteAllocationResult,
    discretize_weights,
)
from .ensembles import (
    average_frontiers,
    assemble_portfolio_ensemble,
    exposure_stack_frontiers,
    risk_percentile_selections,
    stack_portfolios,
    make_portfolio_spec,
)
from .plotting import (
    plot_frontiers,
    plot_frontiers_grid,
    plot_weights,
    plot_frontier_report,
    plot_robust_path,
    plot_param_impact,
    plot_robust_surface,
    plot_assumptions_3d,
)
from .stress import (
    entropy_pooling_stress,
    exp_decay_stress,
    kernel_focus_stress,
    linear_map,
    stress_test,
)
from .utils.performance import performance_report, scenario_pnl
