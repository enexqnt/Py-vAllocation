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
    "covariance_ewma",
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
    "at_least",
    "at_most",
    "between",
    "above",
    "below",
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
    # --- Projection & repricing (Prayer P3-P4) ---
    "project_scenarios",
    "simulate_paths",
    "reprice_exp",
    "reprice_taylor",
    "make_repricing_fn",
    "log2simple",
    "simple2log",
    "compose_repricers",
    "project_mean_covariance",
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
    "stress_invariants",
    "exp_decay_stress",
    "kernel_focus_stress",
    "entropy_pooling_stress",
    "linear_map",
    # --- Performance ---
    "scenario_pnl",
    "performance_report",
    "horizon_report",
    "drawdown_quantile",
]

from .bayesian import RobustBayesPosterior
from .discrete_allocation import (
    DiscreteAllocationInput,
    DiscreteAllocationResult,
    discretize_weights,
)
from .ensembles import (
    assemble_portfolio_ensemble,
    average_frontiers,
    exposure_stack_frontiers,
    make_portfolio_spec,
    risk_percentile_selections,
    stack_portfolios,
)
from .moments import (
    covariance_ewma,
    estimate_moments,
    estimate_sample_moments,
    factor_covariance_poet,
    posterior_moments_black_litterman,
    posterior_moments_niw,
    posterior_moments_niw_with_uncertainty,
    robust_covariance_tyler,
    robust_mean_huber,
    robust_mean_median_of_means,
    shrink_covariance_ledoit_wolf,
    shrink_covariance_nls,
    shrink_covariance_oas,
    shrink_mean_james_stein,
    shrink_mean_jorion,
    sparse_precision_glasso,
)
from .optimization import InfeasibleOptimizationError, MeanCVaR, MeanVariance, RelaxedRiskParity, RobustOptimizer
from .plotting import (
    plot_assumptions_3d,
    plot_frontier_report,
    plot_frontiers,
    plot_frontiers_grid,
    plot_param_impact,
    plot_robust_path,
    plot_robust_surface,
    plot_weights,
)
from .portfolioapi import AssetsDistribution, PortfolioFrontier, PortfolioWrapper, TransactionCosts
from .probabilities import (
    compute_effective_number_scenarios,
    compute_effective_number_scenarios_hhi,
    generate_exp_decay_probabilities,
    generate_gaussian_kernel_probabilities,
    generate_uniform_probabilities,
    silverman_bandwidth,
)
from .stress import (
    entropy_pooling_stress,
    exp_decay_stress,
    kernel_focus_stress,
    linear_map,
    stress_invariants,
    stress_test,
)
from .utils.constraints import Constraints
from .utils.functions import portfolio_variance, portfolio_volatility
from .utils.performance import drawdown_quantile, horizon_report, performance_report, scenario_pnl
from .utils.projection import (
    compose_repricers,
    log2simple,
    make_repricing_fn,
    project_mean_covariance,
    project_scenarios,
    reprice_exp,
    reprice_taylor,
    simple2log,
    simulate_paths,
)
from .views import (
    BlackLittermanProcessor,
    FlexibleViewsProcessor,
    above,
    at_least,
    at_most,
    below,
    between,
    entropy_pooling,
)
