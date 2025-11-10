import unittest

import numpy as np
import pandas as pd
from pyvallocation import (
    average_exposures,
    average_frontiers,
    exposure_stack_frontiers,
    exposure_stacking,
    assemble_portfolio_ensemble,
    make_portfolio_spec,
    stack_portfolios,
)
from pyvallocation.portfolioapi import AssetsDistribution, PortfolioFrontier, PortfolioWrapper


class TestEnsembleUtilities(unittest.TestCase):
    def test_average_exposures_uniform(self):
        samples = np.array([[0.6, 0.2], [0.4, 0.8]])
        averaged = average_exposures(samples)
        expected = samples.mean(axis=1)
        self.assertTrue(np.allclose(averaged, expected))

    def test_average_exposures_weighted(self):
        samples = np.array([[0.5, 0.7], [0.5, 0.3]])
        averaged = average_exposures(samples, weights=[0.25, 0.75])
        expected = samples @ np.array([0.25, 0.75]) / (0.25 + 0.75)
        self.assertTrue(np.allclose(averaged, expected))

    def test_exposure_stacking_identical(self):
        base = np.array([[0.55], [0.45]])
        samples = np.repeat(base, repeats=4, axis=1)
        stacked = exposure_stacking(samples, L=2)
        self.assertTrue(np.allclose(stacked, base.ravel()))

    def test_average_exposures_dataframe_preserves_index(self):
        df = pd.DataFrame(
            [[0.6, 0.4, 0.2], [0.4, 0.6, 0.8]],
            index=["AssetA", "AssetB"],
            columns=["p1", "p2", "p3"],
        )
        averaged = average_exposures(df)
        self.assertIsInstance(averaged, pd.Series)
        self.assertListEqual(list(averaged.index), ["AssetA", "AssetB"])
        expected = df.to_numpy() @ np.full(df.shape[1], 1.0 / df.shape[1])
        self.assertTrue(np.allclose(averaged.values, expected))

    def test_average_exposures_weight_series_alignment(self):
        df = pd.DataFrame(
            [[0.5, 0.7], [0.5, 0.3]],
            index=["AssetA", "AssetB"],
            columns=["one", "two"],
        )
        weights = pd.Series({"two": 3.0, "one": 1.0})
        averaged = average_exposures(df, weights=weights)
        expected = df.to_numpy() @ np.array([1.0, 3.0]) / 4.0
        self.assertTrue(np.allclose(averaged.values, expected))

    def test_exposure_stacking_dataframe_returns_series(self):
        df = pd.DataFrame(
            [[0.55, 0.60], [0.45, 0.40]],
            index=["AssetA", "AssetB"],
            columns=["p1", "p2"],
        )
        stacked = exposure_stacking(df, L=2)
        self.assertIsInstance(stacked, pd.Series)
        self.assertListEqual(list(stacked.index), ["AssetA", "AssetB"])

    def test_average_exposures_series_input(self):
        series = pd.Series([0.6, 0.4], index=["AssetA", "AssetB"], name="portfolio_one")
        averaged = average_exposures(series)
        self.assertIsInstance(averaged, pd.Series)
        self.assertListEqual(list(averaged.index), ["AssetA", "AssetB"])
        self.assertTrue(np.allclose(averaged.values, series.values))

    def test_average_frontiers(self):
        weights1 = np.array([[0.6, 0.5, 0.4], [0.4, 0.5, 0.6]])
        weights2 = np.array([[0.7, 0.3, 0.2], [0.3, 0.7, 0.8]])
        returns = np.array([0.1, 0.12, 0.14])
        risks = np.array([0.15, 0.2, 0.25])
        f1 = PortfolioFrontier(weights=weights1, returns=returns, risks=risks, risk_measure="Vol", asset_names=["A", "B"])
        f2 = PortfolioFrontier(weights=weights2, returns=returns, risks=risks, risk_measure="Vol", asset_names=["A", "B"])

        result = average_frontiers([f1, f2])
        combined = np.hstack([weights1, weights2]).mean(axis=1)
        self.assertTrue(np.allclose(result.values, combined))
        self.assertEqual(list(result.index), ["A", "B"])

    def test_portfolio_frontier_average(self):
        weights = np.array([[0.6, 0.5, 0.4, 0.3], [0.4, 0.5, 0.6, 0.7]])
        returns = np.array([0.1, 0.12, 0.13, 0.14])
        risks = np.array([0.15, 0.18, 0.2, 0.25])
        frontier = PortfolioFrontier(weights=weights, returns=returns, risks=risks, risk_measure="Vol", asset_names=["A", "B"])

        avg_series = frontier.ensemble_average(columns=[0, 3])
        expected_avg = average_exposures(weights[:, [0, 3]])
        self.assertTrue(np.allclose(avg_series.values, expected_avg))

    def test_exposure_stack_frontiers(self):
        weights1 = np.array([[0.6, 0.5], [0.4, 0.5]])
        weights2 = np.array([[0.3, 0.2], [0.7, 0.8]])
        returns = np.array([0.1, 0.11])
        risks = np.array([0.15, 0.16])
        f1 = PortfolioFrontier(weights=weights1, returns=returns, risks=risks, risk_measure="Vol", asset_names=["A", "B"])
        f2 = PortfolioFrontier(weights=weights2, returns=returns, risks=risks, risk_measure="Vol", asset_names=["A", "B"])
        stacked = exposure_stack_frontiers([f1, f2], L=2)
        expected = exposure_stacking(np.hstack([weights1, weights2]), L=2)
        self.assertTrue(np.allclose(stacked.values, expected))
        self.assertEqual(stacked.name, "Exposure Stacking (L=2)")

    def test_stack_portfolios_with_frontiers(self):
        weights1 = np.array([[0.6, 0.5], [0.4, 0.5]])
        weights2 = np.array([[0.3, 0.2], [0.7, 0.8]])
        returns = np.array([0.1, 0.11])
        risks = np.array([0.15, 0.16])
        f1 = PortfolioFrontier(weights=weights1, returns=returns, risks=risks, risk_measure="Vol", asset_names=["A", "B"])
        f2 = PortfolioFrontier(weights=weights2, returns=returns, risks=risks, risk_measure="Vol", asset_names=["A", "B"])
        stacked = stack_portfolios([f1, f2], L=2)
        expected = exposure_stacking(np.hstack([weights1, weights2]), L=2)
        self.assertTrue(np.allclose(stacked.values, expected))
        self.assertEqual(stacked.name, "Exposure Stacking (L=2)")

    def test_stack_portfolios(self):
        p1 = pd.Series({"A": 0.6, "B": 0.4})
        p2 = pd.Series({"A": 0.3, "B": 0.7})
        result = stack_portfolios([p1, p2], L=2)
        expected = exposure_stacking(np.column_stack([p1.values, p2.values]), L=2)
        self.assertTrue(np.allclose(result.values, expected))
        self.assertEqual(list(result.index), ["A", "B"])

    def test_assemble_portfolio_ensemble_selected_workflow(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.normal(0.001, 0.02, size=(120, 4)), columns=list("ABCD"))

        spec_mv = make_portfolio_spec(
            name="MV",
            returns=returns,
            mean_estimator="james_stein",
            cov_estimator="oas",
            optimiser="mean_variance",
            optimiser_kwargs={
                "num_portfolios": 7,
                "constraints": {"long_only": True, "total_weight": 1.0},
            },
            selector="tangency",
            selector_kwargs={"risk_free_rate": 0.0},
        )

        spec_rrp = make_portfolio_spec(
            name="RRP",
            returns=returns,
            mean_estimator="huber",
            cov_estimator="tyler",
            cov_kwargs={"shrinkage": 0.15},
            optimiser="rrp",
            optimiser_kwargs={
                "num_portfolios": 5,
                "max_multiplier": 1.4,
                "lambda_reg": 0.2,
                "constraints": {"long_only": True, "total_weight": 1.0},
            },
            selector="max_return",
        )

        result = assemble_portfolio_ensemble(
            [spec_mv, spec_rrp],
            ensemble=("average", "stack"),
            stack_folds=2,
        )

        self.assertEqual(set(result.frontiers.keys()), {"MV", "RRP"})
        self.assertEqual(list(result.selections.columns), ["MV", "RRP"])
        self.assertIsNotNone(result.average)
        self.assertIsNotNone(result.stacked)
        self.assertTrue(result.stacked.index.equals(result.selections.index))

    def test_assemble_portfolio_ensemble_frontier_mode(self):
        rng = np.random.default_rng(7)
        returns = pd.DataFrame(rng.standard_t(df=6, size=(90, 3)) * 0.015, columns=["X", "Y", "Z"])
        probabilities = pd.Series(np.full(len(returns), 1.0 / len(returns)), index=returns.index)

        spec_mv = make_portfolio_spec(
            name="MVFrontier",
            returns=returns,
            probabilities=probabilities,
            optimiser="mean_variance",
            optimiser_kwargs={
                "num_portfolios": 6,
                "constraints": {"long_only": True, "total_weight": 1.0},
            },
            selector="column",
            selector_kwargs={"index": 0},
            frontier_selection=range(4),
        )

        spec_cvar = make_portfolio_spec(
            name="CVARFrontier",
            returns=returns,
            probabilities=probabilities,
            use_scenarios=True,
            mean_estimator="james_stein",
            cov_estimator="oas",
            optimiser="cvar",
            optimiser_kwargs={
                "num_portfolios": 4,
                "alpha": 0.95,
                "constraints": {"long_only": True, "total_weight": 1.0},
            },
            selector="min_risk",
            frontier_selection=[0, 2],
        )

        result = assemble_portfolio_ensemble(
            [spec_mv, spec_cvar],
            ensemble="stack",
            combine="frontier",
            stack_folds=3,
            stack_kwargs={"maxiters": 30},
        )

        self.assertIn("stack", result.ensembles)

    def test_portfolio_wrapper_make_ensemble_spec(self):
        mu = pd.Series([0.01, 0.015, 0.02], index=["A", "B", "C"])
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]],
            index=mu.index,
            columns=mu.index,
        )
        wrapper = PortfolioWrapper(AssetsDistribution(mu=mu, cov=cov))
        spec = wrapper.make_ensemble_spec(
            "MV",
            optimiser_kwargs={"num_portfolios": 5, "constraints": {"long_only": True, "total_weight": 1.0}},
            selector="min_risk",
        )
        result = wrapper.assemble_ensembles([spec], ensemble="average")
        self.assertIn("average", result.ensembles)
        self.assertEqual(list(result.ensembles["average"].index), ["A", "B", "C"])
        self.assertIsNone(result.stacked)


if __name__ == "__main__":
    unittest.main()
