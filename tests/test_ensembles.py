import unittest

import numpy as np
from pyvallocation import (
    average_exposures,
    average_frontiers,
    exposure_stack_frontiers,
    exposure_stacking,
)
from pyvallocation.portfolioapi import PortfolioFrontier


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


if __name__ == "__main__":
    unittest.main()
