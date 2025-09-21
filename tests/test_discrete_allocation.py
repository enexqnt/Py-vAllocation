import unittest

import numpy as np
import pandas as pd

from pyvallocation import (
    DiscreteAllocationInput,
    allocate_greedy,
    discretize_weights,
)
from pyvallocation.portfolioapi import PortfolioFrontier


class TestDiscreteAllocation(unittest.TestCase):
    def test_input_validation_missing_price(self):
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
        prices = pd.Series({"AAA": 10.0})
        with self.assertRaisesRegex(ValueError, "Missing latest prices"):
            DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=100.0)

    def test_greedy_basic_allocation(self):
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
        prices = pd.Series({"AAA": 10.0, "BBB": 5.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=100.0)
        result = allocate_greedy(inputs)
        self.assertEqual(result.shares, {"AAA": 5, "BBB": 10})
        self.assertAlmostEqual(result.leftover_cash, 0.0)
        self.assertAlmostEqual(result.tracking_error, 0.0)

    def test_greedy_respects_budget(self):
        weights = pd.Series({"AAA": 0.7, "BBB": 0.3})
        prices = pd.Series({"AAA": 40.0, "BBB": 35.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=100.0)
        result = allocate_greedy(inputs)
        spent = sum(prices[k] * v for k, v in result.shares.items())
        self.assertLessEqual(spent, 100.0 + 1e-8)
        self.assertGreaterEqual(result.leftover_cash, 0.0)

    def test_mip_matches_greedy_for_simple_case(self):
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
        prices = pd.Series({"AAA": 10.0, "BBB": 5.0})
        mip_res = discretize_weights(weights, prices, 100.0, method="milp")
        greedy_res = discretize_weights(weights, prices, 100.0, method="greedy")
        self.assertEqual(mip_res.shares, greedy_res.shares)
        self.assertAlmostEqual(mip_res.leftover_cash, greedy_res.leftover_cash)

    def test_frontier_integration(self):
        weights = np.array([[0.6, 0.3], [0.4, 0.7]])
        returns = np.array([0.1, 0.2])
        risks = np.array([0.15, 0.3])
        frontier = PortfolioFrontier(
            weights=weights,
            returns=returns,
            risks=risks,
            risk_measure="Volatility",
            asset_names=["AAA", "BBB"],
        )
        prices = pd.Series({"AAA": 10.0, "BBB": 5.0})
        result = frontier.as_discrete_allocation(0, prices, 100.0)
        spent = sum(prices[k] * v for k, v in result.shares.items())
        self.assertLessEqual(spent, 100.0 + 1e-8)
        self.assertGreaterEqual(result.leftover_cash, 0.0)


if __name__ == "__main__":
    unittest.main()
