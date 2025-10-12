import unittest
from unittest import mock

import numpy as np
import pandas as pd

from pyvallocation import (
    DiscreteAllocationInput,
    allocate_greedy,
    allocate_mip,
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

    def test_greedy_handles_small_target_lots(self):
        weights = pd.Series({"AAA": 0.95, "BBB": 0.05})
        prices = pd.Series({"AAA": 200.0, "BBB": 15.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=100.0)
        result = allocate_greedy(inputs)
        self.assertEqual(result.shares, {"BBB": 1})
        self.assertAlmostEqual(result.leftover_cash, 85.0)

    def test_greedy_skips_unaffordable_assets(self):
        weights = pd.Series({"AAA": 0.6, "BBB": 0.3, "CCC": 0.1})
        prices = pd.Series({"AAA": 150.0, "BBB": 90.0, "CCC": 30.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=120.0)
        result = allocate_greedy(inputs)
        self.assertEqual(result.shares, {"BBB": 1, "CCC": 1})
        self.assertAlmostEqual(result.leftover_cash, 0.0)

    def test_greedy_respects_custom_lot_sizes(self):
        weights = pd.Series({"AAA": 0.4, "BBB": 0.6})
        prices = pd.Series({"AAA": 20.0, "BBB": 12.0})
        lot_sizes = {"AAA": 5, "BBB": 3}
        inputs = DiscreteAllocationInput(
            weights=weights,
            latest_prices=prices,
            total_value=600.0,
            lot_sizes=lot_sizes,
        )
        result = allocate_greedy(inputs)
        for asset, size in lot_sizes.items():
            if asset in result.shares:
                self.assertEqual(result.shares[asset] % size, 0)
        portfolio_value = sum(prices[k] * v for k, v in result.shares.items())
        if portfolio_value > 0:
            achieved = pd.Series(result.shares).reindex(weights.index, fill_value=0) * prices / portfolio_value
            deficits = weights - achieved
            deficits[deficits < 0] = 0.0
            positive_assets = deficits[deficits > 1e-9].index
            if not positive_assets.empty:
                limiting_cost = min(prices[a] * lot_sizes[a] for a in positive_assets)
                self.assertLess(result.leftover_cash, limiting_cost + 1e-9)

    def test_greedy_falls_back_to_mip_when_iterations_exhausted(self):
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
        prices = pd.Series({"AAA": 10.0, "BBB": 5.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=100.0)
        greedy_result = allocate_greedy(inputs, max_iterations=0, fallback_mode="milp")
        mip_result = allocate_mip(inputs)
        self.assertEqual(greedy_result.shares, mip_result.shares)
        self.assertAlmostEqual(greedy_result.leftover_cash, mip_result.leftover_cash)

    def test_greedy_auto_fallback_matches_mip(self):
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
        prices = pd.Series({"AAA": 40.0, "BBB": 35.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=50.0)
        auto_result = allocate_greedy(inputs, max_iterations=0, fallback_mode="auto")
        mip_result = allocate_mip(inputs)
        self.assertEqual(auto_result.shares, mip_result.shares)
        self.assertAlmostEqual(auto_result.leftover_cash, mip_result.leftover_cash)

    def test_greedy_none_mode_returns_partial_allocation(self):
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
        prices = pd.Series({"AAA": 40.0, "BBB": 35.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=50.0)
        result = allocate_greedy(inputs, max_iterations=0, fallback_mode="none")
        self.assertEqual(result.shares, {})
        self.assertAlmostEqual(result.leftover_cash, 50.0)

    def test_greedy_passes_fallback_kwargs(self):
        weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
        prices = pd.Series({"AAA": 40.0, "BBB": 25.0})
        inputs = DiscreteAllocationInput(weights=weights, latest_prices=prices, total_value=80.0)
        expected = allocate_mip(inputs, cash_penalty=0.5)
        with mock.patch("pyvallocation.discrete_allocation.allocate_mip", wraps=allocate_mip) as patched:
            result = allocate_greedy(
                inputs,
                max_iterations=0,
                fallback_mode="auto",
                fallback_kwargs={"cash_penalty": 0.5},
            )
        patched.assert_called_once()
        _, kwargs = patched.call_args
        self.assertAlmostEqual(kwargs["cash_penalty"], 0.5)
        self.assertEqual(result.shares, expected.shares)
        self.assertAlmostEqual(result.leftover_cash, expected.leftover_cash)

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
