"""Shared loaders for the bundled ETF sample dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).with_name("ETF_prices.csv")


def load_prices(path: Path | None = None, *, forward_fill: bool = True) -> pd.DataFrame:
    """Load sample ETF prices (Date index).

    Args:
        path: Optional path to the CSV file.
        forward_fill: Whether to forward-fill missing prices.

    Returns:
        pd.DataFrame: Price DataFrame with Date index.
    """
    data_path = path or DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Sample data not found at {data_path}")
    prices = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    prices = prices.dropna(how="all")
    if forward_fill:
        prices = prices.ffill()
    return prices


def load_returns(path: Path | None = None) -> pd.DataFrame:
    """Load simple returns derived from the sample ETF prices.

    Args:
        path: Optional path to the CSV file.

    Returns:
        pd.DataFrame: Simple return scenarios.
    """
    prices = load_prices(path)
    return prices.pct_change().dropna(how="any")


def load_moments(path: Path | None = None) -> Tuple[pd.Series, pd.DataFrame]:
    """Load mean/covariance moments from the sample returns.

    Args:
        path: Optional path to the CSV file.

    Returns:
        Tuple[pd.Series, pd.DataFrame]: Mean vector and covariance matrix.
    """
    returns = load_returns(path)
    mu = returns.mean()
    cov = returns.cov()
    min_eig = float(np.linalg.eigvalsh(cov.values).min())
    if min_eig <= 0:
        cov = cov + np.eye(len(cov)) * (abs(min_eig) + 1e-6)
    return mu, cov
