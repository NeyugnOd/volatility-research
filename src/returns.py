"""
Return calculations module.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns.

    Financial intuition:
    Log returns are time-additive and preferred in econometrics.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = "log_return"
    return returns