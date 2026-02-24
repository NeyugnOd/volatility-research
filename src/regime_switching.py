"""
Regime switching volatility discussion and basic proxy.
"""

import pandas as pd


def markov_switching_proxy(returns, threshold=2.0):
    """
    Simple volatility regime proxy:
    High volatility regime if |r| > threshold * std
    """

    std = returns.std()
    regime = (abs(returns) > threshold * std).astype(int)

    return pd.Series(regime, index=returns.index)