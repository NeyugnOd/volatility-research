"""
Forecast evaluation module.
"""

import numpy as np


def mse_loss(forecast, realized):
    return np.mean((forecast - realized) ** 2)


def qlike_loss(forecast, realized):
    """
    QLIKE loss:
    L = log(f_t) + r_t^2 / f_t
    """
    return np.mean(np.log(forecast) + realized / forecast)


def train_test_split(returns):
    train = returns[: "2020-12-31"]
    test = returns["2021-01-01": "2024-12-31"]
    return train, test