"""
Stylized facts analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def summary_statistics(returns):
    """
    Compute higher moments.
    """
    return {
        "mean": returns.mean(),
        "std": returns.std(),
        "skewness": skew(returns),
        "kurtosis": kurtosis(returns, fisher=False)
    }


def plot_histogram_with_gaussian(returns):
    """
    Demonstrate heavy tails.
    """
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 1000)

    plt.figure(figsize=(8, 5))
    plt.hist(returns, bins=50, density=True, alpha=0.6)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r', lw=2)
    plt.title("Return Distribution vs Gaussian")
    plt.show()


def plot_squared_returns(returns):
    """
    Visualize volatility clustering.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(returns**2)
    plt.title("Squared Returns (Volatility Clustering)")
    plt.show()


def acf_squared_returns(returns, lags=20):
    """
    Autocorrelation of squared returns.
    """
    plot_acf(returns**2, lags=lags)
    plt.title("ACF of Squared Returns")
    plt.show()


def ljung_box_test(returns, lags=20):
    """
    Test serial correlation in squared returns.
    """
    result = acorr_ljungbox(returns**2, lags=[lags], return_df=True)
    return result