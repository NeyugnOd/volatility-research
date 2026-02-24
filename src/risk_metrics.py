"""
VaR and Expected Shortfall.
"""

import numpy as np
from scipy.stats import norm, t, chi2


# ---------------------------------------------------
# Historical ES
# ---------------------------------------------------

def historical_es(returns, alpha=0.05):
    var = np.percentile(returns, 100 * alpha)
    tail_losses = returns[returns <= var]
    return tail_losses.mean()


# ---------------------------------------------------
# Parametric ES (Gaussian)
# ---------------------------------------------------

def parametric_es(returns, alpha=0.05):
    mu = returns.mean()
    sigma = returns.std()

    z = norm.ppf(alpha)
    es = mu - sigma * norm.pdf(z) / alpha
    return es


# ---------------------------------------------------
# Student-t ES
# ---------------------------------------------------

def student_t_es(returns, nu, alpha=0.05):
    """
    Expected Shortfall under Student-t distribution.
    """
    mu = returns.mean()
    sigma = returns.std()

    t_alpha = t.ppf(alpha, df=nu)

    es = mu - sigma * (
        (nu + t_alpha**2) / (nu - 1)
        * t.pdf(t_alpha, df=nu)
        / alpha
    )

    return es