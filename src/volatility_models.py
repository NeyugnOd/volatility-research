"""
Advanced volatility estimators:
- GARCH(1,1)
- EGARCH(1,1)
- Student-t innovations
"""

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252


# ---------------------------------------------------
# Standard GARCH(1,1) - Gaussian
# ---------------------------------------------------

def fit_garch(returns):
    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        q=1,
        dist="normal"
    )
    res = model.fit(disp="off")

    return _extract_garch_results(res)


# ---------------------------------------------------
# GARCH(1,1) with Student-t Innovations
# ---------------------------------------------------

def fit_garch_student(returns):
    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        q=1,
        dist="t"
    )
    res = model.fit(disp="off")

    results = _extract_garch_results(res)
    results["nu"] = res.params.get("nu", None)

    return results


# ---------------------------------------------------
# EGARCH(1,1)
# ---------------------------------------------------

def fit_egarch(returns, dist="normal"):
    model = arch_model(
        returns * 100,
        vol="EGarch",
        p=1,
        q=1,
        dist=dist
    )
    res = model.fit(disp="off")

    params = res.params
    persistence = params.get("beta[1]", 0)

    cond_vol = res.conditional_volatility / 100 * np.sqrt(TRADING_DAYS)

    return {
        "result": res,
        "params": params,
        "persistence": persistence,
        "conditional_volatility": cond_vol
    }


# ---------------------------------------------------
# Helper
# ---------------------------------------------------

def _extract_garch_results(res):
    params = res.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]

    persistence = alpha + beta
    cond_vol = res.conditional_volatility / 100 * np.sqrt(TRADING_DAYS)

    return {
        "result": res,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "conditional_volatility": cond_vol
    }