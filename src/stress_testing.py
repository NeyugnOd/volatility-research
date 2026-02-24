"""
Stress scenario simulation.
"""

import numpy as np


import numpy as np


def simulate_garch_paths(garch_result, n_days=20, n_sim=1000):
    """
    Simulate future return paths using Monte Carlo GARCH.
    """

    res = garch_result["result"]

    sim = res.forecast(
        horizon=n_days,
        method="simulation",
        simulations=n_sim
    )

    if sim.simulations is None:
        raise ValueError("Simulation failed. Check GARCH fit.")

    # shape: (time, horizon, simulations)
    paths = sim.simulations.values[-1]

    return paths


def stress_shock_scenario(returns, shock_size=-0.05):
    """
    Inject one-day crash shock and recompute volatility.
    """

    stressed = returns.copy()
    stressed.iloc[-1] = shock_size

    return stressed