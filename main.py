from src.data import download_sp500
from src.returns import compute_log_returns
from src.volatility_models import (
    fit_garch,
    fit_garch_student,
    fit_egarch
)
from src.risk_metrics import historical_es
from src.stress_testing import simulate_garch_paths


def main():

    prices = download_sp500()
    returns = compute_log_returns(prices)

    garch = fit_garch(returns)
    garch_t = fit_garch_student(returns)
    egarch = fit_egarch(returns, dist="t")

    print("GARCH persistence:", garch["persistence"])
    print("Student-t dof:", garch_t.get("nu"))

    es = historical_es(returns)
    print("Historical Expected Shortfall:", es)

    sims = simulate_garch_paths(garch, n_days=10)
    print("Simulated stress paths shape:", sims.shape)


if __name__ == "__main__":
    main()