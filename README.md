# Volatility Lab: Empirical Volatility Modeling of the S&P 500

## 1. Introduction

This project builds a professional quantitative research framework for modeling,
forecasting, and evaluating financial volatility using S&P 500 data (2010–2024).

The objective is to replicate stylized facts of financial returns and compare
volatility models under rigorous out-of-sample evaluation.

---

## 2. Stylized Facts

Empirical findings confirm:

- Heavy tails (kurtosis > 3)
- Negative skewness
- Volatility clustering
- Significant autocorrelation in squared returns
- Ljung-Box rejects IID hypothesis

These results validate the use of conditional heteroskedasticity models.

---

## 3. Model Specification

We implement:

- Historical Volatility
- Rolling Volatility (20, 60, 120 days)
- EWMA (RiskMetrics, λ = 0.94)
- GARCH(1,1)

GARCH specification:

σ²ₜ = ω + α ε²ₜ₋₁ + β σ²ₜ₋₁

Persistence: α + β

---

## 4. Forecast Evaluation

Train: 2010–2020  
Test: 2021–2024  

Forecast comparison metrics:

- Mean Squared Error (MSE)
- QLIKE loss

Realized volatility proxy: squared returns.

---

## 5. Event Study (FOMC)

We examine volatility behavior in [-10,+10] windows around FOMC meetings.

Bootstrap inference (1000 resamples) provides confidence intervals for
pre-vs-post volatility differences.

---

## 6. VaR Backtesting

We compute:

- Historical VaR (5%)
- Parametric Gaussian VaR
- GARCH-based VaR

Kupiec unconditional coverage test evaluates model adequacy.

---

## 7. Key Findings

- Volatility is highly persistent (α + β ≈ 0.95–0.99).
- GARCH outperforms EWMA in QLIKE.
- Heavy tails invalidate Gaussian assumptions.
- VaR models exhibit varying coverage performance.

---

## Conclusion

This framework demonstrates a complete volatility research workflow:
data ingestion, stylized facts validation, model estimation,
forecast evaluation, event study, and risk backtesting.
