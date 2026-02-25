import yfinance as yf
import pandas as pd


def download_sp500(start="2010-01-01", end="2024-12-31"):
    """
    Download daily S&P 500 index data (robust to yfinance changes).
    """

    data = yf.download("^GSPC", start=start, end=end, progress=False)

    if data.empty:
        raise ValueError("No data downloaded from Yahoo Finance.")

    # If MultiIndex columns (new yfinance behavior)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Prefer Adj Close if available, else Close
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise KeyError(f"Available columns: {data.columns}")

    prices = prices.dropna()
    prices.name = "sp500"

    return prices
