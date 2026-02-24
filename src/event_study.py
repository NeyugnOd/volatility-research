"""
Event study framework for volatility reaction.
"""

import numpy as np
import pandas as pd


def extract_event_windows(vol_series, event_dates, window=10):
    windows = []

    for date in event_dates:
        if date in vol_series.index:
            idx = vol_series.index.get_loc(date)
            window_slice = vol_series.iloc[idx-window:idx+window+1]
            windows.append(window_slice.values)

    return np.array(windows)


def bootstrap_mean_difference(pre, post, n_boot=1000):
    diffs = []

    for _ in range(n_boot):
        pre_sample = np.random.choice(pre, size=len(pre), replace=True)
        post_sample = np.random.choice(post, size=len(post), replace=True)
        diffs.append(post_sample.mean() - pre_sample.mean())

    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)

    return np.mean(diffs), (lower, upper)