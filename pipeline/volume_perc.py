"""Volume data channel: per-bar percussion density + earnings-day fill bars.

v_{T}.npy holds daily log-volume aligned 1:1 with r_{T}.npy (same dates).
Bar mapping matches techno_album.compose(): dp16 days per 16th note,
bar b covers days [b*16*dp16, (b+1)*16*dp16).
"""
import numpy as np


def _bar_slices(n_days, n_bars, dp16):
    dpb = 16 * dp16
    for b in range(n_bars):
        i0 = b * dpb
        i1 = min((b + 1) * dpb, n_days)
        yield i0, max(i1, i0 + 1)


def perc_density(v_log, n_bars, dp16):
    """Per-bar percussion density in [0,1].

    Percentile rank of bar-mean log-volume across bars, smoothed 3 bars.
    """
    v_log = np.asarray(v_log, float)
    means = np.array([v_log[i0:i1].mean() for i0, i1 in _bar_slices(len(v_log), n_bars, dp16)])
    rank = np.argsort(np.argsort(means)) / max(n_bars - 1, 1)
    pad = np.concatenate([rank[:1], rank, rank[-1:]])
    dens = np.convolve(pad, np.ones(3) / 3.0, mode="valid")
    return np.clip(dens, 0.0, 1.0)


def fill_bars(v_log, n_bars, dp16):
    """Boolean per-bar array marking the earnings-day fills.

    A day's spike is its log-volume above a 21-day local baseline; a bar is a
    fill bar when its largest single-day spike sits in the top decile of
    per-bar maxima (~n_bars/10 fills per track).
    """
    v_log = np.asarray(v_log, float)
    n = len(v_log)
    w = 21
    kern = np.ones(w) / w
    base = np.convolve(np.pad(v_log, w // 2, mode="edge"), kern, mode="valid")[:n]
    spike = v_log - base
    mx = np.array([spike[i0:i1].max() for i0, i1 in _bar_slices(n, n_bars, dp16)])
    thr = np.quantile(mx, 0.9)
    return mx >= thr


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for t in ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "TSLA"]:
        r = np.load(os.path.join(here, f"r_{t}.npy"))
        v = np.load(os.path.join(here, f"v_{t}.npy"))
        assert len(v) == len(r), (t, len(v), len(r))
        n = len(r)
        dp16 = max(2, round(n / 1536))            # same formula as techno_album.compose
        n_bars = ((n // dp16) // 16 * 16) // 16
        d = perc_density(v, n_bars, dp16)
        f = fill_bars(v, n_bars, dp16)
        assert len(d) == n_bars and len(f) == n_bars
        assert d.min() >= 0.0 and d.max() <= 1.0
        print(f"{t}: len(v)={len(v)}==len(r) OK  n_bars={n_bars}  "
              f"density min/mean/max = {d.min():.3f}/{d.mean():.3f}/{d.max():.3f}  "
              f"fills = {int(f.sum())}")
