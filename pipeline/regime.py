"""regime.py — song form from the data's own era structure (Tech(no) US pipeline).

Instead of a fixed intro/breakdown/drop template, the arrangement is read off the
return series itself: 21d rolling vol (percentile-ranked per bar) gives the energy
contour, and the worst-quartile drawdown regime (same rule as techno_album.compose,
dd > q(0.78)) gives the bear/breakdown windows.  Change-points on the bar-level
vol series split the non-bear stretches into eras, and each era becomes a section:

  bear bars                         -> 'bear'      (breakdown: pads+lead only)
  last 1-2 bars of each bear run    -> 'recovery'  (roll build into the drop)
  non-bear eras by vol tercile      -> 'calm' / 'build' / 'peak'
  first 4 bars / last 2 bars forced -> 'intro' / 'outro'

segments() returns contiguous, non-overlapping dicts {bar0, bar1, kind, energy}
with bar1 EXCLUSIVE, covering bars [0, n_bars).  section_mask() maps a kind to
layer gates + density multipliers for the engine; is_bull() is the 252d-MA
z-score key-modulation trigger (same z as compose()'s bar harmony).
"""
import numpy as np

KINDS = ("intro", "calm", "build", "peak", "bear", "recovery", "outro")
ENERGY = {"intro": 0.25, "calm": 0.40, "build": 0.65, "peak": 1.00,
          "bear": 0.15, "recovery": 0.80, "outro": 0.30}

_MIN_SEG = 6      # min bars per vol-era (change-point splitting)
_CP_WIN = 4       # bars: window for the vol-percentile jump test
_CP_JUMP = 0.25   # |d vol_pct| over _CP_WIN bars that opens a new era


def _roll_std(x, w):
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1):i + 1].std()
    return out


def _bar_features(rs, n_bars, dp16):
    """Per-bar (at bar-end day): vol percentile in [0,1] and bear flag."""
    rs = np.asarray(rs, float)
    n = len(rs)
    price = np.cumsum(rs)
    vol21 = _roll_std(rs, 21) * np.sqrt(252)
    dd = np.maximum.accumulate(price) - price
    bear_day = dd > np.quantile(dd, 0.78)          # worst-quartile dd, as v3 compose
    ends = np.minimum((np.arange(n_bars) + 1) * 16 * dp16 - 1, n - 1)
    vol = vol21[ends]
    vol_pct = np.argsort(np.argsort(vol)) / max(n_bars - 1, 1)
    bearb = bear_day[ends]
    return vol_pct, bearb


def _change_points(vol_pct):
    """Era boundaries on the bar-level vol-percentile series.
    A boundary opens where the percentile moved > _CP_JUMP over a _CP_WIN-bar
    window; boundaries closer than _MIN_SEG bars to each other or to either
    edge are dropped (tiny eras get merged away)."""
    nb = len(vol_pct)
    cps = []
    for b in range(_CP_WIN, nb):
        if abs(vol_pct[b] - vol_pct[b - _CP_WIN]) > _CP_JUMP:
            if (not cps or b - cps[-1] >= _MIN_SEG) and b >= _MIN_SEG and nb - b >= _MIN_SEG:
                cps.append(b)
    return [0] + cps + [nb]


def segments(rs, n_bars, dp16):
    """List of dicts {bar0, bar1(excl), kind, energy} tiling bars [0, n_bars)."""
    vol_pct, bearb = _bar_features(rs, n_bars, dp16)
    # bridge 1-bar non-bear blips between bear runs (one noisy bar is no rally)
    for b in range(1, n_bars - 1):
        if bearb[b - 1] and not bearb[b] and bearb[b + 1]:
            bearb[b] = True
    kind = np.empty(n_bars, object)

    # --- non-bear eras: change-point split, classify by vol tercile ---
    bounds = _change_points(vol_pct)
    nb_pct = vol_pct[~bearb] if (~bearb).any() else vol_pct
    t1, t2 = np.quantile(nb_pct, [1 / 3, 2 / 3])
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        sel = ~bearb[s0:s1]
        v = vol_pct[s0:s1][sel].mean() if sel.any() else vol_pct[s0:s1].mean()
        kind[s0:s1] = "calm" if v <= t1 else ("build" if v <= t2 else "peak")

    # --- bear runs + recovery (1-2 bars before each bear -> non-bear transition) ---
    b = 0
    while b < n_bars:
        if bearb[b]:
            j = b
            while j < n_bars and bearb[j]:
                j += 1
            if j - b >= 2:                          # 1-bar dd blips are noise, not eras
                kind[b:j] = "bear"
                if j < n_bars:                      # run actually ends: build back up
                    r = 2 if j - b >= 4 else 1
                    kind[j - r:j] = "recovery"
            b = j
        else:
            b += 1

    # --- forced frame ---
    kind[:4] = "intro"
    kind[-2:] = "outro"

    # --- compress runs into segments, absorbing tiny non-structural slivers ---
    segs = []
    b = 0
    while b < n_bars:
        j = b
        while j < n_bars and kind[j] == kind[b]:
            j += 1
        segs.append(dict(bar0=b, bar1=j, kind=str(kind[b])))
        b = j
    merged = True
    while merged and len(segs) > 1:
        merged = False
        for i, s in enumerate(segs):
            structural = s["kind"] in ("intro", "outro", "recovery", "bear")
            if s["bar1"] - s["bar0"] < 5 and not structural:
                host = segs[i - 1] if i > 0 else segs[i + 1]
                if i > 0:
                    host["bar1"] = s["bar1"]
                else:
                    host["bar0"] = s["bar0"]
                segs.pop(i)
                # re-fuse neighbours that now share a kind
                k = 1
                while k < len(segs):
                    if (segs[k]["kind"] == segs[k - 1]["kind"]
                            and segs[k]["bar0"] == segs[k - 1]["bar1"]):
                        segs[k - 1]["bar1"] = segs[k]["bar1"]
                        segs.pop(k)
                    else:
                        k += 1
                merged = True
                break
    for s in segs:
        s["energy"] = ENERGY[s["kind"]]
    return segs


def section_mask(kind):
    """Layer gates + density multipliers for one section kind.
    hats/perc are density multipliers (0..1.5); stab_density thins the stab
    grammar when stabs are gated on; halftime asks the engine for a half-time
    kick/hat feel.  bear = breakdown: pads+lead only, as v3."""
    base = dict(kick=True, bass=True, stabs=True, lead=True, pads=True,
                hats=1.0, perc=1.0, stab_density=1.0, halftime=False)
    if kind == "intro":
        base.update(bass=False, stabs=False, lead=False, hats=0.3, perc=0.2,
                    stab_density=0.0)
    elif kind == "calm":
        base.update(stabs=False, lead=False, hats=0.4, perc=0.5,
                    stab_density=0.0, halftime=True)
    elif kind == "build":
        base.update(hats=0.8, perc=0.8, stab_density=0.5)
    elif kind == "peak":
        base.update(hats=1.3, perc=1.2)
    elif kind == "bear":
        base.update(kick=False, bass=False, stabs=False, hats=0.15, perc=0.0,
                    stab_density=0.0)
    elif kind == "recovery":                        # roll build handled by engine
        base.update(hats=1.0, perc=1.0, stab_density=0.7)
    elif kind == "outro":
        base.update(kick=False, stabs=False, lead=False, hats=0.3, perc=0.2,
                    stab_density=0.0)
    else:
        raise ValueError(f"unknown section kind: {kind!r}")
    return base


def is_bull(rs, bar, dp16):
    """True when price z-score vs its 252d MA > 1.0 at this bar's end
    (key-modulation trigger; same z as compose()'s bar harmony)."""
    rs = np.asarray(rs, float)
    price = np.cumsum(rs)
    ie = min((bar + 1) * 16 * dp16 - 1, len(rs) - 1)
    win = price[max(0, ie - 251):ie + 1]
    return bool((price[ie] - win.mean()) / max(win.std(), 1e-6) > 1.0)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    rs = np.load(os.path.join(here, "r_NVDA.npy"))
    n = len(rs)
    dp16 = max(2, round(n / 1536))
    n_bars = (n // dp16) // 16 * 16 // 16          # NBAR, as techno_album.compose
    segs = segments(rs, n_bars, dp16)
    print(f"NVDA: n={n} days, dp16={dp16}, n_bars={n_bars}, {len(segs)} segments")
    for s in segs:
        print(f"  bars {s['bar0']:3d}-{s['bar1']:3d}  {s['kind']:9s} energy={s['energy']:.2f}")
    # full coverage, no overlap, contiguous
    assert segs[0]["bar0"] == 0 and segs[-1]["bar1"] == n_bars
    for a, b in zip(segs[:-1], segs[1:]):
        assert a["bar1"] == b["bar0"], (a, b)
    for s in segs:
        assert s["bar1"] > s["bar0"] and s["kind"] in KINDS
        assert 0.0 <= s["energy"] <= 1.0
        section_mask(s["kind"])                     # every kind has a mask
    assert 6 <= len(segs) <= 14, len(segs)
    print("bull @ last bar:", is_bull(rs, n_bars - 1, dp16))
    print("self-test OK")
