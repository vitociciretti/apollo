"""Tech(no) US — motif.py: per-stock theme + development operators.

The "make it feel composed" module.  Every stock gets one short THEME derived
deterministically from its first trading year (252 days of cumulative
log-price), so each track owns a recognizable melodic identity.  develop()
then applies classic motivic operators — inversion, transposition, diminution,
augmentation — selected by the *market data* of the current bar, so the theme
is developed the way a composer would, but the market picks the variation.

Pure functions on numpy arrays / ints; no imports from techno_album.py.

  derive(rs, scale, n=8)                 -> list of n scale degrees (0..13)
  develop(theme, z, bear, volpct, bar_idx) -> [(offset16, dur16, degree, accent)]
  to_midi(degree, scale, root)           -> midi note (two-octave indexing)
"""
import numpy as np

DEG_LO, DEG_HI = 0, 13          # two diatonic octaves of a 7-note scale
MAX_LEAP = 4                    # largest allowed successive leap (degrees)

# 16th-grid rhythm for an 8-note theme: leaves rests, syncopates beats 2 & 4.
GRID8 = [0, 2, 3, 6, 8, 10, 11, 14]


# ---------------------------------------------------------------- theme -----
def _grid(n):
    """Rhythm positions for n notes on a 16-step grid, leaving rests."""
    if n == 8:
        return list(GRID8)
    pos, prev = [], -1
    for i in range(n):
        p = int(round(i * 16 / max(n, 1)))
        p = max(p, prev + 1)            # strictly increasing
        pos.append(min(p, 15))
        prev = pos[-1]
    return pos


def derive(rs, scale, n=8):
    """Derive the stock's theme: n scale-degree ints (0..13), deterministic.

    First 252 days of cumulative log-price, resampled to n points, min-max
    normalized, mapped to degrees 0..9; then musicality is enforced:
    leaps clamped to <= MAX_LEAP degrees, note 0 forced to the tonic, and the
    contour must end within 2 degrees of its start.
    """
    rs = np.asarray(rs, dtype=float)
    price = np.cumsum(rs[:252])
    # resample the year to n points (segment means: robust to daily noise)
    edges = np.linspace(0, len(price), n + 1).astype(int)
    pts = np.array([price[edges[i]:max(edges[i + 1], edges[i] + 1)].mean()
                    for i in range(n)])
    lo, hi = pts.min(), pts.max()
    x = (pts - lo) / max(hi - lo, 1e-12)           # min-max to [0, 1]
    raw = np.rint(x * 9).astype(int)               # degrees 0..9
    theme = [0]                                    # note 0 = tonic
    for d in raw[1:]:
        prev = theme[-1]
        d = int(np.clip(d, prev - MAX_LEAP, prev + MAX_LEAP))  # tame leaps
        theme.append(int(np.clip(d, DEG_LO, DEG_HI)))
    # contour must come home: end within 2 degrees of the start.  Each 1-step
    # pull of the last note is followed by a backward pass re-clamping leaps,
    # so the cadence eases down instead of jumping (note 0 stays the tonic:
    # with theme[0]=0 the last note only ever moves down, so this converges).
    while abs(theme[-1] - theme[0]) > 2:
        theme[-1] += -1 if theme[-1] > theme[0] else 1
        for i in range(n - 2, 0, -1):
            theme[i] = int(np.clip(theme[i],
                                   theme[i + 1] - MAX_LEAP,
                                   theme[i + 1] + MAX_LEAP))
    return theme


# ---------------------------------------------------------- development -----
def _accents(seq):
    """Accent flags: True on contour peaks (local maxima) of the note list."""
    acc = []
    for i, d in enumerate(seq):
        left = seq[i - 1] if i > 0 else d - 1
        right = seq[i + 1] if i + 1 < len(seq) else d - 1
        acc.append(d >= left and d > right or (i == 0 and d > right))
    return acc


def _events(seq, positions, span=16):
    """Lay `seq` notes on `positions`; dur 1-2 sixteenths, accent on peaks."""
    acc = _accents(seq)
    ev = []
    for i, (p, d) in enumerate(zip(positions, seq)):
        nxt = positions[i + 1] if i + 1 < len(positions) else span
        dur = 2 if nxt - p >= 2 else 1
        ev.append((int(p), int(dur), int(np.clip(d, DEG_LO, DEG_HI)), bool(acc[i])))
    return ev


def develop(theme, z, bear, volpct, bar_idx):
    """One bar (16 sixteenths) of the theme, varied by the bar's market data.

    Priority (first match wins):
      bear          -> INVERSION: exact mirror around the theme's compass
                       midpoint (integer pivot lo+hi, so every melodic
                       interval is exactly negated: rises become falls)
      |z| > 1.5     -> TRANSPOSITION by int(z) degrees (stretched price shifts
                       the register), clamped to 0..13
      volpct > 0.8  -> DIMINUTION: theme squeezed into half a bar, played twice
      volpct < 0.3  -> AUGMENTATION: half the theme per bar, halves alternating
                       by bar parity (calm markets stretch time)
      else          -> PRIME form
    Returns [(offset16, dur16, degree, accent)].
    """
    theme = [int(d) for d in theme]
    n = len(theme)
    if bear:
        # exact mirror inside the theme's own compass: pivot = lo+hi is an
        # integer, so every interval is EXACTLY negated (no per-note rounding
        # or clipping — hi+lo-d always stays within [lo, hi] <= [0, 13]).
        p = min(theme) + max(theme)
        seq = [int(p - d) for d in theme]
        return _events(seq, _grid(n))
    if abs(z) > 1.5:
        t = int(z)
        seq = [int(np.clip(d + t, DEG_LO, DEG_HI)) for d in theme]
        return _events(seq, _grid(n))
    if volpct > 0.8:                                # diminution, twice per bar
        half = [min(p, 7) for p in _grid_half(n)]
        ev = _events(theme, half, span=8)
        return ev + [(p + 8, dur, d, a) for (p, dur, d, a) in ev]
    if volpct < 0.3:                                # augmentation, alternating
        h = max(n // 2, 1)
        seq = theme[:h] if bar_idx % 2 == 0 else theme[n - h:]
        pos = [int(round(i * 16 / h)) for i in range(h)]
        pos = [min(p, 15) for p in pos]
        acc = _accents(seq)
        return [(pos[i], 2, int(np.clip(seq[i], DEG_LO, DEG_HI)), bool(acc[i]))
                for i in range(h)]
    return _events(theme, _grid(n))                 # prime form


def _grid_half(n):
    """Strictly increasing positions for n notes inside 8 sixteenths."""
    pos, prev = [], -1
    for i in range(n):
        p = max(int(round(i * 8 / max(n, 1))), prev + 1)
        pos.append(min(p, 7))
        prev = pos[-1]
    return pos


# ------------------------------------------------------------------ midi ----
def to_midi(degree, scale, root):
    """Scale degree (two-octave indexing, 0..13) -> midi note."""
    L = len(scale)
    return int(root) + 12 * (degree // L) + scale[degree % L]


# ------------------------------------------------------------- self-test ----
if __name__ == "__main__":
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    AEOL = [0, 2, 3, 5, 7, 8, 10]
    NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "TSLA"]

    themes = {}
    for t in TICKERS:
        rs = np.load(os.path.join(HERE, f"r_{t}.npy"))
        th = derive(rs, AEOL)
        themes[t] = th
        assert len(th) == 8 and th[0] == 0, t
        assert all(DEG_LO <= d <= DEG_HI for d in th), t
        assert all(abs(a - b) <= MAX_LEAP for a, b in zip(th, th[1:])), t
        assert abs(th[-1] - th[0]) <= 2, t
        notes = [NAMES[to_midi(d, AEOL, 60) % 12] for d in th]
        print(f"{t:6s} degrees {th}  notes {'-'.join(notes)}")
    assert len({tuple(v) for v in themes.values()}) == len(TICKERS), \
        "themes are not all distinct"
    print("all 8 themes distinct: OK")

    # develop() across a data grid
    th = themes["NVDA"]
    n_ev = 0
    for z in (-3.2, -1.6, -0.4, 0.0, 0.7, 1.6, 2.9):
        for bear in (False, True):
            for volpct in (0.0, 0.15, 0.5, 0.85, 1.0):
                for bar_idx in (0, 1, 2, 3):
                    ev = develop(th, z, bear, volpct, bar_idx)
                    assert ev, (z, bear, volpct, bar_idx)
                    offs = [e[0] for e in ev]
                    assert offs == sorted(offs), "offsets not monotone"
                    for (o, dur, d, a) in ev:
                        assert 0 <= o <= 15, (o, z, bear, volpct)
                        assert dur in (1, 2), dur
                        assert DEG_LO <= d <= DEG_HI, d
                        assert isinstance(a, bool)
                        m = to_midi(d, AEOL, 33)
                        assert 0 <= m <= 127
                    n_ev += len(ev)
    print(f"develop() grid: {n_ev} events valid: OK")
    print("motif.py self-test PASSED")
