"""synths.py — per-genre instrument bank for the Tech(no) US album.

Eight genres, eight sonic identities: techno, deephouse, dub, prog,
warehouse, breaks, trance, acid.  Every public function returns a mono
float64 numpy array at SR=44100, peak-normalized <= 1.0, all-finite.

API:
    drum(genre, name, vel)                     name in {kick,hatc,hato,clap,snare,crash,ride}
    bass_note(genre, midi, dur_s, vel, cutoff_hz, slide_from=None, accent=False)
    lead_note(genre, midi, dur_s, vel, cutoff_hz)
"""
import numpy as np
from scipy import signal

SR = 44100
GENRES = ("techno", "deephouse", "dub", "prog", "warehouse", "breaks", "trance", "acid",
          "duet")   # duet: the 09 closer's own voice — not any single-stock genre
_rng = np.random.default_rng(7)

midi2hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- helpers
def _wn(hz):
    """Guard a cutoff into a valid normalized Wn for scipy filters."""
    return float(np.clip(hz / (SR / 2), 0.001, 0.99))


def _lp(y, hz, order=2):
    b, a = signal.butter(order, _wn(hz), "low")
    return signal.lfilter(b, a, y)


def _hp(y, hz, order=2):
    b, a = signal.butter(order, _wn(hz), "high")
    return signal.lfilter(b, a, y)


def _bp(y, lo, hi, order=2):
    w0, w1 = _wn(lo), _wn(hi)
    if w1 <= w0:
        w1 = min(0.99, w0 + 0.01)
    b, a = signal.butter(order, [w0, w1], "band")
    return signal.lfilter(b, a, y)


def _t(dur):
    n = max(16, int(SR * dur))
    return np.arange(n) / SR


def _attack(y, ms=4.0):
    """3-8 ms linear attack ramp to kill clicks."""
    n = min(len(y), max(2, int(SR * ms / 1000)))
    y[:n] *= np.linspace(0.0, 1.0, n)
    return y


def _release(y, ms=6.0):
    """Short fade to zero at the very end (on top of the exp envelopes)."""
    n = min(len(y), max(2, int(SR * ms / 1000)))
    y[-n:] *= np.linspace(1.0, 0.0, n)
    return y


def _finish(y, level):
    """Peak-normalize then scale by level (clipped to 1) -> peak <= 1.0, finite."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.abs(y).max()
    if p > 1e-12:
        y = y / p * float(np.clip(level, 0.0, 1.0))
    return y


def _pitch_body(f0, f1, dur, tau, dec):
    """Sine body with exponential pitch envelope f0 -> f1 and exp amp decay."""
    t = _t(dur)
    phase = 2 * np.pi * (f1 * t + (f0 - f1) * tau * (1 - np.exp(-t / tau)))
    return np.sin(phase) * np.exp(-t * dec)


def _noise_burst(dur, dec):
    t = _t(dur)
    return _rng.standard_normal(len(t)) * np.exp(-t * dec)


def _saw(f, t):
    return 2.0 * ((f * t) % 1.0) - 1.0


def _saw_stack(f, t, cents, weights=None):
    y = np.zeros(len(t))
    if weights is None:
        weights = np.ones(len(cents))
    for c, w in zip(cents, weights):
        y += w * _saw(f * 2 ** (c / 1200.0), t)
    return y / max(np.sum(weights), 1e-9)


def _square_partials(t, base, ratios, dec):
    """Detuned square-wave partial cluster: the 909 metal core."""
    y = np.zeros(len(t))
    for r in ratios:
        y += np.sign(np.sin(2 * np.pi * base * r * t + _rng.uniform(0, np.pi)))
    return y / len(ratios) * np.exp(-t * dec)


_METAL_RATIOS = (1.0, 1.3420, 1.2312, 1.6532, 1.9523, 2.1523)  # 6 detuned partials


def _karplus(f, dur, damp=0.995):
    """Karplus-Strong plucked string."""
    period = max(2, int(round(SR / max(f, 20.0))))
    buf = _rng.uniform(-1.0, 1.0, period)
    n = max(16, int(SR * dur))
    out = np.empty(n)
    for i in range(n):
        j = i % period
        out[i] = buf[j]
        buf[j] = damp * 0.5 * (buf[j] + buf[(i + 1) % period])
    return out


# ---------------------------------------------------------------- drums
def _kick(genre, vel):
    g = genre
    if g == "techno":       # deep punchy: fast 92->44 Hz pitch env, 170 ms
        y = _pitch_body(92, 44, 0.170, 0.020, 15)
        c = _hp(_noise_burst(0.006, 500), 3000, 2)
        y[: len(c)] += 0.35 * c
        y = _attack(y, 3)
    elif g == "deephouse":  # round & boomy: 55 Hz fundamental, 300 ms tail, soft attack
        y = _pitch_body(72, 55, 0.300, 0.045, 7.5)
        y = _lp(y, 350, 2)
        y = _attack(y, 8)
    elif g == "dub":        # deep muffled sub thud, lowpassed 200 Hz
        y = _pitch_body(60, 42, 0.280, 0.040, 10)
        y = _lp(y, 200, 4)
        y = _attack(y, 6)
    elif g == "prog":       # tight mid-punch
        y = _pitch_body(130, 60, 0.110, 0.012, 26)
        knock = _bp(_noise_burst(0.030, 90), 150, 320, 2)
        y[: len(knock)] += 0.7 * knock
        y = _attack(y, 3)
    elif g == "warehouse":  # distorted hard kick: saturate the body
        y = _pitch_body(110, 50, 0.180, 0.016, 14)
        y = np.tanh(7.0 * y)
        y += 0.7 * _hp(y, 350, 2)                      # bring the clipped harmonics forward
        c = _hp(_noise_burst(0.005, 600), 2500, 2)
        y[: len(c)] += 0.5 * c
        y = _attack(y, 3)
    elif g == "breaks":     # 808-ish boom + snappy transient
        y = _pitch_body(85, 48, 0.350, 0.030, 7)
        snap = _hp(_noise_burst(0.012, 350), 2500, 2)
        y[: len(snap)] += 0.55 * snap
        y = _attack(y, 3)
    elif g == "trance":     # clicky attack + tight body
        y = _pitch_body(115, 52, 0.120, 0.014, 22)
        click = _hp(_noise_burst(0.007, 550), 4500, 2)
        y[: len(click)] += 2.2 * click
        y = _attack(y, 3)
    elif g == "acid":       # 909-like: tuned HIGH (60 Hz) and tight, bright click
        y = _pitch_body(112, 60, 0.130, 0.013, 22)
        c = _hp(_noise_burst(0.007, 450), 2500, 2)
        y[: len(c)] += 0.7 * c
        y = _attack(y, 3)
    else:                   # duet: very deep slow sub thud, 38 Hz, soft attack
        y = _pitch_body(85, 38, 0.300, 0.045, 8)
        y = _lp(y, 300, 2)
        y = _attack(y, 7)
    return _release(y, 5), 0.55 + 0.45 * vel


def _hat(genre, closed, vel):
    metallic = genre in ("techno", "acid", "trance", "warehouse", "prog", "breaks")
    dark = genre in ("dub", "deephouse", "duet")
    dur = (0.035 if closed else 0.22)
    if genre == "dub":
        dur = 0.030 if closed else 0.16
    t = _t(dur)
    if metallic:            # 6 detuned square partials, 909 style
        base = 316.0
        dec = 90 if closed else 16
        y = _square_partials(t, base, _METAL_RATIOS, dec)
        y += 0.5 * _rng.standard_normal(len(t)) * np.exp(-t * dec)
        hp = {"techno": 6800, "acid": 9500}.get(genre, 6500)   # acid sizzles, techno stays mid
        y = _hp(y, hp, 4)
    else:                   # darker filtered noise for dub / deephouse
        dec = 70 if closed else 14
        y = _rng.standard_normal(len(t)) * np.exp(-t * dec)
        y = _bp(y, 2500 if dark else 4000, 9000, 2)
        y = _lp(y, 8500, 2)
    y = _attack(y, 3)
    return _release(y, 4), (0.35 + 0.45 * vel) * (1.0 if closed else 0.9)


def _clap(genre, vel):
    bright = genre in ("techno", "trance", "warehouse", "acid")
    t = _t(0.20)
    y = np.zeros(len(t))
    for i, off in enumerate((0.0, 0.010, 0.021, 0.033)):     # pre-echo bursts + tail
        s = int(off * SR)
        seg = _rng.standard_normal(len(t) - s) * np.exp(-np.arange(len(t) - s) / SR * (28 if i == 3 else 90))
        y[s:] += seg * (1.0 if i == 3 else 0.7)
    lo, hi = (900, 4500) if bright else (600, 3200)
    y = _bp(y, lo, hi, 2)
    y = _attack(y, 3)
    return _release(y, 5), 0.5 + 0.5 * vel


def _snare(genre, vel):
    snappy = genre in ("breaks", "prog")
    t = _t(0.16 if snappy else 0.14)
    tone = np.sin(2 * np.pi * (200 if snappy else 180) * t) * np.exp(-t * 34)
    lo, hi = (1500, 6500) if snappy else (1100, 4800)
    noise = _bp(_rng.standard_normal(len(t)), lo, hi, 2) * np.exp(-t * (24 if snappy else 28))
    y = noise + (0.6 if snappy else 0.5) * tone
    y = _attack(y, 3)
    return _release(y, 5), 0.5 + 0.5 * vel


def _crash(genre, vel):
    t = _t(1.1)
    y = _rng.standard_normal(len(t)) * np.exp(-t * 4.2)
    y += 0.3 * _square_partials(t, 520.0, _METAL_RATIOS, 5)
    y = _hp(y, 5500 if genre in ("dub", "deephouse", "duet") else 6500, 4)
    y = _attack(y, 4)
    return _release(y, 20), 0.45 + 0.45 * vel


def _ride(genre, vel):
    dur = 1.1 if genre == "warehouse" else 0.6           # long metallic wash for warehouse
    t = _t(dur)
    dec = 3.0 if genre == "warehouse" else 7.0
    y = _square_partials(t, 470.0, _METAL_RATIOS, dec)
    y += 0.6 * _rng.standard_normal(len(t)) * np.exp(-t * dec * 1.4)
    y = _hp(y, 4500, 4)
    ping = np.sin(2 * np.pi * 3520 * t) * np.exp(-t * dec * 2)
    y += 0.25 * ping
    y = _attack(y, 3)
    return _release(y, 15), 0.35 + 0.45 * vel


def drum(genre, name, vel):
    """Render one drum hit. name in {kick,hatc,hato,clap,snare,crash,ride}."""
    if genre not in GENRES:
        raise ValueError(f"unknown genre {genre!r}")
    vel = float(np.clip(vel, 0.0, 1.0))
    if name == "kick":
        y, lvl = _kick(genre, vel)
    elif name == "hatc":
        y, lvl = _hat(genre, True, vel)
    elif name == "hato":
        y, lvl = _hat(genre, False, vel)
    elif name == "clap":
        y, lvl = _clap(genre, vel)
    elif name == "snare":
        y, lvl = _snare(genre, vel)
    elif name == "crash":
        y, lvl = _crash(genre, vel)
    elif name == "ride":
        y, lvl = _ride(genre, vel)
    else:
        raise ValueError(f"unknown drum {name!r}")
    return _finish(y, lvl)


# ---------------------------------------------------------------- bass
def bass_note(genre, midi, dur_s, vel, cutoff_hz, slide_from=None, accent=False):
    """One bass note in the genre's signature voice."""
    if genre not in GENRES:
        raise ValueError(f"unknown genre {genre!r}")
    vel = float(np.clip(vel, 0.0, 1.0))
    f = midi2hz(midi)
    t = _t(dur_s)
    dur = len(t) / SR
    lvl = 0.55 + 0.45 * vel
    env = np.exp(-t / max(dur * 0.6, 0.05))                    # exponential-ish release

    if genre == "techno":        # 2-voice detuned saw, warm
        y = _saw_stack(f, t, (-6.0, 6.0))
        y = _lp(y, cutoff_hz, 2)
        y = np.tanh(1.4 * y)                                   # gentle warmth
        y *= env
    elif genre == "deephouse":   # sine sub + short plucky top octave
        sub = np.sin(2 * np.pi * f * t) * env
        pluck = _saw(2 * f, t) * np.exp(-t * 22)
        pluck = _lp(pluck, min(cutoff_hz * 1.5, 4000), 2)
        y = sub + 0.35 * pluck
    elif genre == "dub":         # 2-op FM, mod ratio 2, low index, dark
        idx = 0.8 * (0.5 + 0.5 * vel) * np.exp(-t * 3)
        y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 2 * f * t))
        y = _lp(y, min(cutoff_hz, 800), 2)
        y *= env
    elif genre == "prog":        # Karplus-Strong-ish pluck
        y = _karplus(f, dur)
        y = _lp(y, cutoff_hz, 2)
        y *= np.exp(-t / max(dur * 0.8, 0.08))
    elif genre == "warehouse":   # saw through tanh saturation
        y = _saw(f, t)
        y = _lp(y, cutoff_hz, 2)
        y = np.tanh(3.0 * (0.6 + 0.8 * vel) * y)
        y *= env
    elif genre == "breaks":      # sub sine + click transient
        y = np.sin(2 * np.pi * f * t) * env
        click = _hp(_noise_burst(0.008, 400), 2000, 2)
        y[: len(click)] += 0.5 * click
    elif genre == "trance":      # 7-voice supersaw, ~18 cent spread
        y = _saw_stack(f, t, np.linspace(-9.0, 9.0, 7))
        y = _lp(y, cutoff_hz, 2)
        y *= env
    elif genre == "duet":        # hollow 2-op FM (ratio 3): the correlation voice
        idx = 1.3 * (0.4 + 0.6 * vel) * np.exp(-t * 4)
        y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 3 * f * t))
        y = _lp(y, min(cutoff_hz, 2200), 2)
        y *= env
    else:                        # acid: TB-303
        cut = cutoff_hz * (1.7 if accent else 1.0)
        if slide_from is not None:                             # exp glide over ~60 ms
            f_s = midi2hz(slide_from)
            tau = 0.020
            f_inst = f * (f_s / f) ** np.exp(-t / tau)
            phase = 2 * np.pi * np.cumsum(f_inst) / SR
            y = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
        else:
            y = _saw(f, t)
        y = _lp(y, cut, 2)                                     # cascade 2x 2nd-order = 4-pole-ish
        y = _lp(y, cut, 2)
        bpk, apk = signal.iirpeak(_wn(cut), Q=6.0)             # resonance at the corner
        y = y + 1.6 * signal.lfilter(bpk, apk, y)
        y *= np.exp(-t / max(dur * 0.5, 0.05))
        if accent:
            lvl = min(1.0, lvl * 1.25)

    y = _attack(y, 4)
    y = _release(y, 6)
    return _finish(y, lvl)


# ---------------------------------------------------------------- lead
def lead_note(genre, midi, dur_s, vel, cutoff_hz):
    """Brighter counterpart voice per genre."""
    if genre not in GENRES:
        raise ValueError(f"unknown genre {genre!r}")
    vel = float(np.clip(vel, 0.0, 1.0))
    t = _t(dur_s)
    dur = len(t) / SR
    lvl = 0.5 + 0.5 * vel
    env = np.exp(-t / max(dur * 0.7, 0.06))

    if genre == "trance":        # supersaw an octave up
        f = midi2hz(midi + 12)
        y = _saw_stack(f, t, np.linspace(-9.0, 9.0, 7))
        y = _lp(y, min(cutoff_hz * 1.6, 12000), 2)
        y *= env
    elif genre == "deephouse":   # EP-like FM chordal tone
        f = midi2hz(midi)
        idx = 1.6 * np.exp(-t * 6)
        y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
        y += 0.30 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t * 11)   # tine partial
        y += 0.45 * np.sin(2 * np.pi * f * 2 ** (3 / 12) * t + idx * np.sin(2 * np.pi * f * 2 ** (3 / 12) * t))
        y = _lp(y, min(cutoff_hz * 1.4, 9000), 2)
        y *= env
    elif genre == "dub":         # dark FM, spacious
        f = midi2hz(midi)
        idx = 1.5 * np.exp(-t * 2.5)
        y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 2 * f * t))
        y = _lp(y, min(cutoff_hz, 1400), 2)
        y *= env
    elif genre == "duet":        # glassy sine pair, slight octave shimmer
        f = midi2hz(midi)
        y = np.sin(2 * np.pi * f * t) + 0.55 * np.sin(2 * np.pi * 2.005 * f * t)
        y = _lp(y, min(cutoff_hz * 1.2, 8000), 2)
        y *= env
    else:                        # saw pluck variants
        f = midi2hz(midi)
        cents = {"techno": (-7.0, 7.0), "prog": (-5.0, 0.0, 5.0),
                 "warehouse": (-8.0, 8.0), "breaks": (-6.0, 6.0),
                 "acid": (0.0,)}[genre]
        y = _saw_stack(f, t, cents)
        y = _lp(y, min(cutoff_hz * 1.3, 10000), 2)
        if genre == "warehouse":
            y = np.tanh(2.5 * y)
        if genre == "acid":
            bpk, apk = signal.iirpeak(_wn(min(cutoff_hz * 1.3, 10000)), Q=5.0)
            y = y + 1.2 * signal.lfilter(bpk, apk, y)
        y *= np.exp(-t * (14 if genre in ("prog", "breaks") else 8)) + 0.15 * env

    y = _attack(y, 4)
    y = _release(y, 6)
    return _finish(y, lvl)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    DRUMS = ("kick", "hatc", "hato", "clap", "snare", "crash", "ride")

    def centroid(y):
        P = np.abs(np.fft.rfft(y)) ** 2                 # power-weighted spectral centroid
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        s = P.sum()
        return float((fr * P).sum() / s) if s > 1e-12 else 0.0

    def check(tag, y):
        assert isinstance(y, np.ndarray) and y.dtype == np.float64, f"{tag}: dtype {y.dtype}"
        assert y.ndim == 1, f"{tag}: not mono"
        assert np.all(np.isfinite(y)), f"{tag}: non-finite samples"
        p = np.abs(y).max()
        assert p <= 1.0 + 1e-9, f"{tag}: peak {p:.4f} > 1.0"
        return p

    n_checked = 0
    for g in GENRES:
        for d in DRUMS:
            for v in (0.7, 1.0):
                check(f"{g}/{d}", drum(g, d, v)); n_checked += 1
        for m in (40, 60):
            check(f"{g}/bass{m}", bass_note(g, m, 0.5, 0.9, 900)); n_checked += 1
            check(f"{g}/lead{m}", lead_note(g, m, 0.5, 0.9, 2200)); n_checked += 1
        # acid-specific paths
        check(f"{g}/bass-slide", bass_note(g, 52, 0.4, 0.9, 700, slide_from=40)); n_checked += 1
        check(f"{g}/bass-accent", bass_note(g, 40, 0.3, 1.0, 700, accent=True)); n_checked += 1

    cents = {g: centroid(drum(g, "kick", 1.0)) for g in GENRES}
    print(f"\n{'genre':>10s}  {'kick centroid (Hz)':>18s}")
    for g in GENRES:
        print(f"{g:>10s}  {cents[g]:18.1f}")
    spread = max(cents.values()) - min(cents.values())
    print(f"\ncentroid spread: {spread:.1f} Hz  ({n_checked} renders checked, all finite, peak<=1.0)")
    assert spread > 100.0, f"kick centroids too similar: spread {spread:.1f} Hz <= 100 Hz"
    print("self-test PASSED")
