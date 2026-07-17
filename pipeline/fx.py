"""fx.py -- effects rack for the TECH(NO) US pipeline (v2 engine companion).

Stereo convention: a stereo signal is a tuple (L, R) of 1-D float64 numpy
arrays of equal length.  Every function returns finite float64 output.

Performance note on the reverb combs: scipy.signal.lfilter on a sparse
denominator [1, 0, ..., -g] costs O(N * D) regardless of sparsity, which is
far too slow for D ~ 2000 taps on minutes of audio.  Because a feedback comb
only reaches back exactly D samples, it can be computed *exactly* in
D-sized blocks: y[b] = x[b] + g * LP(y[b-1]), with the one-pole damping
lowpass LP run by lfilter (2 coefficients) with carried state.  That keeps
everything vectorized; there is no per-sample python loop anywhere here.
"""
import numpy as np
from scipy import signal

__all__ = ["reverb", "chorus", "saturate", "tape_wobble", "pingpong",
           "sidechain", "vocoder"]


def _f64(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _finite(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- reverb
def _damped_comb(x, D, g, damp):
    """y[n] = x[n] + g * lp(y[n-D]);  lp = one-pole lowpass (coeff damp).
    Exact block recursion in blocks of D samples."""
    N = len(x)
    D = max(1, int(D))
    y = np.empty_like(x)
    m = min(D, N)
    y[:m] = x[:m]                       # y[n-D] == 0 for n < D
    if N <= D:
        return y
    b_lp = np.array([1.0 - damp])
    a_lp = np.array([1.0, -damp])
    zi = np.zeros(1)
    for s in range(D, N, D):
        e = min(s + D, N)
        fb, zi = signal.lfilter(b_lp, a_lp, y[s - D:s - D + (e - s)], zi=zi)
        y[s:e] = x[s:e] + g * fb
    return y


def _allpass(x, D, g):
    """y[n] = -g x[n] + x[n-D] + g y[n-D], exact block recursion."""
    N = len(x)
    D = max(1, int(D))
    y = np.empty_like(x)
    m = min(D, N)
    y[:m] = -g * x[:m]
    for s in range(D, N, D):
        e = min(s + D, N)
        y[s:e] = -g * x[s:e] + x[s - D:e - D] + g * y[s - D:e - D]
    return y


def reverb(L, R, sr, size=0.5, damp=0.5, mix=0.3):
    """Schroeder reverb: 4 parallel damped feedback combs + 2 series allpasses.
    Channel comb delays decorrelated by +/- 11 samples."""
    L, R = _f64(L), _f64(R)
    size = float(np.clip(size, 0.0, 1.0))
    damp = float(np.clip(damp, 0.0, 0.99))
    mix = float(np.clip(mix, 0.0, 1.0))
    base_ms = (25.3, 29.7, 36.1, 44.3)                 # 25-45 ms base
    scale = 0.55 + 0.9 * size                          # delays scaled by size
    g = min(0.97, 0.70 + 0.27 * size)                  # comb feedback
    outs = []
    for ch, x in enumerate((L, R)):
        wet = np.zeros_like(x)
        for i, bm in enumerate(base_ms):
            D = int(bm * scale * sr / 1000.0)
            D += (11 if i % 2 == 0 else -11) * (1 if ch else -1)  # decorrelate
            wet += _damped_comb(x, D, g, damp)
        wet *= 0.25
        wet = _allpass(wet, int(0.005 * sr), 0.5)      # 5 ms
        wet = _allpass(wet, int(0.0017 * sr), 0.5)     # 1.7 ms
        outs.append(_finite((1.0 - mix) * x + mix * wet))
    return outs[0], outs[1]


# ---------------------------------------------------------------- chorus
def chorus(L, R, sr, depth_ms=6, rate_hz=0.6, mix=0.4):
    """Modulated fractional-delay chorus, L/R LFOs 90 degrees apart."""
    L, R = _f64(L), _f64(R)
    N = len(L)
    n = np.arange(N, dtype=np.float64)
    dep = max(0.0, depth_ms) * 1e-3 * sr * 0.5         # LFO amplitude (samples)
    base = dep + 2.0                                   # keep tap causal
    mix = float(np.clip(mix, 0.0, 1.0))
    outs = []
    for x, ph in ((L, 0.0), (R, np.pi / 2)):
        lfo = dep * np.sin(2 * np.pi * rate_hz * n / sr + ph)
        read = np.clip(n - (base + lfo), 0.0, N - 1.0)
        wet = np.interp(read, n, x)
        outs.append(_finite((1.0 - mix) * x + mix * wet))
    return outs[0], outs[1]


# ---------------------------------------------------------------- saturate
def saturate(L, R, drive=1.5):
    """tanh soft clip with makeup so unity peaks stay unity."""
    L, R = _f64(L), _f64(R)
    drive = max(1e-6, float(drive))
    mk = 1.0 / np.tanh(drive)
    return _finite(np.tanh(drive * L) * mk), _finite(np.tanh(drive * R) * mk)


# ---------------------------------------------------------------- tape wobble
def tape_wobble(L, R, sr, depth_cents=8, rate_hz=0.9):
    """Slow pitch wobble: sinusoidal time-varying fractional delay whose
    slope gives +/- depth_cents of pitch deviation."""
    L, R = _f64(L), _f64(R)
    N = len(L)
    n = np.arange(N, dtype=np.float64)
    ratio = 2.0 ** (abs(depth_cents) / 1200.0) - 1.0   # peak pitch deviation
    rate_hz = max(1e-3, float(rate_hz))
    A = ratio * sr / (2 * np.pi * rate_hz)             # delay LFO amplitude
    base = A + 2.0
    d = base + A * np.sin(2 * np.pi * rate_hz * n / sr)
    read = np.clip(n - d, 0.0, N - 1.0)
    return (_finite(np.interp(read, n, L)),
            _finite(np.interp(read, n, R)))


# ---------------------------------------------------------------- ping-pong
def pingpong(L, R, sr, delay_s, fb=0.3, hp_hz=1200):
    """High-passed cross-feedback delay, loop unrolled to max 6 taps."""
    L, R = _f64(L), _f64(R)
    N = len(L)
    dl = int(max(1e-4, delay_s) * sr)
    fb = float(np.clip(fb, 0.0, 0.95))
    if dl < 1 or dl >= N or fb == 0.0:
        return _finite(L.copy()), _finite(R.copy())
    b, a = signal.butter(2, min(hp_hz, 0.45 * sr) / (sr / 2), "high")
    curL = signal.lfilter(b, a, L)
    curR = signal.lfilter(b, a, R)
    wetL = np.zeros(N)
    wetR = np.zeros(N)
    for _ in range(6):                                 # max 6 taps
        nL = np.zeros(N)
        nR = np.zeros(N)
        nL[dl:] = curR[:-dl] * fb                      # cross feedback
        nR[dl:] = curL[:-dl] * fb
        curL, curR = nL, nR
        wetL += curL
        wetR += curR
        if fb ** 2 * max(np.abs(curL).max(initial=0.0),
                         np.abs(curR).max(initial=0.0)) < 1e-6:
            break
    return _finite(L + wetL), _finite(R + wetR)


# ---------------------------------------------------------------- sidechain
def sidechain(L, R, sr, kick_samples, depth=0.55, rel_s=0.085):
    """The v2 techno_album.py duck: gain = 1 - depth*exp(-t/rel_s) over a
    0.30 s window from each kick onset, combined with np.minimum."""
    L, R = _f64(L), _f64(R)
    total = len(L)
    duck = np.ones(total)
    w = int(0.30 * sr)
    tt = np.arange(w) / sr
    shape = 1.0 - depth * np.exp(-tt / max(1e-6, rel_s))
    for ks in np.asarray(kick_samples, dtype=np.int64):
        if ks >= total or ks < 0:
            continue
        e = min(ks + w, total)
        duck[ks:e] = np.minimum(duck[ks:e], shape[:e - ks])
    return _finite(L * duck), _finite(R * duck)


# ---------------------------------------------------------------- vocoder
def vocoder(modulator, carrier, sr, n_bands=16):
    """Classic channel vocoder, mono in -> mono out.
    n_bands log-spaced butterworth (order 2) bandpasses over 80-8000 Hz;
    modulator band envelopes (rectify + 2nd-order 30 Hz lowpass) multiply
    the carrier bands; sum; peak-normalize."""
    mod = _f64(modulator)
    car = _f64(carrier)
    N = len(mod)
    if len(car) < N:                                   # loop carrier to length
        car = np.tile(car, int(np.ceil(N / max(1, len(car)))))
    car = car[:N]
    hi_lim = min(8000.0, 0.45 * sr)
    edges = np.geomspace(80.0, hi_lim, int(n_bands) + 1)
    lp_sos = signal.butter(2, 30.0, "low", fs=sr, output="sos")
    out = np.zeros(N)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sos = signal.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        mb = signal.sosfilt(sos, mod)
        cb = signal.sosfilt(sos, car)
        env = signal.sosfilt(lp_sos, np.abs(mb))       # rectify + lowpass
        out += cb * np.maximum(env, 0.0)
    out = _finite(out)
    peak = np.abs(out).max(initial=0.0)
    if peak > 1e-9:
        out *= 0.9 / peak
    return out


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    import time
    SR = 44100
    rng = np.random.default_rng(7)

    def mk(dur):
        t = np.arange(int(dur * SR)) / SR
        x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(len(t))
        y = 0.5 * np.sin(2 * np.pi * 331 * t) + 0.1 * rng.standard_normal(len(t))
        return x, y

    L, R = mk(2.0)
    N = len(L)

    def check(name, a, b=None, n=N):
        for tag, x in (("L", a),) + ((("R", b),) if b is not None else ()):
            assert isinstance(x, np.ndarray) and x.dtype == np.float64, (name, tag)
            assert x.shape == (n,), (name, tag, x.shape)
            assert np.isfinite(x).all(), (name, tag, "non-finite")
        print(f"OK {name}")

    a, b = reverb(L, R, SR, size=0.7, damp=0.4, mix=0.35); check("reverb", a, b)
    a, b = chorus(L, R, SR); check("chorus", a, b)
    a, b = saturate(L, R, drive=2.0); check("saturate", a, b)
    assert np.abs(a).max() <= 1.0 + 1e-9
    a, b = tape_wobble(L, R, SR); check("tape_wobble", a, b)
    a, b = pingpong(L, R, SR, delay_s=0.25, fb=0.4); check("pingpong", a, b)
    kicks = np.arange(0, N, SR // 2)
    a, b = sidechain(L, R, SR, kicks); check("sidechain", a, b)
    assert a[kicks[1]] == L[kicks[1]] * (1 - 0.55), "duck depth mismatch"
    v = vocoder(L, R, SR, n_bands=16); check("vocoder", v)
    assert np.abs(v).max() <= 0.9 + 1e-9

    # edge cases: silence and short input
    z = np.zeros(SR // 4)
    for f in (lambda: reverb(z, z, SR), lambda: chorus(z, z, SR),
              lambda: pingpong(z, z, SR, 0.5), lambda: tape_wobble(z, z, SR)):
        o = f()
        assert all(np.isfinite(c).all() for c in o)
    print("OK edge-cases")

    L60, R60 = mk(60.0)
    t0 = time.perf_counter()
    a, b = reverb(L60, R60, SR)
    dt = time.perf_counter() - t0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert dt < 10.0, f"reverb too slow: {dt:.2f}s"
    print(f"OK reverb-60s ({dt:.2f}s)")
    print("ALL OK")
