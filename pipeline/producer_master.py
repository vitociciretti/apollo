"""Mastering chain for the album: the automatable part of 'DAW production'.
Per track: tonal EQ (low shelf, mud cut, air shelf) -> bass mono below 120 Hz ->
gentle glue compression (1.5:1 above -12 dB: cohesion, NOT loudness) -> ONE
loudness normalization to target LUFS (proper gated BS.1770 via pyloudnorm)
-> true-peak brickwall limiter (8x-oversampled detector, lookahead + smooth
attack, ceiling 0.97 dBTP-safe) -> small makeup pass only if limiting pulled
the track under the loudness window -> 16-bit with TPDF dither.
The composed section dynamics must survive mastering: no repeated
normalize+limit ratchet, glue kept shallow.  Originals in premaster/.
"""
import numpy as np, wave, os, shutil
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d
import pyloudnorm as pyln

DIR = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
PRE = os.path.join(DIR, "premaster")
os.makedirs(PRE, exist_ok=True)
TARGET_LUFS = -9.5
SR = 44100

_METERS = {}

def lufs(L, R, sr):
    """Gated integrated loudness per ITU-R BS.1770-4 (pyloudnorm)."""
    m = _METERS.get(sr)
    if m is None:
        m = _METERS[sr] = pyln.Meter(sr)
    return float(m.integrated_loudness(np.stack([L, R], axis=1)))

def shelf(x, sr, f0, gain_db, kind):
    """first-order shelf via mixing a filtered band back in"""
    b, a = signal.butter(2, f0/(sr/2), "low" if kind == "low" else "high")
    band = signal.lfilter(b, a, x)
    return x + (10**(gain_db/20) - 1)*band

def peak_cut(x, sr, f0, gain_db, q=1.0):
    w0 = f0/(sr/2)
    b, a = signal.iirpeak(w0, q)
    band = signal.lfilter(b, a, x)
    return x + (10**(gain_db/20) - 1)*band

def compress(L, R, sr, thresh_db=-9, ratio=1.35, atk=0.03, rel=0.2):
    det = np.maximum(np.abs(L), np.abs(R))
    # RMS-ish smoothing of the detector
    a_atk, a_rel = np.exp(-1/(atk*sr)), np.exp(-1/(rel*sr))
    env = np.empty_like(det); e = 0.0
    for i, d in enumerate(det):
        a = a_atk if d > e else a_rel
        e = a*e + (1-a)*d
        env[i] = e
    env_db = 20*np.log10(env + 1e-9)
    over = np.maximum(env_db - thresh_db, 0)
    gain = 10**(-(over*(1 - 1/ratio))/20)
    return L*gain, R*gain

def _tp_det(L, R, os=8):
    """Per-sample TRUE-peak detector: |.| of the os-times oversampled signal,
    folded back to one max per base-rate sample, floored by the sample peak."""
    n = len(L)
    Lu = signal.resample_poly(L, os, 1)[:n*os]
    Ru = signal.resample_poly(R, os, 1)[:n*os]
    det = np.maximum(np.abs(Lu), np.abs(Ru)).reshape(n, os).max(axis=1)
    return np.maximum(det, np.maximum(np.abs(L), np.abs(R)))

def true_peak(L, R, os=8):
    Lu = signal.resample_poly(L, os, 1)
    Ru = signal.resample_poly(R, os, 1)
    return max(np.abs(Lu).max(), np.abs(Ru).max())

def _release_env(g_target, sr, rel_s=0.080):
    """Envelope e with e[i] <= g_target[i] and exponential release toward 1:
    deficit d[i] = max(1-g_target[i], a*d[i-1]) computed vectorized in the
    log domain (running max of log x[j] + j*c, c = 1/(rel_s*sr))."""
    x = np.maximum(1.0 - g_target, 0.0)
    c = 1.0/(rel_s*sr)
    j = np.arange(len(x), dtype=np.float64)
    with np.errstate(divide="ignore"):
        lx = np.log(x) + j*c
    d = np.exp(np.maximum.accumulate(lx) - j*c)
    return 1.0 - d

def transient_shave(L, R, sr, ceiling=0.97):
    """Fast pre-limiter peak shave (~1 ms smoothing): clips the kick/snare
    transient tips that carry almost no loudness, so the true-peak limiter
    barely works and the section RMS contour (the composed dynamics)
    survives loudness normalization."""
    w = max(4, int(0.001*sr))
    det = np.maximum(np.abs(L), np.abs(R))
    g = np.minimum(1.0, ceiling/np.maximum(det, 1e-9))
    g = uniform_filter1d(minimum_filter1d(g, size=w), size=w)
    return L*g, R*g

def limit(L, R, sr, ceiling=0.97):
    """True-peak brickwall limiter. The detector is the 8x-oversampled peak
    (not the sample peak), the gain envelope gets a 2 ms lookahead minimum +
    2 ms averaged attack (still <= required gain pointwise) and an 80 ms
    exponential release — no hard clipping, so no intersample overshoot is
    manufactured. Re-detects up to 3x, then a global trim guarantees the
    true-peak ceiling."""
    la = max(2, int(0.002*sr))
    for _ in range(3):
        det = _tp_det(L, R)
        g_req = np.minimum(1.0, ceiling/np.maximum(det, 1e-9))
        if g_req.min() > 0.9995:
            break
        g_min = minimum_filter1d(g_req, size=2*la+1, mode="nearest")
        g_sm = uniform_filter1d(g_min, size=la, mode="nearest")  # smooth attack
        g = _release_env(g_sm, sr, rel_s=0.050)   # <= g_sm pointwise, fast release
        L, R = L*g, R*g
    tp = true_peak(L, R)
    if tp > ceiling:
        L, R = L*(ceiling/tp), R*(ceiling/tp)
    return L, R

def master(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)/32767
    L, R = raw[0::2].copy(), raw[1::2].copy()
    # EQ: +1.2 dB low shelf 90 Hz, -1.5 dB mud 300 Hz, +1.8 dB air 8 kHz
    for ch in (L, R):
        ch[:] = shelf(ch, sr, 90, 1.2, "low")
        ch[:] = peak_cut(ch, sr, 300, -1.5, q=0.9)
        ch[:] = shelf(ch, sr, 8000, 1.8, "high")
    # bass mono below 120 Hz
    b, a = signal.butter(2, 120/(sr/2), "low")
    side = (L - R)/2
    side_lo = signal.lfilter(b, a, side)
    L, R = L - side_lo, R + side_lo
    # glue (shallow: cohesion, not loudness — deep glue flattens the form)
    L, R = compress(L, R, sr)
    # ONE loudness normalization to target, then the limiter.  No repeated
    # normalize+limit loop: re-boosting after limiting ratchets the loud
    # sections into the ceiling and erases the composed dynamics.
    cur = lufs(L, R, sr)
    g = 10**((TARGET_LUFS - cur)/20)
    L, R = L*g, R*g
    L, R = transient_shave(L, R, sr)   # tips first: loudness-free peaks
    L, R = limit(L, R, sr)
    final = lufs(L, R, sr)
    for _ in range(3):                 # small CAPPED makeups only — never the
        if final >= TARGET_LUFS - 0.35:   # old re-normalize ratchet
            break
        g = 10**(min(TARGET_LUFS - final, 1.2)/20)
        L, R = transient_shave(L*g, R*g, sr)
        L, R = limit(L, R, sr)
        final = lufs(L, R, sr)
    if final > TARGET_LUFS + 0.4:      # too hot for the window: plain trim
        g = 10**((TARGET_LUFS - final)/20)
        L, R = L*g, R*g
        final = lufs(L, R, sr)
    # 16-bit TPDF dither
    dith = (np.random.rand(len(L)) - np.random.rand(len(L)))/32767
    out = np.empty(2*len(L), dtype=np.int16)
    out[0::2] = np.clip((L + dith)*32767, -32767, 32767).astype(np.int16)
    out[1::2] = np.clip((R + dith)*32767, -32767, 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(out.tobytes())
    return final, np.abs(np.stack([L, R])).max(), true_peak(L, R)

if __name__ == "__main__":
    for f in sorted(os.listdir(DIR)):
        if f.endswith(".wav"):
            src = os.path.join(DIR, f)
            pre = os.path.join(PRE, f)
            if os.path.exists(pre):
                shutil.copy2(pre, src)          # re-master from the clean premaster
            else:
                shutil.copy2(src, pre)
            lu, pk, tp = master(src)
            print(f"{f:28s} -> {lu:6.1f} LUFS, peak {pk:.3f}, true peak {tp:.3f}")
