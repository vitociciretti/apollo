"""choir.py -- melodic vocal system for the TECH(NO) US pipeline.

Fix for "voice too robotic": the v1 failure was a monotone SAPI voice
vocoded flat onto a chord.  The fix here is threefold:
  1. sing()  -- pitch-mapped melody: the TTS take is segmented into
     syllable chunks, each chunk phase-vocoder stretched to its note
     duration and pitch-shifted onto its melody note (per-chunk f0
     measured by autocorrelation; shift clamped to +/-8 st from the
     source pitch to keep formants acceptable -- when the melody sits
     far from the speaking register the TTS itself is re-rendered at a
     raised/lowered SSML prosody pitch so the clamp never bites hard).
  2. choir() -- 5 sing() layers at unison/+12/+7/-12/+3 (octave-folded
     into range), onset jitter, cent detune, ramping vibrato, then ONE
     fx.vocoder pass of the stacked voices onto a soft-saw carrier
     stack of the same melody notes, mixed 60 vocoded / 40 dry.
  3. ooh()   -- wordless formant-synth choir (pulse-train glottal
     source through 3 formant bandpasses), no TTS at all.

TTS: tries piper-tts with a locally available voice model first (better
timbre, never downloads anything); falls back to Windows SAPI via
PowerShell System.Speech.  All audio float64 mono internally, SR=44100.
"""
import os
import wave
import hashlib
import subprocess

import numpy as np
from scipy import signal

import fx

SR = 44100
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRATCH_DEFAULT = (r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude"
                    r"\C--Users-vito-ciciretti"
                    r"\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad")

__all__ = ["say", "sing", "choir", "ooh", "SR"]


def _scratch():
    d = os.environ.get("CHOIR_SCRATCH", _SCRATCH_DEFAULT)
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return _HERE


def midi2hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def hz2midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f, dtype=np.float64) / 440.0)


def _finite(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ TTS
_PIPER = None            # None = untried, False = unavailable, else voice


def _find_piper_voice():
    """Look for an already-downloaded piper voice model (never downloads)."""
    dirs = [os.environ.get("PIPER_VOICES", ""), _scratch(), _HERE]
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".onnx") and os.path.exists(
                    os.path.join(d, f + ".json")):
                return os.path.join(d, f)
    return None


def _piper_say(text, out_wav):
    global _PIPER
    if _PIPER is False:
        return False
    if _PIPER is None:
        try:
            from piper import PiperVoice
            mp = _find_piper_voice()
            _PIPER = PiperVoice.load(mp) if mp else False
        except Exception:
            _PIPER = False
    if not _PIPER:
        return False
    try:
        with wave.open(out_wav, "wb") as w:
            _PIPER.synthesize_wav(text, w)
        return True
    except Exception:
        return False


def say(text, out_wav, rate=-2, pitch_pct=0):
    """Render `text` to a mono 16-bit wav at `out_wav`.

    Uses a local piper-tts voice if one is installed, else Windows SAPI
    (System.Speech via PowerShell).  `rate` is the SAPI rate (-10..10);
    `pitch_pct` shifts the SAPI prosody pitch (SSML percent, e.g. 40)."""
    if _piper_say(text, out_wav):
        return out_wav
    esc = str(text).replace("'", "''").replace("<", " ").replace(">", " ")
    esc = esc.replace("&", " and ")
    path = str(out_wav).replace("'", "''")
    if int(round(pitch_pct)) != 0:
        sign = "+" if pitch_pct >= 0 else ""
        body = ('$ssml = \'<speak version="1.0" '
                'xmlns="http://www.w3.org/2001/10/synthesis" '
                'xml:lang="en-US"><prosody pitch="{}{}%">{}</prosody>'
                '</speak>\'; $s.SpeakSsml($ssml);'
                .format(sign, int(round(pitch_pct)), esc))
    else:
        body = "$s.Speak('{}');".format(esc)
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = {rate}; "
        "try {{ $s.SelectVoice('Microsoft Zira Desktop') }} catch {{}}; "
        "$s.SetOutputToWaveFile('{path}', $fmt); "
        "{body} $s.Dispose();"
    ).format(rate=int(rate), path=path, body=body)
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                    "-Command", ps], check=True, capture_output=True)
    return out_wav


def _load_wav_mono(path, sr=SR):
    with wave.open(path, "rb") as w:
        sr0, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483647.0
    else:                                       # 8-bit unsigned
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 127.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    if sr0 != sr:
        n_out = int(round(len(x) * sr / sr0))
        x = np.interp(np.linspace(0.0, len(x) - 1.0, n_out),
                      np.arange(len(x)), x)
    return _finite(x)


_TTS_CACHE = {}


def _tts(text, rate=-2, pitch_pct=0, sr=SR):
    key = (text, int(rate), int(round(pitch_pct)), sr)
    if key in _TTS_CACHE:
        return _TTS_CACHE[key]
    h = hashlib.md5(("%s|%d|%d" % (text, rate, round(pitch_pct)))
                    .encode("utf-8")).hexdigest()[:12]
    p = os.path.join(_scratch(), "tts_%s.wav" % h)
    if not (os.path.exists(p) and os.path.getsize(p) > 1000):
        say(text, p, rate=rate, pitch_pct=pitch_pct)
    x = _load_wav_mono(p, sr)
    _TTS_CACHE[key] = x
    return x


# --------------------------------------------------- pitch (autocorr f0)
def _f0_frames(x, sr, fmin=70.0, fmax=500.0, frame=2048, hop=512,
               ref_hz=None, tol_st=7.0):
    """Per-frame autocorrelation f0 estimates of voiced frames (Hz).
    With ref_hz set, estimates further than tol_st semitones from the
    reference are discarded (kills fricative false positives)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < frame:
        x = np.pad(x, (0, frame + 1 - len(x)))
    win = np.hanning(frame)
    starts = range(0, len(x) - frame, hop)
    rms = np.array([np.sqrt(np.mean(x[s:s + frame] ** 2)) for s in starts])
    if not len(rms) or rms.max() < 1e-6:
        return np.array([])
    gate = 0.2 * rms.max()
    lo = max(2, int(sr / fmax))
    hi = min(int(sr / fmin), frame - 2)
    f0s = []
    for s, r_ in zip(starts, rms):
        if r_ < gate:
            continue
        seg = x[s:s + frame]
        seg = (seg - seg.mean()) * win
        r = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * frame)) ** 2)[:frame]
        if r[0] < 1e-12:
            continue
        r /= r[0]
        sub = r[lo:hi]
        m = sub.max()
        if m < 0.35:                              # unvoiced frame
            continue
        pk = np.where((sub[1:-1] >= sub[:-2]) & (sub[1:-1] >= sub[2:]) &
                      (sub[1:-1] >= 0.85 * m))[0] + 1
        if not len(pk):
            continue
        k = lo + pk[0]                            # smallest strong lag
        for _ in range(2):                        # octave-up disambiguation:
            j = int(round(k / 2.0))               # a strong half-lag peak
            if j < lo + 1 or j + 3 > len(r):      # means k was a subharmonic
                break
            jj = j - 2 + int(np.argmax(r[j - 2:j + 3]))
            if r[jj] >= max(0.8 * r[int(round(k))], 0.3) and \
                    r[jj] >= r[jj - 1] and r[jj] >= r[jj + 1]:
                k = jj
            else:
                break
        if 1 <= k < frame - 1:                    # parabolic refinement
            a, b, c = r[k - 1], r[k], r[k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-12:
                k = k + 0.5 * (a - c) / den
        f0s.append(sr / k)
    f0s = np.array(f0s)
    if ref_hz is not None and len(f0s):
        f0s = f0s[np.abs(hz2midi(f0s) - hz2midi(ref_hz)) <= tol_st]
    return f0s


def _median_f0(x, sr, default=185.0, **kw):
    f = _f0_frames(x, sr, **kw)
    return float(np.median(f)) if len(f) >= 3 else float(default)


# --------------------------------------------- syllable segmentation
def _segment(x, sr, n):
    """Split a TTS take into n syllable-ish chunks: trim silence, then
    place boundaries at energy minima near the equal-split positions
    (equal split is the fallback when no clear minimum exists)."""
    x = np.asarray(x, dtype=np.float64)
    w = max(8, int(0.020 * sr))
    ker = np.hanning(2 * w)
    ker /= ker.sum()
    env = np.convolve(np.abs(x), ker, mode="same")
    thr = 0.05 * (env.max() + 1e-12)
    idx = np.where(env > thr)[0]
    if len(idx) > w:
        x = x[idx[0]:idx[-1] + 1]
        env = env[idx[0]:idx[-1] + 1]
    L = len(x)
    if n <= 1 or L < 4 * n:
        return [x] * max(1, n) if L else [np.zeros(int(0.1 * sr))] * max(1, n)
    min_len = max(4, min(int(0.06 * sr), L // (2 * n)))
    bounds = [0]
    for i in range(1, n):
        c = int(round(i * L / n))
        wnd = max(2, int(L / (3 * n)))
        lo = max(bounds[-1] + min_len, c - wnd)
        hi = min(L - (n - i) * min_len, c + wnd)
        b = c if hi <= lo else lo + int(np.argmin(env[lo:hi]))
        b = int(np.clip(b, bounds[-1] + 1, L - (n - i)))
        bounds.append(b)
    bounds.append(L)
    # melisma rule: a chunk with no voiced core (final fricatives etc.)
    # extends backward over the previous vowel, like a singer holding it
    ref = _median_f0(x, sr)
    chunks = []
    step = max(4, int(0.05 * sr))
    for a, b in zip(bounds[:-1], bounds[1:]):
        ch = x[a:b]
        tries = 0
        while a > 0 and tries < 12 and \
                len(_f0_frames(ch, sr, ref_hz=ref)) < 4:
            a = max(0, a - step)
            ch = x[a:b]
            tries += 1
        chunks.append(ch)
    return chunks


# --------------------------------------------------- phase vocoder core
def _stft(x, n_fft, hop):
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = (np.arange(n_fft)[None, :] +
           hop * np.arange(n_frames)[:, None])
    return np.fft.rfft(xp[idx] * np.hanning(n_fft)[None, :], axis=1)


def _istft(S, n_fft, hop):
    win = np.hanning(n_fft)
    F = S.shape[0]
    N = n_fft + hop * (F - 1)
    frames = np.fft.irfft(S, n=n_fft, axis=1) * win[None, :]
    idx = np.arange(n_fft)[None, :] + hop * np.arange(F)[:, None]
    y = np.zeros(N)
    wsum = np.zeros(N)
    np.add.at(y, idx, frames)
    np.add.at(wsum, idx, (win ** 2)[None, :] * np.ones((F, 1)))
    y /= np.maximum(wsum, 1e-8)
    pad = n_fft // 2
    return y[pad:-pad] if N > 2 * pad else y


def _pv_stretch(x, factor, n_fft=1024, hop=256):
    """Phase-vocoder time-stretch: output length ~= factor * len(x),
    pitch preserved.  scipy-STFT-style framing, hop-ratio stretching."""
    x = np.asarray(x, dtype=np.float64)
    factor = max(float(factor), 1e-3)
    if len(x) < n_fft // 2:
        x = np.pad(x, (0, n_fft // 2 - len(x)))
    S = _stft(x, n_fft, hop)
    F = S.shape[0]
    if F < 2:
        n_out = int(round(len(x) * factor))
        return np.interp(np.linspace(0, len(x) - 1, max(n_out, 2)),
                         np.arange(len(x)), x)
    steps = np.arange(0.0, F - 1.0, 1.0 / factor)
    i0 = steps.astype(int)
    frac = (steps - i0)[:, None]
    mag = (1.0 - frac) * np.abs(S[i0]) + frac * np.abs(S[i0 + 1])
    adv = 2.0 * np.pi * hop * np.arange(S.shape[1]) / n_fft
    dphi = np.angle(S[i0 + 1]) - np.angle(S[i0]) - adv[None, :]
    dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
    inc = adv[None, :] + dphi
    ph = np.empty_like(inc)
    ph[0] = np.angle(S[0])
    ph[1:] = np.cumsum(inc[:-1], axis=0) + ph[0][None, :]
    return _istft(mag * np.exp(1j * ph), n_fft, hop)


def _shift_stretch(x, semitones, out_len, sr):
    """Pitch-shift by `semitones` AND stretch to exactly `out_len`
    samples: PV time-stretch to out_len * rate, then resample by rate."""
    rate = 2.0 ** (float(semitones) / 12.0)
    out_len = max(int(out_len), 8)
    y = _pv_stretch(x, out_len * rate / max(len(x), 1))
    if rate > 1.02:                       # anti-alias before decimation
        fc = min(0.45 * sr / rate, 0.49 * sr)
        sos = signal.butter(4, fc, "low", fs=sr, output="sos")
        y = signal.sosfiltfilt(sos, y)
    pos = np.clip(np.arange(out_len) * rate, 0, len(y) - 1)
    return _finite(np.interp(pos, np.arange(len(y)), y))


def _vibrato(y, sr, cents=15.0, rate_hz=5.0, delay=0.15, ramp=0.20,
             phase=0.0):
    """5 Hz pitch vibrato ramping in after `delay` seconds."""
    n = len(y)
    if n < 8:
        return y
    t = np.arange(n) / sr
    depth = cents * np.clip((t - delay) / max(ramp, 1e-6), 0.0, 1.0)
    r = 2.0 ** (depth * np.sin(2.0 * np.pi * rate_hz * t + phase) / 1200.0)
    pos = np.cumsum(r)
    pos -= pos[0]
    pos = np.clip(pos, 0, n - 1)
    return np.interp(pos, np.arange(n), y)


# ----------------------------------------------------------------- sing
def _parse_melody(melody_midi, note_dur):
    notes, durs = [], []
    for m in melody_midi:
        if isinstance(m, (tuple, list)):
            notes.append(float(m[0]))
            durs.append(float(m[1]))
        else:
            notes.append(float(m))
            durs.append(float(note_dur))
    return notes, durs


def _overlap_concat(segs, sr, xfade_s=0.010):
    """Concatenate segments with equal-power crossfades of xfade_s."""
    xf = max(2, int(xfade_s * sr))
    fade_in = np.sin(0.5 * np.pi * np.linspace(0, 1, xf)) ** 2
    fade_out = fade_in[::-1]
    steps = [len(s) - xf for s in segs]
    total = sum(steps) + xf
    y = np.zeros(total)
    pos = 0
    for s in segs:
        s = s.copy()
        s[:xf] *= fade_in
        s[-xf:] *= fade_out
        y[pos:pos + len(s)] += s
        pos += len(s) - xf
    return y


def _sing_chunk(chunk, note, n_out, sr, f0_fallback, max_shift):
    """Retune one syllable chunk onto `note`, flattening the spoken
    prosody glide: the chunk is processed in ~120 ms sub-blocks, each
    pitch-shifted from its own measured f0 to the note (clamped to
    +/- max_shift), then crossfaded back together at n_out samples."""
    xfb = max(2, int(0.010 * sr))
    n_sub = max(1, int(round(len(chunk) / (0.12 * sr))))
    edges = np.linspace(0, len(chunk), n_sub + 1).astype(int)
    blocks = [chunk[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if not blocks:
        blocks = [chunk]
    base = max(n_out - xfb, 8)
    f0bs, voiced = [], []
    for b in blocks:
        f = _f0_frames(b, sr, ref_hz=f0_fallback)
        voiced.append(len(f) >= 2)
        f0bs.append(float(np.median(f)) if len(f) >= 2 else f0_fallback)
    # a singer stretches the vowel, not the consonant: voiced sub-blocks
    # soak up the note duration, unvoiced ones stay near natural length
    w = np.array([len(b) * (4.0 if v else 1.0)
                  for b, v in zip(blocks, voiced)])
    if not any(voiced):
        w = np.array([float(len(b)) for b in blocks])
    alloc = np.maximum(np.round(base * w / w.sum()).astype(int), 4)
    alloc[-1] = max(base - int(alloc[:-1].sum()), 4)
    outs = []
    for b, a, f0b in zip(blocks, alloc, f0bs):
        s = float(np.clip(note - hz2midi(f0b), -max_shift, max_shift))
        outs.append(_shift_stretch(b, s, int(a) + xfb, sr))
    seg = _overlap_concat(outs, sr) if len(outs) > 1 else outs[0]
    if len(seg) < n_out:
        seg = np.pad(seg, (0, n_out - len(seg)))
    return seg[:n_out]


def sing(text, melody_midi, sr=SR, note_dur=0.5, max_shift=8.0,
         vibrato_cents=0.0, tts_rate=-2, rng=None, _tts_x=None):
    """Sing `text` on `melody_midi` (list of MIDI notes, or (note, dur)
    tuples; plain notes get `note_dur` seconds each) -> mono float64.

    The TTS take is split into len(melody) syllable chunks; each chunk
    is phase-vocoder stretched to its note duration and pitch-shifted
    from its own measured f0 onto the melody note (shift clamped to
    +/- max_shift semitones to keep formants acceptable).  If the melody
    median sits > 5 st from the spoken median pitch, the TTS source is
    re-rendered at a raised/lowered prosody pitch first."""
    notes, durs = _parse_melody(melody_midi, note_dur)
    if not notes:
        return np.zeros(int(0.1 * sr))
    if rng is None:
        rng = np.random.default_rng(0)

    if _tts_x is None:
        x = _tts(text, tts_rate, 0, sr)
        med = _median_f0(x, sr)
        delta = float(np.median(notes)) - hz2midi(med)
        if abs(delta) > 5.0:                    # re-render nearer register
            pct = np.clip((2.0 ** (delta / 12.0) - 1.0) * 100.0, -45.0, 90.0)
            x = _tts(text, tts_rate, pct, sr)
    else:
        x = _tts_x

    med_all = _median_f0(x, sr)
    rms_take = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    chunks = _segment(x, sr, len(notes))
    xf = max(2, int(0.010 * sr))
    segs = []
    for chunk, note, dur in zip(chunks, notes, durs):
        f = _f0_frames(chunk, sr, ref_hz=med_all)
        crms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        f0c = float(np.median(f)) if len(f) >= 4 else med_all
        if crms < 0.25 * rms_take:
            f0c = med_all                  # too quiet: distrust estimate
        n_out = int(dur * sr) + xf
        seg = _sing_chunk(chunk, note, n_out, sr, f0c, max_shift)
        # closed loop: measure what actually came out, fix the residual
        f = _f0_frames(seg, sr, fmin=70.0, fmax=800.0,
                       ref_hz=float(midi2hz(note)), tol_st=6.0)
        if len(f) >= 3:
            err = float(hz2midi(np.median(f))) - note
            if 0.3 < abs(err) < 6.0:
                seg = _shift_stretch(seg, -err, n_out, sr)
        if vibrato_cents > 0:
            seg = _vibrato(seg, sr, vibrato_cents,
                           phase=rng.uniform(0, 2 * np.pi))
        segs.append(seg)
    y = _overlap_concat(segs, sr)
    peak = np.abs(y).max(initial=0.0)
    if peak > 1e-9:
        y *= 0.9 / peak
    return _finite(y)


# ---------------------------------------------------------------- choir
_OFFSETS = (0, 12, 7, -12, 3)
_GAINS = (1.00, 0.26, 0.30, 0.16, 0.24)


def _soft_saw(f_hz, n, sr, rng, cents=6.0):
    """Two slightly detuned saws, lowpassed -> soft choir carrier."""
    t = np.arange(n)
    y = np.zeros(n)
    for c in (-cents, cents):
        f = f_hz * 2.0 ** (c / 1200.0)
        ph = (f * t / sr + rng.uniform()) % 1.0
        y += 2.0 * ph - 1.0
    sos = signal.butter(2, min(4000.0, 0.45 * sr), "low", fs=sr,
                        output="sos")
    return signal.sosfilt(sos, 0.5 * y)


def _carrier_stack(notes, durs, offsets, gains, sr, rng):
    """Soft-saw stack following the melody (same notes as the voices)."""
    xf = max(2, int(0.010 * sr))
    segs = []
    for note, dur in zip(notes, durs):
        n = int(dur * sr) + xf
        seg = np.zeros(n)
        for off, g in zip(offsets, gains):
            if off < 0:                    # keep the sub-octave discreet:
                g = min(g, 0.12)           # the unison must own the pitch
            seg += g * _soft_saw(midi2hz(note + off), n, sr, rng)
        segs.append(seg)
    car = _overlap_concat(segs, sr)
    peak = np.abs(car).max(initial=0.0)
    if peak > 1e-9:
        car *= 0.9 / peak
    return car


def choir(text, melody_midi, sr=SR, voices=5, note_dur=0.5, tts_rate=-2,
          seed=11):
    """Layered melodic choir -> mono float64.

    `voices` sing() layers at unison/+12/+7/-12/+3 (offsets octave-folded
    so no voice shifts more than ~15 st from the TTS pitch), each with
    +/-12 ms onset jitter, +/-8 cent detune and 5 Hz / 15 cent vibrato
    ramping in after 150 ms; the stacked group then gets ONE fx.vocoder
    pass onto a soft-saw carrier stack of the same melody notes, mixed
    60% vocoded / 40% dry."""
    rng = np.random.default_rng(seed)
    notes, durs = _parse_melody(melody_midi, note_dur)
    if not notes:
        return np.zeros(int(0.1 * sr))

    x = _tts(text, tts_rate, 0, sr)
    med = _median_f0(x, sr)
    delta = float(np.median(notes)) - hz2midi(med)
    if abs(delta) > 5.0:
        pct = np.clip((2.0 ** (delta / 12.0) - 1.0) * 100.0, -45.0, 90.0)
        x = _tts(text, tts_rate, pct, sr)
    med_st = hz2midi(_median_f0(x, sr))

    xf = max(2, int(0.010 * sr))
    total = sum(int(d * sr) for d in durs) + xf
    pad = int(0.024 * sr)
    mix = np.zeros(total + pad + 4)
    used_offsets, used_gains = [], []
    for v in range(max(1, int(voices))):
        off = _OFFSETS[v % len(_OFFSETS)]
        g = _GAINS[v % len(_GAINS)]
        # clamp shift range: fold offset by octaves into +/-15 st reach
        while float(np.median(notes)) + off - med_st > 15.0:
            off -= 12
        while float(np.median(notes)) + off - med_st < -15.0:
            off += 12
        used_offsets.append(off)
        used_gains.append(g)
        detune = rng.uniform(-8.0, 8.0) / 100.0          # +/-8 cents
        vm = [(n + off + detune, d) for n, d in zip(notes, durs)]
        vy = sing(text, vm, sr, note_dur,
                  max_shift=(8.0 if off == 0 else 15.0),
                  vibrato_cents=15.0, rng=rng, _tts_x=x)
        onset = int(round((0.012 + rng.uniform(-0.012, 0.012)) * sr))
        n = min(len(vy), len(mix) - onset)
        mix[onset:onset + n] += g * vy[:n]

    peak = np.abs(mix).max(initial=0.0)
    if peak > 1e-9:
        mix *= 0.9 / peak

    # one vocoder pass for the whole voice-group onto the saw stack
    car = _carrier_stack(notes, durs, used_offsets, used_gains, sr, rng)
    if len(car) < len(mix):
        car = np.pad(car, (0, len(mix) - len(car)))
    voc = fx.vocoder(mix, car[:len(mix)], sr, n_bands=20)
    out = 0.6 * voc + 0.4 * mix
    peak = np.abs(out).max(initial=0.0)
    if peak > 1e-9:
        out *= 0.89 / peak
    return _finite(out)


# ----------------------------------------------------------------- ooh
_FORMANTS = {
    "ah": ((700.0, 130.0, 1.00), (1220.0, 90.0, 0.50), (2600.0, 160.0, 0.25)),
    "oo": ((300.0, 80.0, 1.00), (870.0, 90.0, 0.42), (2240.0, 140.0, 0.20)),
}


def _glottal_note(f_hz, dur, sr, rng, vib_cents=15.0, vib_hz=5.0):
    """Pulse-train glottal source for one note (vibrato ramps in).
    Returns (pulse_train, fundamental) -- the fundamental sine is the
    glottal leak that carries the pitch below the first formant."""
    n = max(int(dur * sr), 8)
    t = np.arange(n) / sr
    depth = vib_cents * np.clip((t - 0.15) / 0.20, 0.0, 1.0)
    f = f_hz * 2.0 ** (depth * np.sin(2.0 * np.pi * vib_hz * t +
                                      rng.uniform(0, 2 * np.pi)) / 1200.0)
    ph = np.cumsum(f) / sr + rng.uniform()
    src = np.exp(20.0 * (np.cos(2.0 * np.pi * ph) - 1.0))   # smooth pulses
    src -= src.mean()
    a = max(4, int(0.040 * sr))
    r = max(4, int(0.100 * sr))
    env = np.ones(n)
    env[:min(a, n)] = np.linspace(0, 1, min(a, n))
    env[-min(r, n):] *= np.linspace(1, 0, min(r, n))
    return src * env, np.sin(2.0 * np.pi * ph) * env


def ooh(melody_midi, vowel="ah", sr=SR, voices=5, note_dur=0.5, seed=23):
    """Wordless formant-synth choir (no TTS): per note a pulse-train
    glottal source through 3 formant bandpasses (ah: 700/1220/2600 Hz,
    oo: 300/870/2240), stacked `voices` times with detune/jitter/vibrato
    as in choir().  Standalone pad color and the wordless fallback."""
    rng = np.random.default_rng(seed)
    notes, durs = _parse_melody(melody_midi, note_dur)
    if not notes:
        return np.zeros(int(0.1 * sr))
    formants = _FORMANTS.get(vowel, _FORMANTS["ah"])
    soss = [signal.butter(2, [max(20.0, fc - bw / 2.0),
                              min(fc + bw / 2.0, 0.47 * sr)],
                          btype="band", fs=sr, output="sos")
            for fc, bw, _ in formants]

    xf = max(2, int(0.010 * sr))
    total = sum(int(d * sr) for d in durs) + xf
    pad = int(0.024 * sr)
    mix = np.zeros(total + pad + 4)
    v_off = (0, 0, 0, 12, -12)
    v_gain = (1.0, 0.9, 0.85, 0.14, 0.18)
    for v in range(max(1, int(voices))):
        off = v_off[v % len(v_off)]
        g = v_gain[v % len(v_gain)]
        detune = rng.uniform(-8.0, 8.0) / 100.0
        segs = []
        for note, dur in zip(notes, durs):
            f0 = float(midi2hz(note + off + detune))
            src, fund = _glottal_note(f0, dur + xf / sr, sr, rng)
            seg = np.zeros(len(src))
            for sos, (_, _, fg) in zip(soss, formants):
                seg += fg * signal.sosfilt(sos, src)
            pk = np.abs(seg).max(initial=0.0)
            if pk > 1e-9:
                seg /= pk
            # glottal fundamental leak: exactly one voice carries it,
            # so detuned unisons cannot beat-cancel the fundamental
            seg += (1.1 if v == 0 else 0.0) * fund
            seg += 0.015 * signal.sosfilt(soss[0],
                                          rng.standard_normal(len(src)))
            segs.append(seg)
        vy = _overlap_concat(segs, sr)
        onset = int(round((0.012 + rng.uniform(-0.012, 0.012)) * sr))
        n = min(len(vy), len(mix) - onset)
        mix[onset:onset + n] += g * vy[:n]
    peak = np.abs(mix).max(initial=0.0)
    if peak > 1e-9:
        mix *= 0.9 / peak
    return _finite(mix)


# ------------------------------------------------------------ self-test
if __name__ == "__main__":
    import time

    def _write(path, y, sr=SR):
        y16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(y16.tobytes())

    def _seg_f0(y, sr, notes, durs, xfade_s=0.010):
        """median detected f0 per note segment (autocorr), Hz."""
        out = []
        pos = 0
        for note, dur in zip(notes, durs):
            n = int(dur * sr)
            a = pos + int(0.25 * n)
            b = pos + int(0.85 * n)
            f = _f0_frames(y[a:b], sr, fmin=80.0, fmax=600.0,
                           frame=4096, hop=1024)
            out.append(float(np.median(f)) if len(f) else 0.0)
            pos += n
        return out

    sc = _scratch()
    t0 = time.time()

    TEXT = "converge to the minimum"
    MEL = [65, 63, 61, 60, 58]
    ND = 0.5

    print("rendering choir(%r, %s) ..." % (TEXT, MEL))
    yc = choir(TEXT, MEL, note_dur=ND)
    print("rendering ooh([53, 56, 60]) ...")
    MEL2 = [53, 56, 60]
    yo = ooh(MEL2, vowel="ah", note_dur=ND)

    pc = os.path.join(sc, "choir_selftest.wav")
    po = os.path.join(sc, "ooh_selftest.wav")
    _write(pc, yc)
    _write(po, yo)
    print("wrote", pc)
    print("wrote", po)

    for name, y, mel in (("choir", yc, MEL), ("ooh", yo, MEL2)):
        assert np.all(np.isfinite(y)), name + ": non-finite samples"
        rms = float(np.sqrt(np.mean(y ** 2)))
        assert rms > 1e-4, name + ": silent output (rms=%g)" % rms
        peak = float(np.abs(y).max())
        assert peak <= 1.0, name + ": peak %.3f > 1" % peak
        tot = ND * len(mel)
        dur = len(y) / SR
        assert 0.5 * tot <= dur <= 2.0 * tot, (
            name + ": duration %.2fs vs melody total %.2fs" % (dur, tot))
        print("%s: rms %.3f  peak %.3f  dur %.2fs (melody %.2fs)  OK"
              % (name, rms, peak, dur, tot))

    # listen-proxy: pitch-track the choir output per note segment
    durs = [ND] * len(MEL)
    f0s = _seg_f0(yc, SR, MEL, durs)
    worst = 0.0
    for note, f0 in zip(MEL, f0s):
        assert f0 > 0, "choir: no pitch detected for note %d" % note
        err = abs(hz2midi(f0) - note)
        worst = max(worst, err)
        print("choir note %3d -> detected %6.1f Hz (midi %5.2f)  "
              "err %.2f st" % (note, f0, hz2midi(f0), err))
        assert err <= 1.5, (
            "choir: note %d off by %.2f st (detected %.1f Hz)"
            % (note, err, f0))

    f0s = _seg_f0(yo, SR, MEL2, [ND] * len(MEL2))
    for note, f0 in zip(MEL2, f0s):
        assert f0 > 0, "ooh: no pitch detected for note %d" % note
        err = abs(hz2midi(f0) - note)
        print("ooh   note %3d -> detected %6.1f Hz (midi %5.2f)  "
              "err %.2f st" % (note, f0, hz2midi(f0), err))
        assert err <= 1.5, (
            "ooh: note %d off by %.2f st (detected %.1f Hz)"
            % (note, err, f0))

    print("all self-tests passed (worst choir pitch err %.2f st) in %.1fs"
          % (worst, time.time() - t0))
