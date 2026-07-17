"""vocals.py — melodic choir passes for tracks 01 / 07 / 09 (TECH(NO) US).

Loads each track's RAW render, recomputes its section structure exactly the
way the engine composed it (techno_album.compose for 01/07; the duet's joint
drawdown bar grid for 09), then places 2-3 choir.choir() phrases:

  melody   = the track's motif.derive() theme mapped to midi in the track key
             (roots from techno_album.ALBUM: 01/09 F minor, 07 E minor),
             GRID8 rhythm augmented 2-3x so a phrase spans 2-3 bars
  phrase 1 = first bear-section onset
  last     = the final recovery
  ooh()    = wordless tonic-triad pad under the longest bear section

Voice bus: fx.reverb(size .65, mix .35).  Level is per track: the dense
01/07 mixes get the voice at -3.5/-4 dB vs music RMS plus a -6/-5 dB music
duck under each worded phrase; the sparser 09 duet keeps -9 dB, no duck.
Overwrites the raw renders in TechnoUS (premaster/ stays the source of
truth for mastering: run producer_master.py afterwards).
Usage: python vocals.py [1] [7] [9]   (default: all three)

Lyrics live in lyrics.md; the sung lines are indexed from there.
"""
import os
import re
import wave

import numpy as np

import choir
import fx
import motif
from techno_album import compose, ALBUM, AEOL, SR, OUT

HERE = os.path.dirname(os.path.abspath(__file__))
GAPS8 = [2, 1, 3, 2, 2, 1, 3, 2]      # note gaps (16ths) of motif.GRID8
VOICE_DB = -9.0                       # default voice bus vs music RMS (09 duet)
OOH_GAIN = 0.40                       # ooh pad below the worded phrases
DUCK_FADE = 0.30                      # s: raised-cosine edges of the duck


# ------------------------------------------------------------- lyrics ----
def load_lyrics(path=os.path.join(HERE, "lyrics.md")):
    """{'01_Chipmaker': [line1, ...], ...} from lyrics.md numbered lists."""
    out, key = {}, None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            m = re.match(r"^##\s+(\S+)", ln)
            if m:
                key = m.group(1)
                out[key] = []
                continue
            m = re.match(r"^\d+\.\s+(.*?)(?:\s{2,}\(.*\))?\s*$", ln)
            if m and key:
                out[key].append(m.group(1).strip())
    return out


# ----------------------------------------------------------- structure ----
def structure_album(ticker, root, style):
    """Recompute exactly what compose() composed: bpm, sections, theme."""
    rs = np.load(os.path.join(HERE, f"r_{ticker}.npy"))
    vp = os.path.join(HERE, f"v_{ticker}.npy")
    v_log = np.load(vp) if os.path.exists(vp) else None
    ev = compose(rs, root, style, v_log=v_log)
    bears = [(s["bar0"], s["bar1"]) for s in ev["sections"] if s["kind"] == "bear"]
    recs = [s["bar0"] for s in ev["sections"] if s["kind"] == "recovery"]
    return dict(bpm=ev["bpm"], theme=ev["theme"], bears=bears,
                final_recovery=max(recs), root=root)


def structure_duet():
    """The duet's bar grid, as duet.compose_duet builds it (joint drawdown)."""
    r1 = np.load(os.path.join(HERE, "r_NVDA.npy"))
    r2 = np.load(os.path.join(HERE, "r_AAPL.npy"))
    n = min(len(r1), len(r2)); r1, r2 = r1[:n], r2[:n]
    dp16 = max(2, round(n / 1536))
    NBAR = (n // dp16) // 16 * 16 // 16
    p1, p2 = np.cumsum(r1), np.cumsum(r2)
    dd1 = np.maximum.accumulate(p1) - p1
    dd2 = np.maximum.accumulate(p2) - p2
    bear = (dd1 > np.quantile(dd1, 0.78)) & (dd2 > np.quantile(dd2, 0.78))
    ends = np.minimum((np.arange(NBAR) + 1) * 16 * dp16 - 1, n - 1)
    bearb = bear[ends]
    runs, b = [], 0
    while b < NBAR:
        if bearb[b]:
            j = b
            while j < NBAR and bearb[j]:
                j += 1
            runs.append((b, j)); b = j
        else:
            b += 1
    # duet recovery = the roll-build bar closing each bear run (buildb)
    return dict(bpm=126, theme=motif.derive((r1 + r2) / 2, AEOL),
                bears=runs, final_recovery=runs[-1][1] - 1, root=29)


# -------------------------------------------------------------- render ----
def theme_melody(theme, root, s16, aug):
    """Theme degrees -> [(midi, dur_s)] in the track key, an octave above
    the bass root (root+24 keeps the choir in the F3/E3 singing register),
    GRID8 rhythm augmented `aug`x (aug=3 -> the phrase spans 3 bars)."""
    croot = root + 24
    return [(motif.to_midi(d, AEOL, croot), g * aug * s16)
            for d, g in zip(theme, GAPS8)]


def _load_stereo(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()),
                          dtype=np.int16).astype(np.float64) / 32767
    return x[0::2].copy(), x[1::2].copy()


def _write_stereo(path, L, R):
    mx = max(np.abs(L).max(), np.abs(R).max(), 1e-9)
    out = np.empty(2 * len(L), dtype=np.int16)
    out[0::2] = (L / mx * 0.95 * 32767).astype(np.int16)
    out[1::2] = (R / mx * 0.95 * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(out.tobytes())


def sing_track(title, st, lines, plan, voice_db=VOICE_DB, duck_db=0.0):
    """plan = [(bar, line_idx, aug)]; ooh pad under the longest bear.
    voice_db: worded-phrase bus level vs music RMS (raise for dense mixes).
    duck_db < 0: duck the music bus by this much under each worded phrase
    (raised-cosine edges), so the melody reads through the wall."""
    path = os.path.join(OUT, f"{title}.wav")
    L, R = _load_stereo(path)
    total = len(L)
    s16 = 60 / st["bpm"] / 4
    spb = 16 * s16

    voice = np.zeros(total)
    placed = []
    spans = []
    for i, (bar, li, aug) in enumerate(plan):
        text = lines[li]
        mel = theme_melody(st["theme"], st["root"], s16, aug)
        y = choir.choir(text, mel, seed=11 + 7 * i)
        t0 = bar * spb
        a = int(t0 * SR); e = min(a + len(y), total)
        voice[a:e] += y[:e - a]
        spans.append((a, e))
        placed.append((text, t0))
        print(f"  phrase @ {int(t0//60)}:{t0%60:05.2f}  bar {bar:3d}  "
              f"{text!r}", flush=True)

    # ooh pad under the longest bear section
    b0, b1 = max(st["bears"], key=lambda r: r[1] - r[0])
    dur = (b1 - b0) * spb
    croot = st["root"] + 24
    pad = None
    for j, note in enumerate((croot, croot + 3, croot + 7)):
        y = choir.ooh([(note, dur)], vowel="oo", seed=23 + 5 * j)
        pad = y if pad is None else pad[:len(y)] + y[:len(pad)]
    pk = np.abs(pad).max()
    if pk > 1e-9:
        pad *= 0.9 / pk
    a = int(b0 * spb * SR); e = min(a + len(pad), total)
    voice[a:e] += OOH_GAIN * pad[:e - a]
    print(f"  ooh pad @ {int(b0*spb//60)}:{b0*spb%60:05.2f}  "
          f"bars {b0}-{b1} ({dur:.1f}s)", flush=True)

    # duck the music bus under the worded phrases (NOT under the ooh pad)
    if duck_db < 0:
        env = np.zeros(total)
        nf = int(DUCK_FADE * SR)
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
        for a, e in spans:
            env[a:e] = 1.0
            a0 = max(a - nf, 0)
            env[a0:a] = np.maximum(env[a0:a], ramp[-(a - a0):])
            e1 = min(e + nf, total)
            env[e:e1] = np.maximum(env[e:e1], ramp[::-1][:e1 - e])
        gd = 1.0 + (10 ** (duck_db / 20) - 1.0) * env
        L = L * gd; R = R * gd
        print(f"  music ducked {duck_db:+.1f} dB under "
              f"{len(spans)} phrases", flush=True)

    # voice bus: reverb, then level voice_db under the music RMS
    vL, vR = fx.reverb(voice, voice, SR, size=0.65, mix=0.35)
    act = np.abs(voice) > 1e-3 * max(np.abs(voice).max(), 1e-9)
    rms_v = np.sqrt(np.mean(voice[act] ** 2)) if act.any() else 1e-9
    rms_m = np.sqrt((np.mean(L ** 2) + np.mean(R ** 2)) / 2)
    g = rms_m * 10 ** (voice_db / 20) / max(rms_v, 1e-9)
    _write_stereo(path, L + g * vL, R + g * vR)
    print(f"  voice gain {20*np.log10(g):+.1f} dB "
          f"(music rms {20*np.log10(rms_m):.1f} dBFS) -> {path}", flush=True)
    return placed


if __name__ == "__main__":
    import sys
    only = {a.lstrip("0") or "0" for a in sys.argv[1:]} or {"1", "7", "9"}
    lyr = load_lyrics()
    roots = {t: r for _, t, r, _ in ALBUM}

    if "1" in only:
        print("01_Chipmaker (F minor)", flush=True)
        st = structure_album("NVDA", roots["01_Chipmaker"], "techno")
        # bears 7-17 / 46-48 / 64-66, final recovery 66
        sing_track("01_Chipmaker", st, lyr["01_Chipmaker"], [
            (st["bears"][0][0], 0, 3),        # first bear onset: "We painted..."
            (st["bears"][-1][0], 3, 2),       # last bear: the DeepSeek morning
            (st["final_recovery"], 5, 3),     # recovery: "still the curve climbs"
        ], voice_db=-3.5, duck_db=-6.0)       # dense techno wall: voice up + duck

    if "7" in only:
        print("07_Stream (E minor)", flush=True)
        st = structure_album("NFLX", roots["07_Stream"], "trance")
        sing_track("07_Stream", st, lyr["07_Stream"], [
            (st["bears"][0][0], 0, 3),        # first bear onset
            (st["bears"][-1][0], 2, 3),       # subscriber shock bear
            (st["final_recovery"], 4, 3),     # recovery: "Press play, begin again"
        ], voice_db=-4.0, duck_db=-5.0)       # supersaw wall: voice up + duck

    if "9" in only:
        print("09_Correlation_One (F minor)", flush=True)
        st = structure_duet()
        sing_track("09_Correlation_One", st, lyr["09_Correlation_One"], [
            (st["bears"][0][0], 0, 3),        # first joint-stress onset
            (st["bears"][1][0], 2, 3),        # "everything moves as one"
            (st["final_recovery"], 4, 3),     # final recovery build
        ])
    print("vocals done", flush=True)
