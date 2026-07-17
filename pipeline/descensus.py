"""Descensus Gradientis — track 10, the closer.
Latin-choral hard dance built from the REAL 80-epoch training history of the album's
neural composer (hist_NVDA.npy). The loss curve is the harmonic arc:
  high val NLL -> dissonant clusters; convergence -> resolution to clean F minor
  new best-val epoch -> impact/drop | LR decays (ep 30, 60) -> section transitions
  train/val gap -> supersaw detune width | choir = SAPI Latin vocoded onto the chords
Usage: python descensus.py OUT_DIR   (WAV written there; default TechnoUS)
"""
import numpy as np, wave, os, sys
import synths, fx
from scipy import signal

SR = 44100
SCRATCH = os.path.dirname(os.path.abspath(__file__))
TTS_DIR = sys.argv[2] if len(sys.argv) > 2 else \
    r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d6\scratchpad"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\vito.ciciretti\Downloads\TechnoUS"

hist = np.load(os.path.join(SCRATCH, "hist_NVDA.npy"))   # (80, 2) train, val
EP = len(hist)
train, val = hist[:, 0], hist[:, 1]
vmin, v0 = val.min(), val[0]
diss = np.clip((val - vmin)/(v0 - vmin + 1e-9), 0, 1)     # 1 = epoch-0 chaos, 0 = converged
gap = np.clip((train - val) - (train - val).min(), 0, None)
gap = gap/max(gap.max(), 1e-9)                            # overfitting tension
impr = np.zeros(EP)
b = np.inf
for e in range(EP):
    if val[e] < b - 1e-6: impr[e] = b - val[e]; b = val[e]
best = np.zeros(EP, bool)
best[np.argsort(impr)[-8:]] = True          # only the 8 biggest leaps fire a drop
best[impr == 0] = False
LR_STEPS = {30, 60}

BPM = 145
BEAT = 60/BPM; BAR = 4*BEAT
INTRO, OUTRO = 8, 8
NBARS = INTRO + EP + OUTRO
total = int(NBARS*BAR*SR)
L = np.zeros(total); R = np.zeros(total)
def add(t_s, sig, pl=1.0, pr=1.0):
    st = int(t_s*SR); e = min(st+len(sig), total)
    if st < total: L[st:e] += sig[:e-st]*pl; R[st:e] += sig[:e-st]*pr

ROOT = 41  # F2
midi2hz = lambda m: 440*2**((m-69)/12)
def chord_for(e):
    """F minor triad, cluster tones injected by dissonance level"""
    notes = [ROOT, ROOT+3, ROOT+7, ROOT+12]
    d = diss[e]
    if d > 0.66: notes += [ROOT+1, ROOT+6]      # b2 + tritone: maximal tension
    elif d > 0.33: notes += [ROOT+8]            # b6
    if e >= 60 and d < 0.15: notes += [ROOT+15] # converged: add the high minor 10th, open
    return notes

# ---------------- drums ----------------
kicks = []
for bar in range(NBARS):
    t0 = bar*BAR
    e = bar - INTRO
    in_body = 0 <= e < EP
    if not in_body: continue
    v = 0.8 + 0.2*(1 - diss[e])
    for beat in range(4):
        t = t0 + beat*BEAT
        add(t, synths.drum("trance", "kick", v)); kicks.append(int(t*SR))
        if beat in (1, 3): add(t, synths.drum("trance", "clap", 0.6))
        add(t + BEAT/2, synths.drum("acid", "hato", 0.28 + 0.12*(1-diss[e])))
        add(t + BEAT/4, synths.drum("trance", "hatc", 0.14))
        add(t + 3*BEAT/4, synths.drum("trance", "hatc", 0.14))
    if best[e]:                                             # new best val -> impact
        add(t0, synths.drum("warehouse", "crash", 0.8))
        add(t0, synths.bass_note("breaks", ROOT-12, 1.2, 1.0, 400))  # sub drop hit
    if (e+1) in LR_STEPS or e+1 == EP:                      # roll into every transition
        for s in range(16):
            add(t0 + s*BAR/16, synths.drum("trance", "clap", 0.15 + 0.75*s/15))

# ---------------- bass + supersaw stabs ----------------
for bar in range(NBARS):
    e = bar - INTRO
    if not (0 <= e < EP): continue
    t0 = bar*BAR
    cutoff = 400 + 2600*(1 - diss[e])                       # converging opens the filter
    notes = chord_for(e)
    for eighth in range(8):                                 # offbeat bass
        if eighth % 2 == 1:
            add(t0 + eighth*BEAT/2, synths.bass_note("trance", ROOT, BEAT*0.45, 0.85, cutoff))
    for pos in (0, 3, 6):                                   # supersaw chord stabs (3-3-2)
        t = t0 + pos*BEAT/2
        for m in notes[1:]:
            s = synths.lead_note("trance", m+12, BEAT*0.9, 0.5/len(notes), cutoff+1200)
            add(t, s, 0.9, 1.1) if pos % 2 else add(t, s, 1.1, 0.9)

# ---------------- choir: vocoded Latin ----------------
def load_tts(i):
    with wave.open(os.path.join(TTS_DIR, f"tts_{i}.wav")) as w:
        sr0 = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)/32767
        if w.getnchannels() == 2: x = x.reshape(-1, 2).mean(1)
    n_out = int(len(x)*SR/sr0)
    return np.interp(np.linspace(0, len(x)-1, n_out), np.arange(len(x)), x)

def choir(phrase, notes, dur_s):
    """vocode one spoken phrase onto a sustained chord -> choral wall"""
    mod = load_tts(phrase)
    reps = int(np.ceil(dur_s*SR/len(mod)))
    mod = np.tile(mod, max(reps, 1))[:int(dur_s*SR)]
    carrier = sum(synths.lead_note("trance", m+12, dur_s, 0.8/len(notes), 4000) for m in notes)
    n = min(len(mod), len(carrier))
    v = fx.vocoder(mod[:n], carrier[:n], SR, n_bands=20)
    env = np.minimum(1, np.minimum(np.arange(n)/(0.4*SR), (n-np.arange(n))/(0.6*SR)))
    return v*env

PLACE = [  # (phrase idx, at bar, dur bars, chord epoch)
    (0, 0, 6, 0),                      # intro: "Descende, gradiens" over epoch-0 chaos chord
    (1, INTRO+30, 4, 30),              # LR step 1: "Minue iacturam"
    (2, INTRO+60, 4, 60),              # LR step 2: "Converge ad minimum"
    (3, INTRO+EP, 3, EP-1),            # outro: "Octoginta aetates"
    (4, INTRO+EP+4, 4, EP-1),          # outro: "Nox est stochastica"
]
for ph, at_bar, nb, ce in PLACE:
    v = choir(ph, chord_for(min(ce, EP-1)), nb*BAR)
    vL, vR = fx.reverb(v*0.8, v*0.8, SR, size=0.85, damp=0.4, mix=0.45)
    add(at_bar*BAR, vL, 1.0, 0.0); add(at_bar*BAR, vR, 0.0, 1.0)

# ---------------- detune-width pad from the overfitting gap ----------------
tf = np.arange(total)/SR
bar_of = np.clip((tf/BAR - INTRO).astype(int), 0, EP-1)
gap_t = gap[bar_of]
pad = np.sin(2*np.pi*midi2hz(ROOT+12)*(1+0.002*gap_t)*tf) + np.sin(2*np.pi*midi2hz(ROOT+12)*(1-0.002*gap_t)*tf)
body = ((tf > INTRO*BAR) & (tf < (INTRO+EP)*BAR))
L += pad*0.05*body; R += pad*0.05*body

# ---------------- bus fx + master prep ----------------
L, R = fx.sidechain(L, R, SR, kicks)
for ks in kicks:
    d = synths.drum("trance", "kick", 0.9); e = min(ks+len(d), total)
    L[ks:e] += d[:e-ks]*0.3; R[ks:e] += d[:e-ks]*0.3
L, R = fx.reverb(L, R, SR, size=0.5, damp=0.5, mix=0.14)
b, a = signal.butter(2, 25/(SR/2), "high")
L, R = signal.lfilter(b, a, L), signal.lfilter(b, a, R)
fade = np.minimum(1, np.minimum(tf/0.3, (tf[-1]-tf)/3.0))
L *= fade; R *= fade
mx = max(np.abs(L).max(), np.abs(R).max())
audio = np.empty(2*total, dtype=np.int16)
audio[0::2] = (L/mx*0.95*32767).astype(np.int16)
audio[1::2] = (R/mx*0.95*32767).astype(np.int16)
out = os.path.join(OUT_DIR, "10_Descensus_Gradientis.wav")
with wave.open(out, "w") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(audio.tobytes())
print(f"wrote {out}: {total/SR:.0f}s @ {BPM} BPM | {int(best.sum())} best-val impacts | "
      f"diss {diss[0]:.2f}->{diss[-1]:.2f} | choir at bars {[p[1] for p in PLACE]}")
