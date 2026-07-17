"""Descensus Gradientis — track 10, the closer (v2: the choir sings the loss).

Latin-choral hard dance built from the REAL 80-epoch training history of the
album's neural composer (hist_NVDA.npy).  The loss curve is the harmonic arc:
  high val NLL -> dissonant clusters; convergence -> resolution to clean F minor
  new best-val epoch -> impact/drop (only the 8 biggest leaps fire)
  LR decays (ep 30, 60) -> section transitions | train/val gap -> pad detune

v2 replaces the flat vocoded chants with MELODIC choir (choir.py):
  each Latin phrase is sung by choir.choir() on a DESCENDING melody read off
  the actual val-NLL curve — the epoch window is resampled to 5-7 notes and
  mapped onto F harmonic minor, high loss = high degree, converged = tonic,
  so the choir literally sings gradient descent down to F.
  ooh() formant-choir sustains open (intro, 'oo') and close (outro, 'ah')
  the track.  Drums/bass keep the v4 synths trance/warehouse kit.

Usage: python descensus.py [OUT_DIR]   (RAW WAV written there; default TechnoUS)
"""
import numpy as np, wave, os, sys
import synths, fx
import choir as choirlib
from scipy import signal

SR = 44100
SCRATCH = os.path.dirname(os.path.abspath(__file__))
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

# ---------------- drums (v4 synths trance/warehouse kit) ----------------
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

# ---------------- choir v2: the loss curve, sung ----------------
FHARM = [0, 2, 3, 5, 7, 8, 11]        # F harmonic minor intervals
CROOT = 53                            # F3 — the tonic the choir descends to
def deg2midi(d): return CROOT + 12*(d // 7) + FHARM[d % 7]

def curve_melody(e0, e1, n_notes, beats=2.0, end_tonic=False):
    """Resample val NLL over epochs [e0, e1) to n_notes and map it onto
    F harmonic minor: normalized loss 1 -> degree 9 (chaos, high), 0 -> the
    tonic.  The melody IS the descent.  Returns [(midi, dur_s)]."""
    w = val[max(0, e0):min(max(e1, e0 + 2), EP)]
    v = np.interp(np.linspace(0, len(w) - 1, n_notes), np.arange(len(w)), w)
    vn = np.clip((v - vmin)/(v0 - vmin + 1e-9), 0, 1)
    # local contrast: the window's own min-max shape, worth up to 2 degrees,
    # so converged (flat-tail) phrases still sing the stochastic wiggle
    # around the tonic instead of collapsing to a monotone chant
    wig = (v - v.min())/max(np.ptp(v), 1e-12)*2.0
    degs = np.clip(np.rint(vn*9 + wig), 0, 9).astype(int)
    if end_tonic:
        degs[-1] = 0                  # converged: land ON the F
    return [(deg2midi(int(d)), beats*BEAT) for d in degs]

LATIN = ["Descende gradiens", "Minue iacturam", "Converge ad minimum",
         "Octoginta aetates", "Nox est stochastica"]
PLACE = [  # (phrase idx, at bar, epoch window, n notes, end on tonic)
    (0, 1,          (0, 12),  6, False),  # intro: descend from epoch-0 chaos
    (1, INTRO+30,   (30, 40), 5, False),  # LR step 1: "Minue iacturam"
    (2, INTRO+60,   (60, 70), 6, False),  # LR step 2: "Converge ad minimum"
    (3, INTRO+EP,   (68, 80), 5, False),  # outro: "Octoginta aetates"
    (4, INTRO+EP+4, (72, 80), 7, True),   # outro: "Nox est stochastica" -> F
]

voice = np.zeros(total)
placements = []
for i, (ph, at_bar, (e0, e1), nn, endt) in enumerate(PLACE):
    mel = curve_melody(e0, e1, nn, beats=2.0, end_tonic=endt)
    y = choirlib.choir(LATIN[ph], mel, seed=31 + 7*i)
    a = int(at_bar*BAR*SR); e_ = min(a + len(y), total)
    voice[a:e_] += y[:e_ - a]
    placements.append((LATIN[ph], at_bar*BAR,
                       "-".join(str(m) for m, _ in mel)))

# ooh sustains: 'oo' opens the intro, 'ah' closes the converged outro
for (bar0, nb, chord, vowel, sd) in (
        (0.5,        INTRO - 1,  (CROOT, CROOT+3, CROOT+7),  "oo", 51),
        (INTRO+EP+1, OUTRO - 2,  (CROOT, CROOT+7, CROOT+12), "ah", 61)):
    dur = nb*BAR
    pad = None
    for j, note in enumerate(chord):
        y = choirlib.ooh([(note, dur)], vowel=vowel, seed=sd + 3*j)
        pad = y if pad is None else pad[:len(y)] + y[:len(pad)]
    pad *= 0.9/max(np.abs(pad).max(), 1e-9)
    a = int(bar0*BAR*SR); e_ = min(a + len(pad), total)
    voice[a:e_] += 0.6*pad[:e_ - a]

# ---------------- detune-width pad from the overfitting gap ----------------
tf = np.arange(total)/SR
bar_of = np.clip((tf/BAR - INTRO).astype(int), 0, EP-1)
gap_t = gap[bar_of]
pad = np.sin(2*np.pi*midi2hz(ROOT+12)*(1+0.002*gap_t)*tf) + np.sin(2*np.pi*midi2hz(ROOT+12)*(1-0.002*gap_t)*tf)
body = ((tf > INTRO*BAR) & (tf < (INTRO+EP)*BAR))
L += pad*0.05*body; R += pad*0.05*body

# ---------------- bus fx + voice bus + master prep ----------------
L, R = fx.sidechain(L, R, SR, kicks)
for ks in kicks:
    d = synths.drum("trance", "kick", 0.9); e_ = min(ks+len(d), total)
    L[ks:e_] += d[:e_-ks]*0.3; R[ks:e_] += d[:e_-ks]*0.3
L, R = fx.reverb(L, R, SR, size=0.5, damp=0.5, mix=0.14)
# voice bus: big hall, un-ducked, ~7 dB under the music RMS (featured choral)
vL, vR = fx.reverb(voice, voice, SR, size=0.8, damp=0.4, mix=0.42)
act = np.abs(voice) > 1e-3*max(np.abs(voice).max(), 1e-9)
rms_v = np.sqrt(np.mean(voice[act]**2)) if act.any() else 1e-9
rms_m = np.sqrt((np.mean(L**2) + np.mean(R**2))/2)
gv = rms_m*10**(-7.0/20)/max(rms_v, 1e-9)
L += gv*vL; R += gv*vR
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
      f"diss {diss[0]:.2f}->{diss[-1]:.2f} | voice gain {20*np.log10(gv):+.1f} dB")
for txt, t_s, mel in placements:
    print(f"  choir @ {int(t_s//60)}:{t_s%60:05.2f}  {txt!r}  midi {mel}")
