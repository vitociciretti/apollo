"""MIDI companion for 10_Descensus_Gradientis — mirrors descensus.py's event
grammar (conductor + bass/stabs/choir/pads/drums) so the release ships 10 MIDs.
"""
import numpy as np, os, mido

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"

hist = np.load(os.path.join(SCRATCH, "hist_NVDA.npy"))
EP = len(hist)
train, val = hist[:, 0], hist[:, 1]
vmin, v0 = val.min(), val[0]
diss = np.clip((val - vmin)/(v0 - vmin + 1e-9), 0, 1)
impr = np.zeros(EP); b = np.inf
for e in range(EP):
    if val[e] < b - 1e-6: impr[e] = b - val[e]; b = val[e]
best = np.zeros(EP, bool)
best[np.argsort(impr)[-8:]] = True
best[impr == 0] = False
LR_STEPS = {30, 60}

BPM = 145; INTRO, OUTRO = 8, 8
ROOT = 41
def chord_for(e):
    notes = [ROOT, ROOT+3, ROOT+7, ROOT+12]
    d = diss[e]
    if d > 0.66: notes += [ROOT+1, ROOT+6]
    elif d > 0.33: notes += [ROOT+8]
    if e >= 60 and d < 0.15: notes += [ROOT+15]
    return notes

FHARM = [0, 2, 3, 5, 7, 8, 11]; CROOT = 53
def deg2midi(d): return CROOT + 12*(d // 7) + FHARM[d % 7]
def curve_melody(e0, e1, n_notes, end_tonic=False):
    w = val[max(0, e0):min(max(e1, e0 + 2), EP)]
    v = np.interp(np.linspace(0, len(w) - 1, n_notes), np.arange(len(w)), w)
    vn = np.clip((v - vmin)/(v0 - vmin + 1e-9), 0, 1)
    wig = (v - v.min())/max(np.ptp(v), 1e-12)*2.0
    degs = np.clip(np.rint(vn*9 + wig), 0, 9).astype(int)
    if end_tonic: degs[-1] = 0
    return [deg2midi(int(d)) for d in degs]

PLACE = [(1, (0, 12), 6, False), (INTRO+30, (30, 40), 5, False),
         (INTRO+60, (60, 70), 6, False), (INTRO+EP, (68, 80), 5, False),
         (INTRO+EP+4, (72, 80), 7, True)]
LATIN = ["Descende gradiens", "Minue iacturam", "Converge ad minimum",
         "Octoginta aetates", "Nox est stochastica"]

TPQ = 480; T16 = TPQ//4
mid = mido.MidiFile(ticks_per_beat=TPQ)
meta = mido.MidiTrack(); mid.tracks.append(meta)
meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
meta.append(mido.MetaMessage("marker", text="intro@0", time=0))
meta.append(mido.MetaMessage("marker", text=f"descent@{INTRO}", time=0))
for s in sorted(LR_STEPS):
    meta.append(mido.MetaMessage("marker", text=f"lr_step@{INTRO+s}", time=0))
meta.append(mido.MetaMessage("marker", text=f"outro@{INTRO+EP}", time=0))

def track_from(events, chan, program=None, name=""):
    """events: list of (t16, dur16, note, vel)"""
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    if program is not None:
        tr.append(mido.Message("program_change", channel=chan, program=program, time=0))
    msgs = []
    for t16, dur16, note, vel in events:
        msgs.append((int(t16*T16), 1, mido.Message("note_on", channel=chan, note=int(note), velocity=int(vel), time=0)))
        msgs.append((int((t16+dur16)*T16), 0, mido.Message("note_off", channel=chan, note=int(note), velocity=0, time=0)))
    msgs.sort(key=lambda x: (x[0], x[1]))
    prev = 0
    for tk, _, msg in msgs:
        msg.time = tk - prev; prev = tk
        tr.append(msg)

bass, stabs, choir, pads, drums = [], [], [], [], []
for e in range(EP):
    bar16 = (INTRO + e)*16
    v = int(90 + 25*(1 - diss[e]))
    for beat in range(4):
        t = bar16 + beat*4
        drums.append((t, 2, 36, v))                       # kick
        if beat in (1, 3): drums.append((t, 2, 39, 70))   # clap
        drums.append((t+2, 1, 46, int(35 + 15*(1-diss[e]))))  # open hat offbeat
        drums.append((t+1, 1, 42, 22)); drums.append((t+3, 1, 42, 22))
    if best[e]:
        drums.append((bar16, 8, 49, 100))                 # crash on new best val
        bass.append((bar16, 8, ROOT-12, 110))             # sub drop hit
    if (e+1) in LR_STEPS or e+1 == EP:                    # clap roll into transitions
        for s in range(16):
            drums.append((bar16 + s, 1, 39, int(20 + 75*s/15)))
    for eighth in range(1, 8, 2):                         # offbeat bass
        bass.append((bar16 + eighth*2, 1.8, ROOT, 95))
    for pos in (0, 3, 6):                                 # supersaw stabs (3-3-2)
        for m in chord_for(e)[1:]:
            stabs.append((bar16 + pos*2, 3.6, m+12, 60))
    pads.append((bar16, 16, ROOT+12, int(30 + 25*(train[e]-val[e] > 0))))

for (at_bar, (e0, e1), nn, endt), txt in zip(PLACE, LATIN):
    meta.append(mido.MetaMessage("marker", text=f"choir:{txt}@{at_bar}", time=0))
    for j, m in enumerate(curve_melody(e0, e1, nn, endt)):
        choir.append((at_bar*16 + j*8, 8, m, 85))         # 2-beat notes

track_from(bass,  0, program=38, name="bass")
track_from(stabs, 1, program=81, name="stabs")
track_from(choir, 2, program=52, name="choir")
track_from(pads,  3, program=90, name="pads")
track_from(drums, 9, name="drums")
path = os.path.join(OUT, "10_Descensus_Gradientis.mid")
mid.save(path)
print(f"wrote {path}: {sum(len(t) for t in mid.tracks)} messages, "
      f"{mid.length:.1f}s nominal")
