"""Tech(no) US — album pipeline.
Each track is composed from a stock's real 2010-2026 daily history:
  vol -> filter cutoff + kick velocity + BPM (per track)
  price z-score vs 252d MA -> bar harmony | big days -> octave jumps, stab gates
  kurtosis -> detune/brightness | worst-quartile drawdowns -> breakdowns
  recoveries -> clap-roll builds + crash | >4 sigma days -> crash
Outputs per track: WAV demo master + standard MIDI file (bass/stabs/pads/drums + CC74 cutoff).
"""
import numpy as np, wave, os
from scipy import signal
import mido

OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
os.makedirs(OUT, exist_ok=True)
SCRATCH = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)

ALBUM = [  # ticker, title, root pitch class name, root midi (bass octave)
    ("NVDA",  "01_Chipmaker",        29),  # F
    ("AAPL",  "02_Cupertino",        33),  # A
    ("MSFT",  "03_Redmond",          26),  # D
    ("GOOGL", "04_PageRank",         31),  # G
    ("AMZN",  "05_Everything_Store", 36),  # C
    ("META",  "06_Social_Graph",     34),  # Bb
    ("NFLX",  "07_Stream",           28),  # E
    ("TSLA",  "08_Ludicrous",        35),  # B
]
PHRYG = [0, 1, 3, 5, 7, 8, 10]; AEOL = [0, 2, 3, 5, 7, 8, 10]
midi2hz = lambda m: 440*2**((m-69)/12)

def roll(x, w, fn):
    out = np.empty(len(x))
    for i in range(len(x)): out[i] = fn(x[max(0, i-w):i+1])
    return out
def _kurt(x):
    v = x.var()
    return ((x-x.mean())**4).mean()/v**2 if v > 1e-12 else 3.0

# ---------------- composition: returns -> events ----------------
def compose(rs, root):
    n = len(rs)
    dp16 = max(2, round(n/1536))
    N16 = (n//dp16)//16*16; NBAR = N16//16
    price = np.cumsum(rs)
    ma252 = roll(price, 252, np.mean); sd252 = np.maximum(roll(price, 252, np.std), 1e-6)
    vol21 = roll(rs, 21, np.std)*np.sqrt(252); kurt63 = roll(rs, 63, _kurt)
    dd = np.maximum.accumulate(price) - price
    bear = dd > np.quantile(dd, 0.78)
    sigma = rs.std()
    annvol = rs.std()*np.sqrt(252)
    bpm = int(np.clip(120 + 40*(annvol-0.20), 122, 134))

    bar = []
    for b in range(NBAR):
        ie = min((b+1)*16*dp16 - 1, n-1)
        z = (price[ie]-ma252[ie])/sd252[ie]
        deg = 5 if z < -1 else (3 if z > 0.5 else 0)
        bar.append(dict(deg=deg, vol=vol21[ie], kurt=kurt63[ie], bear=bool(bear[ie])))
    volz = np.array([bb["vol"] for bb in bar]); volz = (volz-volz.min())/max(np.ptp(volz), 1e-12)
    kurtz = np.array([bb["kurt"] for bb in bar]); kurtz = np.argsort(np.argsort(kurtz))/max(NBAR-1, 1)
    bearb = np.array([bb["bear"] for bb in bar])
    buildb = np.zeros(NBAR, bool)
    for b in range(NBAR-1):
        if bearb[b] and not bearb[b+1]: buildb[b] = True

    ev = dict(bpm=bpm, dp16=dp16, N16=N16, NBAR=NBAR, root=root, bear=bear, rs=rs,
              notes=[], drums=[], cc=[], volz=volz, kurtz=kurtz, bearb=bearb)
    for k in range(N16):
        b16 = k % 16; bn = k//16; B = bar[bn]
        scale = PHRYG if B["bear"] else AEOL
        vz, kz = volz[bn], kurtz[bn]
        intro = bn < 4; outro = bn >= NBAR-2
        i0, i1 = k*dp16, (k+1)*dp16
        vel = int(70 + 45*vz)
        if b16 == 0:
            ev["cc"].append((k, 74, int(30 + 80*vz)))          # filter cutoff automation
        if b16 % 4 == 0 and not B["bear"] and not outro:
            ev["drums"].append((k, 36, min(127, vel+15)))      # kick
        if b16 % 4 == 2:
            ev["drums"].append((k, 46, int(45+30*vz)))         # open hat offbeat
        else:
            ev["drums"].append((k, 42, int(30+25*vz)))         # closed hat
        if b16 in (4, 12) and not B["bear"]:
            ev["drums"].append((k, 39, int(75+25*vz)))         # clap
        if buildb[bn]:
            ev["drums"].append((k, 39, int(30+90*(b16/15))))   # clap-roll build
        if b16 == 0 and bn > 0 and bearb[bn-1] and not B["bear"]:
            ev["drums"].append((k, 49, 100))                   # recovery crash
        if np.abs(rs[i0:i1]).max()/sigma > 4:
            ev["drums"].append((k, 49, 85))                    # shock crash
        if not B["bear"] and not intro and not outro:          # rolling bass
            octv = 12 if (b16 % 4 == 2 and np.abs(rs[i0:i1]).max() > sigma) else 0
            m = root + scale[B["deg"] % 7] + octv
            v = int(vel*(1.15 if b16 % 2 == 1 else 0.85))
            ev["notes"].append((k, 0.9, m, min(127, v), 0))    # (t16, dur16, midi, vel, chan)
        if b16 in (3, 7, 10, 14) and not intro:                # gated syncopated stabs
            if np.abs(rs[i0:i1]).sum() > np.abs(rs).mean()*dp16:
                deg = B["deg"]
                for c in (scale[deg % 7], scale[(deg+2) % 7]+12, scale[(deg+4) % 7]+12):
                    ev["notes"].append((k, 1.8, root+24+c, int(55+40*kz), 1))
    return ev

# ---------------- per-ticker spectral peaks (for the pads) ----------------
def spectral_peaks(rs, root, nsel=6):
    a = np.abs(rs) - np.abs(rs).mean()
    f, P = signal.welch(a, fs=1.0, nperseg=1024, noverlap=768)
    band = (f > 1/600) & (f < 1/40)
    f, P = f[band], P[band]
    pk, _ = signal.find_peaks(P)
    pk = pk[np.argsort(P[pk])[::-1]]
    sel = []
    for i in pk:
        if all(abs(np.log(f[i]/f[j])) > 0.18 for j in sel): sel.append(i)
        if len(sel) == nsel: break
    pcs = {(root + s) % 12 for s in AEOL}
    out = []
    for i in sorted(sel, key=lambda i: f[i]):
        hz = f[i]*2.2e5
        m = 12*np.log2(hz/440) + 69
        cands = [mm for mm in range(int(m)-6, int(m)+7) if mm % 12 in pcs]
        mq = min(cands, key=lambda mm: abs(mm-m)) - 24         # 2 octaves down: pads, not whistles
        out.append((1/f[i], midi2hz(mq), mq, np.sqrt(P[i]/P[sel[0]])))
    return out

# ---------------- synthesis: events -> stereo wav ----------------
SR = 44100
def synth(ev, peaks, path):
    bpm, N16 = ev["bpm"], ev["N16"]
    S16 = 60/bpm/4
    total = int(N16*S16*SR)
    L = np.zeros(total); R = np.zeros(total)
    def add(st, sig, pl=1.0, pr=1.0):
        e = min(st+len(sig), total)
        L[st:e] += sig[:e-st]*pl; R[st:e] += sig[:e-st]*pr
    def kick(a):
        t = np.arange(int(SR*0.20))/SR
        return a*(np.sin(2*np.pi*(110*np.exp(-t*16)+45)*t)*np.exp(-t*14) + 0.5*np.sin(2*np.pi*900*t)*np.exp(-t*300))
    def hat(a, dur, hp, dec):
        t = np.arange(int(SR*dur))/SR
        b, aa = signal.butter(4, hp/(SR/2), "high")
        return a*signal.lfilter(b, aa, rng.standard_normal(len(t)))*np.exp(-t*dec)
    def clap(a):
        t = np.arange(int(SR*0.15))/SR
        b, aa = signal.butter(2, [900/(SR/2), 3500/(SR/2)], "band")
        return a*signal.lfilter(b, aa, rng.standard_normal(len(t)))*np.exp(-t*30)*(1+0.7*(np.sin(2*np.pi*55*t) > 0))
    def crash(a):
        t = np.arange(int(SR*0.9))/SR
        b, aa = signal.butter(4, 6000/(SR/2), "high")
        return a*signal.lfilter(b, aa, rng.standard_normal(len(t)))*np.exp(-t*4.5)
    def sawstack(hz, dur, det, cutoff, a):
        t = np.arange(int(SR*dur))/SR
        y = np.zeros(len(t))
        for dt in (1-det, 1+det):
            f0 = hz*dt
            for h in range(1, int(min(cutoff*2, SR*0.45)/f0)+1):
                y += np.sin(2*np.pi*h*f0*t)/h
        b, aa = signal.butter(2, min(cutoff/(SR/2), 0.95), "low")
        env = np.minimum(1, np.minimum(t/0.004, np.maximum(0, (dur-t)/(dur*0.35))))
        return a*env*signal.lfilter(b, aa, y)

    cutmap = {k: v for k, _, v in [(c[0], c[1], c[2]) for c in ev["cc"]]}
    cut = 200
    kicks = []
    DRUM = {36: lambda v: kick(v), 39: lambda v: clap(v), 42: lambda v: hat(v, 0.025, 8000, 160),
            46: lambda v: hat(v, 0.18, 7000, 18), 49: lambda v: crash(v)}
    for k, note, vel in ev["drums"]:
        st = int(k*S16*SR); v = vel/127
        d = DRUM[note](v*(0.95 if note == 36 else 0.55 if note == 39 else 0.28 if note == 46 else 0.16 if note == 42 else 0.55))
        pan = (0.8, 1.0) if note == 46 else (1.0, 0.8) if note == 42 else (1.0, 1.0)
        add(st, d, *pan)
        if note == 36: kicks.append(st)
    for k, dur, m, vel, chan in ev["notes"]:
        if k in cutmap: cut = 160 + 28*cutmap[k]
        st = int(k*S16*SR); v = vel/127
        if chan == 0:
            add(st, sawstack(midi2hz(m), S16*dur, 0.005, cut, 0.16*v*1.6))
        else:
            s = sawstack(midi2hz(m), S16*dur, 0.008, 900+cut, 0.055*v*1.6)
            pan = 0.35 if (k//4) % 2 else -0.35
            add(st, s, 1-max(pan, 0), 1+min(pan, 0))

    # pads: Hilbert band envelopes, breakdown-gated, scale-quantized
    rs, bear = ev["rs"], ev["bear"]; n = len(rs)
    a_ = np.abs(rs) - np.abs(rs).mean()
    t_full = np.arange(total)/SR
    day_axis = np.linspace(0, ev["N16"]*ev["dp16"]-1, total)
    bg = np.clip(np.interp(day_axis, np.arange(n), signal.filtfilt(*signal.butter(2, 0.02), bear.astype(float))), 0, 1)
    lfo = 1 + 0.15*np.sin(2*np.pi*0.11*t_full)
    for j, (T, hz, mq, amp) in enumerate(peaks):
        f0 = 1.0/T
        b, bb2 = signal.butter(3, [f0/1.35*2, min(f0*1.35*2, 0.99)], btype="band")
        env = np.abs(signal.hilbert(signal.filtfilt(b, bb2, a_)))
        env = signal.filtfilt(*signal.butter(2, 0.05), env)
        env = np.maximum(env, 0); env /= max(env.max(), 1e-12)
        voice = np.sin(2*np.pi*hz*0.997*t_full) + np.sin(2*np.pi*hz*1.003*t_full)
        pad = np.interp(day_axis, np.arange(n), env)*voice*0.5*(0.16*bg)*lfo*amp
        pan = j/max(len(peaks)-1, 1)
        L += pad*(1-0.5*pan); R += pad*(0.5+0.5*pan)

    # sidechain duck + kick restore
    duck = np.ones(total); w = int(0.30*SR); tt = np.arange(w)/SR
    shape = 1 - 0.55*np.exp(-tt/0.085)
    for ks in kicks:
        e = min(ks+w, total); duck[ks:e] = np.minimum(duck[ks:e], shape[:e-ks])
    L *= duck; R *= duck
    for ks in kicks:
        d = kick(0.9); e = min(ks+len(d), total)
        L[ks:e] += d[:e-ks]*0.35; R[ks:e] += d[:e-ks]*0.35
    # ping-pong delay on highs
    db, da = signal.butter(2, 1200/(SR/2), "high")
    hiL, hiR = signal.lfilter(db, da, L), signal.lfilter(db, da, R)
    dl = int(S16*3*SR)
    L[dl:] += hiR[:-dl]*0.22; R[2*dl:] += hiL[:-2*dl]*0.15
    # master
    b, aa = signal.butter(2, 25/(SR/2), "high")
    L, R = signal.lfilter(b, aa, L), signal.lfilter(b, aa, R)
    fade = np.minimum(1, np.minimum(t_full/0.3, (t_full[-1]-t_full)/4.0))
    L *= fade; R *= fade
    mx = max(np.abs(L).max(), np.abs(R).max())
    audio = np.empty(2*total, dtype=np.int16)
    audio[0::2] = (L/mx*0.95*32767).astype(np.int16)
    audio[1::2] = (R/mx*0.95*32767).astype(np.int16)
    with wave.open(path, "w") as w_:
        w_.setnchannels(2); w_.setsampwidth(2); w_.setframerate(SR)
        w_.writeframes(audio.tobytes())
    return total/SR

# ---------------- MIDI export: events -> .mid ----------------
def write_midi(ev, peaks, path):
    TPQ = 480; T16 = TPQ//4
    mid = mido.MidiFile(ticks_per_beat=TPQ)
    meta = mido.MidiTrack(); mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(ev["bpm"]), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    def track_from(events, chan, program=None, name=""):
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if program is not None:
            tr.append(mido.Message("program_change", channel=chan, program=program, time=0))
        msgs = []
        for e in events:
            if e[0] == "cc":
                _, t16, cc, val = e
                msgs.append((int(t16*T16), 0, mido.Message("control_change", channel=chan, control=cc, value=val, time=0)))
            else:
                _, t16, dur16, note, vel = e
                on, off = int(t16*T16), int((t16+dur16)*T16)
                msgs.append((on, 1, mido.Message("note_on", channel=chan, note=note, velocity=vel, time=0)))
                msgs.append((off, 0, mido.Message("note_off", channel=chan, note=note, velocity=0, time=0)))
        msgs.sort(key=lambda x: (x[0], x[1]))
        prev = 0
        for tick, _, msg in msgs:
            msg.time = tick - prev; prev = tick
            tr.append(msg)

    bass = [("n", k, d, m, v) for k, d, m, v, ch in ev["notes"] if ch == 0]
    bass += [("cc", k, cc, val) for k, cc, val in ev["cc"]]
    stabs = [("n", k, d, m, v) for k, d, m, v, ch in ev["notes"] if ch == 1]
    drums = [("n", k, 0.5, note, vel) for k, note, vel in ev["drums"]]
    # pads in MIDI: one long note per contiguous breakdown region per top-3 peaks
    pads = []
    bb = ev["bearb"]; b0 = None
    for b in range(len(bb)+1):
        if b < len(bb) and bb[b] and b0 is None: b0 = b
        elif (b == len(bb) or not bb[b]) and b0 is not None:
            for (_, _, mq, amp) in peaks[:3]:
                pads.append(("n", b0*16, (b-b0)*16, mq, int(50+30*amp)))
            b0 = None
    track_from(bass,  0, program=38, name="bass")
    track_from(stabs, 1, program=81, name="stabs")
    track_from(pads,  2, program=90, name="pads")
    track_from(drums, 9, name="drums")
    mid.save(path)

# ---------------- run the album ----------------
print(f"{'track':22s} {'bpm':>4s} {'len':>6s} {'bars':>5s} {'bd':>3s}  top pad periods (d)")
for ticker, title, root in ALBUM:
    rs = np.load(os.path.join(SCRATCH, f"r_{ticker}.npy"))
    ev = compose(rs, root)
    peaks = spectral_peaks(rs, root % 12)
    dur = synth(ev, peaks, os.path.join(OUT, f"{title}.wav"))
    write_midi(ev, peaks, os.path.join(OUT, f"{title}.mid"))
    per = ", ".join(f"{p[0]:.0f}" for p in peaks[:3])
    print(f"{title:22s} {ev['bpm']:4d} {int(dur//60)}:{int(dur%60):02d} {ev['NBAR']:5d} {ev['bearb'].sum():3d}  {per}")
print("album written to", OUT)
