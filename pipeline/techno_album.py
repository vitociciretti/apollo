"""Tech(no) US — album pipeline, v4 (the creativity grammar).
Each track is composed from a stock's real 2010-2026 daily history (yfinance
adjusted closes).  The market still writes everything; v4 adds FORM, MELODY
and DYNAMICS, all read off the data:

  FORM    regime.segments() turns vol-era change-points + drawdown regimes into
          sections (intro/calm/build/peak/bear/recovery/outro); every bar gets
          its section's layer mask + density (regime.section_mask).  calm eras
          play halftime (kick on 1 and 3, half hats, half-rate bass); a 1-bar
          16th clap/snare roll fills INTO every peak section and a crash marks
          every section downbeat.
  MELODY  motif.derive() gives each stock one theme from its first trading
          year; motif.develop() varies it bar-by-bar (inversion in drawdowns,
          transposition on stretched z, diminution in high vol, augmentation
          in calm) on a new 'lead' voice (chan 2).  In bear sections the lead
          IS the featured voice: the inverted theme over the pads.
  VOLUME  volume_perc.perc_density (daily log-volume) scales hat velocities
          and gates the 16th-note perc ladder; volume_perc.fill_bars marks
          earnings-spike bars that get a last-beat drum fill.
  KEY     sections that are bull (252d-MA z > 1) end-to-end and not bear
          modulate melody+stabs to the relative major (root+3, major scale).
  DYNAMICS per-section gain staging on the music bus (energy .15->1.0 mapped
          to gain 0.55->1.0) BEFORE the fx chain, so quiet sections are
          genuinely quiet; a low-mix soundfont pad (sf_render) doubles the
          lead for body.

v3's engine survives underneath: instruments from synths.py, per-genre fx
buses from fx.py, harmonic color from harmony.py, Hilbert-envelope pads from
each ticker's spectral peaks.  duet.py still imports STYLES/PHRYG/AEOL/
midi2hz/roll/_kurt/spectral_peaks/synth/write_midi/SR and stays v3-form
(its ev carries no 'sections', so v4 staging simply doesn't engage).
Outputs per track: RAW WAV (mastering happens later) + standard MIDI file
(bass/stabs/lead/pads/drums + CC74 cutoff).
"""
import numpy as np, wave, os
from scipy import signal
import mido
import synths, fx, harmony, regime, motif, volume_perc

try:
    import sf_render
    _SF_OK = True
except Exception:
    _SF_OK = False

OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
os.makedirs(OUT, exist_ok=True)
SCRATCH = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)

# genre styles: groove + articulation parameters consumed by compose()/synth()
STYLES = {
    #            bass mode      kick      swing  stab positions    pad   res  bpm base
    "techno":    dict(bass="roll",    kick="four",  swing=0.00, stabs=(3, 7, 10, 14), pad=1.0, res=0.0, bpm0=128),
    "deephouse": dict(bass="offbeat", kick="four",  swing=0.16, stabs=(2, 6, 10, 14), pad=0.8, res=0.0, bpm0=120),
    "dub":       dict(bass="offbeat", kick="four",  swing=0.08, stabs=(2, 10),        pad=1.5, res=0.0, bpm0=118),
    "trance":    dict(bass="arp",     kick="four",  swing=0.00, stabs=(3, 7, 11, 15), pad=1.3, res=0.0, bpm0=136),
    "prog":      dict(bass="arp",     kick="four",  swing=0.05, stabs=(3, 7, 10, 14), pad=1.1, res=0.0, bpm0=128),
    "warehouse": dict(bass="roll",    kick="four",  swing=0.00, stabs=(3, 6, 10, 13), pad=0.7, res=0.0, bpm0=131),
    "breaks":    dict(bass="roll",    kick="break", swing=0.10, stabs=(3, 7, 10, 14), pad=0.9, res=0.0, bpm0=126),
    "acid":      dict(bass="acid",    kick="four",  swing=0.00, stabs=(7, 14),        pad=0.6, res=8.0, bpm0=134),
}

ALBUM = [  # ticker, title, root midi (bass octave), style
    ("NVDA",  "01_Chipmaker",        29, "techno"),     # F  — peak-time: the flagship story
    ("AAPL",  "02_Cupertino",        33, "deephouse"),  # A  — low vol, steady groove, swung
    ("MSFT",  "03_Redmond",          26, "dub"),        # D  — lowest vol: spacious dub echoes
    ("GOOGL", "04_PageRank",         31, "prog"),       # G  — long steady runs: progressive arps
    ("AMZN",  "05_Everything_Store", 36, "warehouse"),  # C  — relentless logistics: driving
    ("META",  "06_Social_Graph",     34, "breaks"),     # Bb — broke the 4/4: breakbeat
    ("NFLX",  "07_Stream",           28, "trance"),     # E  — binge arcs: uplifting trance
    ("TSLA",  "08_Ludicrous",        35, "acid"),       # B  — highest vol: 303 squelch
]
PHRYG = list(harmony.PHRYGIAN); AEOL = list(harmony.AEOLIAN)   # kept for duet.py
ARPSEQ = list(harmony.ARPSEQ)                           # trance/prog arpeggio degree cycle
MAJOR = [0, 2, 4, 5, 7, 9, 11]                          # relative-major modulation scale
midi2hz = lambda m: 440*2**((m-69)/12)

def roll(x, w, fn):
    out = np.empty(len(x))
    for i in range(len(x)): out[i] = fn(x[max(0, i-w):i+1])
    return out
def _kurt(x):
    v = x.var()
    return ((x-x.mean())**4).mean()/v**2 if v > 1e-12 else 3.0

def longest_bear_run(bearb):
    """(start_bar, end_bar_exclusive) of the longest consecutive bear-bar run."""
    best_len, best = 0, (None, None)
    i, n = 0, len(bearb)
    while i < n:
        if bearb[i]:
            j = i
            while j < n and bearb[j]: j += 1
            if j - i > best_len: best_len, best = j - i, (i, j)
            i = j
        else:
            i += 1
    return best

# 16th-note perc gate ladder: odd 16ths open up as volume density rises.
# positions b16 = 1,3,5,7,9,11,13,15 -> thresholds (low thr = plays first)
_PERC_THR = [0.90, 0.45, 0.75, 0.30, 0.85, 0.40, 0.70, 0.25]

# ---------------- composition: returns -> events ----------------
def compose(rs, root, style_name, v_log=None):
    st = STYLES[style_name]
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
    bpm = int(np.clip(st["bpm0"] + 30*(annvol - 0.35), st["bpm0"]-6, st["bpm0"]+8))

    bar = []
    for b in range(NBAR):
        ie = min((b+1)*16*dp16 - 1, n-1)
        z = (price[ie]-ma252[ie])/sd252[ie]
        deg = 5 if z < -1 else (3 if z > 0.5 else 0)
        bar.append(dict(deg=deg, z=z, vol=vol21[ie], kurt=kurt63[ie], bear=bool(bear[ie])))
    volz = np.array([bb["vol"] for bb in bar]); volz = (volz-volz.min())/max(np.ptp(volz), 1e-12)
    volpct = np.argsort(np.argsort([bb["vol"] for bb in bar]))/max(NBAR-1, 1)
    kurtz = np.array([bb["kurt"] for bb in bar]); kurtz = np.argsort(np.argsort(kurtz))/max(NBAR-1, 1)
    bearb = np.array([bb["bear"] for bb in bar])

    # ---- v4 FORM: data-driven sections ----
    sections = regime.segments(rs, NBAR, dp16)
    kindb = np.empty(NBAR, object); energyb = np.empty(NBAR)
    for s in sections:
        kindb[s["bar0"]:s["bar1"]] = s["kind"]; energyb[s["bar0"]:s["bar1"]] = s["energy"]
    masks = {k: regime.section_mask(k) for k in set(kindb)}
    crash_bars = {s["bar0"] for s in sections if s["bar0"] > 0}   # section downbeats
    # 1-bar drum fill INTO every peak section
    sec_fill = np.zeros(NBAR, bool)
    for s in sections:
        if s["kind"] == "peak" and s["bar0"] > 0:
            sec_fill[s["bar0"]-1] = True

    # ---- v4 VOLUME: perc density + earnings fills ----
    if v_log is not None:
        dens = volume_perc.perc_density(v_log, NBAR, dp16)
        vfill = volume_perc.fill_bars(v_log, NBAR, dp16)
    else:
        dens = np.full(NBAR, 0.6); vfill = np.zeros(NBAR, bool)
    # earnings fill in the bar's last beat; never stack with a section fill,
    # only where the kit is actually playing
    vol_fill = np.array([bool(vfill[b]) and not sec_fill[b] and masks[kindb[b]]["kick"]
                         for b in range(NBAR)])

    # ---- v4 KEY MODULATION: whole-section bull -> relative major ----
    bull = np.array([regime.is_bull(rs, b, dp16) for b in range(NBAR)])
    modb = np.zeros(NBAR, bool)
    for s in sections:
        if s["kind"] != "bear" and bool(bull[s["bar0"]:s["bar1"]].all()):
            modb[s["bar0"]:s["bar1"]] = True

    # ---- v4 MELODY: the stock's theme ----
    theme = motif.derive(rs, AEOL)

    ev = dict(bpm=bpm, dp16=dp16, N16=N16, NBAR=NBAR, root=root, bear=bear, rs=rs,
              style=st, style_name=style_name,
              notes=[], drums=[], cc=[], volz=volz, kurtz=kurtz, bearb=bearb,
              sections=sections, kindb=list(kindb), energyb=energyb, theme=theme,
              modb=modb, dens=dens, sec_fill=sec_fill, vol_fill=vol_fill,
              n_mod=int(modb.sum()), n_fill=int(sec_fill.sum()+vol_fill.sum()))
    prev_acid = None                                       # (k, midi) of last acid 16th
    for k in range(N16):
        b16 = k % 16; bn = k//16; B = bar[bn]
        km = masks[kindb[bn]]; halftime = km["halftime"]
        recov = kindb[bn] == "recovery"
        scale = harmony.scale_for(style_name, B["bear"], build=recov)
        vz, kz = volz[bn], kurtz[bn]
        i0, i1 = k*dp16, (k+1)*dp16
        big = np.abs(rs[i0:i1]).max() > sigma
        vel = int(70 + 45*vz)
        if b16 == 0:
            ev["cc"].append((k, 74, int(30 + 80*vz)))          # filter cutoff automation
        # ---- drums ----
        if sec_fill[bn]:                                       # full-bar roll INTO the peak
            ev["drums"].append((k, 39 if k % 2 == 0 else 38, int(40 + 85*(b16/15))))
        if km["kick"]:
            if halftime:            kick_here = b16 in (0, 8)  # halftime: 1 and 3 only
            elif st["kick"] == "four": kick_here = b16 % 4 == 0
            else:                   kick_here = b16 in (0, 10)
            if kick_here:
                kv = int((vel+15)*(0.82 + 0.18*energyb[bn]))   # section energy in the kick
                ev["drums"].append((k, 36, min(127, kv)))
            if st["kick"] == "break" and b16 in (4, 12) and not halftime:
                ev["drums"].append((k, 38, int(85+25*vz)))     # breakbeat snare
            if st["kick"] == "four" and b16 in (4, 12) and not halftime and not sec_fill[bn]:
                ev["drums"].append((k, 39, int(75+25*vz)))     # clap on 2 & 4
        hmul = km["hats"]*(0.55 + 0.60*dens[bn])               # volume drives hat level
        if b16 % 4 == 2:
            hv = int((45+30*vz)*hmul)
            if hv >= 8: ev["drums"].append((k, 46, min(127, hv)))   # open hat offbeat
        elif b16 % 2 == 0:
            hv = int((30+25*vz)*hmul)
            if hv >= 8: ev["drums"].append((k, 42, min(127, hv)))   # closed hat, 8th grid
        else:                                                  # 16th-note perc: volume-gated
            if (not halftime and km["perc"] > 0
                    and dens[bn]*km["perc"] > _PERC_THR[(b16-1)//2]):
                hv = int((26+25*vz)*hmul)
                if hv >= 8: ev["drums"].append((k, 42, min(127, hv)))
        if (style_name == "warehouse" and b16 % 4 == 2 and km["kick"]
                and dens[bn]*km["perc"] > 0.25):
            ev["drums"].append((k, 51, int((45+30*vz)*km["perc"])))  # offbeat ride
        if recov:
            ev["drums"].append((k, 39, int(30+90*(b16/15))))   # recovery roll build
        if vol_fill[bn] and b16 >= 12:                         # earnings fill, last beat
            ev["drums"].append((k, 39 if k % 2 == 0 else 38, int(55 + 65*((b16-12)/3))))
        if b16 == 0 and bn in crash_bars:
            ev["drums"].append((k, 49, 100))                   # section downbeat crash
        if np.abs(rs[i0:i1]).max()/sigma > 4:
            ev["drums"].append((k, 49, 85))                    # shock crash
        # ---- bass ----
        if km["bass"]:
            mode = st["bass"]
            if mode == "roll":
                if not (halftime and k % 2 == 1):              # halftime: 8ths, not 16ths
                    octv = 12 if (b16 % 4 == 2 and big) else 0
                    m = root + scale[B["deg"] % 7] + octv
                    v = int(vel*(1.15 if b16 % 2 == 1 else 0.85))
                    ev["notes"].append((k, 0.9, m, min(127, v), 0, None))
            elif mode == "offbeat":                            # house/dub: 8th offbeats, longer
                if (b16 in (2, 10)) if halftime else (b16 % 4 == 2):
                    m = root + scale[B["deg"] % 7] + (12 if big else 0)
                    ev["notes"].append((k, 1.8, m, min(127, vel), 0, None))
            elif mode == "arp":                                # trance/prog: run the arp cycle
                if not (halftime and k % 2 == 1):
                    deg = harmony.bass_degree(style_name, B["deg"], k)
                    octv = 12*((k % 3 == 2) + (1 if big else 0))
                    v = int(vel*(1.2 if b16 % 4 == 0 else 0.8))
                    ev["notes"].append((k, 0.85, root+12+scale[deg]+octv, min(127, v), 0, None))
            elif mode == "acid":                               # 303: 16ths, data-driven accents
                if not (halftime and k % 2 == 1):
                    accent = big or (b16 in (3, 11))
                    m = root + scale[B["deg"] % 7] + (12 if b16 % 8 == 6 else 0)
                    v = 120 if accent else int(vel*0.75)       # vel>110 = accent -> filter bite
                    sf = prev_acid[1] if (prev_acid is not None and prev_acid[0] == k-1
                                          and prev_acid[1] != m) else None
                    ev["notes"].append((k, 0.55 if accent else 0.9, m, min(127, v), 0, sf))
                    prev_acid = (k, m)
        # ---- lead: the theme, developed by the bar's data (chan 2) ----
        if b16 == 0 and km["lead"]:
            in_bear = kindb[bn] == "bear"
            if in_bear or B["bear"]:
                # drawdown -> REAL (chromatic) inversion: mirror the PRIME
                # theme in MIDI space around the fixed midpoint of its own
                # compass in the track's home scale.  Degree-space mirrors
                # don't negate semitone intervals (scale steps are uneven);
                # this does, exactly — so the ostinato cell's inversion is
                # literally audible (and countable) in every bear bar, and
                # consecutive bear bars chain with negated wrap intervals too.
                home = harmony.scale_for(style_name, False)
                hroot = root + 36
                pivot = (motif.to_midi(min(theme), home, hroot)
                         + motif.to_midi(max(theme), home, hroot))
                for off, dur, dgr, acc in motif.develop(theme, 0.0, False, 0.5, bn):
                    lm = pivot - motif.to_midi(dgr, home, hroot)
                    lv = int(52 + 34*vz) + (24 if acc else 0) + 16
                    ev["notes"].append((k+off, dur*1.9, lm, min(127, lv), 2, None))
            else:
                if modb[bn]:  msc, lroot = MAJOR, root + 3 + 36   # relative major re-center
                else:         msc, lroot = scale,  root + 36
                for off, dur, dgr, acc in motif.develop(theme, B["z"], False,
                                                        volpct[bn], bn):
                    lm = motif.to_midi(dgr, msc, lroot)
                    lv = int(52 + 34*vz) + (24 if acc else 0)
                    ev["notes"].append((k+off, dur*0.95, lm, min(127, lv), 2, None))
        # ---- stabs ----
        keep = st["stabs"]
        if km["stab_density"] < 1.0:
            keep = keep[:max(1, int(round(len(keep)*km["stab_density"])))] \
                   if km["stab_density"] > 0 else ()
        if km["stabs"] and b16 in keep:
            if np.abs(rs[i0:i1]).sum() > np.abs(rs).mean()*dp16:
                if modb[bn]:  vroot, vsc = root + 3 + 24, MAJOR
                else:         vroot, vsc = root + 24, scale
                for c in harmony.voicing(style_name, vroot, vsc, B["deg"]):
                    ev["notes"].append((k, 3.2 if style_name == "dub" else 1.8,
                                        c, int(55+40*kz), 1, None))
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
        mq = min(cands, key=lambda mm: abs(mm-m)) - 24
        out.append((1/f[i], midi2hz(mq), mq, np.sqrt(P[i]/P[sel[0]])))
    return out

# ---------------- synthesis: events -> stereo wav ----------------
SR = 44100
_DRUM_NAME = {36: "kick", 38: "snare", 39: "clap", 42: "hatc",
              46: "hato", 49: "crash", 51: "ride"}
_DRUM_GAIN = {36: 0.90, 38: 0.50, 39: 0.45, 42: 0.20,
              46: 0.16, 49: 0.45, 51: 0.20}
_HAT_TRIM = {"acid": 1.5, "techno": 0.8}   # acid sizzles up top, techno stays dark

def _fx_chain(genre, L, R, sr, s16):
    """Per-genre fx chain for the non-drum bus (applied before the sidechain)."""
    if genre == "techno":
        L, R = fx.saturate(L, R, 1.3)
        L, R = fx.reverb(L, R, sr, size=0.35, mix=0.18)
    elif genre == "deephouse":
        L, R = fx.chorus(L, R, sr, mix=0.4)
        L, R = fx.reverb(L, R, sr, size=0.45, mix=0.25)
    elif genre == "dub":
        L, R = fx.tape_wobble(L, R, sr)
        L, R = fx.reverb(L, R, sr, size=0.85, damp=0.6, mix=0.5)
        L, R = fx.pingpong(L, R, sr, 3*s16, fb=0.45)
    elif genre == "prog":
        L, R = fx.reverb(L, R, sr, size=0.5, mix=0.22)
        L, R = fx.pingpong(L, R, sr, 3*s16, fb=0.3)
    elif genre == "warehouse":
        L, R = fx.saturate(L, R, 1.8)
        L, R = fx.reverb(L, R, sr, size=0.3, mix=0.15)
    elif genre == "breaks":
        L, R = fx.saturate(L, R, 1.2)
        L, R = fx.reverb(L, R, sr, size=0.4, mix=0.2)
    elif genre == "trance":
        L, R = fx.chorus(L, R, sr, mix=0.5)
        L, R = fx.reverb(L, R, sr, size=0.6, mix=0.3)
        L, R = fx.pingpong(L, R, sr, 3*s16, fb=0.35)
    elif genre == "acid":
        L, R = fx.saturate(L, R, 2.6)
        L, R = fx.reverb(L, R, sr, size=0.2, mix=0.12)
    elif genre == "duet":
        L, R = fx.chorus(L, R, sr, mix=0.35)
        L, R = fx.reverb(L, R, sr, size=0.6, mix=0.28)
        L, R = fx.pingpong(L, R, sr, 3*s16, fb=0.3)
    return L, R

def synth(ev, peaks, path):
    st = ev["style"]; genre = ev["style_name"]
    bpm, N16 = ev["bpm"], ev["N16"]
    S16 = 60/bpm/4
    swing = st["swing"]
    total = int(N16*S16*SR)
    dL = np.zeros(total); dR = np.zeros(total)               # drum bus (never ducked)
    mL = np.zeros(total); mR = np.zeros(total)               # music bus (bass/stabs/lead/pads)
    def t16(k):                                              # swung 16th start sample
        return int((k + (swing if k % 2 else 0))*S16*SR)
    def add(bl, br, stt, sig, pl=1.0, pr=1.0):
        e = min(stt+len(sig), total)
        if e > stt:
            bl[stt:e] += sig[:e-stt]*pl; br[stt:e] += sig[:e-stt]*pr

    # ---- drums: per-genre kit from synths.drum ----
    kicks = []
    for k, note, vel in ev["drums"]:
        stt = t16(k); v = vel/127
        name = _DRUM_NAME[note]
        if name == "kick" and genre == "dub":
            v = min(v, 0.75)                                 # dub macro: kick never peaks
        d = synths.drum(genre, name, v)*_DRUM_GAIN[note]
        if note in (42, 46):
            d = d*_HAT_TRIM.get(genre, 1.0)
        pan = (0.8, 1.0) if note == 46 else (1.0, 0.8) if note == 42 else (1.0, 1.0)
        add(dL, dR, stt, d, *pan)
        if note == 36: kicks.append(stt)

    # ---- bass + stabs + lead: per-genre voices from synths ----
    cutmap = {c[0]: c[2] for c in ev["cc"]}
    cut = 200
    sf_events = []                                           # lead notes for the sf double
    for e in ev["notes"]:
        k, dur, m, vel, chan = e[:5]
        sf = e[5] if len(e) > 5 else None
        if k in cutmap: cut = 160 + 28*cutmap[k]
        stt = t16(k); v = vel/127
        if chan == 0:
            c_hz = cut
            if genre == "acid":                              # 16-bar triangle LFO x0.9..x2.4
                tri = 1 - abs(2*((k % 256)/256.0) - 1)
                c_hz = cut*(0.9 + 1.5*tri)
            sig = synths.bass_note(genre, m, S16*dur, v, c_hz,
                                   slide_from=sf, accent=(vel > 110))
            add(mL, mR, stt, sig*0.38)
        elif chan == 2:                                      # v4 lead: the developed theme
            sig = synths.lead_note(genre, m, S16*dur, v, 1200 + cut)
            pan = 0.22 if (k//16) % 2 else -0.22
            add(mL, mR, stt, sig*0.22, 1-max(pan, 0), 1+min(pan, 0))
            sf_events.append((stt/SR, S16*dur*1.6, m, v))
        else:
            sig = synths.lead_note(genre, m, S16*dur, v, 900 + cut)
            pan = 0.35 if (k//4) % 2 else -0.35
            add(mL, mR, stt, sig*0.15, 1-max(pan, 0), 1+min(pan, 0))

    # soundfont warm-pad double under the lead, low mix (body, not a new voice)
    if _SF_OK and sf_events:
        try:
            sfl, sfr = sf_render.render_notes(sf_events, 89)   # GM Pad 2 (warm)
            e = min(len(sfl), total)
            mL[:e] += sfl[:e]*0.06; mR[:e] += sfr[:e]*0.06
        except Exception:
            pass

    # trance macro: pads x2 during the longest breakdown
    padgain = np.ones(total)
    if genre == "trance":
        lb0, lb1 = longest_bear_run(np.asarray(ev["bearb"], bool))
        if lb0 is not None:
            s0 = min(int(lb0*16*S16*SR), total); s1 = min(int(lb1*16*S16*SR), total)
            padgain[s0:s1] = 2.0

    # pads: Hilbert band envelopes, breakdown-gated, scale-quantized, style-weighted
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
        pad = np.interp(day_axis, np.arange(n), env)*voice*0.5*(0.16*bg)*lfo*amp*st["pad"]*padgain
        pan = j/max(len(peaks)-1, 1)
        mL += pad*(1-0.5*pan); mR += pad*(0.5+0.5*pan)       # pads ride the genre fx bus too

    # v4 DYNAMICS: per-section gain staging on the music bus, BEFORE fx —
    # energy .15..1.0 -> gain 0.55..1.0, crossfaded across bar centers
    secs = ev.get("sections")
    if secs:
        eb = np.asarray(ev["energyb"], float)
        gains = 0.55 + 0.45*np.clip((eb - 0.15)/0.85, 0, 1)
        xbar = np.arange(total)/(16*S16*SR)                  # fractional bar position
        genv = np.interp(xbar, np.arange(len(gains)) + 0.5, gains)
        mL *= genv; mR *= genv

    # per-genre fx chain on the music bus, then sidechain duck + kick restore
    mL, mR = _fx_chain(genre, mL, mR, SR, S16)               # s16 = one 16th in seconds
    mL, mR = fx.sidechain(mL, mR, SR, kicks)
    L = dL + mL; R = dR + mR
    kv = min(0.9, 0.75) if genre == "dub" else 0.9
    kd = synths.drum(genre, "kick", kv)*0.35                 # kick restore over the duck
    for ks in kicks:
        e = min(ks+len(kd), total)
        L[ks:e] += kd[:e-ks]; R[ks:e] += kd[:e-ks]
    # master
    b, aa = signal.butter(2, 25/(SR/2), "high")
    L, R = signal.lfilter(b, aa, L), signal.lfilter(b, aa, R)
    fade = np.minimum(1, np.minimum(t_full/0.3, (t_full[-1]-t_full)/4.0))
    L *= fade; R *= fade
    L = np.nan_to_num(L); R = np.nan_to_num(R)
    mx = max(np.abs(L).max(), np.abs(R).max(), 1e-9)
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
    swing = ev["style"]["swing"]
    mid = mido.MidiFile(ticks_per_beat=TPQ)
    meta = mido.MidiTrack(); mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(ev["bpm"]), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    # section map as markers on the conductor track (v4 form, if present)
    for s in ev.get("sections", []):
        meta.append(mido.MetaMessage("marker", text=f"{s['kind']}@{s['bar0']}", time=0))
    def tick(t): return int((t + (swing if int(t) % 2 else 0))*T16)

    def track_from(events, chan, program=None, name=""):
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if program is not None:
            tr.append(mido.Message("program_change", channel=chan, program=program, time=0))
        msgs = []
        for e in events:
            if e[0] == "cc":
                _, tt, cc, val = e
                msgs.append((tick(tt), 0, mido.Message("control_change", channel=chan, control=cc, value=val, time=0)))
            else:
                _, tt, dur16, note, vel = e
                msgs.append((tick(tt), 1, mido.Message("note_on", channel=chan, note=note, velocity=vel, time=0)))
                msgs.append((tick(tt+dur16), 0, mido.Message("note_off", channel=chan, note=note, velocity=0, time=0)))
        msgs.sort(key=lambda x: (x[0], x[1]))
        prev = 0
        for tk, _, msg in msgs:
            msg.time = tk - prev; prev = tk
            tr.append(msg)

    bass = [("n", e[0], e[1], e[2], e[3]) for e in ev["notes"] if e[4] == 0]
    bass += [("cc", k, cc, val) for k, cc, val in ev["cc"]]
    stabs = [("n", e[0], e[1], e[2], e[3]) for e in ev["notes"] if e[4] == 1]
    lead = [("n", e[0], e[1], e[2], e[3]) for e in ev["notes"] if e[4] == 2]
    drums = [("n", k, 0.5, note, vel) for k, note, vel in ev["drums"]]
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
    track_from(lead,  2, program=81, name="lead")            # v4: the theme voice
    track_from(pads,  3, program=90, name="pads")
    track_from(drums, 9, name="drums")
    mid.save(path)

# ---------------- per-track report helpers ----------------
def _section_rms(path, ev):
    """Per-section RMS (dBFS) of a rendered wav, using the section bar spans."""
    with wave.open(path) as w_:
        nf = w_.getnframes()
        x = np.frombuffer(w_.readframes(nf), dtype=np.int16).astype(np.float64)/32767
    x = x.reshape(-1, 2).mean(axis=1)
    S16 = 60/ev["bpm"]/4; spb = 16*S16*SR
    out = []
    for s in ev["sections"]:
        a, b = int(s["bar0"]*spb), min(int(s["bar1"]*spb), len(x))
        if b > a:
            r = np.sqrt(np.mean(x[a:b]**2))
            out.append((s["kind"], 20*np.log10(max(r, 1e-9))))
    return out

def render_track(ticker, title, root, style):
    rs = np.load(os.path.join(SCRATCH, f"r_{ticker}.npy"))
    vpath = os.path.join(SCRATCH, f"v_{ticker}.npy")
    v_log = np.load(vpath) if os.path.exists(vpath) else None
    ev = compose(rs, root, style, v_log=v_log)
    peaks = spectral_peaks(rs, root % 12)
    wav = os.path.join(OUT, f"{title}.wav")
    dur = synth(ev, peaks, wav)
    write_midi(ev, peaks, os.path.join(OUT, f"{title}.mid"))
    secstr = " ".join(f"{s['kind']}:{s['bar1']-s['bar0']}" for s in ev["sections"])
    rms = _section_rms(wav, ev)
    loud = [d for _, d in rms if d > -55]
    rng_db = (max(loud) - min(loud)) if len(loud) > 1 else 0.0
    print(f"{title:22s} {style:>10s} {ev['bpm']:4d} {int(dur//60)}:{int(dur%60):02d} "
          f"bars={ev['NBAR']:3d} mod={ev['n_mod']:3d} fills={ev['n_fill']:2d} "
          f"rmsrange={rng_db:4.1f}dB", flush=True)
    print(f"    sections: {secstr}", flush=True)
    return ev, dur

# ---------------- run the album ----------------
if __name__ == "__main__":
    print(f"{'track':22s} {'style':>10s} {'bpm':>4s} {'len':>6s}", flush=True)
    for ticker, title, root, style in ALBUM:
        render_track(ticker, title, root, style)
    print("album written to", OUT)
