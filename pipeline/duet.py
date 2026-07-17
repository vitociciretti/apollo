"""Correlation One — the duet. NVDA drives the bass, AAPL the lead.
Rolling 63d correlation controls harmony: high rho -> unison/octave (voices lock),
mid rho -> consonant intervals (3rd/5th), low rho -> tension intervals (2nds/tritone).
Breakdown = JOINT drawdown (both in worst quartile). You hear diversification die.
Rendered with the dedicated "duet" voice bank (deep 38 Hz sub kick, dark hats,
FM ratio-3 bass, glassy sine lead, chorus+reverb+pingpong bus) so the closer
does not read as a clone of 01_Chipmaker's techno kit.
"""
import numpy as np, os
from techno_album import (STYLES, PHRYG, AEOL, midi2hz, roll, _kurt,
                          spectral_peaks, synth, write_midi, SR)
from scipy import signal

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
r1 = np.load(os.path.join(SCRATCH, "r_NVDA.npy"))   # bass voice
r2 = np.load(os.path.join(SCRATCH, "r_AAPL.npy"))   # lead voice
n = min(len(r1), len(r2)); r1, r2 = r1[:n], r2[:n]
ROOT = 29  # F

def rolling_corr(a, b, w):
    out = np.zeros(len(a))
    for i in range(len(a)):
        s = max(0, i-w+1)
        if i - s > 10:
            out[i] = np.corrcoef(a[s:i+1], b[s:i+1])[0, 1]
    return np.nan_to_num(out)

def compose_duet(r1, r2, root):
    st = dict(STYLES["techno"]); st["stabs"] = (3, 7, 10, 14); st["pad"] = 1.2
    n = len(r1)
    dp16 = max(2, round(n/1536))
    N16 = (n//dp16)//16*16; NBAR = N16//16
    p1, p2 = np.cumsum(r1), np.cumsum(r2)
    rho = rolling_corr(r1, r2, 63)
    def zscore(p):
        ma = roll(p, 252, np.mean); sd = np.maximum(roll(p, 252, np.std), 1e-6)
        return (p - ma)/sd
    z1, z2 = zscore(p1), zscore(p2)
    vol = (roll(r1, 21, np.std) + roll(r2, 21, np.std))*np.sqrt(252)/2
    kurt = (roll(r1, 63, _kurt) + roll(r2, 63, _kurt))/2
    dd1 = np.maximum.accumulate(p1) - p1; dd2 = np.maximum.accumulate(p2) - p2
    bear = (dd1 > np.quantile(dd1, 0.78)) & (dd2 > np.quantile(dd2, 0.78))  # JOINT stress only
    s1, s2 = r1.std(), r2.std()
    annvol = (s1 + s2)*np.sqrt(252)/2
    bpm = 126

    bar = []
    for b in range(NBAR):
        ie = min((b+1)*16*dp16 - 1, n-1)
        deg = 5 if z1[ie] < -1 else (3 if z1[ie] > 0.5 else 0)
        bar.append(dict(deg=deg, z2=z2[ie], rho=rho[ie], vol=vol[ie], kurt=kurt[ie], bear=bool(bear[ie])))
    volz = np.array([bb["vol"] for bb in bar]); volz = (volz-volz.min())/max(np.ptp(volz), 1e-12)
    kurtz = np.array([bb["kurt"] for bb in bar]); kurtz = np.argsort(np.argsort(kurtz))/max(NBAR-1, 1)
    bearb = np.array([bb["bear"] for bb in bar])
    buildb = np.zeros(NBAR, bool)
    for b in range(NBAR-1):
        if bearb[b] and not bearb[b+1]: buildb[b] = True

    ev = dict(bpm=bpm, dp16=dp16, N16=N16, NBAR=NBAR, root=root, bear=bear, rs=(r1+r2)/2,
              style=st, style_name="duet",     # its own voice bank: NOT the 01 techno kit
              notes=[], drums=[], cc=[], volz=volz, kurtz=kurtz, bearb=bearb)
    CONS = [0, 3, 7, 12]          # consonant offsets vs bass (m3, P5, octave)
    TENS = [1, 2, 6, 8, 10]       # tension offsets (2nds, tritone, b6, b7)
    for k in range(N16):
        b16 = k % 16; bn = k//16; B = bar[bn]
        scale = PHRYG if B["bear"] else AEOL
        vz = volz[bn]
        intro = bn < 4; outro = bn >= NBAR-2
        i0, i1 = k*dp16, (k+1)*dp16
        big1 = np.abs(r1[i0:i1]).max() > s1
        big2 = np.abs(r2[i0:i1]).max() > s2
        vel = int(70 + 45*vz)
        if b16 == 0:
            ev["cc"].append((k, 74, int(30 + 80*vz)))
        # drums (techno grammar, joint regime)
        if b16 % 4 == 0 and not B["bear"] and not outro:
            ev["drums"].append((k, 36, min(127, vel+15)))
        ev["drums"].append((k, 46 if b16 % 4 == 2 else 42,
                            int((45+30*vz) if b16 % 4 == 2 else (30+25*vz))))
        if b16 in (4, 12) and not B["bear"]:
            ev["drums"].append((k, 39, int(75+25*vz)))
        if buildb[bn]:
            ev["drums"].append((k, 39, int(30+90*(b16/15))))
        if b16 == 0 and bn > 0 and bearb[bn-1] and not B["bear"]:
            ev["drums"].append((k, 49, 100))
        if np.abs(r1[i0:i1]).max()/s1 > 4 or np.abs(r2[i0:i1]).max()/s2 > 4:
            ev["drums"].append((k, 49, 85))
        if B["bear"] or intro or outro: continue
        # bass: NVDA voice (rolling 16ths)
        octv = 12 if (b16 % 4 == 2 and big1) else 0
        bass_m = root + scale[B["deg"] % 7]
        v = int(vel*(1.15 if b16 % 2 == 1 else 0.85))
        ev["notes"].append((k, 0.9, bass_m + octv, min(127, v), 0))
        # lead: AAPL voice, 8ths, interval vs bass gated by correlation
        if b16 % 2 == 0:
            raw_deg = int(round((np.clip(B["z2"], -2.5, 2.5) + 2.5)/5 * 6))
            raw = root + 24 + scale[raw_deg % 7]
            rel = (raw - bass_m) % 12
            rho_ = B["rho"]
            if rho_ > 0.75:                      # locked: unison/octave with the bass
                off = 0 if rel < 6 else 12
                m = bass_m + 24 + off
            elif rho_ > 0.40:                    # consonant: snap to nearest of CONS
                off = min(CONS, key=lambda c: min((rel-c) % 12, (c-rel) % 12))
                m = bass_m + 24 + off
            else:                                # decoupled: snap to nearest tension
                off = min(TENS, key=lambda c: min((rel-c) % 12, (c-rel) % 12))
                m = bass_m + 24 + off
            lv = int(vel*(1.1 if big2 else 0.8))
            ev["notes"].append((k, 1.7, m, min(127, lv), 1))
    return ev, rho, bear

ev, rho, bear = compose_duet(r1, r2, ROOT)
peaks = spectral_peaks((np.abs(r1)+np.abs(r2))/2, ROOT % 12)
dur = synth(ev, peaks, os.path.join(OUT, "09_Correlation_One.wav"))
write_midi(ev, peaks, os.path.join(OUT, "09_Correlation_One.mid"))
locked = np.mean(rho[rho != 0] > 0.75)
print(f"09_Correlation_One: {int(dur//60)}:{int(dur%60):02d} @ {ev['bpm']} BPM, "
      f"{ev['bearb'].sum()}/{ev['NBAR']} joint-stress bars, "
      f"rho>0.75 {locked:.0%} of days, mean rho {rho[rho != 0].mean():.2f}")
np.save(os.path.join(SCRATCH, "duet_rho.npy"), rho)
