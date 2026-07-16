"""Techno rendering of an RNN-sampled return path.
Mapping: vol->filter cutoff+kick velocity | price z->bar harmony | kurtosis->detune/brightness
bear regime->breakdown (kick+bass drop out, pads swell) | recovery->snare roll + crash | >4sigma day->crash
Clean mix: no grit noise, brief highpassed hats only, no soft-clip distortion on master."""
import numpy as np, wave, sys
from scipy import signal

rng = np.random.default_rng(11)
rs = np.load(sys.argv[1]); out_path = sys.argv[2]

SR = 44100
BPM = 128
S16 = 60/BPM/4                    # 16th note = 0.1171875 s
DAYS_PER_16TH = 3
N16 = len(rs) // DAYS_PER_16TH    # 1536
NBAR = N16 // 16                  # 96 bars
total = int(N16 * S16 * SR)
print(f"{N16} 16ths, {NBAR} bars, {total/SR:.1f}s")

n = len(rs)
price = np.cumsum(rs)
def roll(x, w, fn):
    out = np.empty(len(x))
    for i in range(len(x)): out[i] = fn(x[max(0, i-w):i+1])
    return out
def _kurt(x):
    v = x.var()
    return ((x-x.mean())**4).mean()/v**2 if v > 1e-12 else 3.0
ma252  = roll(price, 252, np.mean)
sd252  = np.maximum(roll(price, 252, np.std), 1e-6)
vol21  = roll(rs, 21, np.std)*np.sqrt(252)
kurt63 = roll(rs, 63, _kurt)
dd     = np.maximum.accumulate(price) - price
bear   = dd > np.quantile(dd, 0.78)   # adaptive: worst ~22% of this path's drawdowns = breakdown
sigma  = rs.std()

# per-bar state (1 bar = 48 days ~ 2.3 months)
bar = []
for b in range(NBAR):
    ie = min((b+1)*16*DAYS_PER_16TH - 1, n-1)
    z = (price[ie]-ma252[ie])/sd252[ie]
    deg = 5 if z < -1 else (3 if z > 0.5 else 0)     # harmony: i / iv-ish / VI-ish root motion
    bar.append(dict(z=z, deg=deg, vol=vol21[ie], kurt=kurt63[ie], bear=bool(bear[ie])))
volz  = np.array([bb["vol"] for bb in bar]);  volz = (volz-volz.min())/max(np.ptp(volz), 1e-12)
kurtz = np.array([bb["kurt"] for bb in bar]); kurtz = np.argsort(np.argsort(kurtz))/max(NBAR-1, 1)
bearb = np.array([bb["bear"] for bb in bar])
buildb = np.zeros(NBAR, bool)                        # last bear bar before recovery -> snare roll
for b in range(NBAR-1):
    if bearb[b] and not bearb[b+1]: buildb[b] = True
print(f"bear bars {bearb.sum()}/{NBAR}, builds {buildb.sum()}")

PHRYG = [0, 1, 3, 5, 7, 8, 10]; AEOL = [0, 2, 3, 5, 7, 8, 10]
ROOT = 29                                            # F1
midi2hz = lambda m: 440*2**((m-69)/12)

L = np.zeros(total); R = np.zeros(total)
def add(bufL, bufR, start, sig, panL=1.0, panR=1.0):
    e = min(start+len(sig), total)
    bufL[start:e] += sig[:e-start]*panL; bufR[start:e] += sig[:e-start]*panR

# ---------- clean drum voices ----------
def kick(amp):
    t = np.arange(int(SR*0.20))/SR
    body = np.sin(2*np.pi*(110*np.exp(-t*16)+45)*t)*np.exp(-t*14)
    click = np.sin(2*np.pi*900*t)*np.exp(-t*300)*0.5
    return amp*(body+click)
def hat_c(amp):                                       # closed: 25 ms, HP 8 kHz
    t = np.arange(int(SR*0.025))/SR
    b, a = signal.butter(4, 8000/(SR/2), "high")
    return amp*signal.lfilter(b, a, rng.standard_normal(len(t)))*np.exp(-t*160)
def hat_o(amp):                                       # open: 180 ms, HP 7 kHz
    t = np.arange(int(SR*0.18))/SR
    b, a = signal.butter(4, 7000/(SR/2), "high")
    return amp*signal.lfilter(b, a, rng.standard_normal(len(t)))*np.exp(-t*18)
def clap(amp):
    t = np.arange(int(SR*0.15))/SR
    b, a = signal.butter(2, [900/(SR/2), 3500/(SR/2)], "band")
    no = signal.lfilter(b, a, rng.standard_normal(len(t)))
    env = np.exp(-t*30)*(1 + 0.7*(np.sin(2*np.pi*55*t) > 0))   # multi-burst clap feel
    return amp*no*env
def crash(amp):
    t = np.arange(int(SR*0.9))/SR
    b, a = signal.butter(4, 6000/(SR/2), "high")
    return amp*signal.lfilter(b, a, rng.standard_normal(len(t)))*np.exp(-t*4.5)

def sawstack(hz, dur, detune, cutoff, amp):
    """detuned saw pair -> lowpass; additive, band-limited"""
    t = np.arange(int(SR*dur))/SR
    y = np.zeros(len(t))
    for dt in (1-detune, 1+detune):
        f = hz*dt
        for h in range(1, int(min(cutoff*2, SR/2*0.9)/f)+1):
            y += np.sin(2*np.pi*h*f*t)/h
    b, a = signal.butter(2, min(cutoff/(SR/2), 0.95), "low")
    y = signal.lfilter(b, a, y)
    env = np.minimum(1, np.minimum(t/0.004, np.maximum(0, (dur-t)/(dur*0.35))))
    return amp*env*y

# ---------- sequence ----------
kick_samples = []
for k in range(N16):
    b16 = k % 16; bnum = k//16
    st = int(k*S16*SR)
    B = bar[bnum]
    scale = PHRYG if B["bear"] else AEOL
    vz, kz = volz[bnum], kurtz[bnum]
    intro = bnum < 4; outro = bnum >= NBAR-2

    # drums
    if b16 % 4 == 0 and not B["bear"] and not outro:
        d = kick(0.85+0.15*vz); add(L, R, st, d); kick_samples.append(st)
    if b16 % 4 == 2:
        h = hat_o(0.16+0.10*vz); add(L, R, st, h, 0.8, 1.0)
    else:
        h = hat_c(0.07+0.06*vz); add(L, R, st, h, 1.0, 0.8)
    if b16 in (4, 12) and not B["bear"]:
        c = clap(0.5+0.2*vz); add(L, R, st, c)
    if buildb[bnum]:                                   # snare-roll build
        c = clap(0.15+0.75*(b16/15)); add(L, R, st, c)
    if b16 == 0 and bnum > 0 and bearb[bnum-1] and not B["bear"]:
        d = crash(0.55); add(L, R, st, d)
    i0, i1 = k*DAYS_PER_16TH, (k+1)*DAYS_PER_16TH
    if np.abs(rs[i0:i1]).max()/sigma > 4:
        d = crash(0.4); add(L, R, st, d)

    # rolling bass: 16ths, rest under the kick beat's first 16th in high-vol bars
    if not B["bear"] and not intro and not outro:
        deg = B["deg"]
        octv = 12 if (b16 % 4 == 2 and np.abs(rs[i0:i1]).max() > sigma) else 0   # data-driven octave jump
        m = ROOT + scale[deg % 7] + octv
        accent = 1.25 if b16 % 2 == 1 else 0.9                                    # off-beat accent = roll
        cutoff = 160 + 2800*vz + 500*kz
        bs = sawstack(midi2hz(m), S16*0.9, 0.004+0.006*kz, cutoff, 0.16*accent)
        add(L, R, st, bs)

    # stabs: sparse, syncopated (positions 3, 7, 10, 14 gated by data)
    if b16 in (3, 7, 10, 14) and not intro:
        gate = np.abs(rs[i0:i1]).sum() > np.abs(rs).mean()*DAYS_PER_16TH          # play only on active days
        if gate:
            deg = B["deg"]
            chord = [scale[deg % 7], scale[(deg+2) % 7]+12, scale[(deg+4) % 7]+12]
            stab = sum(sawstack(midi2hz(ROOT+24+c), S16*1.8, 0.008, 900+2500*kz, 0.055) for c in chord)
            pan = 0.35 if (k//4) % 2 else -0.35
            add(L, R, st, stab, 1-max(pan, 0), 1+min(pan, 0))

# ---------- spectral pads: breakdown-only, quantized to F minor (no more drone/buzz) ----------
PEAKS = [(519.5, 105.9), (319.7, 172.0), (207.8, 264.7), (153.9, 357.3), (125.9, 436.8), (63.9, 860.2)]
FMIN_PCS = {5, 7, 8, 10, 0, 1, 3}                             # F minor pitch classes
def quantize_fmin(hz):
    m = 12*np.log2(hz/440) + 69
    cands = [mm for mm in range(int(m)-6, int(m)+7) if mm % 12 in FMIN_PCS]
    return 440*2**((min(cands, key=lambda mm: abs(mm-m)) - 69)/12)
a_ = np.abs(rs) - np.abs(rs).mean()
t_full = np.arange(total)/SR
day_axis = np.linspace(0, n-1, total)
bear_smooth = signal.filtfilt(*signal.butter(2, 0.02), bear.astype(float))   # ~gentle fade in/out of regime
bear_gain = np.clip(np.interp(day_axis, np.arange(n), bear_smooth), 0, 1)
lfo = 1 + 0.15*np.sin(2*np.pi*0.11*t_full)                                   # slow movement, not static
for j, (T, hz) in enumerate(PEAKS):
    f0 = 1.0/T
    b, bb2 = signal.butter(3, [f0/1.35*2, min(f0*1.35*2, 0.99)], btype="band")
    env = np.abs(signal.hilbert(signal.filtfilt(b, bb2, a_)))
    env = signal.filtfilt(*signal.butter(2, 0.05), env)
    env = np.maximum(env, 0); env /= max(env.max(), 1e-12)
    hq = quantize_fmin(hz)
    voice = np.sin(2*np.pi*hq*0.997*t_full) + np.sin(2*np.pi*hq*1.003*t_full)  # detuned pair -> lush, not buzzy
    pad = np.interp(day_axis, np.arange(n), env)*voice*0.5*(0.16*bear_gain)*lfo
    pan = j/(len(PEAKS)-1)
    L += pad*(1-0.5*pan); R += pad*(0.5+0.5*pan)

# ---------- sidechain pump on bass/pads/stabs is approximated on full mix minus drums:
# simpler & musical: duck everything after each kick, drums re-added on top
duck = np.ones(total)
w = int(0.30*SR); tt = np.arange(w)/SR
shape = 1 - 0.55*np.exp(-tt/0.085)
for ks in kick_samples:
    e = min(ks+w, total)
    duck[ks:e] = np.minimum(duck[ks:e], shape[:e-ks])
L *= duck; R *= duck
for ks in kick_samples:
    d = kick(0.9); e = min(ks+len(d), total)
    L[ks:e] += d[:e-ks]*0.35; R[ks:e] += d[:e-ks]*0.35        # restore kick punch over the duck

# ---------- stereo delay on the high band (dotted-8th ping-pong), clean ----------
db, da = signal.butter(2, 1200/(SR/2), "high")
hiL, hiR = signal.lfilter(db, da, L), signal.lfilter(db, da, R)
dl = int(S16*3*SR)
echoL = np.zeros(total); echoR = np.zeros(total)
echoL[dl:] += hiR[:-dl]*0.22; echoR[2*dl:] += hiL[:-2*dl]*0.15
L += echoL; R += echoR

# ---------- master: DC block, fade, normalize (no distortion stage) ----------
b, a = signal.butter(2, 25/(SR/2), "high")
L, R = signal.lfilter(b, a, L), signal.lfilter(b, a, R)
fade = np.minimum(1, np.minimum(t_full/0.3, (t_full[-1]-t_full)/4.0))
L *= fade; R *= fade
mx = max(np.abs(L).max(), np.abs(R).max())
audio = np.empty(2*total, dtype=np.int16)
audio[0::2] = (L/mx*0.95*32767).astype(np.int16)
audio[1::2] = (R/mx*0.95*32767).astype(np.int16)
with wave.open(out_path, "w") as w_:
    w_.setnchannels(2); w_.setsampwidth(2); w_.setframerate(SR)
    w_.writeframes(audio.tobytes())
print(f"wrote {out_path}: {total/SR:.0f}s stereo @ {BPM} BPM")
