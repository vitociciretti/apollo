import numpy as np, wave
from scipy import signal
from arch import arch_model

rng = np.random.default_rng(7)
r = np.load(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\r.npy")

# ---------- 1. sample from the fitted distribution ----------
BURN = 252
DAYS_PER_8TH = 7
N8 = 576                      # 8th notes: 180 s at 96 BPM
NSIM = N8 * DAYS_PER_8TH + BURN
am = arch_model(100*r, vol="GARCH", p=1, o=1, q=1, dist="skewt")
res = am.fit(disp="off")
sim = am.simulate(res.params, NSIM)
rs = sim["data"].values / 100
rs = rs[BURN:]; n = len(rs)
price = np.cumsum(rs)

# rolling stats
def roll(x, w, fn):
    out = np.empty(len(x))
    for i in range(len(x)): out[i] = fn(x[max(0, i-w):i+1])
    return out
ma252   = roll(price, 252, np.mean)
sd252   = np.maximum(roll(price, 252, np.std), 1e-6)
vol21   = roll(rs, 21, np.std) * np.sqrt(252)
def _kurt(x):
    v = x.var()
    return ((x - x.mean())**4).mean()/v**2 if v > 1e-12 else 3.0
kurt63  = roll(rs, 63, _kurt)                       # 4th moment -> timbre
runmax  = np.maximum.accumulate(price)
bear    = (runmax - price) > 0.25          # >25% drawdown in log terms -> minor key
sigma   = rs.std()

print(f"sampled {n} days ({n/252:.1f} trading yrs) | ann.vol {rs.std()*np.sqrt(252):.1%} | bear share {bear.mean():.0%}")

# ---------- 2. musical mapping ----------
SR = 22050
BPM = 96; T8 = 60/BPM/2                     # 0.3125 s per 8th
midi2hz = lambda m: 440*2**((m-69)/12)
MAJ = [0, 2, 4, 7, 9]; MIN = [0, 3, 5, 7, 10]
ROOT = 65                                    # F4
def scale_note(z, minor):                    # z in [-2.5, 2.5] -> 2-octave pentatonic
    deg = int(round((np.clip(z, -2.5, 2.5) + 2.5) / 5 * 9))
    s = MIN if minor else MAJ
    return ROOT + 12*(deg//5) + s[deg % 5]

notes = []
for k in range(N8):
    i0, i1 = k*DAYS_PER_8TH, (k+1)*DAYS_PER_8TH
    iend = i1 - 1
    z = (price[iend] - ma252[iend]) / sd252[iend]
    notes.append(dict(
        midi=scale_note(z, bear[iend]),
        vol=vol21[iend],
        kurt=kurt63[iend],
        shock=np.abs(rs[i0:i1]).max()/sigma,
        minor=bool(bear[iend])))

volz = np.array([nt["vol"] for nt in notes])
volz = (volz - volz.min()) / (volz.max() - volz.min())   # 0..1 loudness driver
kz = np.array([nt["kurt"] for nt in notes])
kz = np.argsort(np.argsort(kz)) / (len(kz)-1)            # percentile rank 0..1 -> timbre driver

# ---------- 3. synthesis ----------
total = int(N8 * T8 * SR)
L = np.zeros(total); R = np.zeros(total)
def add(buf, start, sig):
    e = min(start + len(sig), total)
    buf[start:e] += sig[:e-start]

def tone(hz, dur, amp, timbre=0.0):
    # timbre 0 = pure sine (thin tails); 1 = band-limited saw + grit (fat tails)
    t = np.arange(int(SR*dur))/SR
    env = np.minimum(1, np.minimum(t/0.012, np.maximum(0, (dur-t)/(dur*0.6))))
    y = np.sin(2*np.pi*hz*t)
    for h in range(2, 7):                     # saw harmonics fade in with timbre
        if h*hz < SR/2:
            y += (timbre/h) * np.sin(2*np.pi*h*hz*t)
    if timbre > 0.6:                          # extreme-kurtosis grit
        y += 0.12*(timbre-0.6)/0.4 * rng.standard_normal(len(t)) * np.exp(-t*8)
    return amp*env*y

def kick(amp):
    t = np.arange(int(SR*0.18))/SR
    return amp*np.sin(2*np.pi*(120*np.exp(-t*18)+42)*t)*np.exp(-t*22)
def hat(amp):
    t = np.arange(int(SR*0.05))/SR
    no = rng.standard_normal(len(t))
    b, a = signal.butter(2, 0.55, "high"); return amp*signal.lfilter(b, a, no)*np.exp(-t*70)
def snare(amp):
    t = np.arange(int(SR*0.12))/SR
    no = rng.standard_normal(len(t))
    b, a = signal.butter(2, [0.12, 0.6], "band")
    return amp*(signal.lfilter(b, a, no)*np.exp(-t*28) + 0.4*np.sin(2*np.pi*190*t)*np.exp(-t*35))
def crash(amp):
    t = np.arange(int(SR*1.2))/SR
    no = rng.standard_normal(len(t))
    b, a = signal.butter(2, 0.3, "high"); return amp*signal.lfilter(b, a, no)*np.exp(-t*3.5)

# lead melody + drums on the 8th grid
for k, nt in enumerate(notes):
    s = int(k*T8*SR)
    v = 0.25 + 0.55*volz[k]
    lead = tone(midi2hz(nt["midi"]), T8*0.95, 0.5*v, timbre=kz[k])
    add(L, s, lead); add(R, s, lead)
    if k % 2 == 0:                       # kick on beats, velocity = vol
        d = kick(0.55 + 0.45*volz[k]); add(L, s, d); add(R, s, d)
    if k % 8 == 4:                       # snare on 2 & 4
        d = snare(0.4 + 0.3*volz[k]); add(L, s, d); add(R, s, d)
    h = hat(0.10 + 0.12*volz[k]); add(L, s, h*0.7); add(R, s, h)
    if nt["shock"] > 4:                  # >4 sigma day -> crash
        d = crash(0.5*min(nt["shock"]/6, 1)); add(L, s, d*0.9); add(R, s, d*0.9)

# spectral pads: six peaks, Hilbert band envelopes from simulated |r|, panned
PEAKS = [(519.5, 211.7), (319.7, 344.1), (207.8, 529.4),
         (153.9, 714.7), (125.9, 873.5), (63.9, 1720.4)]   # one octave down for warmth
a_ = np.abs(rs) - np.abs(rs).mean()
t_full = np.arange(total)/SR
day_axis = np.linspace(0, n-1, total)
for j, (T, hz) in enumerate(PEAKS):
    f0 = 1.0/T
    b, bb = signal.butter(3, [f0/1.35*2, min(f0*1.35*2, 0.99)], btype="band")
    env = np.abs(signal.hilbert(signal.filtfilt(b, bb, a_)))
    env = signal.filtfilt(*signal.butter(2, 0.05), env)
    env = np.maximum(env, 0); env /= max(env.max(), 1e-12)
    pad = np.interp(day_axis, np.arange(n), env) * np.sin(2*np.pi*hz*t_full) * 0.10
    pan = j/(len(PEAKS)-1)
    L += pad*(1-0.6*pan); R += pad*(0.4+0.6*pan)

# master: fade, soft-clip, normalize
fade = np.minimum(1, np.minimum(t_full/1.0, (t_full[-1]-t_full)/3.0))
L *= fade; R *= fade
mx = max(np.abs(L).max(), np.abs(R).max())
L, R = np.tanh(1.4*L/mx), np.tanh(1.4*R/mx)
mx2 = max(np.abs(L).max(), np.abs(R).max())
audio = np.empty(2*total, dtype=np.int16)
audio[0::2] = (L/mx2*0.9*32767).astype(np.int16)
audio[1::2] = (R/mx2*0.9*32767).astype(np.int16)

path = r"C:\Users\vito.ciciretti\Downloads\nvda_song_3min.wav"
with wave.open(path, "w") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(audio.tobytes())
print(f"wrote {path}: {total/SR:.0f}s stereo, {sum(nt['minor'] for nt in notes)} minor-key 8ths, "
      f"{sum(nt['shock']>4 for nt in notes)} crash events")
