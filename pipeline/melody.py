import numpy as np, json, wave, base64
from scipy import signal
from scipy.signal.windows import dpss

r = np.load(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\r.npy")
N = len(r)

def mtspec(x):
    x = x - x.mean()
    tapers = dpss(len(x), 4, 7)
    P = np.array([np.abs(np.fft.rfft(x*t))**2 for t in tapers]).mean(axis=0)
    f = np.fft.rfftfreq(len(x), d=1.0)
    return f[1:], P[1:]

fr, Pr = mtspec(r)
fa, Pa = mtspec(np.abs(r))

# --- GPH estimator of fractional d for |r| ---
fp, Pp = signal.periodogram(np.abs(r)-np.abs(r).mean(), fs=1.0)
fp, Pp = fp[1:], Pp[1:]
m = int(N**0.6)  # standard bandwidth
lam = 2*np.pi*fp[:m]
Y = np.log(Pp[:m])
X = np.log(4*np.sin(lam/2)**2)
A = np.vstack([np.ones(m), -X]).T
coef, res_, *_ = np.linalg.lstsq(A, Y, rcond=None)
d_hat = coef[1]
se = np.pi/np.sqrt(24*m)
print(f"GPH: d = {d_hat:.3f} +/- {se:.3f} (m={m})  -> t = {d_hat/se:.1f}")

# --- log-binned spectra for plotting (decimate to ~120 pts per series) ---
def logbin(f, P, nb=120):
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), nb+1)
    fc, pc = [], []
    for i in range(nb):
        m_ = (f >= edges[i]) & (f < edges[i+1])
        if m_.any():
            fc.append(f[m_].mean()); pc.append(P[m_].mean())
    return np.array(fc), np.array(pc)

frb, Prb = logbin(fr, Pr)
fab, Pab = logbin(fa, Pa)

# --- melody: significant peaks of |r| spectrum -> pitches ---
# top distinct peaks (merge peaks closer than 20% in freq)
from scipy.signal import find_peaks
pk, _ = find_peaks(Pa)
pk = pk[(fa[pk] > 1/600) & (fa[pk] < 1/40)]  # significant low-freq band only
pk = pk[np.argsort(Pa[pk])[::-1]]
sel = []
for i in pk:
    if all(abs(np.log(fa[i]/fa[j])) > 0.18 for j in sel):
        sel.append(i)
    if len(sel) == 6: break
sel = sorted(sel, key=lambda i: fa[i])

SCALE = 2.2e5  # daily cycles -> audible Hz
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def note_name(hz):
    n = int(round(12*np.log2(hz/440.0))) + 69
    return f"{NOTES[n%12]}{n//12-1}"

peaks = []
for i in sel:
    hz = fa[i]*SCALE
    peaks.append(dict(period=float(1/fa[i]), f=float(fa[i]), hz=float(hz),
                      note=note_name(hz), amp=float(np.sqrt(Pa[i]/Pa[sel].max()))))
    print(f"peak T={1/fa[i]:7.1f} d  -> {hz:7.1f} Hz  ({note_name(hz)})  amp={peaks[-1]['amp']:.2f}")

# --- synthesize: arpeggio low->high, then the full chord ---
sr = 16000
def tone(hz, dur, amp):
    t = np.arange(int(sr*dur))/sr
    env = np.minimum(1, np.minimum(t/0.03, (dur-t)/0.25))
    y = np.sin(2*np.pi*hz*t) + 0.35*np.sin(4*np.pi*hz*t) + 0.15*np.sin(6*np.pi*hz*t)
    return amp*env*y

parts = [tone(p['hz'], 0.7, 0.55*p['amp']+0.25) for p in peaks]
chord_dur = 3.0
chord = sum(tone(p['hz'], chord_dur, 0.35*p['amp']+0.12) for p in peaks)
audio = np.concatenate(parts + [np.zeros(sr//5), chord])
audio = (audio/np.abs(audio).max()*0.85*32767).astype(np.int16)

wav_path = r"C:\Users\vito.ciciretti\Downloads\nvda_melody.wav"
with wave.open(wav_path, 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(audio.tobytes())
print(f"wrote {wav_path} ({len(audio)/sr:.1f}s)")

with open(wav_path, 'rb') as fh:
    b64 = base64.b64encode(fh.read()).decode()
print(f"wav base64 size: {len(b64)/1024:.0f} KB")

out = dict(
    d_hat=float(d_hat), d_se=float(se),
    ret=dict(f=frb.round(6).tolist(), P=(Prb/Prb.mean()).round(4).tolist()),
    vol=dict(f=fab.round(6).tolist(), P=(Pab/Prb.mean()).round(4).tolist()),
    peaks=peaks, wav=b64,
)
with open(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\plotdata.json", 'w') as fh:
    json.dump(out, fh)
print("plotdata.json written")
