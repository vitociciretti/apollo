import numpy as np, wave
from scipy import signal

rng = np.random.default_rng(42)
r = np.load(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\r.npy")
N = len(r)

PEAKS = [(519.5, 423.5), (319.7, 688.2), (207.8, 1058.7),
         (153.9, 1429.3), (125.9, 1746.9), (63.9, 3440.8)]
SR = 16000

def sonify(vol_series, path, days_per_sec=63):
    """band-envelope sonification of an |r|-like series"""
    a = vol_series - vol_series.mean()
    n = len(a)
    n_audio = int(n / days_per_sec * SR)
    t = np.arange(n_audio) / SR
    day_axis = np.linspace(0, n - 1, n_audio)
    mix = np.zeros(n_audio)
    for T, hz in PEAKS:
        f0 = 1.0 / T
        b, bb = signal.butter(3, [f0/1.35*2, min(f0*1.35*2, 0.99)], btype="band")
        env = np.abs(signal.hilbert(signal.filtfilt(b, bb, a)))
        env = signal.filtfilt(*signal.butter(2, 0.05), env)
        env = np.maximum(env, 0); env /= env.max()
        voice = np.sin(2*np.pi*hz*t) + 0.3*np.sin(4*np.pi*hz*t)
        mix += np.interp(day_axis, np.arange(n), env) * voice / np.sqrt(hz/440)
    rv = np.array([vol_series[max(0,i-21):i+1].std() for i in range(n)])
    rv = np.nan_to_num(rv / max(rv.max(), 1e-12))
    mix += 0.6 * np.interp(day_axis, np.arange(n), rv) * np.sin(2*np.pi*211.75*t)
    fade = np.minimum(1, np.minimum(t/0.5, (t[-1]-t)/1.0))
    audio = (mix*fade / np.abs(mix*fade).max() * 0.85 * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(audio.tobytes())
    print(f"wrote {path} ({n_audio/SR:.0f}s, {n} simulated days)")

NSIM = 2016  # 8 trading years

# --- 1. Wiener-Khinchin sampler: phase-randomized surrogate of |r| ---
a = np.abs(r)
A = np.fft.rfft(a - a.mean())
# resample amplitude spectrum onto the new length via interpolation
f_old = np.fft.rfftfreq(N); f_new = np.fft.rfftfreq(NSIM)
amp = np.interp(f_new, f_old, np.abs(A)) * np.sqrt(NSIM/N)
phases = rng.uniform(0, 2*np.pi, len(f_new))
phases[0] = 0
surr = np.fft.irfft(amp * np.exp(1j*phases), n=NSIM) + a.mean()
surr = np.maximum(surr, 0)  # |r| is nonnegative; surrogate is Gaussian so clip
print(f"surrogate: mean {surr.mean():.5f} vs real {a.mean():.5f}, std {surr.std():.5f} vs {a.std():.5f}")
sonify(surr, r"C:\Users\vito.ciciretti\Downloads\nvda_melody_sampled_wk.wav")

# --- 2. GJR-GARCH(1,1,1) skew-t: fit and simulate ---
from arch import arch_model
am = arch_model(100*r, vol="GARCH", p=1, o=1, q=1, dist="skewt")
res = am.fit(disp="off")
print(res.params.round(4).to_dict())
sim = am.simulate(res.params, NSIM)
r_sim = sim["data"].values / 100
print(f"sim ann.vol {r_sim.std()*np.sqrt(252):.1%} vs real {r.std()*np.sqrt(252):.1%}, "
      f"kurt {((r_sim-r_sim.mean())**4).mean()/r_sim.var()**2:.1f} vs {((r-r.mean())**4).mean()/r.var()**2:.1f}")
sonify(np.abs(r_sim), r"C:\Users\vito.ciciretti\Downloads\nvda_melody_sampled_garch.wav")
