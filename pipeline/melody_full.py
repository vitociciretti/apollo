import numpy as np, wave
from scipy import signal

r = np.load(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\r.npy")
N = len(r)
a = np.abs(r) - np.abs(r).mean()

# the six peaks (period days -> Hz) from the spectral analysis
peaks = [(519.5, 423.5), (319.7, 688.2), (207.8, 1058.7),
         (153.9, 1429.3), (125.9, 1746.9), (63.9, 3440.8)]

sr = 16000
SEC_PER_DAY = 66.0 / N          # 16.5y -> 66 s
n_audio = int(N * SEC_PER_DAY * sr)
t = np.arange(n_audio) / sr
day_axis = np.linspace(0, N - 1, n_audio)

mix = np.zeros(n_audio)
for T, hz in peaks:
    f0 = 1.0 / T
    # narrow bandpass around the cycle frequency (octave-ish band, min width guard)
    lo, hi = f0 / 1.35, f0 * 1.35
    b, bb = signal.butter(3, [lo * 2, min(hi * 2, 0.99)], btype="band")  # *2: Wn in Nyquist units, fs=1/day -> Nyq=0.5
    band = signal.filtfilt(b, bb, a)
    env = np.abs(signal.hilbert(band))
    env = signal.filtfilt(*signal.butter(2, 0.05), env)   # smooth the envelope
    env = np.maximum(env, 0)
    env = env / env.max()
    env_a = np.interp(day_axis, np.arange(N), env)        # resample to audio timeline
    voice = np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(4 * np.pi * hz * t)
    # louder cycles weighted a bit by 1/sqrt(hz) so high notes don't pierce
    mix += env_a * voice / np.sqrt(hz / 440)

# soft drone following 21-day rolling vol, one octave under the lowest note
rv = np.array([r[max(0, i - 21):i + 1].std() for i in range(N)])
rv = rv / rv.max()
rv_a = np.interp(day_axis, np.arange(N), rv)
mix += 0.6 * rv_a * np.sin(2 * np.pi * 211.75 * t)

# fade in/out, normalize
fade = np.minimum(1, np.minimum(t / 0.5, (t[-1] - t) / 1.0))
mix *= fade
audio = (mix / np.abs(mix).max() * 0.85 * 32767).astype(np.int16)

path = r"C:\Users\vito.ciciretti\Downloads\nvda_melody_full.wav"
with wave.open(path, "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(audio.tobytes())
print(f"wrote {path}: {n_audio/sr:.1f}s, {N} days, {1/SEC_PER_DAY:.0f} trading days per second")

# landmark timestamps for the listener
import pandas as pd
df = pd.read_excel(r"C:\Users\vito.ciciretti\Downloads\prices.xlsx").dropna().sort_values("Dates")
dates = df["Dates"].iloc[1:].reset_index(drop=True)  # aligned with returns
for label, d in [("Vol-mageddon Feb 2018", "2018-02-05"), ("COVID crash", "2020-03-16"),
                 ("2022 bear", "2022-09-01"), ("DeepSeek Jan 2025", "2025-01-27")]:
    idx = (dates - pd.Timestamp(d)).abs().idxmin()
    print(f"  {label}: at ~{idx * SEC_PER_DAY:.0f}s")
