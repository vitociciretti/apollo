import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal.windows import dpss

df = pd.read_excel(r"C:\Users\vito.ciciretti\Downloads\prices.xlsx")
df = df.dropna().sort_values("Dates").reset_index(drop=True)
p = df["NVDA"].values
r = np.diff(np.log(p))
n = len(r)
print(f"n returns = {n}, span {df.Dates.iloc[0].date()} .. {df.Dates.iloc[-1].date()}")
print(f"ann. vol = {r.std()*np.sqrt(252):.1%}, mean daily = {r.mean():.5f}")

def analyze(x, name):
    x = x - x.mean()
    N = len(x)
    out = {}

    # --- raw periodogram + Fisher g-test ---
    f, Pxx = signal.periodogram(x, fs=1.0, detrend=False)
    f, Pxx = f[1:], Pxx[1:]  # drop DC
    m = len(Pxx)
    g = Pxx.max() / Pxx.sum()
    # Fisher's exact p-value (first term dominates)
    k = np.arange(1, min(m, int(1/g)) + 1)
    from scipy.special import gammaln
    logterms = gammaln(m+1) - gammaln(k+1) - gammaln(m-k+1) + (m-1)*np.log1p(-k*g)
    pval = np.sum((-1)**(k-1) * np.exp(logterms))
    kmax = Pxx.argmax()
    print(f"\n=== {name} ===")
    print(f"Fisher g-test: max peak at period {1/f[kmax]:.2f} days, g={g:.5f}, p={pval:.4f}")

    # false-alarm threshold: peak significant at 5% if g > g_crit
    # approx g_crit = 1 - (0.05/m)**(1/(m-1))
    g_crit = 1 - (0.05/m)**(1/(m-1))
    nsig = (Pxx/Pxx.sum() > g_crit).sum()
    print(f"peaks above 5% white-noise false-alarm level: {nsig}")

    # --- Welch ---
    fw, Pw = signal.welch(x, fs=1.0, nperseg=512, noverlap=384, detrend=False)
    fw, Pw = fw[1:], Pw[1:]

    # --- Wiener-Khinchin: FFT of Tukey-windowed sample ACF ---
    L = 512  # max lag
    ac = np.correlate(x, x, mode="full")[N-1:N+L] / N
    lagwin = 0.5*(1+np.cos(np.pi*np.arange(L+1)/L))  # Tukey-Hanning lag window
    acw = ac * lagwin
    nfft = 4096
    Pwk = np.fft.rfft(np.concatenate([acw, acw[1:][::-1]]), n=nfft).real
    fwk = np.fft.rfftfreq(nfft, d=1.0)
    Pwk = np.maximum(Pwk, 0)

    # --- multitaper (Thomson), NW=4, K=7 ---
    NW, K = 4, 7
    tapers = dpss(N, NW, K)
    mt = np.array([np.abs(np.fft.rfft(x*t))**2 for t in tapers]).mean(axis=0)
    fmt = np.fft.rfftfreq(N, d=1.0)
    fmt, mt = fmt[1:], mt[1:]

    # top peaks on multitaper (smooth, good line resolution)
    from scipy.signal import find_peaks
    pk, props = find_peaks(mt, height=0)
    order = pk[np.argsort(mt[pk])[::-1][:5]]
    print("top-5 multitaper peaks (period in days, share of total power):")
    for i in order:
        print(f"  T = {1/fmt[i]:8.2f} d   f = {fmt[i]:.4f}/d   share = {mt[i]/mt.sum():.4%}")

    # spectral slope at low freq (long memory): log P ~ -2d' log f
    lo = (fmt > 0.001) & (fmt < 0.05)
    slope = np.polyfit(np.log(fmt[lo]), np.log(mt[lo]), 1)[0]
    print(f"low-freq log-log spectral slope: {slope:.3f}  (white noise ~ 0; long memory < 0)")
    return dict(f=fmt, P=mt, fw=fw, Pw=Pw, fwk=fwk, Pwk=Pwk, pval=pval)

res_r  = analyze(r, "returns r_t")
res_a  = analyze(np.abs(r), "|r_t| (volatility proxy)")
res_s  = analyze(r**2, "r_t^2")

# variance ratio check: how flat is the return spectrum really
f, Pxx = signal.periodogram(r - r.mean(), fs=1.0)
f, Pxx = f[1:], Pxx[1:]
half = Pxx[f <= 0.25].sum() / Pxx.sum()
print(f"\nshare of return variance below f=0.25 (periods > 4 days): {half:.1%} (white noise -> 50%)")

np.save(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\r.npy", r)
