"""Triptych music video: Julia set + price panel + LSTM training panel, synced to the track.
Usage: make_video3.py TICKER TRACKNAME [--testframe out.png]
"""
import numpy as np, sys, os, subprocess, wave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg, pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
ticker, track = sys.argv[1], sys.argv[2]
TESTFRAME = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--testframe" else None

rs = np.load(os.path.join(SCRATCH, f"r_{ticker}.npy"))
hist = np.load(os.path.join(SCRATCH, f"hist_{ticker}.npy"))  # (80, 2) train/val NLL
NEP = len(hist)
wav_path = os.path.join(OUT, f"{track}.wav")

# same composition math as techno_album.compose
n = len(rs)
dp16 = max(2, round(n/1536))
N16 = (n//dp16)//16*16
ndays = N16*dp16
if TESTFRAME:
    dur = 200.0
else:
    with wave.open(wav_path) as w:
        dur = w.getnframes()/w.getframerate()

price = np.exp(np.cumsum(rs[:ndays]))
dd_full = np.maximum.accumulate(np.cumsum(rs)) - np.cumsum(rs)
bear = (dd_full > np.quantile(dd_full, 0.78))[:ndays]
vol21 = np.nan_to_num(pd.Series(rs[:ndays]).rolling(21, min_periods=1).std().values)*np.sqrt(252)
sigma = rs.std()
crash_days = [i for i in range(ndays) if abs(rs[i]) > 4*sigma]

dates = pd.to_datetime(np.load(os.path.join(SCRATCH, f"dates_{ticker}.npy"), allow_pickle=True))[:ndays] if \
    os.path.exists(os.path.join(SCRATCH, f"dates_{ticker}.npy")) else pd.bdate_range("2010-01-05", periods=ndays)

# bear regions
regions = []
b0 = None
for i in range(ndays+1):
    if i < ndays and bear[i] and b0 is None: b0 = i
    elif (i == ndays or not bear[i]) and b0 is not None: regions.append((b0, i)); b0 = None

FPS = 24; W_, H_ = 1280, 720
frames = int(dur*FPS)
BG, INK, MUT = "#0d0d0d", "#ffffff", "#898781"
BLUE, GREEN, RED = "#3987e5", "#00c853", "#e66767"

# ---------- Julia parameter path (precompute c per frame) ----------
lp = np.cumsum(rs[:ndays])                       # log price
ma252 = pd.Series(lp).rolling(252, min_periods=20).mean()
sd252 = pd.Series(lp).rolling(252, min_periods=20).std()
z252 = np.nan_to_num(((pd.Series(lp) - ma252)/sd252).rolling(21, min_periods=1).mean().values)
volpct = pd.Series(vol21).rank(pct=True).rolling(21, min_periods=1).mean().values

fdays = np.linspace(0, ndays-1, frames if not TESTFRAME else 100)
prog = np.linspace(0, 1, frames if not TESTFRAME else 100, endpoint=False)
zf = np.interp(fdays, np.arange(ndays), z252)
vf = np.interp(fdays, np.arange(ndays), volpct)
theta = 2*np.pi*(0.12 + 0.5*prog) + 0.7*np.tanh(zf/2)
rad = 0.755 + 0.035*vf
c_path = (rad*np.exp(1j*theta)).astype(np.complex64)

JN, JIT = 400, 60
gx = np.linspace(-1.6, 1.6, JN, dtype=np.float32)
Z0 = (gx[None, :] + 1j*gx[:, None]).astype(np.complex64).ravel()

def julia_frame(c):
    z = Z0.copy()
    live = np.arange(z.size)
    out = np.zeros(z.size, np.float32)
    for i in range(JIT):
        z = z*z + c
        a2 = z.real*z.real + z.imag*z.imag
        esc = a2 > 16.0
        if esc.any():
            out[live[esc]] = i + 1 - np.log2(0.5*np.log(a2[esc]))
            keep = ~esc
            z = z[keep]; live = live[keep]
            if live.size == 0: break
    # interior stays 0 (dark) so the boundary filaments glow
    return out.reshape(JN, JN)

# ---------- figure layout ----------
fig = plt.figure(figsize=(W_/100, H_/100), dpi=100)
fig.patch.set_facecolor(BG)

# header
fig.text(0.03, 0.955, "TECH(NO) US", color=INK, fontsize=16, fontweight="bold", family="monospace")
fig.text(0.205, 0.955, track.replace("_", " ").lstrip("0123456789 "), color=BLUE, fontsize=16, family="monospace")
fig.text(0.03, 0.915, "left: fractal of the moment | right: the data and the model",
         color=MUT, fontsize=9.5, family="monospace")

# left: julia panel (square)
JW = 0.44
axj = fig.add_axes([0.03, 0.09, JW, JW*W_/H_])
axj.set_facecolor(BG); axj.set_xticks([]); axj.set_yticks([])
for s in axj.spines.values(): s.set_color("#2c2c2a")
im = axj.imshow(np.zeros((JN, JN)), cmap="magma", vmin=0, vmax=JIT*0.55,
                extent=[-1.6, 1.6, -1.6, 1.6], origin="lower", interpolation="bilinear")
fig.text(0.03, 0.045, "the market as julia set   c(t) <- vol, trend",
         color=MUT, fontsize=9, family="monospace")

# right top: price + vol strip
ax = fig.add_axes([0.545, 0.585, 0.435, 0.265])
axv = fig.add_axes([0.545, 0.50, 0.435, 0.06])
# right bottom: training panel
axt = fig.add_axes([0.545, 0.115, 0.435, 0.295])
for a in (ax, axv, axt):
    a.set_facecolor(BG)
    for s in a.spines.values(): s.set_color("#2c2c2a")
    a.tick_params(colors=MUT, labelsize=8)
ax.set_yscale("log"); ax.set_xlim(0, ndays); ax.set_ylim(price.min()*0.85, price.max()*1.2)
ax.set_xticks([])
axv.set_xlim(0, ndays); axv.set_ylim(0, vol21.max()*1.1); axv.set_xticks([]); axv.set_yticks([])
ax.set_ylabel("growth of $1 (log)", color=MUT, fontsize=9)
axv.set_ylabel("vol", color=MUT, fontsize=8)
date_txt = fig.text(0.98, 0.955, "", color=MUT, fontsize=12, ha="right", family="monospace")
state_txt = fig.text(0.98, 0.915, "", color=RED, fontsize=10, ha="right", family="monospace", fontweight="bold")

line, = ax.plot([], [], color=BLUE, lw=1.4)
dot, = ax.plot([], [], "o", color=INK, ms=5)
vline, = axv.plot([], [], color=GREEN, lw=1.1)
spans = [ax.axvspan(a_, b_, color=RED, alpha=0.0, lw=0) for a_, b_ in regions]
crash_sc = ax.scatter([], [], color=RED, s=22, zorder=5)
x = np.arange(ndays)

# training panel statics
axt.set_xlim(0, NEP)
lo, hi = hist.min(), hist.max()
pad = 0.06*(hi - lo + 1e-9)
axt.set_ylim(lo - pad, hi + pad)
axt.set_xlabel("epoch", color=MUT, fontsize=8)
axt.set_ylabel("NLL", color=MUT, fontsize=9)
tr_line, = axt.plot([], [], color=MUT, lw=1.2, label="train")
va_line, = axt.plot([], [], color=BLUE, lw=1.5, label="val")
va_dot, = axt.plot([], [], "o", color=INK, ms=5)
ep_txt = axt.text(0.97, 0.90, "", color=INK, fontsize=9, ha="right", family="monospace",
                  transform=axt.transAxes)
leg = axt.legend(loc="upper left", fontsize=8, frameon=False)
for t in leg.get_texts(): t.set_color(MUT)
fig.text(0.545, 0.045, "the composer learning   LSTM p(r_t+1 | F_t)",
         color=MUT, fontsize=9, family="monospace")

ep_x = np.arange(1, NEP+1)

def render(fi, nframes):
    p = fi/nframes
    di = min(int(p*ndays), ndays-1)
    im.set_data(julia_frame(c_path[fi]))
    line.set_data(x[:di+1], price[:di+1])
    dot.set_data([di], [price[di]])
    vline.set_data(x[:di+1], vol21[:di+1])
    date_txt.set_text(str(dates[di].date()))
    state_txt.set_text("BREAKDOWN" if bear[di] else "")
    for (a_, b_), sp in zip(regions, spans):
        sp.set_alpha(0.13 if di >= a_ else 0.0)
    shown = [d for d in crash_days if d <= di]
    if shown:
        crash_sc.set_offsets(np.c_[shown, price[shown]])
    ep = max(1, min(NEP, int(np.floor(p*NEP)) + 1))
    tr_line.set_data(ep_x[:ep], hist[:ep, 0])
    va_line.set_data(ep_x[:ep], hist[:ep, 1])
    va_dot.set_data([ep], [hist[ep-1, 1]])
    ep_txt.set_text(f"epoch {ep}/{NEP}  val NLL {hist[ep-1,1]:.4f}")

if TESTFRAME:
    render(62, 100)
    fig.savefig(TESTFRAME, dpi=100, facecolor=BG)
    print(f"wrote {TESTFRAME}")
    sys.exit(0)

ffexe = imageio_ffmpeg.get_ffmpeg_exe()
out_mp4 = os.path.join(OUT, f"{track}.mp4")
cmd = [ffexe, "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W_}x{H_}", "-r", str(FPS), "-i", "pipe:",
       "-i", wav_path, "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for fi in range(frames):
    render(fi, frames)
    fig.canvas.draw()
    proc.stdin.write(fig.canvas.buffer_rgba())
proc.stdin.close(); proc.wait()
print(f"wrote {out_mp4} ({frames} frames, {dur:.0f}s)")
