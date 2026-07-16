"""Duet video: both price lines + rolling correlation panel, synced to the track."""
import numpy as np, os, subprocess, wave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg, pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
track = "09_Correlation_One"
r1 = np.load(os.path.join(SCRATCH, "r_NVDA.npy"))
r2 = np.load(os.path.join(SCRATCH, "r_AAPL.npy"))
rho = np.load(os.path.join(SCRATCH, "duet_rho.npy"))
n = min(len(r1), len(r2)); r1, r2 = r1[:n], r2[:n]
dp16 = max(2, round(n/1536)); N16 = (n//dp16)//16*16; ndays = N16*dp16
with wave.open(os.path.join(OUT, f"{track}.wav")) as w:
    dur = w.getnframes()/w.getframerate()

p1, p2 = np.exp(np.cumsum(r1[:ndays])), np.exp(np.cumsum(r2[:ndays]))
d1 = np.maximum.accumulate(np.cumsum(r1)) - np.cumsum(r1)
d2 = np.maximum.accumulate(np.cumsum(r2)) - np.cumsum(r2)
bear = ((d1 > np.quantile(d1, 0.78)) & (d2 > np.quantile(d2, 0.78)))[:ndays]
rho = rho[:ndays]
dates = pd.to_datetime(np.load(os.path.join(SCRATCH, "dates_NVDA.npy"), allow_pickle=True))[:ndays]

regions = []
b0 = None
for i in range(ndays+1):
    if i < ndays and bear[i] and b0 is None: b0 = i
    elif (i == ndays or not bear[i]) and b0 is not None: regions.append((b0, i)); b0 = None

FPS = 24; W_, H_ = 1280, 720
frames = int(dur*FPS)
BG, INK, MUT = "#0d0d0d", "#ffffff", "#898781"
BLUE, GREEN, RED = "#3987e5", "#00c853", "#e66767"

fig = plt.figure(figsize=(W_/100, H_/100), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.07, 0.30, 0.88, 0.58]); axr = fig.add_axes([0.07, 0.07, 0.88, 0.17])
for a in (ax, axr):
    a.set_facecolor(BG)
    for s in a.spines.values(): s.set_color("#2c2c2a")
    a.tick_params(colors=MUT, labelsize=9)
ax.set_yscale("log"); ax.set_xlim(0, ndays)
ax.set_ylim(min(p1.min(), p2.min())*0.85, max(p1.max(), p2.max())*1.2)
ax.set_xticks([])
axr.set_xlim(0, ndays); axr.set_ylim(-0.3, 1.05); axr.set_xticks([])
axr.axhline(0.75, color=RED, lw=0.8, ls="--", alpha=0.6)
axr.set_ylabel("corr 63d", color=MUT, fontsize=9)
ax.set_ylabel("growth of $1 (log)", color=MUT, fontsize=10)
fig.text(0.07, 0.945, "TECH(NO) US", color=INK, fontsize=17, fontweight="bold", family="monospace")
fig.text(0.24, 0.945, "Correlation One", color=BLUE, fontsize=17, family="monospace")
fig.text(0.07, 0.905, "bass: chipmaker   lead: cupertino   harmony: their correlation",
         color=MUT, fontsize=10, family="monospace")
date_txt = fig.text(0.95, 0.945, "", color=MUT, fontsize=13, ha="right", family="monospace")
state_txt = fig.text(0.95, 0.905, "", color=RED, fontsize=11, ha="right", family="monospace", fontweight="bold")

l1, = ax.plot([], [], color=BLUE, lw=1.5)
l2, = ax.plot([], [], color=GREEN, lw=1.5)
lr, = axr.plot([], [], color="#d55181", lw=1.4)
spans = [ax.axvspan(a_, b_, color=RED, alpha=0.0, lw=0) for a_, b_ in regions]
x = np.arange(ndays)

ffexe = imageio_ffmpeg.get_ffmpeg_exe()
out_mp4 = os.path.join(OUT, f"{track}.mp4")
cmd = [ffexe, "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W_}x{H_}", "-r", str(FPS), "-i", "pipe:",
       "-i", os.path.join(OUT, f"{track}.wav"), "-c:v", "libx264", "-preset", "fast", "-crf", "22",
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for fi in range(frames):
    di = min(int(fi/frames*ndays), ndays-1)
    l1.set_data(x[:di+1], p1[:di+1]); l2.set_data(x[:di+1], p2[:di+1])
    lr.set_data(x[:di+1], rho[:di+1])
    date_txt.set_text(str(dates[di].date()))
    state_txt.set_text("UNISON" if rho[di] > 0.75 else ("JOINT STRESS" if bear[di] else ""))
    for (a_, b_), sp in zip(regions, spans):
        sp.set_alpha(0.13 if di >= a_ else 0.0)
    fig.canvas.draw()
    proc.stdin.write(fig.canvas.buffer_rgba())
proc.stdin.close(); proc.wait()
print(f"wrote {out_mp4} ({frames} frames, {dur:.0f}s)")
