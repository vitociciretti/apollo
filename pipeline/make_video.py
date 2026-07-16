"""Music video: price evolution synced to the track. Usage: make_video.py TICKER TRACKNAME"""
import numpy as np, sys, os, subprocess, wave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg, pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
ticker, track = sys.argv[1], sys.argv[2]
rs = np.load(os.path.join(SCRATCH, f"r_{ticker}.npy"))
wav_path = os.path.join(OUT, f"{track}.wav")

# same composition math as techno_album.compose
n = len(rs)
dp16 = max(2, round(n/1536))
N16 = (n//dp16)//16*16
ndays = N16*dp16
with wave.open(wav_path) as w:
    dur = w.getnframes()/w.getframerate()

price = np.exp(np.cumsum(rs[:ndays]))
dd_full = np.maximum.accumulate(np.cumsum(rs)) - np.cumsum(rs)
bear = (dd_full > np.quantile(dd_full, 0.78))[:ndays]
vol21 = np.nan_to_num(pd.Series(rs[:ndays]).rolling(21, min_periods=1).std().values)*np.sqrt(252)
sigma = rs.std()
crash_days = [i for i in range(ndays) if abs(rs[i]) > 4*sigma]

# dates for readout
import yfinance as yf
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
BLUE, GREEN, RED = "#3987e5", "#008300", "#e66767"

fig = plt.figure(figsize=(W_/100, H_/100), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.07, 0.22, 0.88, 0.66]); axv = fig.add_axes([0.07, 0.07, 0.88, 0.11])
for a in (ax, axv):
    a.set_facecolor(BG)
    for s in a.spines.values(): s.set_color("#2c2c2a")
    a.tick_params(colors=MUT, labelsize=9)
ax.set_yscale("log"); ax.set_xlim(0, ndays); ax.set_ylim(price.min()*0.85, price.max()*1.2)
ax.set_xticks([]); axv.set_xlim(0, ndays); axv.set_ylim(0, vol21.max()*1.1); axv.set_xticks([]); axv.set_yticks([])
ax.set_ylabel("growth of $1 (log)", color=MUT, fontsize=10)
axv.set_ylabel("vol", color=MUT, fontsize=9)
fig.text(0.07, 0.945, "TECH(NO) US", color=INK, fontsize=17, fontweight="bold", family="monospace")
fig.text(0.24, 0.945, track.replace("_", " ").lstrip("0123456789 "), color=BLUE, fontsize=17, family="monospace")
date_txt = fig.text(0.95, 0.945, "", color=MUT, fontsize=13, ha="right", family="monospace")
state_txt = fig.text(0.95, 0.90, "", color=RED, fontsize=11, ha="right", family="monospace", fontweight="bold")

line, = ax.plot([], [], color=BLUE, lw=1.6)
dot, = ax.plot([], [], "o", color=INK, ms=6)
vline, = axv.plot([], [], color=GREEN, lw=1.2)
spans = [ax.axvspan(a_, b_, color=RED, alpha=0.0, lw=0) for a_, b_ in regions]
crash_sc = ax.scatter([], [], color=RED, s=30, zorder=5)
x = np.arange(ndays)

ffexe = imageio_ffmpeg.get_ffmpeg_exe()
out_mp4 = os.path.join(OUT, f"{track}.mp4")
cmd = [ffexe, "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W_}x{H_}", "-r", str(FPS), "-i", "pipe:",
       "-i", wav_path, "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

shown_crashes = []
for fi in range(frames):
    di = min(int(fi/frames*ndays), ndays-1)
    line.set_data(x[:di+1], price[:di+1])
    dot.set_data([di], [price[di]])
    vline.set_data(x[:di+1], vol21[:di+1])
    date_txt.set_text(str(dates[di].date()))
    inbear = bear[di]
    state_txt.set_text("BREAKDOWN" if inbear else "")
    for (a_, b_), sp in zip(regions, spans):
        sp.set_alpha(0.13 if di >= a_ else 0.0)
    while shown_crashes != crash_days[:len(shown_crashes)+1] and len(shown_crashes) < len(crash_days) and crash_days[len(shown_crashes)] <= di:
        shown_crashes.append(crash_days[len(shown_crashes)])
    if shown_crashes:
        crash_sc.set_offsets(np.c_[shown_crashes, price[shown_crashes]])
    fig.canvas.draw()
    proc.stdin.write(fig.canvas.buffer_rgba())
proc.stdin.close(); proc.wait()
print(f"wrote {out_mp4} ({frames} frames, {dur:.0f}s)")
