"""Descensus Gradientis video: julia set descending the REAL val-loss curve
(chaos -> order as the LSTM converges) | training curves revealed epoch-by-epoch.
Usage: make_video_descensus.py [--testframe out.png]
"""
import numpy as np, sys, os, subprocess, wave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\vito.ciciretti\Downloads\TechnoUS"
track = "10_Descensus_Gradientis"
TESTFRAME = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--testframe" else None

hist = np.load(os.path.join(SCRATCH, "hist_NVDA.npy"))   # (80, 2) train/val NLL
NEP = len(hist)
train, val = hist[:, 0], hist[:, 1]
vmin, v0 = val.min(), val[0]
diss = np.clip((val - vmin)/(v0 - vmin + 1e-9), 0, 1)    # 1 = epoch-0 chaos, 0 = converged

wav_path = os.path.join(OUT, f"{track}.wav")
if TESTFRAME:
    dur = 159.0
else:
    with wave.open(wav_path) as w:
        dur = w.getnframes()/w.getframerate()

# track structure (must match descensus.py): 8 intro bars + 80 epoch bars + 8 outro
BPM = 145; BAR = 4*60/BPM; INTRO = 8

FPS = 24; W_, H_ = 1280, 720
frames = int(dur*FPS)
BG, INK, MUT = "#0d0d0d", "#ffffff", "#898781"
BLUE, GREEN, RED = "#3987e5", "#00c853", "#e66767"

# ---------- Julia parameter path: c descends the val-loss curve ----------
nf = frames if not TESTFRAME else 100
t_f = np.linspace(0, dur, nf, endpoint=False)
ep_f = np.clip(t_f/BAR - INTRO, 0.0, NEP - 1e-6)          # continuous epoch position
diss_f = np.interp(ep_f, np.arange(NEP), diss)            # loss-driven chaos level
prog = np.linspace(0, 1, nf, endpoint=False)
theta = 2*np.pi*(0.10 + 0.45*prog) + 0.6*diss_f           # slow orbit, chaos twists it
rad = 0.60 + 0.22*diss_f                                  # radius shrinks with the loss
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
    return out.reshape(JN, JN)

# ---------- figure layout ----------
fig = plt.figure(figsize=(W_/100, H_/100), dpi=100)
fig.patch.set_facecolor(BG)

fig.text(0.03, 0.955, "TECH(NO) US", color=INK, fontsize=16, fontweight="bold", family="monospace")
fig.text(0.205, 0.955, "Descensus Gradientis", color=BLUE, fontsize=16, family="monospace")
fig.text(0.03, 0.915, "left: julia set descending the val-loss curve | right: the composer, learning",
         color=MUT, fontsize=9.5, family="monospace")

# left: julia panel (square)
JW = 0.44
axj = fig.add_axes([0.03, 0.09, JW, JW*W_/H_])
axj.set_facecolor(BG); axj.set_xticks([]); axj.set_yticks([])
for s in axj.spines.values(): s.set_color("#2c2c2a")
im = axj.imshow(np.zeros((JN, JN)), cmap="magma", vmin=0, vmax=JIT*0.55,
                extent=[-1.6, 1.6, -1.6, 1.6], origin="lower", interpolation="bilinear")
fig.text(0.03, 0.045, "gradient descent as fractal   |c| <- val NLL   chaos -> order",
         color=MUT, fontsize=9, family="monospace")

# right: training curves revealed epoch by epoch
axt = fig.add_axes([0.545, 0.115, 0.435, 0.72])
axt.set_facecolor(BG)
for s in axt.spines.values(): s.set_color("#2c2c2a")
axt.tick_params(colors=MUT, labelsize=8)
axt.set_xlim(0, NEP)
lo, hi = hist.min(), hist.max()
pad = 0.06*(hi - lo + 1e-9)
axt.set_ylim(lo - pad, hi + pad)
axt.set_xlabel("epoch", color=MUT, fontsize=8)
axt.set_ylabel("NLL", color=MUT, fontsize=9)
for lr_ep in (30, 60):
    axt.axvline(lr_ep, color=MUT, lw=0.8, ls=":", alpha=0.5)
    axt.text(lr_ep, hi + pad*0.2, "lr/10", color=MUT, fontsize=7, ha="center", family="monospace")
tr_line, = axt.plot([], [], color=MUT, lw=1.2, label="train")
va_line, = axt.plot([], [], color=BLUE, lw=1.5, label="val")
va_dot, = axt.plot([], [], "o", color=INK, ms=5)
best_line = axt.axhline(v0, color=GREEN, lw=0.9, ls="--", alpha=0.7)
leg = axt.legend(loc="upper left", fontsize=8, frameon=False)
for t in leg.get_texts(): t.set_color(MUT)

ep_big = fig.text(0.98, 0.955, "", color=INK, fontsize=13, ha="right", family="monospace")
diss_txt = fig.text(0.98, 0.915, "", color=RED, fontsize=10, ha="right", family="monospace",
                    fontweight="bold")
phase_txt = axt.text(0.97, 0.90, "", color=MUT, fontsize=9, ha="right", family="monospace",
                     transform=axt.transAxes)
fig.text(0.545, 0.045, "hist_NVDA.npy   LSTM p(r_t+1 | F_t)   80 epochs, the choir sings the loss",
         color=MUT, fontsize=9, family="monospace")

ep_x = np.arange(1, NEP+1)

def render(fi, nframes):
    im.set_data(julia_frame(c_path[fi]))
    t = fi/nframes*dur
    e_cont = t/BAR - INTRO
    if e_cont < 0:
        phase = "INTRO"
        ep = 0
    elif e_cont >= NEP:
        phase = "CONVERGED"
        ep = NEP
    else:
        phase = ""
        ep = int(e_cont) + 1
    ep_show = max(ep, 1) if e_cont >= 0 else 0
    if ep_show:
        tr_line.set_data(ep_x[:ep_show], train[:ep_show])
        va_line.set_data(ep_x[:ep_show], val[:ep_show])
        va_dot.set_data([ep_show], [val[ep_show-1]])
        best_line.set_ydata([val[:ep_show].min()]*2)
        d = diss[ep_show-1]
        diss_txt.set_text(f"dissonance {d:.2f}")
        diss_txt.set_color(RED if d > 0.66 else ("#e6a817" if d > 0.33 else GREEN))
    ep_big.set_text(f"epoch {min(ep, NEP)}/{NEP}" if e_cont >= 0 else "epoch 0/80")
    phase_txt.set_text(phase)

if TESTFRAME:
    render(30, 100)
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
