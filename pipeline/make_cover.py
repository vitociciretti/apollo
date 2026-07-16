"""Album cover: 3000x3000, the eight growth-of-$1 curves as light streaks."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SCRATCH = os.path.dirname(os.path.abspath(__file__))
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "TSLA"]
NEON = ["#3987e5", "#00c853", "#d55181", "#c98500", "#199e70", "#d95926", "#9085e9", "#e66767"]

fig = plt.figure(figsize=(10, 10), dpi=300)
fig.patch.set_facecolor("#0a0a0c")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#0a0a0c"); ax.axis("off")

for t, c in zip(TICKERS, NEON):
    r = np.load(os.path.join(SCRATCH, f"r_{t}.npy"))
    x = np.linspace(0, 1, len(r))
    y = np.cumsum(r)
    y = (y - y.min())/(y.max() - y.min())          # normalize each to [0,1]
    for lw, al in [(6, 0.05), (3, 0.12), (1.1, 0.9)]:   # glow: wide faint underlayers
        ax.plot(x, 0.16 + 0.62*y, color=c, lw=lw, alpha=al, solid_capstyle="round")

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
try:
    fp = font_manager.FontProperties(family="Arial Black", weight="black")
    font_manager.findfont(fp, fallback_to_default=False)
except Exception:
    fp = font_manager.FontProperties(family="sans-serif", weight="bold")
ax.text(0.5, 0.925, "TECH(NO)", fontproperties=fp, fontsize=64, color="#ffffff",
        ha="center", va="center")
ax.text(0.5, 0.845, "US", fontproperties=fp, fontsize=40, color="#3987e5",
        ha="center", va="center")
ax.text(0.5, 0.065, "EIGHT STOCKS  ·  SIXTEEN YEARS  ·  NO COMPOSER",
        family="monospace", fontsize=13, color="#898781", ha="center", va="center")

out = r"C:\Users\vito.ciciretti\Downloads\TechnoUS\cover_3000.png"
fig.savefig(out, dpi=300, facecolor="#0a0a0c")
print("wrote", out)
from PIL import Image
im = Image.open(out)
print("size:", im.size, "mode:", im.mode)
