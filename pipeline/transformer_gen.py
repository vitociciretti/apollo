"""Joint multivariate generator: decoder-only transformer over 8 tickers.

Compare against the univariate LSTM (train_rnn.py). One model sees all 8
tickers jointly (16 input dims), heads emit per-ticker Student-t params,
and sampling bootstraps whole 8-dim standardized-residual vectors from the
in-sample empirical rows -- preserving crisis correlation exactly.

Outputs (all in pipeline/):
  r_TF_{TICKER}.npy        sampled daily log-returns, 4032 days each
  tf_weights.pt            best-val model weights
  hist_TF.npy              per-epoch [train_nll, val_nll]
  transformer_scorecard.md scorecard, also printed as markdown
"""
import os, time
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(12)
HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "TSLA"]
K = len(TICKERS)

# ---------------- data: align on common dates (tail-trim to min length) ----
rs, ds = {}, {}
for t in TICKERS:
    rs[t] = np.load(os.path.join(HERE, f"r_{t}.npy")).astype(np.float64)
    ds[t] = np.load(os.path.join(HERE, f"dates_{t}.npy"))
n = min(len(rs[t]) for t in TICKERS)
ref = ds[TICKERS[0]][-n:]
for t in TICKERS:
    assert (ds[t][-n:] == ref).all(), f"date grid mismatch for {t}"
    rs[t] = rs[t][-n:]
print(f"aligned {K} tickers on {n} common days: {str(ref[0])[:10]} .. {str(ref[-1])[:10]}")

R = np.stack([rs[t] for t in TICKERS], axis=1)            # (n, 8) raw log-returns
S = R.std(axis=0)                                         # per-ticker scale
Y = (R / S).astype(np.float32)                            # standardized returns (n, 8)
X = np.concatenate([Y, np.log(np.abs(Y) + 1e-3)], axis=1).astype(np.float32)  # (n, 16)

# ---------------- model ----------------------------------------------------
L, STRIDE, BURN_W = 256, 16, 32
D_MODEL, N_HEAD, N_LAYER = 64, 4, 4

class TFGen(nn.Module):
    """Decoder-only transformer -> per-ticker Student-t(mu, sigma, nu)."""
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(2 * K, D_MODEL)
        pe = torch.zeros(L, D_MODEL)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-np.log(10000.0) / D_MODEL))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=4 * D_MODEL,
            dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYER)
        self.head = nn.Linear(D_MODEL, 2 * K)             # per ticker: mu, log_sigma
        self.lognu = nn.Parameter(torch.full((K,), 1.5))  # nu = 2.05 + softplus, per ticker

    def forward(self, xb):                                # xb: (B, T, 16), T <= L
        T = xb.shape[1]
        h = self.inp(xb) + self.pe[:T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=xb.device), 1)
        h = self.enc(h, mask=mask)
        p = self.head(h)                                  # (B, T, 16)
        mu = 0.05 * torch.tanh(p[..., :K])
        sigma = torch.exp(p[..., K:].clamp(-4, 2))
        nu = 2.05 + nn.functional.softplus(self.lognu)    # (K,)
        return mu, sigma, nu

def nll(mu, sigma, nu, target):
    """Student-t NLL, elementwise; nu broadcasts over the last (ticker) dim."""
    z = (target - mu) / sigma
    return -(torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
             - 0.5 * torch.log(np.pi * nu) - torch.log(sigma)
             - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))

# ---------------- windows + train ------------------------------------------
Xw, Yw = [], []
for i in range(0, n - L - 1, STRIDE):
    Xw.append(X[i:i + L]); Yw.append(Y[i + 1:i + L + 1])
Xw = torch.tensor(np.array(Xw)); Yw = torch.tensor(np.array(Yw))
ntr = int(0.85 * len(Xw))
perm = torch.randperm(len(Xw))
tr, va = perm[:ntr], perm[ntr:]
print(f"windows: {len(Xw)} (train {ntr}, val {len(va)})")

model = TFGen()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, 30, 0.3)
EPOCHS = 80
best, best_state, hist = 1e9, None, []
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); ep_loss, nb = 0.0, 0
    for b in torch.split(tr[torch.randperm(len(tr))], 32):
        mu, sg, nu, = model(Xw[b])
        loss = nll(mu, sg, nu, Yw[b])[:, BURN_W:].sum(-1).mean()  # sum over tickers
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); ep_loss += loss.item(); nb += 1
    sched.step()
    model.eval()
    with torch.no_grad():
        mu, sg, nu = model(Xw[va])
        vloss = nll(mu, sg, nu, Yw[va])[:, BURN_W:].sum(-1).mean().item()
    hist.append([ep_loss / nb, vloss])
    if np.isfinite(vloss) and vloss < best:
        best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
    if ep % 10 == 0 or ep == EPOCHS - 1:
        print(f"ep {ep:3d}  train {ep_loss/nb:8.4f}  val {vloss:8.4f}  "
              f"({time.time()-t0:.0f}s)")
model.load_state_dict(best_state)
np.save(os.path.join(HERE, "hist_TF.npy"), np.array(hist))
torch.save(best_state, os.path.join(HERE, "tf_weights.pt"))
print(f"best val NLL {best:.4f}  ({time.time()-t0:.0f}s)")

# ---------------- in-sample standardized residual rows ----------------------
model.eval()
Zrows = []
with torch.no_grad():
    for i in range(0, n - L - 1, L - BURN_W):             # cover every day once past burn-in
        xb = torch.tensor(X[i:i + L]).unsqueeze(0)
        yb = torch.tensor(Y[i + 1:i + L + 1])
        mu, sg, _ = model(xb)
        z = ((yb - mu[0]) / sg[0]).numpy()[BURN_W:]
        Zrows.append(z)
Z = np.concatenate(Zrows, axis=0)                         # (~n, 8) empirical residual rows
Z = Z[np.isfinite(Z).all(axis=1)]
Z = np.clip(Z, -8, 8)
print(f"residual bank: {Z.shape[0]} rows, corr(z_NVDA, z_AAPL) = "
      f"{np.corrcoef(Z[:,0], Z[:,1])[0,1]:.3f}")

# ---------------- autoregressive sampling (bootstrap residual vectors) ------
NSIM, WARM = 4032, 252
rng = np.random.default_rng(0)
ctx = X[-WARM:].copy()                                    # rolling context, (<=256, 16)
out = np.empty((NSIM, K))
t0 = time.time()
with torch.no_grad():
    for step in range(NSIM):
        xb = torch.tensor(ctx[-L:], dtype=torch.float32).unsqueeze(0)
        mu, sg, _ = model(xb)
        m = mu[0, -1].numpy(); s = sg[0, -1].numpy()
        z = Z[rng.integers(len(Z))]                       # joint 8-dim residual row
        y = np.clip(m + s * z, -8, 8)                     # standardized units
        out[step] = y
        ctx = np.concatenate([ctx, np.concatenate(
            [y, np.log(np.abs(y) + 1e-3)]).astype(np.float32)[None]], axis=0)[-L:]
        if (step + 1) % 1000 == 0:
            print(f"  sampled {step+1}/{NSIM} ({time.time()-t0:.0f}s)")
assert np.isfinite(out).all(), "NaN/inf in samples"
Rsim = out * S                                            # back to raw log-return units
for j, t in enumerate(TICKERS):
    np.save(os.path.join(HERE, f"r_TF_{t}.npy"), Rsim[:, j])
print(f"saved r_TF_*.npy ({NSIM} days each, {time.time()-t0:.0f}s sampling)")

# ---------------- scorecard -------------------------------------------------
def acf(v, lag):
    v = v - v.mean(); return float((v[:-lag] * v[lag:]).mean() / v.var())

def gph(v):
    from scipy.signal import periodogram
    a = np.abs(v) - np.abs(v).mean()
    f, P = periodogram(a); f, P = f[1:], P[1:]
    m = int(len(v) ** 0.6); lam = 2 * np.pi * f[:m]
    Xg = np.log(4 * np.sin(lam / 2) ** 2)
    return float(np.polyfit(-Xg, np.log(P[:m]), 1)[0])

def kurt(v):
    return float(((v - v.mean()) ** 4).mean() / v.var() ** 2)

lines = []
lines.append("# Transformer joint generator - scorecard\n")
lines.append(f"Decoder-only transformer, {N_LAYER} layers, d_model {D_MODEL}, "
             f"{N_HEAD} heads, context {L}. Joint 8-ticker Student-t heads, "
             f"cross-sectional dependence via bootstrap of in-sample 8-dim "
             f"standardized-residual rows. Trained on {n} common days "
             f"({str(ref[0])[:10]} .. {str(ref[-1])[:10]}), best val NLL "
             f"{best:.4f} (sum over 8 tickers). Sample length {NSIM} days.\n")
lines.append("## Per-ticker: real vs transformer sample\n")
lines.append("| ticker | annvol real | annvol TF | kurt real | kurt TF | "
             "ACF|r|(1) real | ACF|r|(1) TF | ACF|r|(21) real | ACF|r|(21) TF | "
             "GPH d real | GPH d TF |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for j, t in enumerate(TICKERS):
    rr, rv = R[:, j], Rsim[:, j]
    ar, av = np.abs(rr), np.abs(rv)
    lines.append(
        f"| {t} | {rr.std()*np.sqrt(252):.1%} | {rv.std()*np.sqrt(252):.1%} "
        f"| {kurt(rr):.1f} | {kurt(rv):.1f} "
        f"| {acf(ar,1):.3f} | {acf(av,1):.3f} "
        f"| {acf(ar,21):.3f} | {acf(av,21):.3f} "
        f"| {gph(rr):.3f} | {gph(rv):.3f} |")

iN, iA = TICKERS.index("NVDA"), TICKERS.index("AAPL")
def cross(Rm):
    a, b = Rm[:, iN], Rm[:, iA]
    full = float(np.corrcoef(a, b)[0, 1])
    vol = np.abs(a) + np.abs(b)
    hot = vol >= np.quantile(vol, 0.9)
    crisis = float(np.corrcoef(a[hot], b[hot])[0, 1])
    return full, crisis
fr, cr = cross(R); fs, cs = cross(Rsim)
lines.append("\n## Cross-sectional dependence: corr(NVDA, AAPL)\n")
lines.append("| measure | real | transformer sample |")
lines.append("|---|---|---|")
lines.append(f"| full-sample corr | {fr:.3f} | {fs:.3f} |")
lines.append(f"| crisis corr (top-decile joint vol days) | {cr:.3f} | {cs:.3f} |")

lines.append("\n## NVDA baseline: univariate LSTM (from docs/technical_documentation.tex)\n")
lines.append("Real-NVDA column there uses the full 2010-2026 sample; the "
             "transformer's real column above uses the 8-ticker common sample "
             "(post META IPO), so real values differ slightly.\n")
lines.append("| metric | NVDA real (docs) | LSTM sample | transformer sample |")
lines.append("|---|---|---|---|")
rv = Rsim[:, iN]; av = np.abs(rv)
lines.append(f"| annualized vol | 45.2% | 53.4% | {rv.std()*np.sqrt(252):.1%} |")
lines.append(f"| kurtosis | 9.8 | 10.1 | {kurt(rv):.1f} |")
lines.append(f"| ACF\\|r\\|(1) | 0.171 | 0.131 | {acf(av,1):.3f} |")
lines.append(f"| ACF\\|r\\|(21) | 0.092 | 0.089 | {acf(av,21):.3f} |")
lines.append(f"| GPH d | 0.397 | 0.516 | {gph(rv):.3f} |")

md = "\n".join(lines) + "\n"
with open(os.path.join(HERE, "transformer_scorecard.md"), "w") as fh:
    fh.write(md)
print("\n" + md)
