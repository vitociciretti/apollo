import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(12)

r = np.load(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\r.npy")
s = r.std()
x = np.stack([r/s, np.log(np.abs(r)/s + 1e-3)], axis=1).astype(np.float32)  # features
y = (r/s).astype(np.float32)                                                # target: next-day standardized return

class NeuralVol(nn.Module):
    """LSTM -> conditional Student-t(mu, sigma, nu) for next-day return"""
    def __init__(self, h=64):
        super().__init__()
        self.lstm = nn.LSTM(2, h, batch_first=True)
        self.head = nn.Linear(h, 2)          # mu, log_sigma
        self.lognu = nn.Parameter(torch.tensor(1.5))  # nu = 2 + softplus -> fat tails allowed
    def forward(self, xb, hc=None):
        o, hc = self.lstm(xb, hc)
        p = self.head(o)
        mu = 0.05*torch.tanh(p[..., 0])       # tiny conditional mean, bounded
        sigma = torch.exp(p[..., 1].clamp(-4, 2))
        nu = 2.05 + nn.functional.softplus(self.lognu)
        return mu, sigma, nu, hc

def nll(mu, sigma, nu, target):
    z = (target - mu) / sigma
    return -(torch.lgamma((nu+1)/2) - torch.lgamma(nu/2) - 0.5*torch.log(np.pi*nu)
             - torch.log(sigma) - (nu+1)/2*torch.log1p(z**2/nu))

# windows: input x[t], predict y[t+1]
L, STRIDE, BURN_W = 256, 16, 32
Xw, Yw = [], []
for i in range(0, len(x)-L-1, STRIDE):
    Xw.append(x[i:i+L]); Yw.append(y[i+1:i+L+1])
Xw = torch.tensor(np.array(Xw)); Yw = torch.tensor(np.array(Yw))
ntr = int(0.85*len(Xw))
perm = torch.randperm(len(Xw))
tr, va = perm[:ntr], perm[ntr:]
print(f"windows: {len(Xw)} (train {ntr}, val {len(va)})")

model = NeuralVol()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, 30, 0.3)
best, best_state = 1e9, None
for ep in range(80):
    model.train()
    ep_loss = 0
    for b in torch.split(tr[torch.randperm(len(tr))], 32):
        mu, sg, nu, _ = model(Xw[b])
        loss = nll(mu, sg, nu, Yw[b])[:, BURN_W:].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); ep_loss += loss.item()
    sched.step()
    model.eval()
    with torch.no_grad():
        mu, sg, nu, _ = model(Xw[va])
        vloss = nll(mu, sg, nu, Yw[va])[:, BURN_W:].mean().item()
    if vloss < best:
        best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
    if ep % 10 == 0 or ep == 79:
        print(f"ep {ep:3d}  train {ep_loss/max(1,len(tr)//32):.4f}  val {vloss:.4f}  nu {2.05+np.log1p(np.exp(model.lognu.item())):.2f}")
model.load_state_dict(best_state)
print(f"best val NLL {best:.4f}")
torch.save(best_state, r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\neuralvol.pt")

# ---- autoregressive sampling: warm up on last year of real data, then free-run ----
model.eval()
NSIM, BURN = 4608, 252   # 1536 16th-notes x 3 days = exactly 3:00 at 128 BPM
with torch.no_grad():
    seed = torch.tensor(x[-252:]).unsqueeze(0)
    _, _, _, hc = model(seed[:, :-1])
    xt = seed[:, -1:]
    out = []
    for tstep in range(NSIM + BURN):
        mu, sg, nu, hc = model(xt, hc)
        zdist = torch.distributions.StudentT(nu)
        rt = (mu + sg * zdist.sample(mu.shape)).clamp(-8, 8)   # standardized units, clip 8 std
        out.append(rt.item())
        xt = torch.stack([rt, torch.log(rt.abs() + 1e-3)], dim=-1)
r_rnn = np.array(out[BURN:]) * s

# ---- diagnostics vs real ----
def acf(v, lag):
    v = v - v.mean(); return (v[:-lag]*v[lag:]).mean()/v.var()
def stats(v, name):
    k = ((v-v.mean())**4).mean()/v.var()**2
    a = np.abs(v)
    print(f"{name:8s} annvol {v.std()*np.sqrt(252):6.1%}  kurt {k:5.1f}  "
          f"ACF|r|: lag1 {acf(a,1):.3f}  lag5 {acf(a,5):.3f}  lag21 {acf(a,21):.3f}  lag63 {acf(a,63):.3f}  lag126 {acf(a,126):.3f}")
stats(r, "real")
stats(r_rnn, "RNN")

# GPH d on sampled |r|
from scipy.signal import periodogram
def gph(v):
    a = np.abs(v) - np.abs(v).mean()
    f, P = periodogram(a); f, P = f[1:], P[1:]
    m = int(len(v)**0.6); lam = 2*np.pi*f[:m]
    X = np.log(4*np.sin(lam/2)**2)
    return np.polyfit(-X, np.log(P[:m]), 1)[0]
print(f"GPH d: real {gph(r):.3f}  RNN {gph(r_rnn):.3f}")

np.save(r"C:\Users\VITO~1.CIC\AppData\Local\Temp\claude\C--Users-vito-ciciretti\95ba8a9d-607d-45c9-93af-54d0cc16b4d6\scratchpad\r_rnn.npy", r_rnn)
print("saved r_rnn.npy")
