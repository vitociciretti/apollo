"""Train the LSTM density model for one ticker, saving per-epoch train/val NLL history.
Usage: train_hist.py TICKER [epochs]   -> writes hist_TICKER.npy  (epochs x 2: train, val)
"""
import numpy as np, torch, torch.nn as nn, sys, os
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(6)

SCRATCH = os.path.dirname(os.path.abspath(__file__))
ticker = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
r = np.load(os.path.join(SCRATCH, f"r_{ticker}.npy"))
s = r.std()
x = np.stack([r/s, np.log(np.abs(r)/s + 1e-3)], axis=1).astype(np.float32)
y = (r/s).astype(np.float32)

class NeuralVol(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.lstm = nn.LSTM(2, h, batch_first=True)
        self.head = nn.Linear(h, 2)
        self.lognu = nn.Parameter(torch.tensor(1.5))
    def forward(self, xb, hc=None):
        o, hc = self.lstm(xb, hc)
        p = self.head(o)
        mu = 0.05*torch.tanh(p[..., 0])
        sigma = torch.exp(p[..., 1].clamp(-4, 2))
        nu = 2.05 + nn.functional.softplus(self.lognu)
        return mu, sigma, nu, hc

def nll(mu, sigma, nu, target):
    z = (target - mu)/sigma
    return -(torch.lgamma((nu+1)/2) - torch.lgamma(nu/2) - 0.5*torch.log(torch.tensor(np.pi)*nu)
             - torch.log(sigma) - (nu+1)/2*torch.log1p(z**2/nu))

L, STRIDE, BURN_W = 256, 16, 32
Xw, Yw = [], []
for i in range(0, len(x)-L-1, STRIDE):
    Xw.append(x[i:i+L]); Yw.append(y[i+1:i+L+1])
Xw = torch.tensor(np.array(Xw)); Yw = torch.tensor(np.array(Yw))
ntr = int(0.85*len(Xw))
perm = torch.randperm(len(Xw))
tr, va = perm[:ntr], perm[ntr:]

model = NeuralVol()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, 30, 0.3)
hist = np.zeros((EPOCHS, 2))
for ep in range(EPOCHS):
    model.train(); tl = 0; nb = 0
    for b in torch.split(tr[torch.randperm(len(tr))], 32):
        mu, sg, nu, _ = model(Xw[b])
        loss = nll(mu, sg, nu, Yw[b])[:, BURN_W:].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); tl += loss.item(); nb += 1
    sched.step()
    model.eval()
    with torch.no_grad():
        mu, sg, nu, _ = model(Xw[va])
        vl = nll(mu, sg, nu, Yw[va])[:, BURN_W:].mean().item()
    hist[ep] = [tl/max(nb, 1), vl]
np.save(os.path.join(SCRATCH, f"hist_{ticker}.npy"), hist)
print(f"{ticker}: {EPOCHS} epochs, final val NLL {hist[-1,1]:.4f}")
