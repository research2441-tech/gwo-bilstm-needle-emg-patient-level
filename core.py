"""Models, training, hyperparameter decoding and optimisers shared by all experiments."""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
N_CLASSES = 3

# ----------------------------------------------------------------- search space
# The optimisers move in the unit hypercube u in [0,1]^5; decode() maps a point to
# a Bi-LSTM configuration.  Every optimiser sees exactly the same decoder.
DIM = 5


def decode(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return dict(hidden=int(round(8 + u[0] * 88)),          # 8 .. 96 units per direction
                layers=1 if u[1] < 0.5 else 2,
                dropout=float(0.5 * u[2]),
                lr=float(10 ** (-3.5 + 1.5 * u[3])),        # 3.2e-4 .. 1e-2
                wd=float(10 ** (-6 + 3 * u[4])))            # 1e-6 .. 1e-3


DEFAULT = dict(hidden=64, layers=1, dropout=0.2, lr=1e-3, wd=1e-4)


# ----------------------------------------------------------------- networks
class BiLSTM(nn.Module):
    def __init__(self, d_in, hidden, layers, dropout, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        k = 2 if bidirectional else 1
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(k * hidden, N_CLASSES)
        self.bi = bidirectional

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        z = torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1]
        return self.head(self.drop(z))


class TCN(nn.Module):
    """Small temporal convolutional baseline over the frame sequence."""
    def __init__(self, d_in, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, hidden, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.ReLU(), nn.BatchNorm1d(hidden))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        z = self.net(x.transpose(1, 2)).mean(2)
        return self.head(self.drop(z))


def build(kind, cfg, d_in):
    if kind == "bilstm":
        return BiLSTM(d_in, cfg["hidden"], cfg["layers"], cfg["dropout"], True)
    if kind == "lstm":
        return BiLSTM(d_in, cfg["hidden"], cfg["layers"], cfg["dropout"], False)
    if kind == "tcn":
        return TCN(d_in, cfg["hidden"], cfg["dropout"])
    raise ValueError(kind)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------- training
def class_weights(y):
    c = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = c.sum() / (N_CLASSES * np.maximum(c, 1))
    return torch.tensor(w, dtype=torch.float32)


def train(kind, cfg, Xtr, ytr, Xva, yva, seed, max_epochs=30, patience=5, batch=64):
    """Adam training with class-balanced cross-entropy and early stopping on the
    validation loss.  Returns the best model (in eval mode) and its history."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(kind, cfg, Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    w = class_weights(ytr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva).long()
    best, best_state, wait, hist = 1e9, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(max_epochs):
        m.train()
        idx = rng.permutation(len(Xt))
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            loss = F.cross_entropy(m(Xt[j]), yt[j], weight=w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = F.cross_entropy(m(Xv), yv, weight=w).item()
        hist.append(vl)
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state); m.eval()
    return m, hist


@torch.no_grad()
def logits(m, X, batch=1024):
    out = [m(torch.from_numpy(X[i:i + batch])) for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


# ----------------------------------------------------------------- metrics
def balanced_accuracy(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(N_CLASSES) if np.any(y == c)]))


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(z, y):
    """Temperature scaling: minimise NLL over T on held-out logits."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    nll = [-np.mean(np.log(softmax(z, t)[np.arange(len(y)), y] + 1e-12)) for t in grid]
    return float(grid[int(np.argmin(nll))])


def ece(prob, y, bins=10):
    conf = prob.max(1); pred = prob.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return float(e)


def brier(prob, y):
    oh = np.eye(N_CLASSES)[y]
    return float(np.mean(np.sum((prob - oh) ** 2, 1)))


# ----------------------------------------------------------------- optimisers
class GWO:
    """Grey Wolf Optimiser as defined by Mirjalili et al. (2014): every wolf moves
    to the average of three positions steered by the alpha, beta and delta wolves,
    with a decreasing linearly from 2 to 0 over the iterations."""
    name = "GWO"

    def __init__(self, dim, pop, iters, rng):
        self.d, self.n, self.T, self.rng = dim, pop, iters, rng

    def run(self, f):
        X = self.rng.random((self.n, self.d))
        fit = np.array([f(x) for x in X])                 # maximisation
        trace = list(fit)
        for t in range(self.T):
            order = np.argsort(-fit)
            alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
            a = 2.0 * (1 - t / self.T)
            newX = np.empty_like(X)
            for i in range(self.n):
                parts = []
                for leader in (alpha, beta, delta):
                    A = 2 * a * self.rng.random(self.d) - a
                    C = 2 * self.rng.random(self.d)
                    D = np.abs(C * leader - X[i])
                    parts.append(leader - A * D)
                newX[i] = np.clip(np.mean(parts, 0), 0, 1)
            X = newX
            fit = np.array([f(x) for x in X])
            trace += list(fit)
        return trace


class PSO:
    name = "PSO"

    def __init__(self, dim, pop, iters, rng, w=0.7, c1=1.5, c2=1.5, vmax=0.25):
        self.d, self.n, self.T, self.rng = dim, pop, iters, rng
        self.w, self.c1, self.c2, self.vmax = w, c1, c2, vmax

    def run(self, f):
        X = self.rng.random((self.n, self.d)); V = np.zeros_like(X)
        fit = np.array([f(x) for x in X]); trace = list(fit)
        P, Pf = X.copy(), fit.copy(); g = P[np.argmax(Pf)].copy()
        for _ in range(self.T):
            r1, r2 = self.rng.random(X.shape), self.rng.random(X.shape)
            V = np.clip(self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (g - X), -self.vmax, self.vmax)
            X = np.clip(X + V, 0, 1)
            fit = np.array([f(x) for x in X]); trace += list(fit)
            imp = fit > Pf; P[imp], Pf[imp] = X[imp], fit[imp]
            g = P[np.argmax(Pf)].copy()
        return trace


class RandomSearch:
    name = "RS"

    def __init__(self, dim, budget, rng):
        self.d, self.B, self.rng = dim, budget, rng

    def run(self, f):
        return [f(self.rng.random(self.d)) for _ in range(self.B)]


class BayesOpt:
    """Gaussian-process Bayesian optimisation (Optuna GPSampler, log-EI)."""
    name = "BO"

    def __init__(self, dim, budget, seed, n_startup=6):
        self.d, self.B, self.seed, self.ns = dim, budget, seed, n_startup

    def run(self, f):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.GPSampler(seed=self.seed, n_startup_trials=self.ns)
        st = optuna.create_study(direction="maximize", sampler=sampler)
        trace = []

        def obj(tr):
            u = np.array([tr.suggest_float(f"u{k}", 0.0, 1.0) for k in range(self.d)])
            v = f(u); trace.append(v); return v
        st.optimize(obj, n_trials=self.B)
        return trace
