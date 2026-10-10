"""Patient-level nested cross-validation with equal-budget hyperparameter search.

Outer loop : 5 folds, StratifiedGroupKFold over patients (no patient on both sides).
Inner loop : 2 folds, StratifiedGroupKFold over the outer-training patients only.
Fitness    : mean inner balanced accuracy minus a complexity penalty (Eq. in paper).
Final model: retrained on all outer-training patients (15 % of them held out, again
             by patient, for early stopping and temperature fitting), 3 seeds.

usage: python exp_tune.py OPT FOLD OUTDIR      OPT in {GWO, PSO, RS, BO, DEFAULT}
"""
import sys, os, json, time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

BUDGET = 20          # fitness evaluations per optimiser and outer fold
POP = 5              # population for GWO / PSO -> 5 + 3*5 = 20 evaluations
LAMBDA = 0.02        # weight of the complexity penalty
P_MIN, P_MAX = core.n_params(core.BiLSTM(10, 8, 1, 0)), core.n_params(core.BiLSTM(10, 96, 2, 0))
SEEDS = (0, 1, 2)
REP = int(os.environ.get("REP", "0"))
OUTER_SEED = 2026 + REP
SUFFIX = "" if REP == 0 else f"_r{REP}"


def load():
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "segments.npz"))
    return d["X"], d["y"], d["patient"], d["muscle"]


def patient_labels(y, pid):
    u, first = np.unique(pid, return_index=True)
    return u, y[first]


def outer_folds(y, pid):
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    return list(sgk.split(np.zeros(len(y)), y, pid))


def standardise(Xtr, *others):
    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    f = lambda A: ((A - mu) / sd).astype(np.float32)
    return [f(Xtr)] + [f(A) for A in others]


def group_holdout(y, pid, frac, seed):
    """Split indices by patient, stratified by class, keeping frac of patients aside."""
    rng = np.random.default_rng(seed)
    pats, lab = patient_labels(y, pid)
    hold = []
    for c in np.unique(lab):
        pc = pats[lab == c]; k = max(1, int(round(frac * len(pc))))
        hold += list(rng.choice(pc, k, replace=False))
    m = np.isin(pid, hold)
    return np.where(~m)[0], np.where(m)[0]


def complexity(cfg):
    p = core.n_params(core.BiLSTM(10, cfg["hidden"], cfg["layers"], 0))
    return (np.log(p) - np.log(P_MIN)) / (np.log(P_MAX) - np.log(P_MIN))


def make_fitness(X, y, pid, log, lam=LAMBDA, leaky=False):
    if leaky:   # ablation: inner folds drawn over segments, so inner patients overlap
        from sklearn.model_selection import StratifiedKFold
        inner = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=7).split(np.zeros(len(y)), y))
    else:
        sgk = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=7)
        inner = list(sgk.split(np.zeros(len(y)), y, pid))

    def f(u):
        cfg = core.decode(u); t0 = time.time(); bas = []
        for k, (a, b) in enumerate(inner):
            Xa, Xb = standardise(X[a], X[b])
            m, _ = core.train("bilstm", cfg, Xa, y[a], Xb, y[b], seed=k, max_epochs=20, patience=4, batch=128)
            bas.append(core.balanced_accuracy(y[b], core.logits(m, Xb).argmax(1)))
        ba = float(np.mean(bas)); c = float(complexity(cfg))
        fit = ba - lam * c
        log.append(dict(u=list(map(float, u)), cfg=cfg, ba=ba, complexity=c, fitness=fit, sec=time.time() - t0))
        return fit
    return f


def final_fit(cfg, X, y, pid, tr, te, kind="bilstm"):
    out = []
    for s in SEEDS:
        a, h = group_holdout(y[tr], pid[tr], 0.15, seed=100 + s)
        a, h = tr[a], tr[h]
        Xa, Xh, Xt = standardise(X[a], X[h], X[te])
        t0 = time.time()
        m, hist = core.train(kind, cfg, Xa, y[a], Xh, y[h], seed=s, max_epochs=30, patience=5, batch=128)
        out.append(dict(seed=s, epochs=len(hist), train_sec=time.time() - t0, params=core.n_params(m),
                        z_hold=core.logits(m, Xh).tolist(), y_hold=y[h].tolist(),
                        z_test=core.logits(m, Xt).tolist()))
    return out


def main(opt, fold, outdir):
    X, y, pid, mus = load()
    tr, te = outer_folds(y, pid)[fold]
    assert not set(pid[tr]) & set(pid[te])
    rng = np.random.default_rng(1000 * fold + 17)
    log, t0 = [], time.time()
    if opt == "DEFAULT":
        cfg, trace = dict(core.DEFAULT), []
    else:
        lam = 0.0 if opt == "GWO_L0" else LAMBDA
        f = make_fitness(X[tr], y[tr], pid[tr], log, lam=lam, leaky=(opt == "GWO_LEAK"))
        iters = BUDGET // POP - 1
        o = {"GWO": lambda: core.GWO(core.DIM, POP, iters, rng),
             "GWO_L0": lambda: core.GWO(core.DIM, POP, iters, rng),
             "GWO_LEAK": lambda: core.GWO(core.DIM, POP, iters, rng),
             "PSO": lambda: core.PSO(core.DIM, POP, iters, rng),
             "RS": lambda: core.RandomSearch(core.DIM, BUDGET, rng),
             "BO": lambda: core.BayesOpt(core.DIM, BUDGET, seed=fold)}[opt]()
        trace = o.run(f)
        best = log[int(np.argmax([r["fitness"] for r in log]))]
        cfg = best["cfg"]
    search_sec = time.time() - t0
    finals = final_fit(cfg, X, y, pid, tr, te)
    res = dict(opt=opt, fold=fold, rep=REP, cfg=cfg, trace=list(map(float, trace)), log=log, search_sec=search_sec,
               test_idx=te.tolist(), finals=finals)
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, f"{opt}{SUFFIX}_f{fold}.json"), "w"))
    print(opt, fold, cfg, "search", round(search_sec), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
