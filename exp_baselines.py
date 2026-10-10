"""Baselines on exactly the same patient-level outer folds.

SVM-RBF and random forest use segment descriptors (mean and standard deviation of
the ten frame features over the 32 frames); their hyperparameters are chosen by a
patient-grouped inner 2-fold search.  The unidirectional LSTM and the temporal
convolutional network use the default recurrent configuration and the same
early-stopping protocol as the Bi-LSTM.

usage: python exp_baselines.py FOLD OUTDIR
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import core, exp_tune as E


def summary_feats(X):
    return np.concatenate([X.mean(1), X.std(1)], 1)


def inner_search(make, grid, F, y, pid):
    sgk = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=7)
    folds = list(sgk.split(F, y, pid)); best, bp = -1, None
    for p in grid:
        s = []
        for a, b in folds:
            sc = StandardScaler().fit(F[a]); m = make(**p).fit(sc.transform(F[a]), y[a])
            s.append(core.balanced_accuracy(y[b], m.predict(sc.transform(F[b]))))
        if np.mean(s) > best:
            best, bp = float(np.mean(s)), p
    return bp, best


def main(fold, outdir):
    X, y, pid, mus = E.load()
    tr, te = E.outer_folds(y, pid)[fold]
    F = summary_feats(X)
    res = dict(fold=fold, test_idx=te.tolist(), models={})
    cw = "balanced"
    specs = {
        "SVM": (lambda **p: SVC(kernel="rbf", probability=True, class_weight=cw, random_state=0, **p),
                [dict(C=c, gamma=g) for c in (1, 10, 100) for g in ("scale", 0.01, 0.1)]),
        "RF": (lambda **p: RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=0, n_jobs=1, **p),
               [dict(max_depth=d, min_samples_leaf=l) for d in (None, 12) for l in (1, 5)]),
    }
    for name, (make, grid) in specs.items():
        t0 = time.time()
        p, ib = inner_search(make, grid, F[tr], y[tr], pid[tr])
        sc = StandardScaler().fit(F[tr]); m = make(**p).fit(sc.transform(F[tr]), y[tr])
        prob = m.predict_proba(sc.transform(F[te]))
        res["models"][name] = dict(params=p, inner_ba=ib, sec=time.time() - t0,
                                   finals=[dict(seed=0, z_test=np.log(prob + 1e-9).tolist())])
    for kind in ("lstm", "tcn"):
        t0 = time.time()
        res["models"][kind.upper()] = dict(params=core.DEFAULT, finals=E.final_fit(core.DEFAULT, X, y, pid, tr, te, kind=kind),
                                           sec=time.time() - t0)
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, f"baselines{E.SUFFIX}_f{fold}.json"), "w"))
    print("baselines", fold, {k: v.get("params") for k, v in res["models"].items()}, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
