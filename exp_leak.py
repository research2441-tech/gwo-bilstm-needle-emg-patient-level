"""Leakage experiment: the same default Bi-LSTM and SVM evaluated with
(a) segment-level random 5-fold CV (segments of one patient on both sides) and
(b) recording-level 5-fold CV (recordings separated, patients with several
recordings can still straddle the split).  Patient-level results come from the
nested protocol.  usage: python exp_leak.py MODE OUTDIR   MODE in {segment, record, patient}
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import core, exp_tune as E
from exp_baselines import summary_feats


def main(mode, outdir):
    X, y, pid, mus = E.load()
    rec = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "segments.npz"))["record"]
    if mode == "segment":
        folds = list(StratifiedKFold(5, shuffle=True, random_state=E.OUTER_SEED).split(X, y))
    elif mode == "patient":   # identical procedure, patient-grouped folds (reference for the audit)
        folds = E.outer_folds(y, pid)
    else:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=E.OUTER_SEED).split(X, y, rec))
    F = summary_feats(X)
    out = dict(mode=mode, folds=[])
    for k, (tr, te) in enumerate(folds):
        shared = len(set(pid[tr]) & set(pid[te]))
        rng = np.random.default_rng(k)
        h = rng.choice(tr, int(0.15 * len(tr)), replace=False)       # leaky hold-out, as in the original protocols
        a = np.setdiff1d(tr, h)
        Xa, Xh, Xt = E.standardise(X[a], X[h], X[te])
        m, _ = core.train("bilstm", core.DEFAULT, Xa, y[a], Xh, y[h], seed=0, max_epochs=30, patience=5, batch=128)
        z = core.logits(m, Xt)
        sc = StandardScaler().fit(F[tr]); svm = SVC(kernel="rbf", C=10, class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
        ps = svm.predict(sc.transform(F[te]))
        out["folds"].append(dict(fold=k, shared_patients=shared, n_test_patients=len(set(pid[te])),
                                 bilstm_ba=core.balanced_accuracy(y[te], z.argmax(1)),
                                 bilstm_acc=float(np.mean(z.argmax(1) == y[te])),
                                 svm_ba=core.balanced_accuracy(y[te], ps), svm_acc=float(np.mean(ps == y[te]))))
        print(mode, out["folds"][-1], flush=True)
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, f"leak_{mode}.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
