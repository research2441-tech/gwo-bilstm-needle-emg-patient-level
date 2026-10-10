"""Sensitivity to the segment length: 0.5 s, 1 s and 2 s windows (hop = half the
window, frame length fixed at 128 samples), default Bi-LSTM, first repetition of
the patient-level outer folds, three seeds.
usage: python exp_window.py WIN_SECONDS OUTDIR"""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, core, exp_tune as E

HERE = os.path.dirname(os.path.abspath(__file__))


def build_segments(win):
    seg = int(prep.FSR * win); hop = seg // 2; T = seg // prep.FRAME
    files = sorted(glob.glob(os.path.join(HERE, "..", "..", "mdata", "**", "*.asc"), recursive=True))
    X, y, pid = [], [], []
    for f in files:
        info = prep.parse(f)
        x = np.array(prep.NUM.findall(open(f).read()), dtype=np.float64)[:prep.N_EXPECTED]
        xc = prep.condition(x)
        for s in range(1 + (len(xc) - seg) // hop):
            X.append(prep.frame_features(xc[s * hop: s * hop + seg].reshape(T, prep.FRAME)).astype(np.float32))
            y.append(prep.CLASSES[info["cls"]]); pid.append(info["patient"])
    return np.stack(X), np.array(y), np.array(pid)


def main(win, outdir):
    X, y, pid = build_segments(win)
    # outer folds are defined on patients, so they are identical to the main experiment
    sgk_folds = []
    from sklearn.model_selection import StratifiedGroupKFold
    pats, first = np.unique(pid, return_index=True)
    res = dict(win=win, n_segments=int(len(X)), T=int(X.shape[1]), folds=[])
    for k, (tr, te) in enumerate(E.outer_folds(y, pid)):
        fin = E.final_fit(core.DEFAULT, X, y, pid, tr, te)
        for f in fin:
            z = np.array(f["z_test"])
            pp = np.unique(pid[te])
            zb = np.stack([z[pid[te] == p].mean(0) for p in pp]); yb = np.array([y[te][pid[te] == p][0] for p in pp])
            res["folds"].append(dict(fold=k, seed=f["seed"], seg_ba=core.balanced_accuracy(y[te], z.argmax(1)),
                                     pat_ba=core.balanced_accuracy(yb, zb.argmax(1)), epochs=f["epochs"], sec=f["train_sec"]))
    json.dump(res, open(os.path.join(outdir, f"window_{win}.json"), "w"))
    print("window", win, np.mean([r["seg_ba"] for r in res["folds"]]), np.mean([r["pat_ba"] for r in res["folds"]]), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
