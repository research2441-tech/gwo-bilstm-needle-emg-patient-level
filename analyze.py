"""Compute every number reported in the manuscript from the saved run files.

Patient-level decision: the segment logits of a patient are averaged,
z_bar_p = mean_s z_s, and pi_p = softmax(z_bar_p / T).  T is fitted on the
hold-out patients of the outer-training set (never on test patients).
Out-of-fold predictions of the five outer folds are pooled per repetition and
seed; mean and standard deviation are taken over the repetition x seed runs.
Fold-level values (averaged over seeds) feed the Friedman and Wilcoxon tests.
"""
import os, sys, json, glob, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, exp_tune as E

RUNS = os.path.dirname(os.path.abspath(__file__))  # run files are stored as run_*.json
X, Y, PID, MUS = E.load()
REC = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "segments.npz"))["record"]
CLS = ["Healthy", "Myopathy", "Neuropathy"]


def patient_view(idx, z):
    pats = np.unique(PID[idx])
    zb = np.stack([z[PID[idx] == p].mean(0) for p in pats])
    yb = np.array([Y[idx][PID[idx] == p][0] for p in pats])
    nseg = np.array([(PID[idx] == p).sum() for p in pats])
    return pats, zb, yb, nseg


def holdout_patients(tr, seed):
    a, h = E.group_holdout(Y[tr], PID[tr], 0.15, seed=100 + seed)
    return tr[h]


def metrics(y, prob):
    pred = prob.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    sens = np.diag(cm) / np.maximum(cm.sum(1), 1)
    spec = np.array([(cm.sum() - cm[c].sum() - cm[:, c].sum() + cm[c, c]) / max(cm.sum() - cm[c].sum(), 1) for c in range(3)])
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + sens > 0, 2 * prec * sens / np.maximum(prec + sens, 1e-12), 0)
    # multiclass MCC (Gorodkin)
    t_k, p_k, c, s = cm.sum(1), cm.sum(0), np.trace(cm), cm.sum()
    den = np.sqrt((s ** 2 - (p_k ** 2).sum()) * (s ** 2 - (t_k ** 2).sum()))
    mcc = (c * s - (t_k * p_k).sum()) / den if den > 0 else 0.0
    nll = -np.mean(np.log(prob[np.arange(len(y)), y] + 1e-12))
    return dict(ba=float(sens.mean()), acc=float(np.mean(pred == y)), macro_f1=float(f1.mean()), mcc=float(mcc),
                sens=sens.tolist(), spec=spec.tolist(), ece=core.ece(prob, y), brier=core.brier(prob, y), nll=float(nll),
                cm=cm.tolist())


def tuned_runs(name, rep):
    suf = "" if rep == 0 else f"_r{rep}"
    out = []
    for f in range(5):
        p = os.path.join(RUNS, f"run_{name}{suf}_f{f}.json")
        if not os.path.exists(p):
            return None
        out.append(json.load(open(p)))
    return out


def baseline_runs(name, rep):
    suf = "" if rep == 0 else f"_r{rep}"
    out = []
    for f in range(5):
        p = os.path.join(RUNS, f"run_baselines{suf}_f{f}.json")
        if not os.path.exists(p):
            return None
        r = json.load(open(p)); m = r["models"][name]
        out.append(dict(fold=f, test_idx=r["test_idx"], finals=m["finals"], cfg=m.get("params"), sec=m.get("sec")))
    return out


def evaluate(runs, rep, calibrate=True):
    """Return per-seed pooled metrics (segment and patient level) and per-fold patient BA."""
    E.OUTER_SEED = 2026 + rep
    folds = E.outer_folds(Y, PID)
    seeds = sorted({f["seed"] for f in runs[0]["finals"]})
    per_seed, fold_ba, fold_seg = [], np.zeros((5, len(seeds))), np.zeros((5, len(seeds)))
    pooled_store = {}
    for si, s in enumerate(seeds):
        seg_y, seg_p, pat_y, pat_pr, pat_raw, pat_ids, vote = [], [], [], [], [], [], []
        rec_y, rec_p = [], []
        for r in runs:
            k = r["fold"]; tr, te = folds[k]
            assert list(te) == r["test_idx"]
            fin = [f for f in r["finals"] if f["seed"] == s][0]
            z = np.array(fin["z_test"])
            seg_y.append(Y[te]); seg_p.append(core.softmax(z))
            pats, zb, yb, _ = patient_view(te, z)
            T = 1.0
            if calibrate and "z_hold" in fin:
                h = holdout_patients(tr, s)
                _, zh, yh, _ = patient_view(h, np.array(fin["z_hold"]))
                T = core.fit_temperature(zh, yh)
            pat_y.append(yb); pat_pr.append(core.softmax(zb, T)); pat_raw.append(core.softmax(zb)); pat_ids.append(pats)
            # majority vote over segments
            pv = np.array([np.bincount(z[PID[te] == p].argmax(1), minlength=3) / (PID[te] == p).sum() for p in pats])
            vote.append(pv)
            recs = np.unique(REC[te])
            rec_y.append(np.array([Y[te][REC[te] == q][0] for q in recs]))
            rec_p.append(core.softmax(np.stack([z[REC[te] == q].mean(0) for q in recs]), T))
            fold_ba[k, si] = core.balanced_accuracy(yb, zb.argmax(1))
            fold_seg[k, si] = core.balanced_accuracy(Y[te], z.argmax(1))
        sy, sp = np.concatenate(seg_y), np.concatenate(seg_p)
        py, pp, pr, pv = map(np.concatenate, (pat_y, pat_pr, pat_raw, vote))
        per_seed.append(dict(seed=s, segment=metrics(sy, sp), patient=metrics(py, pp), patient_uncal=metrics(py, pr),
                             patient_vote=metrics(py, pv + 1e-9),
                             recording=metrics(np.concatenate(rec_y), np.concatenate(rec_p))))
        pooled_store[s] = dict(y=py.tolist(), p=pp.tolist(), p_raw=pr.tolist(), ids=np.concatenate(pat_ids).tolist(),
                               seg_y=sy.tolist(), seg_p=sp.tolist())
    return per_seed, fold_ba.mean(1), fold_seg.mean(1), pooled_store


def summarize(per):
    out = {}
    for lvl in ("segment", "recording", "patient", "patient_uncal", "patient_vote"):
        out[lvl] = {}
        for m in ("ba", "acc", "macro_f1", "mcc", "ece", "brier", "nll"):
            v = np.array([p[lvl][m] for p in per])
            out[lvl][m] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
        for m in ("sens", "spec"):
            v = np.array([p[lvl][m] for p in per])
            out[lvl][m] = [v.mean(0).tolist(), (v.std(0, ddof=1) if len(v) > 1 else v.std(0)).tolist()]
        out[lvl]["cm"] = np.sum([p[lvl]["cm"] for p in per], 0).tolist()
    return out


def bootstrap_ci(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.array(y); p = np.array(p); vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        vals.append(core.balanced_accuracy(y[i], p[i].argmax(1)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bayes_signed_rank(d, rope=0.01, n_samples=20000, seed=0):
    """Bayesian signed-rank test (Benavoli et al.): probabilities that the first method
    is practically worse, equivalent (|diff| < rope) or better."""
    rng = np.random.default_rng(seed)
    z = np.concatenate([[0.0], np.asarray(d, float)])          # prior pseudo-observation at 0
    S = z[:, None] + z[None, :]
    left, mid, right = (S < -2 * rope), (np.abs(S) <= 2 * rope), (S > 2 * rope)
    W = rng.dirichlet(np.ones(len(z)), size=n_samples)
    cnt = np.zeros(3)
    for w in W:
        ww = np.outer(w, w)
        v = np.array([ww[left].sum(), ww[mid].sum(), ww[right].sum()])
        cnt[np.argmax(v)] += 1
    p = cnt / n_samples
    return dict(p_worse=float(p[0]), p_rope=float(p[1]), p_better=float(p[2]))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def extra_analyses(R):
    """Evidence curve, selective prediction, muscle subgroups, window sensitivity."""
    rng = np.random.default_rng(0)
    ev = {}; cov_grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]; sel = {c: [] for c in cov_grid}
    sub = {}
    for name in ("GWO", "DEFAULT"):
        curves = []
        for rep in (0, 1):
            runs = tuned_runs(name, rep)
            if runs is None:
                continue
            E.OUTER_SEED = 2026 + rep; folds = E.outer_folds(Y, PID)
            for s in (0, 1, 2):
                ys, zs, mus, ids = [], [], [], []
                curve = {m: [] for m in (1, 2, 3, 5, 7, 9)}
                for r in runs:
                    tr, te = folds[r["fold"]]
                    fin = [f for f in r["finals"] if f["seed"] == s][0]
                    z = np.array(fin["z_test"])
                    h = holdout_patients(tr, s)
                    _, zh, yh, _ = patient_view(h, np.array(fin["z_hold"]))
                    T = core.fit_temperature(zh, yh)
                    for p in np.unique(PID[te]):
                        m_ = PID[te] == p
                        zp = z[m_]
                        ys.append(Y[te][m_][0]); zs.append(core.softmax(zp.mean(0, keepdims=True), T)[0])
                        mus.append("both" if len(set(MUS[te][m_])) > 1 else MUS[te][m_][0]); ids.append(p)
                        for m in curve:
                            k = min(m, len(zp))
                            reps = [zp[rng.choice(len(zp), k, replace=False)].mean(0).argmax() for _ in range(10)]
                            curve[m].append((Y[te][m_][0], reps))
                # evidence curve: balanced accuracy when only m segments per patient are pooled
                c = {}
                for m, lst in curve.items():
                    yy = np.array([a for a, _ in lst]); bas = []
                    for j in range(10):
                        pp = np.array([b[j] for _, b in lst]); bas.append(core.balanced_accuracy(yy, pp))
                    c[m] = float(np.mean(bas))
                curves.append(c)
                ys, P = np.array(ys), np.array(zs)
                if name == "GWO":
                    conf = P.max(1); order = np.argsort(-conf)
                    for cv in cov_grid:
                        keep = order[: int(round(cv * len(order)))]
                        sel[cv].append((core.balanced_accuracy(ys[keep], P[keep].argmax(1)), float(np.mean(P[keep].argmax(1) == ys[keep])),
                                        float(np.mean(ys[keep] == 1)), float(np.min(conf[keep]))))
                    mus = np.array(mus)
                    for cls in (1, 2):
                        for mu in ("D", "B"):
                            msk = (ys == cls) & (mus == mu)
                            sub.setdefault(f"{cls}{mu}", []).append((float(np.mean(P[msk].argmax(1) == cls)), int(msk.sum())))
        ev[name] = {m: [float(np.mean([c[m] for c in curves])), float(np.std([c[m] for c in curves], ddof=1))] for m in curves[0]}
    R["evidence"] = ev
    R["selective"] = {str(c): {"ba": [float(np.mean([v[0] for v in sel[c]])), float(np.std([v[0] for v in sel[c]], ddof=1))],
                               "acc": [float(np.mean([v[1] for v in sel[c]])), float(np.std([v[1] for v in sel[c]], ddof=1))],
                               "myo_share": float(np.mean([v[2] for v in sel[c]])), "min_conf": float(np.mean([v[3] for v in sel[c]]))}
                      for c in cov_grid}
    R["subgroup"] = {k: {"sens": [float(np.mean([a for a, _ in v])), float(np.std([a for a, _ in v], ddof=1))], "n": int(np.mean([n for _, n in v]))}
                     for k, v in sub.items()}
    W = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "run_window_*.json"))):
        w = json.load(open(p))
        W[str(w["win"])] = dict(seg=[float(np.mean([f["seg_ba"] for f in w["folds"]])), float(np.std([f["seg_ba"] for f in w["folds"]], ddof=1))],
                                pat=[float(np.mean([f["pat_ba"] for f in w["folds"]])), float(np.std([f["pat_ba"] for f in w["folds"]], ddof=1))],
                                n_segments=w["n_segments"], T=w["T"], sec=float(np.mean([f["sec"] for f in w["folds"]])))
    R["window"] = W


def main():
    R = {"models": {}, "fold_ba": {}, "search": {}, "leak": {}, "xmuscle": {}}
    tuned = ["DEFAULT", "RS", "PSO", "BO", "GWO", "GWO_L0", "GWO_LEAK"]
    base = ["SVM", "RF", "LSTM", "TCN"]
    store = {}
    for name in tuned + base:
        per_all, fba, fseg = [], [], []
        for rep in (0, 1):
            runs = tuned_runs(name, rep) if name in tuned else baseline_runs(name, rep)
            if runs is None:
                continue
            per, fb, fs, st = evaluate(runs, rep, calibrate=True)
            if rep == 0:
                v = np.array([q["patient"]["ba"] for q in per])
                R.setdefault("rep0_patient_ba", {})[name] = [float(v.mean()), float(v.std(ddof=1))]
                R.setdefault("rep0_fold_ba", {})[name] = list(map(float, fb))
            per_all += per; fba += list(fb); fseg += list(fs)
            store[(name, rep)] = st
        if not per_all:
            continue
        R["models"][name] = summarize(per_all)
        R["models"][name]["n_runs"] = len(per_all)
        R["fold_ba"][name] = fba
        R["models"][name]["fold_seg_ba"] = fseg
        # bootstrap CI on the pooled patient predictions of repetition 0, seed 0
        st = store.get((name, 0))
        if st:
            s0 = st[sorted(st)[0]]
            R["models"][name]["ci95_patient_ba"] = bootstrap_ci(s0["y"], s0["p"])
        if name in tuned and name != "DEFAULT":
            srch = []
            for rep in (0, 1):
                runs = tuned_runs(name, rep)
                if runs is None:
                    continue
                for r in runs:
                    srch.append(dict(rep=rep, fold=r["fold"], cfg=r["cfg"], sec=r["search_sec"],
                                     trace=r["trace"] if r["trace"] else [x["fitness"] for x in r["log"]],
                                     best_ba=max(x["ba"] for x in r["log"]),
                                     best_fit=max(x["fitness"] for x in r["log"]),
                                     best_inner_ba=[x for x in r["log"] if x["fitness"] == max(y["fitness"] for y in r["log"])][0]["ba"],
                                     log=[dict(u=x["u"], ba=x["ba"], fit=x["fitness"], c=x["complexity"], cfg=x["cfg"]) for x in r["log"]]))
            R["search"][name] = srch
        if name in tuned or name in ("LSTM", "TCN"):
            runs = tuned_runs(name, 0) if name in tuned else baseline_runs(name, 0)
            pr = [f["params"] for r in runs for f in r["finals"]]
            R["models"][name]["params"] = [float(np.mean(pr)), float(np.min(pr)), float(np.max(pr))]
            R["models"][name]["train_sec"] = float(np.mean([f["train_sec"] for r in runs for f in r["finals"]]))
            R["models"][name]["epochs"] = float(np.mean([f["epochs"] for r in runs for f in r["finals"]]))
    # statistics across optimisers (fold level, 10 folds)
    opts = [o for o in ["DEFAULT", "RS", "PSO", "BO", "GWO"] if o in R["fold_ba"] and len(R["fold_ba"][o]) == len(R["fold_ba"]["GWO"])]
    if len(opts) >= 3:
        M = np.array([R["fold_ba"][o] for o in opts]).T
        fr = stats.friedmanchisquare(*M.T)
        ranks = np.mean([stats.rankdata(-row) for row in M], 0)
        R["friedman"] = dict(opts=opts, stat=float(fr.statistic), p=float(fr.pvalue), mean_rank=ranks.tolist(), n=int(M.shape[0]))
        comp = [o for o in opts if o != "GWO"]
        ps, ds = [], []
        for o in comp:
            d = np.array(R["fold_ba"]["GWO"]) - np.array(R["fold_ba"][o])
            w = stats.wilcoxon(d, zero_method="zsplit") if np.any(d != 0) else None
            ps.append(float(w.pvalue) if w else 1.0)
            # matched-pairs rank-biserial effect size
            rk = stats.rankdata(np.abs(d)); rpos = rk[d > 0].sum(); rneg = rk[d < 0].sum()
            ds.append(float((rpos - rneg) / rk.sum()) if rk.sum() > 0 else 0.0)
        R["bayes_gwo"] = {o: bayes_signed_rank(np.array(R["fold_ba"]["GWO"]) - np.array(R["fold_ba"][o])) for o in comp}
        for o in ("SVM", "RF", "LSTM", "TCN"):
            if o in R["fold_ba"] and len(R["fold_ba"][o]) == len(R["fold_ba"]["GWO"]):
                R["bayes_gwo"][o] = bayes_signed_rank(np.array(R["fold_ba"]["GWO"]) - np.array(R["fold_ba"][o]))
        R["wilcoxon_gwo"] = dict(vs=comp, p=ps, p_holm=holm(ps), rank_biserial=ds,
                                 mean_diff=[float(np.mean(np.array(R["fold_ba"]["GWO"]) - np.array(R["fold_ba"][o]))) for o in comp])
    # leakage
    for mode in ("segment", "record", "patient"):
        p = os.path.join(RUNS, f"run_leak_{mode}.json")
        if os.path.exists(p):
            L = json.load(open(p))["folds"]
            R["leak"][mode] = {k: [float(np.mean([f[k] for f in L])), float(np.std([f[k] for f in L], ddof=1))]
                               for k in ("bilstm_ba", "bilstm_acc", "svm_ba", "svm_acc", "shared_patients", "n_test_patients")}
    # cross-muscle
    for rep in (0, 1):
        suf = "" if rep == 0 else f"_r{rep}"
        for f in range(5):
          for tag in ("xmuscle", "wmuscle"):
            p = os.path.join(RUNS, f"run_{tag}{suf}_f{f}.json")
            if not os.path.exists(p):
                continue
            for run in json.load(open(p))["runs"]:
                key = f"{run['src']}->{run['dst']}|{run['cfg_name']}"
                te = np.array(run["test_idx"])
                for fin in run["finals"]:
                    z = np.array(fin["z_test"])
                    pats, zb, yb, _ = patient_view(te, z)
                    R["xmuscle"].setdefault(key, {"seg": [], "pat": [], "n_pat": []})
                    R["xmuscle"][key]["seg"].append(core.balanced_accuracy(Y[te], z.argmax(1)))
                    R["xmuscle"][key]["pat"].append(core.balanced_accuracy(yb, zb.argmax(1)))
                    R["xmuscle"][key]["n_pat"].append(len(pats))
    for k, v in R["xmuscle"].items():
        R["xmuscle"][k] = {m: [float(np.mean(v[m])), float(np.std(v[m], ddof=1))] for m in v}
    # error consistency across the nine models (repetition 0, seed 0 of each model)
    allm = ["SVM", "RF", "LSTM", "TCN", "DEFAULT", "RS", "PSO", "BO", "GWO"]
    if all((m, 0) in store for m in allm):
        ref = store[("GWO", 0)][sorted(store[("GWO", 0)])[0]]
        ids = ref["ids"]; yy = np.array(ref["y"])
        wrong = []
        for m in allm:
            st = store[(m, 0)][sorted(store[(m, 0)])[0]]
            mp = dict(zip(st["ids"], np.array(st["p"]).argmax(1)))
            wrong.append([mp[i] != t for i, t in zip(ids, yy)])
        wrong = np.array(wrong)
        nw = wrong.sum(0)
        R["consistency"] = dict(n_patients=int(len(ids)), never=int((nw == 0).sum()), all9=int((nw == 9).sum()),
                                some=int(((nw > 0) & (nw < 9)).sum()),
                                all9_by_class=[int(((nw == 9) & (yy == c)).sum()) for c in range(3)],
                                errors_per_model=[int(w.sum()) for w in wrong],
                                share_errors_hard=float(np.mean([(w & (nw >= 7)).sum() / max(w.sum(), 1) for w in wrong])),
                                hist=[int((nw == k).sum()) for k in range(10)])
    # store pooled predictions of GWO (rep 0, seed 0) for reliability diagrams and confusion matrix
    for name in ("GWO", "DEFAULT", "SVM"):
        st = store.get((name, 0))
        if st:
            R.setdefault("pooled", {})[name] = st[sorted(st)[0]]
    extra_analyses(R)
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
    print(json.dumps({k: (v["patient"]["ba"], v["segment"]["ba"]) for k, v in R["models"].items()}, indent=0))
    print("friedman", R.get("friedman")); print("wilcoxon", R.get("wilcoxon_gwo")); print("leak", R["leak"])
    print("xmuscle", {k: v["pat"] for k, v in R["xmuscle"].items()})


if __name__ == "__main__":
    main()
