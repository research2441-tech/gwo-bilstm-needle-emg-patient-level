"""Result figures, grayscale only, generated from results.json and the run files."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = HERE
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figs")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "mathtext.fontset": "stix", "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False})
CLS = ["Healthy", "Myopathy", "Neuropathy"]
GRAYS = ["0.15", "0.45", "0.7", "0.88"]
HATCH = ["", "///", "...", "xxx"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_signals():
    import prep
    meta = json.load(open(os.path.join(WORK, "segments_meta.json")))
    d = np.load(os.path.join(WORK, "segments.npz"))
    root = os.environ.get("EMG_DATA_DIR", "data")
    picks = {}
    for m in meta:
        key = m["cls"]
        if key not in picks and m["muscle"] == "B" and m["rec"] > 0:
            picks[key] = m
    order = ["Hea", "Myo", "Neu"]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 3.9), gridspec_kw=dict(width_ratios=[2.0, 1], hspace=0.6, wspace=0.42))
    names = ["feature " + str(i) for i in range(10)]
    flabels = ["log RMS", "log MAV", "log WL", "ZC rate", "SSC rate", "15–100 Hz", "100–300 Hz", "300–700 Hz", "0.7–1.5 kHz", "1.5–2 kHz"]
    for i, c in enumerate(order):
        m = picks[c]
        f = glob.glob(os.path.join(root, "**", m["file"]), recursive=True)[0]
        x = np.array(prep.NUM.findall(open(f).read()), dtype=float)[:prep.N_EXPECTED]
        xc = prep.condition(x)
        t = np.arange(len(xc)) / prep.FSR
        ax = axes[i, 0]
        sl = (t >= 1.0) & (t < 2.0)
        ax.plot(t[sl], xc[sl], color="black", lw=0.35)
        ax.set_xlim(1.0, 2.0)
        ax.set_ylabel("Amplitude (a.u.)", labelpad=1, fontsize=6.5)
        ax.set_title(f"{CLS[i]} (biceps brachii, one 1 s segment)", fontsize=7.5, loc="left", pad=2)
        if i == 2:
            ax.set_xlabel("Time (s)")
        seg = prep.frame_features(xc[prep.FSR:2 * prep.FSR].reshape(prep.T, prep.FRAME))
        Z = (seg - d["X"].reshape(-1, 10).mean(0)) / d["X"].reshape(-1, 10).std(0)
        ax2 = axes[i, 1]
        im = ax2.imshow(Z.T, aspect="auto", cmap="gray_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax2.set_yticks(range(10)); ax2.set_yticklabels(flabels, fontsize=5.6)
        ax2.set_xticks([0, 15, 31]); ax2.set_xticklabels(["1", "16", "32"], fontsize=6)
        if i == 2:
            ax2.set_xlabel("Frame index $t$")
        ax2.set_title("Standardised descriptors", fontsize=7.0, loc="left", pad=2)
    cb = fig.colorbar(im, ax=axes[:, 1], fraction=0.05, pad=0.03)
    cb.set_label("z-score", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    save(fig, "fig_signals.png")


def fig_leak(R):
    L = R["leak"]; M = R["models"]
    parts = ["Segment-level", "Recording-level", "Patient-level"]
    bil = [L["segment"]["bilstm_ba"], L["record"]["bilstm_ba"], L["patient"]["bilstm_ba"]]
    svm = [L["segment"]["svm_ba"], L["record"]["svm_ba"], L["patient"]["svm_ba"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(3); w = 0.36
    for k, (lab, v) in enumerate((("Bi-LSTM (default)", bil), ("SVM (RBF)", svm))):
        b = ax.bar(x + (k - 0.5) * w, [a[0] for a in v], w, yerr=[a[1] for a in v], color=GRAYS[k * 2 + 0] if k == 0 else "0.75",
                   edgecolor="black", lw=0.5, hatch=HATCH[k * 2], capsize=2, error_kw=dict(lw=0.6), label=lab)
        for xi, a in zip(x + (k - 0.5) * w, v):
            ax.text(xi, a[0] + a[1] + 0.012, f"{a[0]:.3f}", ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(x); ax.set_xticklabels(parts)
    ax.set_ylabel("Segment balanced accuracy"); ax.set_ylim(0.5, 1.08)
    ax.legend(loc="upper right", fontsize=6.5, ncol=1, bbox_to_anchor=(1.0, 1.02))
    save(fig, "fig_leak.png")


def fig_conv(R):
    S = R["search"]; opts = ["RS", "PSO", "BO", "GWO"]; styles = ["-", "--", "-.", "-"]; cols = ["0.6", "0.35", "0.5", "0.0"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for o, ls, c in zip(opts, styles, cols):
        T = np.array([np.maximum.accumulate(r["trace"]) for r in S[o]])
        mu = T.mean(0); se = T.std(0, ddof=1) / np.sqrt(len(T))
        ev = np.arange(1, T.shape[1] + 1)
        ax.plot(ev, mu, ls=ls, color=c, lw=1.4 if o == "GWO" else 1.0, label=o)
        ax.fill_between(ev, mu - se, mu + se, color=c, alpha=0.12, lw=0)
    ax.set_xlabel("Fitness evaluations"); ax.set_ylabel(r"Best-so-far inner fitness $\mathcal{J}$")
    ax.set_xlim(1, 20); ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)
    save(fig, "fig_conv.png")


def fig_space(R):
    S = R["search"]; opts = ["RS", "PSO", "BO", "GWO"]
    import core
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, o in zip(axes, opts):
        P, B = [], []
        for r in S[o]:
            for x in r["log"]:
                P.append(core.n_params(core.BiLSTM(10, x["cfg"]["hidden"], x["cfg"]["layers"], 0))); B.append(x["ba"])
        sel = [core.n_params(core.BiLSTM(10, r["cfg"]["hidden"], r["cfg"]["layers"], 0)) for r in S[o]]
        selb = [r["best_inner_ba"] for r in S[o]]
        ax.scatter(P, B, s=5, facecolor="0.65", edgecolor="none", label="evaluated")
        ax.scatter(sel, selb, s=16, facecolor="black", edgecolor="white", lw=0.4, marker="D", label="selected")
        ax.set_xscale("log"); ax.set_title(o, fontsize=7.5, pad=2)
        ax.set_xlabel(r"Parameters $N_{\boldsymbol{\theta}}$")
    axes[0].set_ylabel("Inner balanced accuracy")
    axes[0].legend(loc="lower left", fontsize=6.0, handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig_space.png")


def fig_box(R):
    names = ["SVM", "RF", "LSTM", "TCN", "DEFAULT", "RS", "PSO", "BO", "GWO"]
    labels = ["SVM", "RF", "LSTM", "TCN", "Bi-LSTM\ndefault", "Bi-LSTM\nRS", "Bi-LSTM\nPSO", "Bi-LSTM\nBO", "Bi-LSTM\nGWO"]
    data = [R["fold_ba"][n] for n in names]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color="black", lw=1.0),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), flierprops=dict(marker="", ms=0))
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor("0.85" if k < 4 else ("0.6" if names[k] != "GWO" else "0.35")); b.set_edgecolor("black"); b.set_linewidth(0.6)
    rng = np.random.default_rng(0)
    for k, v in enumerate(data):
        ax.scatter(k + 1 + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color="black", zorder=3)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("Patient-level BA per outer fold")
    save(fig, "fig_box.png")


def fig_cm(R):
    cm = np.array(R["models"]["GWO"]["patient"]["cm"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLS, fontsize=6.8); ax.set_yticklabels(CLS, fontsize=6.8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for s in ax.spines.values():
        s.set_visible(True)
    save(fig, "fig_cm.png")


def fig_rel(R):
    P = R["pooled"]["GWO"]; y = np.array(P["y"])
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), sharey=True, gridspec_kw=dict(wspace=0.12))
    import core
    for ax, key, title in zip(axes, ("p_raw", "p"), ("Before scaling", "After scaling")):
        p = np.array(P[key]); conf = p.max(1); acc = (p.argmax(1) == y).astype(float)
        edges = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (conf > lo) & (conf <= hi)
            if s.sum() >= 1:
                xs.append(conf[s].mean()); ys.append(acc[s].mean()); ns.append(s.sum())
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
        ax.bar(edges[:-1] + 0.05, [acc[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any() else 0 for lo, hi in zip(edges[:-1], edges[1:])],
               0.1, color="0.8", edgecolor="black", lw=0.4)
        ax.plot(xs, ys, "o-", color="black", ms=2.5, lw=0.8)
        ax.set_title(f"{title} (ECE {core.ece(p, y):.3f})", fontsize=6.8, pad=2)
        ax.set_xlim(0.3, 1.0); ax.set_ylim(0, 1.0); ax.set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    save(fig, "fig_rel.png")


def fig_xm(R):
    X = R["xmuscle"]
    groups = ["D->B", "B->D"]; gl = ["Deltoid → biceps", "Biceps → deltoid"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.2; x = np.arange(2)
    series = [("DEFAULT", "seg", "Default, segment"), ("GWO", "seg", "GWO, segment"), ("DEFAULT", "pat", "Default, patient"), ("GWO", "pat", "GWO, patient")]
    for k, (cfg, lvl, lab) in enumerate(series):
        v = [X[f"{g}|{cfg}"][lvl] for g in groups]
        ax.bar(x + (k - 1.5) * w, [a[0] for a in v], w, yerr=[a[1] for a in v], color=["0.9", "0.6", "0.9", "0.3"][k],
               hatch=["", "", "///", "///"][k], edgecolor="black", lw=0.5, capsize=1.5, error_kw=dict(lw=0.5), label=lab)
    ref = R["models"]["GWO"]["patient"]["ba"][0]
    ax.axhline(ref, color="black", lw=0.7, ls=":")
    ax.text(1.62, ref + 0.012, "same-muscle\npatient BA", fontsize=5.8, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(gl); ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.2)
    save(fig, "fig_xm.png")


def fig_evidence(R):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.9), gridspec_kw=dict(wspace=0.42))
    ax = axes[0]
    for name, ls, c, lab in (("GWO", "-", "0.0", "GWO-tuned"), ("DEFAULT", "--", "0.5", "Default")):
        E = R["evidence"][name]; ms_ = sorted(int(k) for k in E)
        ax.errorbar(ms_, [E[str(m)][0] for m in ms_], yerr=[E[str(m)][1] for m in ms_], ls=ls, color=c, marker="o", ms=2.5, lw=0.9, capsize=1.5, elinewidth=0.5, label=lab)
    ax.set_xlabel("Segments pooled per patient"); ax.set_ylabel("Patient-level BA")
    ax.set_xticks([1, 3, 5, 7, 9]); ax.legend(loc="lower right", fontsize=6.0)
    ax.set_title("(a)", fontsize=7.5, loc="left", pad=2)
    ax = axes[1]
    S = R["selective"]; cv = sorted(float(k) for k in S)
    ax.errorbar(cv, [S[str(c)]["ba"][0] for c in cv], yerr=[S[str(c)]["ba"][1] for c in cv], color="0.0", marker="o", ms=2.5, lw=0.9, capsize=1.5, elinewidth=0.5, label="Balanced acc.")
    ax.errorbar(cv, [S[str(c)]["acc"][0] for c in cv], yerr=[S[str(c)]["acc"][1] for c in cv], color="0.5", ls="--", marker="s", ms=2.5, lw=0.9, capsize=1.5, elinewidth=0.5, label="Accuracy")
    ax.invert_xaxis(); ax.set_xlabel("Coverage (patients decided)"); ax.set_ylabel("Performance on retained")
    ax.legend(loc="upper left", fontsize=6.0)
    ax.set_title("(b)", fontsize=7.5, loc="left", pad=2)
    save(fig, "fig_evidence.png")


def fig_consistency(R):
    C = R["consistency"]; h = C["hist"]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.bar(range(10), h, color="0.55", edgecolor="black", lw=0.5, width=0.7)
    for k, v in enumerate(h):
        ax.text(k, v + 1.5, str(v), ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(range(10)); ax.set_xlabel("Number of models (out of nine) that misclassify the patient")
    ax.set_ylabel("Patients"); ax.set_ylim(0, max(h) * 1.18)
    save(fig, "fig_consistency.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[2:] if len(sys.argv) > 2 else ["all"]
    if "signals" in which or "all" in which:
        fig_signals()
    if "all" in which or "results" in which:
        R = json.load(open(os.path.join(WORK, "results.json")))
        for f in (fig_leak, fig_conv, fig_space, fig_box, fig_cm, fig_rel, fig_xm, fig_evidence, fig_consistency):
            try:
                f(R)
            except Exception as e:
                print("skip", f.__name__, repr(e))
