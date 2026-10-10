"""Schematic figures (grayscale, no overlapping elements)."""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 8,
                     "mathtext.fontset": "stix"})


def box(ax, x, y, w, h, text, fill="white", lw=0.9, style="round,pad=0.02,rounding_size=0.06", fs=7.6, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec="black", lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, ls="-", lw=0.9, rad=0.0, head=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, lw=lw, ls=ls, color="black",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def fig_framework():
    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    g = "#ececec"; W_, G_ = 14.6, 2.18
    xs = [0.6 + k * (W_ + G_) for k in range(6)]
    y1, h1 = 31, 12
    labels = ["Needle EMG\nrecording\n32 768 Hz, 5 s",
              "Conditioning\nLP 3 kHz, \u2193 8\nnotch 50 Hz\nHP 15 Hz",
              "Segmentation\n1 s windows\n0.5 s hop",
              "Frame\ndescriptors\n" + r"$T=32,\ D=10$",
              "Bi-LSTM\nencoder " + r"$\boldsymbol{\theta}^{\star}$",
              "Segment\nlogits\n" + r"$\mathbf{z}_s\in\mathbb{R}^{3}$"]
    B = [box(ax, x, y1, W_, h1, t, fill=(g if k == 4 else "white"), fs=7.0) for k, (x, t) in enumerate(zip(xs, labels))]
    for a, b in zip(B[:-1], B[1:]):
        arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2))
    # training loss above the encoder and logits
    L = box(ax, xs[4], 50, xs[5] + W_ - xs[4], 8.5, "Class-balanced cross-entropy " + r"$\mathcal{L}(\mathbf{z}_s, y_s)$" + "\n(training only)", fs=7.0)
    arrow(ax, (xs[5] + W_ / 2, y1 + h1), (xs[5] + W_ / 2, 50))
    arrow(ax, (xs[4] + W_ / 2, 50), (xs[4] + W_ / 2, y1 + h1), ls=(0, (3, 2)))
    ax.text(xs[4] - 0.8, 46.5, r"$\partial\mathcal{L}/\partial\boldsymbol{\theta}$" + "\n(backprop)", ha="right", va="center", fontsize=6.8)
    # patient-level decision row
    y2, h2 = 12, 12
    P1 = box(ax, xs[5], y2, W_, h2, "Patient pooling\n" + r"$\bar{\mathbf{z}}_p=\mathrm{mean}_{s\in\mathcal{S}_p}\,\mathbf{z}_s$", fs=7.0)
    P2 = box(ax, xs[4], y2, W_, h2, "Temperature\nscaling " + r"$T^{\star}$" + "\n(hold-out\npatients)", fs=7.0)
    P3 = box(ax, xs[3], y2, W_, h2, "Calibrated\ndiagnosis " + r"$\boldsymbol{\pi}_p$" + "\nHea / Myo / Neu", fill=g, fs=7.0)
    arrow(ax, (xs[5] + W_ / 2, y1), (xs[5] + W_ / 2, y2 + h2))
    arrow(ax, (P1[0], y2 + h2 / 2), (P2[0] + W_, y2 + h2 / 2))
    arrow(ax, (P2[0], y2 + h2 / 2), (P3[0] + W_, y2 + h2 / 2))
    # hyperparameter search panel (inner folds only)
    px0, px1, py0, py1 = 0.6, 46.5, 1.0, 26.5
    box(ax, px0, py0, px1 - px0, py1 - py0, "", style="round,pad=0.02,rounding_size=0.08", ls="--", lw=0.8)
    ax.text((px0 + px1) / 2, py1 - 2.4, "Hyperparameter search on inner folds only", ha="center", va="center", fontsize=7.0, style="italic")
    T1 = box(ax, 2.6, 11.5, 16, 9.5, "Grey Wolf Optimiser\n" + r"$\mathbf{u}\in[0,1]^{5}$", fs=7.0)
    T2 = box(ax, 28.5, 11.5, 16, 9.5, "Fitness\n" + r"$\mathcal{J}=\overline{\mathrm{BA}}-\lambda\kappa$", fs=7.0)
    T3 = box(ax, 2.6, 2.8, 41.9, 6.2, "Inner patient-grouped folds of outer-training patients", fs=6.8)
    arrow(ax, (T1[0] + T1[2], 18.3), (T2[0], 18.3))
    arrow(ax, (T2[0], 14.2), (T1[0] + T1[2], 14.2))
    ax.text(23.55, 19.2, "candidate", ha="center", va="bottom", fontsize=6.3)
    ax.text(23.55, 13.4, "score", ha="center", va="top", fontsize=6.3)
    arrow(ax, (36.5, T3[1] + T3[3]), (36.5, T2[1]))
    # selected configuration feeds the encoder
    arrow(ax, (px1, py1 - 4), (xs[4] + 3.0, y1))
    ax.text(51.5, 28.6, r"selected $\boldsymbol{\theta}^{\star}$", ha="center", va="center", fontsize=6.9,
            bbox=dict(fc="white", ec="none", pad=0.4))
    fig.savefig(os.path.join(OUT, "fig_framework.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_protocol():
    fig, ax = plt.subplots(figsize=(3.45, 3.5))
    ax.set_xlim(0, 50); ax.set_ylim(0, 54); ax.axis("off")
    ax.text(25, 53.0, "241 patients (50 Hea, 98 Myo, 93 Neu)", ha="center", va="center", fontsize=7.6, fontweight="bold")
    # outer folds as 5 blocks of patients
    for k in range(5):
        x = 3 + k * 9
        fill = "#bdbdbd" if k == 2 else "white"
        box(ax, x, 40.5, 8, 6, f"P{k + 1}", fill=fill, fs=7.2, style="square,pad=0")
    ax.text(25, 49.6, "outer split by patient (5 folds × 2 repetitions)", ha="center", va="center", fontsize=6.9, style="italic")
    # outer train / test
    tr = box(ax, 3, 25.5, 30, 9, "Outer-training patients\n(4 folds)", fs=7.0)
    te = box(ax, 36, 25.5, 11, 9, "Outer-test\npatients\n(1 fold)", fill="#bdbdbd", fs=7.0)
    arrow(ax, (11, 40.5), (11, tr[1] + tr[3]))
    arrow(ax, (27, 40.5), (39.5, te[1] + te[3]), rad=0.15)
    # inner tuning and hold-out
    i1 = box(ax, 3, 11, 14, 10, "Inner 2-fold\n(by patient):\nGWO / PSO /\nRS / BO", fs=6.8)
    i2 = box(ax, 19, 11, 14, 10, "15 % hold-out\npatients:\nearly stop,\n" + r"$T^{\star}$" + " fit", fs=6.8)
    arrow(ax, (9, tr[1]), (10, i1[1] + i1[3]))
    arrow(ax, (27, tr[1]), (26, i2[1] + i2[3]))
    f1 = box(ax, 3, 1, 30, 7, "Final model, 3 seeds", fs=7.0)
    arrow(ax, (10, i1[1]), (12, f1[1] + f1[3]))
    arrow(ax, (26, i2[1]), (24, f1[1] + f1[3]))
    r1 = box(ax, 36, 1, 11, 7, "Patient-level\nmetrics", fs=6.8)
    arrow(ax, (f1[0] + f1[2], 4.5), (r1[0], 4.5))
    arrow(ax, (41.5, te[1]), (41.5, r1[1] + r1[3]))
    fig.savefig(os.path.join(OUT, "fig_protocol.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_framework(); fig_protocol()
