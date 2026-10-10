# Patient-level nested evaluation of a GWO-tuned Bi-LSTM for needle-EMG diagnosis

Code accompanying the manuscript *Patient-Level Nested Evaluation of a Grey Wolf–Optimized
Bidirectional LSTM for Myopathy and Neuropathy Diagnosis from Clinical Needle Electromyography*.

Every number, table and figure in the manuscript is produced by the scripts in this folder from
the raw result files in `runs/` (included). Nothing is entered by hand.

## Data

Messekher, A., Mekaoui, S., Kedir-Talha, M., Kediha, M. I., & Mostefaoui, F. (2023).
*Electromyographic data recordings during isometric contractions of the biceps brachii and
deltoid collected from patients and healthy subjects.* Mendeley Data, V1.
https://doi.org/10.17632/543xpjycj9.1 (CC BY 4.0)

Download the archive, unzip it, and pass the folder that contains `EMG Dataset Reposistory`
(spelling as in the archive) to `prep.py`.

## Files

| File | Purpose |
|---|---|
| `prep.py` | Conditioning (LP 3 kHz, ↓8, 50 Hz notch, HP 15 Hz), 1 s segments, 32 × 10 frame descriptors, patient IDs |
| `core.py` | Bi-LSTM / LSTM / TCN, training loop, decoder of the search space, GWO, PSO, random search, GP Bayesian optimisation, metrics |
| `exp_tune.py` | Patient-level nested CV with equal-budget search (`GWO`, `PSO`, `RS`, `BO`, `DEFAULT`, ablations `GWO_L0`, `GWO_LEAK`) |
| `exp_baselines.py` | SVM, random forest, LSTM and TCN on the same outer folds |
| `exp_leak.py` | Leakage audit with segment-, recording- and patient-level partitions |
| `exp_xmuscle.py` | Cross-muscle transfer and within-muscle control |
| `exp_window.py` | Segment-length sensitivity (0.5, 1, 2 s) |
| `analyze.py` | All metrics, pooling, temperature scaling, statistics → `results.json` |
| `efficiency.py` | Parameter counts and CPU latency → `efficiency.json` |
| `fig_diagrams.py`, `fig_results.py` | All figures (grayscale) |
| `run_all.sh` | Runs the complete pipeline in order |
| `run_*.json` | Raw outputs of every run (logits, selected configurations, search logs) |
| `results.json`, `efficiency.json` | Analysis outputs from which every manuscript number is taken |

## Reproduce

```bash
pip install -r requirements.txt
python prep.py "/path/to/Dataset Reposistory" segments.npz
export EMG_DATA_DIR="/path/to/Dataset Reposistory"   # used by exp_window.py and fig_results.py
sh run_all.sh            # about 3 h on two CPU cores
```

Seeds: outer partitions 2026 (repetition 1) and 2027 (repetition 2, `REP=1`), inner partitions 7,
hold-out selection and initialisation 0, 1, 2. Software used for the manuscript: Python 3.13,
PyTorch 2.14.1 (CPU), scikit-learn 1.9.1, SciPy 1.18.1, Optuna 5.0.0.

## Licence

MIT for the code. The data remain under their CC BY 4.0 licence.
