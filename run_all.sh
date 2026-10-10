#!/bin/sh
# Complete pipeline. Expects work/segments.npz produced by prep.py.
set -e
mkdir -p runs figs  # working folders created at run time
for rep in 0 1; do
  for f in 0 1 2 3 4; do
    for o in DEFAULT RS PSO BO GWO; do REP=$rep python3 exp_tune.py $o $f runs; done
    REP=$rep python3 exp_baselines.py $f runs
    REP=$rep python3 exp_xmuscle.py $f runs
    REP=$rep python3 exp_xmuscle.py $f runs within
  done
done
for f in 0 1 2 3 4; do for o in GWO_L0 GWO_LEAK; do python3 exp_tune.py $o $f runs; done; done
for m in segment record patient; do python3 exp_leak.py $m runs; done
for w in 0.5 1.0 2.0; do python3 exp_window.py $w runs; done
for f in runs/*.json; do cp "$f" "run_$(basename "$f")"; done
python3 analyze.py
python3 efficiency.py
python3 fig_diagrams.py figs
python3 fig_results.py figs
