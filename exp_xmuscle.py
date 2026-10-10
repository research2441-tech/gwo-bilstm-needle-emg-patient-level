"""Cross-muscle transfer under patient separation.

For every outer fold the model is trained on one muscle (deltoid or biceps) of the
outer-training patients and tested on the other muscle of the outer-test patients,
so the test set differs from training in both patients and muscle.  The
configuration is either the default one or the configuration GWO selected for that
fold.  usage: python exp_xmuscle.py FOLD OUTDIR
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, exp_tune as E


def main(fold, outdir, pairs=(("D", "B"), ("B", "D")), tag="xmuscle"):
    X, y, pid, mus = E.load()
    tr, te = E.outer_folds(y, pid)[fold]
    gwo = json.load(open(os.path.join(outdir, f"GWO{E.SUFFIX}_f{fold}.json")))["cfg"]
    res = dict(fold=fold, runs=[])
    for src, dst in pairs:
        a_idx = tr[mus[tr] == src]; t_idx = te[mus[te] == dst]
        for name, cfg in (("DEFAULT", core.DEFAULT), ("GWO", gwo)):
            fin = E.final_fit(cfg, X, y, pid, a_idx, t_idx)
            res["runs"].append(dict(src=src, dst=dst, cfg_name=name, test_idx=t_idx.tolist(), finals=fin))
    json.dump(res, open(os.path.join(outdir, f"{tag}{E.SUFFIX}_f{fold}.json"), "w"))
    print(tag, fold, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "within":
        # control: same muscle for training and testing, same data volume, still patient-disjoint
        main(int(sys.argv[1]), sys.argv[2], pairs=(("D", "D"), ("B", "B")), tag="wmuscle")
    else:
        main(int(sys.argv[1]), sys.argv[2])
