"""Parameter count and single-thread CPU latency for the configurations reported."""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, prep

torch.set_num_threads(1)
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")))
out = {}


def lat(m, n=200, batch=1):
    x = torch.randn(batch, 32, 10)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        t = time.perf_counter()
        for _ in range(n):
            m(x)
    return (time.perf_counter() - t) / n * 1000


cfgs = {"DEFAULT": core.DEFAULT}
for o in ("RS", "PSO", "BO", "GWO"):
    # median configuration (by parameter count) selected across the ten outer folds
    sel = sorted(R["search"][o], key=lambda r: core.n_params(core.BiLSTM(10, r["cfg"]["hidden"], r["cfg"]["layers"], 0)))
    cfgs[o] = sel[len(sel) // 2]["cfg"]
for name, cfg in cfgs.items():
    torch.manual_seed(0)
    m = core.build("bilstm", cfg, 10).eval()
    out[name] = dict(cfg=cfg, params=core.n_params(m), ms_segment=lat(m), ms_patient36=lat(m, 100, 36))
for kind in ("lstm", "tcn"):
    m = core.build(kind, core.DEFAULT, 10).eval()
    out[kind.upper()] = dict(params=core.n_params(m), ms_segment=lat(m), ms_patient36=lat(m, 100, 36))
# descriptor extraction cost for one 1 s segment
x = np.random.randn(prep.SEG)
t = time.perf_counter()
for _ in range(200):
    prep.frame_features(x.reshape(prep.T, prep.FRAME))
out["descriptor_ms_segment"] = (time.perf_counter() - t) / 200 * 1000
raw = np.random.randn(prep.N_EXPECTED)
t = time.perf_counter()
for _ in range(10):
    prep.condition(raw)
out["conditioning_ms_recording"] = (time.perf_counter() - t) / 10 * 1000
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficiency.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
