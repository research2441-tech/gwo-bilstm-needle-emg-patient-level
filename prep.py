"""Signal conditioning and frame-level descriptor extraction for the needle-EMG cohort.

Each 5 s recording (32 768 Hz) is low-pass filtered at 3 kHz, decimated by 8
(4096 Hz), notch filtered at 50 Hz and high-pass filtered at 15 Hz, following the
acquisition notes shipped with the dataset.  The conditioned signal is cut into
1 s segments with 0.5 s hop; every segment is split into T = 32 frames of 128
samples and each frame is summarised by D = 10 descriptors.  Patient identity is
kept with every segment so that all later splits can be done at patient level.
"""
import os, re, sys, glob, json
import numpy as np
from scipy import signal

FS = 32768
DEC = 8
FSR = FS // DEC          # 4096 Hz
SEG = FSR                # 1 s
HOP = FSR // 2           # 0.5 s
FRAME = 128              # 31.25 ms
T = SEG // FRAME         # 32 frames
BANDS = [(15, 100), (100, 300), (300, 700), (700, 1500), (1500, 2048)]
CLASSES = {"Hea": 0, "Myo": 1, "Neu": 2}

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
N_EXPECTED = 5 * FS      # 163 840 samples (5 s)
PAT = re.compile(r"EMG _(\d+) _(\d+)_ ([LR])([DB])_(\w+)\.asc")


def parse(path):
    f = os.path.basename(path)
    folder_cls = os.path.basename(os.path.dirname(path))[:3]
    muscle_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    m = PAT.match(f)
    if m:
        rec, sub, side, mus, cls = m.groups()
        return dict(rec=int(rec), patient=f"{cls}_{int(sub):03d}", side=side, muscle=mus, cls=cls)
    # three myopathy biceps files carry hospital identifiers instead of the
    # repository naming scheme; each is treated as a separate patient
    mus = "B" if "Biceps" in muscle_dir else "D"
    hosp = re.match(r"YY_EMG_\d+_(\d+)_", f)
    return dict(rec=-1, patient=f"{folder_cls}_X{hosp.group(1)}", side="?", muscle=mus, cls=folder_cls)


def condition(x):
    b, a = signal.butter(3, 3000 / (FS / 2), "low")
    x = signal.filtfilt(b, a, x)
    x = signal.resample_poly(x, 1, DEC)
    b, a = signal.iirnotch(50.0, 30.0, FSR)
    x = signal.filtfilt(b, a, x)
    b, a = signal.butter(3, 15 / (FSR / 2), "high")
    return signal.filtfilt(b, a, x)


def frame_features(fr):
    """fr: (T, FRAME) -> (T, 10)"""
    eps = 1e-6
    rms = np.sqrt(np.mean(fr ** 2, 1))
    mav = np.mean(np.abs(fr), 1)
    wl = np.sum(np.abs(np.diff(fr, axis=1)), 1)
    zc = np.mean(np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)), 1)
    d = np.diff(fr, axis=1)
    ssc = np.mean(np.abs(np.diff(np.signbit(d).astype(np.int8), axis=1)), 1)
    spec = np.abs(np.fft.rfft(fr * np.hanning(FRAME), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / FSR)
    tot = spec.sum(1) + eps
    bands = [np.log(spec[:, (freqs >= lo) & (freqs < hi)].sum(1) / tot + eps) for lo, hi in BANDS]
    return np.stack([np.log(rms + eps), np.log(mav + eps), np.log(wl + eps), zc, ssc] + bands, 1)


def main(root, out):
    files = sorted(glob.glob(os.path.join(root, "**", "*.asc"), recursive=True))
    X, y, pid, rid, mus, side = [], [], [], [], [], []
    meta = []
    for k, f in enumerate(files):
        info = parse(f)
        # numbers are whitespace separated, but a few negative values are written
        # without a separator (e.g. "-7741.4-11768.8"); a regex tokeniser handles both
        x = np.array(NUM.findall(open(f).read()), dtype=np.float64)[:N_EXPECTED]
        xc = condition(x)
        n_seg = 1 + (len(xc) - SEG) // HOP
        for s in range(n_seg):
            seg = xc[s * HOP: s * HOP + SEG]
            X.append(frame_features(seg.reshape(T, FRAME)).astype(np.float32))
            y.append(CLASSES[info["cls"]]); pid.append(info["patient"]); rid.append(k)
            mus.append(info["muscle"]); side.append(info["side"])
        meta.append(dict(file=os.path.basename(f), n_samples=len(x), **info))
        if k % 50 == 0:
            print(k, len(files), flush=True)
    np.savez_compressed(out, X=np.stack(X), y=np.array(y), patient=np.array(pid), record=np.array(rid),
                        muscle=np.array(mus), side=np.array(side))
    json.dump(meta, open(out.replace(".npz", "_meta.json"), "w"), indent=1)
    print("segments", len(X), "records", len(files), "patients", len(set(pid)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
