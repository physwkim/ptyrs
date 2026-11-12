"""Epochs and time to the truth LLK of ptyrs and PyNX on one llk_trace scenario, by
summarize.py's targets: the ptyrs side of the M3 comparison.

    target/release/examples/llk_trace <dir> <algo> <ncycle> <nprobe> <frame> <npos> <step> --gpu
    python -I ptyrs-engine/scripts/compare_pynx.py <dir> <dm|ap> --cuda --no-cpu
    python -I refs/ptychi/trace_m3.py <dir> <algo> [--epoch-s SERIES=SECONDS ...] [--start-epoch N]

Reads <dir>/llk_truth.txt (llk_trace's truth), llk_rust.txt (series "ptyrs") and every
llk_pynx_<algo>_<name>.txt (series "pynx_<name>"; ML's from the engine's compare_pynx_ml.py),
each normalised by nb_obs as PyNX records it.
A record of cycle c is the LLK of the state after FIRST + c updates plus --start-epoch, the
epochs that made init.h5 (llk_trace ml: its ML_WARMUP AP cycles). Targets are
m3ref.TARGET_FACTORS of the truth LLK, hit at the first epoch at or below. Time to target is
epoch x the series' seconds per epoch (--epoch-s, a clean GPU2 run without LLK: gpu_time for
ptyrs, compare_pynx.py --time for PyNX CUDA), as pty-chi's wall_s leaves out evaluate().
Unlike m3ref.time_to_target no Poisson row is skipped: a trace has no poisson_floor_share, and
a rounding-dominated Poisson LLK is orders above every target, so it cannot make a hit.
Writes <dir>/m3_<algo>.json and prints a line per series and LLK.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ref  # noqa: E402

LLKS = ("poisson", "gaussian", "euclidian")
# Updates behind the record of cycle 0: AP and ML take the LLK before the cycle's update, DM
# after it, except PyNX CUDA DM, which takes it inside the stack loop before the update.
FIRST = {"ap": 0, "ml": 0, "dm": 1}
FIRST_PYNX_CUDA = {"dm": 0}


def read_trace(path):
    t = np.loadtxt(path, ndmin=2)
    assert (t[:, 0] == np.arange(len(t))).all(), f"{path} must hold every cycle"
    return t[:, 1:4]


def curve(llk, first):
    return [{"epoch": first + c, **dict(zip(LLKS, map(float, row)))} for c, row in enumerate(llk)]


def to_target(rows, truth, epoch_s):
    """First epoch at or below each factor of each truth LLK, and its time; None if never."""
    out = {}
    for key in LLKS:
        out[key] = {}
        for f in m3ref.TARGET_FACTORS:
            hit = next((r["epoch"] for r in rows if r[key] <= f * truth[key]), None)
            out[key][f"{f:g}"] = None if hit is None else {
                "epoch": hit, "wall_s": None if epoch_s is None else hit * epoch_s}
    return out


def main():
    d, algo = sys.argv[1], sys.argv[2]
    flags = sys.argv[3:]
    start = int(flags[flags.index("--start-epoch") + 1]) if "--start-epoch" in flags else 0
    epoch_s = dict(flags[i + 1].split("=") for i, a in enumerate(flags) if a == "--epoch-s")
    epoch_s = {k: float(v) for k, v in epoch_s.items()}

    truth = dict(zip(LLKS, map(float, np.loadtxt(f"{d}/llk_truth.txt")[:3])))
    sources = {"ptyrs": f"{d}/llk_rust.txt"}
    prefix = f"{d}/llk_pynx_{algo}_"
    for path in sorted(glob.glob(f"{prefix}*.txt")):
        sources[f"pynx_{path[len(prefix):-4]}"] = path
    unknown = set(epoch_s) - set(sources)
    assert not unknown, f"--epoch-s for series without a trace: {sorted(unknown)}"

    series = {}
    for name, path in sources.items():
        first = FIRST_PYNX_CUDA.get(algo, FIRST[algo]) if name == "pynx_cuda" else FIRST[algo]
        rows = curve(read_trace(path), start + first)
        s = epoch_s.get(name)
        series[name] = {
            "source": os.path.basename(path), "epoch_s": s, "curve": rows,
            "final_over_truth": {k: rows[-1][k] / truth[k] for k in LLKS},
            "to_target": to_target(rows, truth, s)}

    out = {"algorithm": algo, "start_epoch": start, "llk_truth": truth, "series": series}
    with open(f"{d}/m3_{algo}.json", "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)

    print(f"{algo}: truth " + " ".join(f"{k} {v:.6f}" for k, v in truth.items()))
    for name, r in series.items():
        s = r["epoch_s"]
        for key in LLKS:
            t = r["to_target"][key]
            hits = " ".join(f"{f}:{t[f]['epoch'] if t[f] else '-'}"
                            for f in ("1.01", "1.05", "1.2"))
            to1 = t["1.01"]["wall_s"] if t["1.01"] and s is not None else None
            print(f"{name:14s} {key:9s} {hits:22s} final {r['final_over_truth'][key]:.4f}x "
                  + (f"{1e3 * s:.3f} ms/ep" if s is not None else "no timing")
                  + (f", 1.01x in {to1:.3f} s" if to1 is not None else ""))


if __name__ == "__main__":
    main()
