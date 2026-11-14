"""Time to target, ptyrs GPU against PyNX CUDA in float32 on CXIDB 65 (compare_cxidb.py's OUT):
DM, RAAR or ML with compare_cxidb.py's matched settings, both sides driven from Python, every run
through gpu2.sh.

    python -I time_to_target_pynx.py OUT {dm,raar,ml} [--long 1000] [--runs 3] [--module DIR]
        [--gpu2 PATH] [--log OUT/ttt]

time_to_target.py's rules: one long run of --long cycles per side sets the target, 1.01 times the
lower of the two final LLKs, and each side then runs from the same start until --runs runs that
gpu2.sh calls clean have reached it. The LLK is PyNX's normalised one, which both sides record:
the Gaussian for DM and RAAR, the Poisson for ML, the one ML minimises (PyNX's ML gradient is
Poisson only), as its Gaussian rises on these counts. The runs alternate ABBA (ptyrs, PyNX, PyNX,
ptyrs, ...), each a process of its own in a gpu2.sh hold of at most HOLD_S seconds, on CPUs
48-71 with 24 threads. Prints every run with the load average at its end, the medians of the
clean runs, and both long runs' curves; logs and a JSON summary go to --log.

A run (--worker) imports its side (imports_s), reads OUT/scan.cxi and the start (read_s) and
puts the state on the GPU (setup_s, from the read, synchronised; for PyNX its CUDA context and
kernels, built from pycuda's cache), then runs --long cycles in one call recording both LLKs every
cycle, of the estimate each cycle starts from (`llk C G P`: after C cycles). With a target, the
first C at or below it is the cycles to target (`target C`, or `target none`); a new engine from
the same start then runs C cycles in one call with no LLK: `wall_s` is their synchronised wall
time and `end_llk G P` the LLKs they end at. DM and RAAR carry their iterate within a call, so no
run is split. ptyrs's Python GpuPtycho has no synchronize: a read of its probe synchronises it.

The settings are compare_cxidb.py's: dm at alpha DM_ALPHA (PyNX's default, DmParams'), raar at
beta RAAR_BETA with no phase-ramp removal, both from init.h5; ml from OUT/init_ml.h5, compare_
cxidb.py's start for ML (WARMUP ptyrs GPU AP cycles from init.h5), with PyNX's Hann filter off
(compare_pynx_ml.set_hann), the gradient ptyrs takes: stock PyNX ML stops moving within a few
cycles on this data.
"""
import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories
sys.path.insert(0, HERE)
import compare_cxidb as cx  # noqa: E402

FACTOR = 1.01
DM_ALPHA = 0.02
RAAR_BETA = cx.RAAR_BETA
LLK = ("gaussian", "poisson")
TARGET = {"dm": "gaussian", "raar": "gaussian", "ml": "poisson"}
HOLD_S = 295  # under gpu2.sh's five-minute hold
CPUS, THREADS = "48-71", "24"
SIDES = ("ptyrs", "pynx")
LINES = {"imports_s", "read_s", "setup_s", "llk", "target", "wall_s", "end_llk", "load"}
SCRATCH = os.path.expanduser("~/work/ptyrs-scratch")


class Ptyrs:
    """GpuPtycho on device 0 (gpu2.sh's GPU2)."""

    def __init__(self, algo, module):
        sys.path.insert(0, module)
        import h5py  # noqa: F401  read_s times the files alone
        import ptyrs
        self.ptyrs, self.algo = ptyrs, algo

    def start(self, data, start):
        self.g = self.ptyrs.GpuPtycho(data.scan(self.ptyrs), *start, 0)
        self.g.probe

    def run(self, n, llk):
        k = dict(calc_llk=int(llk))
        if self.algo == "dm":
            self.g.dm(n, alpha=DM_ALPHA, **k)
        elif self.algo == "raar":
            self.g.raar(n, beta=RAAR_BETA, zero_phase_ramp=False, **k)
        else:
            self.g.ml(n, update_object=True, update_probe=True, zero_phase_ramp=True, **k)
        self.g.probe
        if llk:
            return cx.history(self.g, n)[:, [cx.KEYS.index(k) for k in LLK]].tolist()

    def llk(self):
        return [self.g.llk()[k] for k in LLK]


class Pynx:
    """PyNX's CUDA operators (compare_pynx_ml.build)."""

    def __init__(self, algo, module):
        self.pm = cx.sibling("ptyrs/scripts/engine", "compare_pynx_ml")
        self.cuop, self.algo = self.pm.cuop, algo

    def start(self, data, start):
        cuop = self.cuop
        self.p = self.pm.build(data.iobs(), (data.x, data.y), data.distance, data.pixel,
                               data.wavelength, *start)
        op = cuop.LLK()  # the first operator makes the CUDA context and compiles the kernels
        if self.algo == "ml":
            self.pm.set_hann(False)
        op.prepare_data(self.p)  # what every operator does first: the state onto the GPU
        cuop.default_processing_unit.finish()

    def run(self, n, llk):
        cuop, c = self.cuop, int(llk)
        if self.algo == "dm":
            op = cuop.DM(calc_llk=c, alpha=DM_ALPHA)
        elif self.algo == "raar":
            op = cuop.RAAR(beta=RAAR_BETA, calc_llk=c, zero_phase_ramp=False)
        else:
            op = cuop.ML(update_object=True, update_probe=True, calc_llk=c)
        self.p = op ** n * self.p
        cuop.default_processing_unit.finish()
        h = self.p.history
        if llk:
            return [[float(h[f"llk_{k}"][i]) for k in LLK] for i in range(n)]

    def llk(self):
        self.p = self.cuop.LoopStack(self.cuop.ObjProbe2LLK()) * self.p
        return [self.p.get_llk(k) for k in LLK]


def load():
    return "{:.2f} {:.2f} {:.2f}".format(*os.getloadavg())


def worker(args):
    """One run of one side, printing the lines `parse` reads."""
    side = (Ptyrs if args.side == "ptyrs" else Pynx)(args.algo, args.module)
    print(f"imports_s {time.perf_counter() - T0}")
    t = time.perf_counter()
    data = cx.Data(args.out)
    start = data.obj, data.probe
    if args.algo == "ml":
        start = cx.read_modes(f"{args.out}/init_ml.h5")
    print(f"read_s {time.perf_counter() - t}")
    side.start(data, start)
    print(f"setup_s {time.perf_counter() - t}", flush=True)
    curve = side.run(args.long, True)
    for c, v in enumerate(curve):
        print(f"llk {c} {v[0]} {v[1]}")
    print(f"load {load()}")
    if args.target is None:
        return
    key = LLK.index(TARGET[args.algo])
    n = next((c for c, v in enumerate(curve) if v[key] <= args.target), None)
    print(f"target {'none' if n is None else n}", flush=True)
    if n is None:
        return
    side.start(data, start)
    t = time.perf_counter()
    side.run(n, False)
    print(f"wall_s {time.perf_counter() - t}")
    print(f"load {load()}")
    print("end_llk {} {}".format(*side.llk()))


def parse(log):
    """A worker's log: the last `load` is the one after the timed cycles."""
    out = {"llk": []}
    with open(log) as f:
        for line in f:
            w = line.split()
            if len(w) < 2 or w[0] not in LINES:
                continue
            if w[0] == "llk":
                out["llk"].append([float(x) for x in w[2:]])
            elif w[0] == "end_llk":
                out["end_llk"] = [float(x) for x in w[1:]]
            elif w[0] == "target":
                out["cycles"] = None if w[1] == "none" else int(w[1])
            elif w[0] == "load":
                out["load"] = " ".join(w[1:])
            else:
                out[w[0]] = float(w[1])
    return out


def run(args, side, label, target):
    """One run under gpu2.sh: its parsed lines, with `clean`, `status` and `log`."""
    log = os.path.join(args.log, f"{args.algo}_{side}_{label}.log")
    cmd = ["timeout", str(HOLD_S), "taskset", "-c", CPUS, sys.executable, "-I",
           os.path.abspath(__file__), args.out, args.algo, "--worker", side, "--long",
           str(args.long), "--module", args.module]
    if target is not None:
        cmd += ["--target", repr(target)]
    env = dict(os.environ, **{k: THREADS for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                   "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")})
    r = subprocess.run([args.gpu2, log, *cmd], capture_output=True, text=True, env=env)
    status = (r.stdout.strip().splitlines() or ["?"])[-1]
    if r.returncode:
        raise RuntimeError(f"{side} {label}: {status}, log {log}")
    out = parse(log)
    out.update(clean=status.startswith("clean"), status=status, log=log)
    return out


def median(rows, key):
    v = [r.get(key) for r in rows]
    return None if None in v else statistics.median(v)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("out")
    a.add_argument("algo", choices=tuple(TARGET))
    a.add_argument("--long", type=int, default=1000)
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--module", default=os.path.join(SCRATCH, "py"))
    a.add_argument("--gpu2", default=os.path.join(SCRATCH, "tools", "gpu2.sh"))
    a.add_argument("--log", help="default OUT/ttt")
    a.add_argument("--worker", choices=SIDES, dest="side", help=argparse.SUPPRESS)
    a.add_argument("--target", type=float, help=argparse.SUPPRESS)
    args = a.parse_args()
    if args.side:
        return worker(args)

    args.log = args.log or os.path.join(args.out, "ttt")
    os.makedirs(args.log, exist_ok=True)
    name, key = TARGET[args.algo], LLK.index(TARGET[args.algo])
    print(f"{args.algo}: {args.long} cycles per long run, {name} LLK ({cx.load_avg()})",
          flush=True)
    long, final = {}, {}
    for side in SIDES:
        r = run(args, side, "long", None)
        long[side] = r["llk"]
        final[side] = r["llk"][-1][key]
        print(f"{side:5s} long run: {name} LLK {final[side]:.6f} after {len(r['llk']) - 1} "
              f"cycles, {r['status']}, load {r['load']}", flush=True)
    target = FACTOR * min(final.values())
    print(f"target {FACTOR:g} x {min(final.values()):.6f} = {target:.6f}")
    for side in SIDES:
        n = next((c for c, v in enumerate(long[side]) if v[key] <= target), None)
        print(f"{side:5s} long run reaches it at cycle {n}")
    every = max(1, args.long // 20)
    print(f"long runs' {name} LLK by cycle: cycle ptyrs pynx")
    for c in sorted({*range(0, args.long, every), args.long - 1}):
        print(f"  {c:5d} {long['ptyrs'][c][key]:.6f} {long['pynx'][c][key]:.6f}")

    summary = {"algo": args.algo, "long_cycles": args.long, "llk": name, "final_llk": final,
               "target": target, "curves": {"columns": LLK, **long},
               "runs": {s: [] for s in SIDES}}
    tries = {s: 0 for s in SIDES}
    k = 0
    while any(len(summary["runs"][s]) < args.runs and tries[s] < 3 * args.runs for s in SIDES):
        side = SIDES[(0, 1, 1, 0)[k % 4]]
        k += 1
        if len(summary["runs"][side]) == args.runs or tries[side] == 3 * args.runs:
            continue
        r = run(args, side, f"run{tries[side]}", target)
        tries[side] += 1
        wall, n = r.get("wall_s"), r.get("cycles")
        r["s_per_cycle"] = None if wall is None else wall / n
        print(f"{side:5s} run {tries[side] - 1}: {r['status']}, setup {r['setup_s']:.3f} s "
              f"(files {r['read_s']:.3f} s, imports {r['imports_s']:.3f} s), cycles {n}, wall "
              + ("-" if wall is None else f"{wall:.3f} s ({1e3 * wall / n:.3f} ms/cycle), "
                 f"end {name} LLK {r['end_llk'][key]:.6f}") + f", load {r['load']}", flush=True)
        del r["llk"]
        if r["clean"]:
            summary["runs"][side].append(r)
    print("median of the clean runs: cycles to target, wall to target, time per cycle, setup "
          "(files)")
    for side in SIDES:
        rows = summary["runs"][side]
        wall, per = median(rows, "wall_s"), median(rows, "s_per_cycle")
        print(f"{side:5s} {len(rows)} clean: {median(rows, 'cycles')} cycles, "
              + ("-" if wall is None else f"{wall:.3f} s, {1e3 * per:.3f} ms/cycle")
              + f", setup {median(rows, 'setup_s'):.3f} s ({median(rows, 'read_s'):.3f} s)")
    path = os.path.join(args.log, f"{args.algo}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(path)
    if any(len(summary["runs"][s]) < args.runs for s in SIDES):
        sys.exit(f"fewer than {args.runs} clean runs on a side")


if __name__ == "__main__":
    main()
