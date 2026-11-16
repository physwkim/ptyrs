"""Time to target, ptyrs GPU against pty-chi GPU in float32 on CXIDB 65 (compare_cxidb.py's
OUT): DESIGN B7's four M3 cases at their batches, both sides driven from Python from init.h5.

    python -I time_to_target_ptychi.py OUT CASE long [--epochs 2000] [--gpu UUID] [--cpus 48-71]
    python -I time_to_target_ptychi.py OUT CASE time [--llk KIND] [--of SIDE] [--target LLK]
        [--runs 3] [--hold 295] [--gpu2 PATH]
    common: [--module DIR] [--log OUT/ttt-ptychi]

CASE is lsqml_gaussian (batch 32), lsqml_poisson (ptyrs 32, pty-chi 64), bh_mini96 (pty-chi's GD)
or bh_cg_full (one batch of every frame, CG). time_to_target_pynx.py's rules, with the long runs
apart from the timed ones. `long` runs both sides --epochs epochs at once, a process each on --gpu
(not the timing GPU), and records the three PyNX LLKs of the start and after every epoch into
--log/CASE_long.json; per LLK it prints the target, 1.01 times the lower final, the epoch each
side first reaches it, a second target at 1.01 times the final of a side that never reaches the
first, and a side whose LLK ends above its start. CASE's LLK is the one its fermat runs were timed
on (refs/ptychi/summarize.py: the noise model's, Gaussian for BH); --llk takes another.

`time` takes the --llk target at 1.01 times --of's final (default the lower), or --target, and
each side's epochs to it, C, from its long record; the side's runs go through gpu2.sh, ABBA, until
--runs are counted (clean, CPUs 48-71 under BUSY busy over the run). A run (--worker) imports its
side (imports_s), reads OUT (read_s) and puts the state on the GPU (setup_s, from the read,
synchronised; pty-chi's lazy CUDA and cuFFT set-up lands in its first epoch, epoch1_s), then runs
C epochs with no LLK: ptyrs one call, a read of its probe synchronising it; pty-chi one task.run(1)
each, synchronised, as compare_cxidb.py. wall_s is their wall time and end_llk the LLKs they end
at, which must be the long record's (both sides are deterministic). The steady epoch follows:
ptyrs (call of 60 - call of 20) / 40 epochs, engine_m3_bench's; pty-chi the median of its epochs
3..max(C, 12), run on past C when C < 12.

pty-chi's tasks are compare_cxidb.ptychi_task's (its mask as valid_pixel_mask) with compare_lsqml.
ptychi_task's or compare_cxidb.bh_task's options, switched to run_m3.py's float32 on the GPU with
its data on the device and random_seed SEED (LSQML Poisson's alpha jitter stays, seeded); its
batches in the engine's random_batches order. Its LLK is m3ref.llk_sums over its valid pixels in
float64 on the GPU; ptyrs's the GpuPtycho record, of the state after each epoch.
"""
import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories
sys.path.insert(0, HERE)
import compare_cxidb as cx  # noqa: E402
import time_to_target_pynx as tp  # noqa: E402

SEED = 7
LLK = ("poisson", "gaussian", "euclidian")
# batch None: every frame, one batch
CASES = {
    "lsqml_gaussian": dict(algo="lsqml", noise="gaussian", llk="gaussian", ptyrs=32, ptychi=32),
    "lsqml_poisson": dict(algo="lsqml", noise="poisson", llk="poisson", ptyrs=32, ptychi=64),
    "bh_mini96": dict(algo="bh", llk="gaussian", ptyrs=96, ptychi=96),
    "bh_cg_full": dict(algo="bh", llk="gaussian", ptyrs=None, ptychi=None),
}
SIDES = ("ptyrs", "pty-chi")
STEADY_CALLS = (20, 60)
STEADY_MIN = 12
GPU1 = "GPU-fdf6033d-b702-4876-ce32-4b60eba986ef"
LINES = {"imports_s", "read_s", "setup_s", "llk", "wall_s", "epoch_s", "end_llk", "steady_s",
         "long_s", "load0", "load", "busy"}


class Ptyrs:
    """GpuPtycho on device 0."""

    def __init__(self, case, module):
        sys.path.insert(0, module)
        import h5py  # noqa: F401  read_s times the files alone
        import ptyrs
        self.ptyrs, self.case = ptyrs, case

    def start(self, data, epochs):
        del epochs
        self.batch = self.case["ptyrs"] or len(data.counts)
        self.g = self.ptyrs.GpuPtycho(data.scan(self.ptyrs), data.obj, data.probe, 0)
        self.g.probe

    def run(self, n, llk):
        c = int(llk)
        if self.case["algo"] == "lsqml":
            cost = {"gaussian": "amplitude", "poisson": "poisson"}[self.case["noise"]]
            self.g.lsqml(n, c, self.batch, cost, SEED)
        else:
            self.g.bh(n, c, self.batch, cx.BH_RHO, SEED)
        self.g.probe
        if llk:
            h = self.g.history
            assert list(h["cycle"]) == list(range(n)), h["cycle"]
            return [[float(h[k][i]) for k in LLK] for i in range(n)]

    def llk(self):
        d = self.g.llk()
        return [d[k] for k in LLK]

    def to_target(self, n):
        t = time.perf_counter()
        self.run(n, False)
        return time.perf_counter() - t, []

    def steady(self, walls):
        del walls
        t = {}
        for c in STEADY_CALLS:
            t0 = time.perf_counter()
            self.run(c, False)
            t[c] = time.perf_counter() - t0
        a, b = STEADY_CALLS
        return (t[b] - t[a]) / (b - a)


class Ptychi:
    """A pty-chi task on the GPU, float32."""

    def __init__(self, case, module):
        del module
        import torch
        import ptychi.api  # noqa: F401
        import pynx.ptycho.ptycho  # noqa: F401  m3ref.pynx_positions
        self.torch, self.case = torch, case
        self.lsqml = cx.sibling("refs/py", "compare_lsqml")
        self.m3ref = self.lsqml.m3ref
        self.rb = cx.sibling("refs/f64", "random_batches")

    def start(self, data, epochs):
        import ptychi.api as api
        import ptychi.api.task as at
        from torch.utils.data import DataLoader

        c, n = self.case, len(data.counts)
        batch = c["ptychi"] or n
        if c["algo"] == "lsqml":
            build = lambda *s: self.lsqml.ptychi_task(*s, c["noise"], batch, epochs)
        else:
            build = lambda *s: cx.bh_task(*s, cx.corners(data), batch, epochs)
        real = at.PtychographyTask

        def f32_gpu(opts, **kw):
            r = opts.reconstructor_options
            r.default_device = api.Devices.GPU
            r.default_dtype = api.Dtypes.FLOAT32
            r.random_seed = SEED
            opts.data_options.save_data_on_device = True
            kw["diffraction_data"] = kw["diffraction_data"].astype(np.float32)
            for k in ("object_data", "probe_data"):
                kw[k] = kw[k].astype(np.complex64)
            return real(opts, **kw)

        at.PtychographyTask = f32_gpu
        try:
            self.task, _ = cx.ptychi_task(data, data.obj, data.probe, build)
        finally:
            at.PtychographyTask = real
        rec = self.task.reconstructor
        self.sampler = self.rb.Sampler(n, batch, SEED)
        rec.dataloader = DataLoader(rec.dataset, batch_sampler=self.sampler)
        self.torch.cuda.synchronize()

    def epoch(self):
        t = time.perf_counter()
        e = self.sampler.epoch
        self.task.run(1)
        self.torch.cuda.synchronize()
        assert self.sampler.epoch == e + 1, (self.sampler.epoch, e)
        return time.perf_counter() - t

    def llk(self, chunk=64):
        torch = self.torch
        fm = self.task.reconstructor.forward_model
        saved = fm.intermediate_variables
        pat = self.task.dataset.patterns
        valid = self.task.dataset.valid_pixel_mask
        n = pat.shape[0]
        tot = None
        with torch.no_grad():
            for s in range(0, n, chunk):
                idx = torch.arange(s, min(n, s + chunk))
                y = fm.forward(idx)
                v = valid.to(y.device)
                part = self.m3ref.llk_sums(y[:, v], pat[idx.to(pat.device)].to(y.device)[:, v])
                tot = part if tot is None else {k: tot[k] + part[k] for k in tot}
        fm.intermediate_variables = saved
        nb = tot["nb_obs"].item()
        return [tot[k].item() / nb for k in LLK]

    def to_target(self, n):
        walls = [self.epoch() for _ in range(n)]
        return sum(walls), walls

    def steady(self, walls):
        walls = list(walls)
        while len(walls) < STEADY_MIN:
            walls.append(self.epoch())
        return statistics.median(walls[2:])


def worker(args):
    """One run of one side, printing the lines `parse` reads."""
    cpu = tp.cpu_times()
    print(f"load0 {tp.load()}")
    case = CASES[args.case]
    side = (Ptyrs if args.side == "ptyrs" else Ptychi)(case, args.module)
    print(f"imports_s {time.perf_counter() - T0}")
    t = time.perf_counter()
    data = cx.Data(args.out)
    print(f"read_s {time.perf_counter() - t}")
    side.start(data, args.epochs)
    print(f"setup_s {time.perf_counter() - t}", flush=True)
    if args.stage == "long":
        print("llk 0 {} {} {}".format(*side.llk()), flush=True)
        if args.side == "ptyrs":
            t = time.perf_counter()
            curve = side.run(args.epochs, True)
            print(f"long_s {time.perf_counter() - t}")  # with its LLK record
            for e, v in enumerate(curve, 1):
                print("llk {} {} {} {}".format(e, *v))
        else:
            wall = 0.0
            for e in range(1, args.epochs + 1):
                wall += side.epoch()
                print("llk {} {} {} {}".format(e, *side.llk()), flush=e % 50 == 0)
            print(f"long_s {wall}")  # without its LLK
    else:
        wall, walls = side.to_target(args.to)
        print(f"wall_s {wall}")
        for e, w in enumerate(walls, 1):
            print(f"epoch_s {e} {w}")
        print("end_llk {} {} {}".format(*side.llk()), flush=True)
        print(f"steady_s {side.steady(walls)}")
    print(f"load {tp.load()}")
    print(f"busy {tp.busy_since(cpu)}")


def parse(log):
    """A worker's log: the last `load` is the one at its end."""
    out = {"llk": [], "epoch_s": []}
    with open(log) as f:
        for line in f:
            w = line.split()
            if len(w) < 2 or w[0] not in LINES:
                continue
            if w[0] == "llk":
                e = int(w[1])
                assert e == len(out["llk"]), (log, e)
                out["llk"].append([float(x) for x in w[2:]])
            elif w[0] == "epoch_s":
                out["epoch_s"].append(float(w[2]))
            elif w[0] == "end_llk":
                out["end_llk"] = [float(x) for x in w[1:]]
            elif w[0] == "load":
                out["load"] = " ".join(w[1:])
            else:
                out[w[0]] = float(w[1])
    return out


def command(args, side, stage, to=None):
    cmd = [sys.executable, "-I", os.path.abspath(__file__), args.out, args.case, stage,
           "--worker", side, "--epochs", str(args.epochs), "--module", args.module]
    if to is not None:
        cmd += ["--to", str(to)]
    return cmd


def env():
    return dict(os.environ, **{k: tp.THREADS for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")})


def first(curve, key, target):
    return next((e for e, v in enumerate(curve) if v[key] <= target), None)


def long_runs(args):
    """Both sides' long runs at once on --gpu; the curves, written to --log/CASE_long.json."""
    e = dict(env(), CUDA_VISIBLE_DEVICES=args.gpu)
    procs = {}
    for side in SIDES:
        log = os.path.join(args.log, f"{args.case}_long_{side}.log")
        with open(log, "w") as f:
            f.write(f"loadavg {tp.load()}\n")
            f.flush()
            procs[side] = (subprocess.Popen(["taskset", "-c", args.cpus,
                                             *command(args, side, "long")],
                                            stdout=f, stderr=subprocess.STDOUT, env=e), log)
    out = {}
    for side, (p, log) in procs.items():
        if p.wait():
            sys.exit(f"{side} long run: rc {p.returncode}, log {log}")
        r = parse(log)
        assert len(r["llk"]) == args.epochs + 1, (log, len(r["llk"]))
        out[side] = r
    summary = {"case": args.case, "epochs": args.epochs, "gpu": args.gpu, "columns": LLK,
               "curves": {s: out[s]["llk"] for s in SIDES},
               "setup_s": {s: out[s]["setup_s"] for s in SIDES},
               "long_s": {s: out[s]["long_s"] for s in SIDES},
               "load": {s: out[s]["load"] for s in SIDES}}
    with open(os.path.join(args.log, f"{args.case}_long.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def targets(final, of):
    return tp.FACTOR * (min(final.values()) if of == "lower" else final[of])


def report_long(args, s):
    curves = s["curves"]
    print(f"{args.case}: {s['epochs']} epochs per side on {s['gpu']}, LLK after each epoch; "
          "their wall time (ptyrs with its LLK record, pty-chi without its LLK) "
          + ", ".join(f"{x} {s['long_s'][x]:.1f} s" for x in SIDES))
    start = [abs(curves["pty-chi"][0][k] / curves["ptyrs"][0][k] - 1) for k in range(len(LLK))]
    print("start LLK, pty-chi from ptyrs: " + ", ".join(
        f"{n} {curves['ptyrs'][0][k]:.6f} ({start[k]:.1e})" for k, n in enumerate(LLK)))
    for k, name in enumerate(LLK):
        final = {x: curves[x][-1][k] for x in SIDES}
        print(f"{name}: final " + ", ".join(f"{x} {final[x]:.6f}" for x in SIDES))
        for x in SIDES:
            c = [v[k] for v in curves[x]]
            low = min(range(len(c)), key=c.__getitem__)
            if c[-1] > c[0]:
                print(f"  {x} ends above its start {c[0]:.6f} (lowest {c[low]:.6f} at epoch "
                      f"{low})")
            elif low < len(c) - 1 and c[-1] > tp.FACTOR * c[low]:
                print(f"  {x} rises after its lowest {c[low]:.6f} at epoch {low}")
        for of in ("lower", *SIDES):
            t = targets(final, of)
            if of != "lower" and final[of] == min(final.values()):
                continue
            hits = {x: first(curves[x], k, t) for x in SIDES}
            print(f"  target {of} {t:.6f}: " + ", ".join(
                f"{x} {'never' if hits[x] is None else f'epoch {hits[x]}'}" for x in SIDES))
            if of == "lower" and None not in hits.values():
                break
    k = LLK.index(args.llk)
    every = max(1, s["epochs"] // 20)
    print(f"{args.llk} LLK by epoch: epoch " + " ".join(SIDES))
    for e in sorted({*range(0, s["epochs"], every), s["epochs"]}):
        print(f"  {e:5d} " + " ".join(f"{curves[x][e][k]:.6f}" for x in SIDES))


def run(args, side, label, to):
    """One timed run under gpu2.sh: its parsed lines, with `clean`, `status` and `log`."""
    log = os.path.join(args.log, f"{args.case}_{args.llk}_{label}_{side}.log")
    cmd = ["timeout", str(args.hold), "taskset", "-c", tp.CPUS, *command(args, side, "time", to)]
    r = subprocess.run([args.gpu2, log, *cmd], capture_output=True, text=True, env=env())
    status = (r.stdout.strip().splitlines() or ["?"])[-1]
    if r.returncode:
        raise RuntimeError(f"{side} {label}: {status}, log {log}")
    out = parse(log)
    out.update(clean=status.startswith("clean"), status=status, log=log)
    return out


def spread(rows, key):
    v = [r[key] for r in rows]
    return f"{statistics.median(v):.3f} s ({min(v):.3f}-{max(v):.3f})"


def timed_runs(args):
    with open(os.path.join(args.log, f"{args.case}_long.json")) as f:
        s = json.load(f)
    k = LLK.index(args.llk)
    final = {x: s["curves"][x][-1][k] for x in SIDES}
    target = args.target if args.target is not None else targets(final, args.of)
    to = {x: first(s["curves"][x], k, target) for x in SIDES}
    of = "given" if args.target is not None else f"1.01 x {args.of} final"
    print(f"{args.case}: {args.llk} LLK target {target:.6f} ({of}), epochs to it in the long "
          "runs " + ", ".join(f"{x} {to[x]}" for x in SIDES) + f" ({cx.load_avg()})", flush=True)
    sides = [x for x in SIDES if to[x]]
    for x in SIDES:
        if not to[x]:
            print(f"{x}: {'never reaches it' if to[x] is None else 'starts at it'}, no runs")
    summary = {"case": args.case, "llk": args.llk, "of": args.of, "target": target,
               "epochs_to_target": to, "runs": {x: [] for x in sides},
               "discarded": {x: [] for x in sides}}
    tries = {x: 0 for x in sides}
    i = 0
    while any(len(summary["runs"][x]) < args.runs and tries[x] < 3 * args.runs for x in sides):
        side = SIDES[(0, 1, 1, 0)[i % 4]]
        i += 1
        if side not in sides:
            continue
        if len(summary["runs"][side]) == args.runs or tries[side] == 3 * args.runs:
            continue
        waited = tp.quiet()
        r = run(args, side, f"run{tries[side]}", to[side])
        tries[side] += 1
        end = r["end_llk"][k]
        r.update(waited_s=waited, counted=r["clean"] and r["busy"] < tp.BUSY,
                 reached=end <= target, long_end=s["curves"][side][to[side]][k])
        print(f"{side:7s} run {tries[side] - 1}: {r['status']}, CPUs {tp.CPUS} "
              f"{100 * r['busy']:.1f}% busy, {'counted' if r['counted'] else 'discarded'}, "
              f"setup {r['setup_s']:.3f} s (files {r['read_s']:.3f} s, imports "
              f"{r['imports_s']:.3f} s), {to[side]} epochs {r['wall_s']:.3f} s"
              + (f" (epoch 1 {r['epoch_s'][0]:.3f} s)" if r["epoch_s"] else "")
              + f", end {args.llk} {end:.6f} (long run {r['long_end']:.6f}), steady epoch "
              f"{1e3 * r['steady_s']:.3f} ms, load {r['load']}", flush=True)
        del r["llk"]
        summary["runs" if r["counted"] else "discarded"][side].append(r)
    print("counted runs: time to target, steady epoch, setup; median (range)")
    for x in sides:
        rows = summary["runs"][x]
        if not rows:
            print(f"{x:7s} none counted")
            continue
        print(f"{x:7s} {len(rows)} counted, {to[x]} epochs: {spread(rows, 'wall_s')}, steady "
              f"{1e3 * statistics.median(r['steady_s'] for r in rows):.3f} ms, setup "
              f"{spread(rows, 'setup_s')}, unreached {sum(not r['reached'] for r in rows)}")
    if all(summary["runs"].get(x) for x in SIDES):
        m = {x: statistics.median(r["wall_s"] for r in summary["runs"][x]) for x in SIDES}
        print(f"pty-chi / ptyrs time to target {m['pty-chi'] / m['ptyrs']:.1f}")
    path = os.path.join(args.log, f"{args.case}_{args.llk}_{args.of}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(path)
    if any(len(summary["runs"][x]) < args.runs for x in sides):
        sys.exit(f"fewer than {args.runs} counted runs on a side")


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("out")
    a.add_argument("case", choices=tuple(CASES))
    a.add_argument("stage", choices=("long", "time"))
    a.add_argument("--epochs", type=int, default=2000, help="the long runs' epochs")
    a.add_argument("--gpu", default=GPU1, help="the long runs' GPU")
    a.add_argument("--cpus", default=tp.CPUS, help="the long runs' CPUs")
    a.add_argument("--llk", choices=LLK, help="default the case's")
    a.add_argument("--of", choices=("lower", *SIDES), default="lower",
                   help="the final the target is 1.01 times")
    a.add_argument("--target", type=float, help="the target LLK in place of --of's")
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--hold", type=int, default=tp.HOLD_S, help="a timed run's limit, s")
    a.add_argument("--module", default=os.path.join(tp.SCRATCH, "py"))
    a.add_argument("--gpu2", default=os.path.join(tp.SCRATCH, "tools", "gpu2.sh"))
    a.add_argument("--log", help="default OUT/ttt-ptychi")
    a.add_argument("--worker", choices=SIDES, dest="side", help=argparse.SUPPRESS)
    a.add_argument("--to", type=int, help=argparse.SUPPRESS)
    args = a.parse_args()
    if args.side:
        return worker(args)
    args.llk = args.llk or CASES[args.case]["llk"]
    args.log = args.log or os.path.join(args.out, "ttt-ptychi")
    os.makedirs(args.log, exist_ok=True)
    if args.stage == "long":
        report_long(args, long_runs(args))
    else:
        timed_runs(args)


if __name__ == "__main__":
    main()
