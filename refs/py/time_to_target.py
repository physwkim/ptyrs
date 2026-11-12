"""Time to target, ptyrs against pty-chi on GPU2 in float32: BH-CG at full batch or rPIE on one M3
dataset from its init as it is, matched settings on both sides, every run through gpu2.sh.

    python -I time_to_target.py DATA_DIR {bh,rpie} [--long 2000] [--runs 3] [--every 50]
        [--name fermat_clean] [--example PATH] [--gpu2 PATH] [--out DIR]

A long run of --long epochs on each side sets the target, 1.01 times the lower of the two final
Gaussian LLKs (PyNX definition, m3ref.py). Each side then runs from the start until --runs runs
that gpu2.sh calls clean have reached it, or --long epochs. Prints every run and the medians of
the epochs to target, the wall time to target and the setup time; logs and a JSON summary go to
--out.

ptyrs is ptyrs-py's gpu_target example (cargo build --release -p ptyrs-py --example gpu_target),
the GPU engine without the binding: its wall time is a new GpuPtycho running the epochs to target
with no LLK pass. pty-chi runs in this script (--worker), one task.run per epoch, its wall time
synchronised around it and evaluate() outside it, as in m3ref.run_epochs. Setup on both sides
is reading the files to the state on the GPU, synchronised; pty-chi's Python imports are timed
apart.

Matched settings as in compare_rpie.py and refs/f64/bh_step.py: pty-chi features ptyrs does not
implement are off (object-probe ambiguity removal, mode orthogonalisation), PATCH shifts at the
PyNX integer corners, consecutive rPIE batches, and gpu_target's parameters read from the pty-chi
options (ptyrs_args). rPIE takes pty-chi's rPIE test settings
(test_2d_ptycho_rpie_synthesisdictlearn.py, SDL probe off): pty-chi's defaults diverge on
fermat_clean on both sides (compare_rpie.py --ptychi-defaults).
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
sys.path[:0] = [HERE, os.path.join(HERE, "..", "ptychi")]
import m3ref  # noqa: E402
from compare_rpie import rpie_params  # noqa: E402

FACTOR = 1.01
SCRATCH = os.path.expanduser("~/work/ptyrs-scratch")
KEYS = {"imports_s", "read_s", "context_s", "setup_s", "llk", "target", "wall_s", "end_llk"}


def ptychi_options(method, nframe):
    import ptychi.api as api

    if method == "bh":
        opts = api.BHOptions()
        r = opts.reconstructor_options
        r.method = "CG"
        r.batch_size = nframe
        opts.probe_options.rho = 0.01  # run_m3.py bh_sweep's choice at full batch
        r.forward_model_options.pad_for_shift = 4  # test_2d_ptycho_bh.py, as bh_step.py
    else:
        opts = api.RPIEOptions()
        r = opts.reconstructor_options
        opts.object_options.step_size = 0.1
        opts.object_options.alpha = 1.0
        opts.probe_options.step_size = 1.0
        opts.probe_options.alpha = 1.0
        r.batch_size = round(0.1 * nframe)
    o, pr = opts.object_options, opts.probe_options
    o.optimizable = pr.optimizable = True
    opts.probe_position_options.optimizable = False
    o.remove_object_probe_ambiguity.enabled = False
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    r.default_dtype = api.Dtypes.FLOAT32
    r.default_device = api.Devices.GPU
    r.allow_nondeterministic_algorithms = False
    r.random_seed = 0
    opts.data_options.fft_shift = True
    opts.data_options.save_data_on_device = True
    return opts


def ptyrs_args(method, opts):
    """gpu_target's method and parameters equal to the pty-chi options `opts`."""
    r, pr = opts.reconstructor_options, opts.probe_options
    if method == "rpie":
        p = rpie_params(opts)
        return ["rpie", *(repr(float(p[k])) for k in
                          ("obj_step", "probe_step", "obj_alpha", "probe_alpha")),
                str(p["batch_size"])]
    assert r.method == "CG" and pr.optimizable and opts.object_options.optimizable
    return [
        "bh",
        str(r.batch_size),  # BHReconstructorOptions.batch_size: one batch of every frame, CG
        repr(pr.rho),  # BHProbeOptions.rho: the probe variable is P / (rho sqrt(ny nx))
        "7",  # seed of the engine's frame order in its one batch (pty-chi's is consecutive)
    ]


def ptychi_task(ds, obj, probe, opts):
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask
    from torch.utils.data import DataLoader

    ny, nx = ds.frame
    opts.data_options.wavelength_m = ds.geometry["wavelength_m"]
    o = opts.object_options
    o.pixel_size_m = ds.pixel
    # Corners as given: the view of corner c starts at c + origin - (n - 1) / 2.
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    # PyNX view corners, the ones the engine preprocesses the positions to.
    corners = (np.array(obj.shape[-2:]) - ds.frame) // 2 + np.round(ds.pos_px)
    task = PtychographyTask(
        opts,
        diffraction_data=ds.counts.astype(np.float32),
        object_data=obj.astype(np.complex64),
        probe_data=ds.probe_to_ptychi(probe)[None].astype(np.complex64),
        probe_position_y_px=corners[:, 0],
        probe_position_x_px=corners[:, 1],
    )
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset, batch_size=rec.batch_size, shuffle=False)
    return task


def worker(args):
    """One pty-chi run, printing gpu_target's lines."""
    import torch
    import ptychi.api  # noqa: F401
    import pynx.ptycho.ptycho  # noqa: F401  m3ref.Dataset's position preprocessing

    print(f"imports_s {time.perf_counter() - T0}")
    t = time.perf_counter()
    ds = m3ref.Dataset(args.name, args.data_dir)
    obj, probe = ds.modes("init")
    print(f"read_s {time.perf_counter() - t}")
    task = ptychi_task(ds, obj, probe, ptychi_options(args.method, ds.nframe))
    torch.cuda.synchronize()
    print(f"setup_s {time.perf_counter() - t}")
    print(f"llk 0 {m3ref.evaluate(task)['gaussian']}", flush=True)
    every = 1 if args.target is not None else args.every
    wall = 0.0
    for e in range(1, args.long + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        task.run(n_epochs=1)
        torch.cuda.synchronize()
        wall += time.perf_counter() - t
        if e % every and e < args.long:
            continue
        v = m3ref.evaluate(task)["gaussian"]
        print(f"llk {e} {v}", flush=True)
        if args.target is not None and v <= args.target:
            print(f"target {e}\nwall_s {wall}\nend_llk {v}")
            return
        if not np.isfinite(v):
            break
    if args.target is not None:
        print("target none")


def parse(log):
    out = {"llk": []}
    with open(log) as f:
        for line in f:
            w = line.split()
            if len(w) < 2 or w[0] not in KEYS:
                continue
            if w[0] == "llk":
                out["llk"].append((int(w[1]), float(w[2])))
            elif w[0] == "target":
                out["epochs"] = None if w[1] == "none" else int(w[1])
            else:
                out[w[0]] = float(w[1])
    return out


def run(args, side, label, target):
    """One run under gpu2.sh: its parsed lines, with `clean` and `log`."""
    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, f"{args.name}_{args.method}_{side}_{label}.log")
    t = "-" if target is None else repr(target)
    if side == "ptyrs":
        cmd = [args.example, args.cxi, args.init, str(args.long), t, *args.ptyrs]
    else:
        cmd = [sys.executable, "-I", os.path.abspath(__file__), args.data_dir, args.method,
               "--worker", "--name", args.name, "--long", str(args.long), "--every",
               str(args.every)] + ([] if target is None else ["--target", t])
    r = subprocess.run([args.gpu2, log, *cmd], capture_output=True, text=True)
    status = (r.stdout.strip().splitlines() or ["?"])[-1]
    if r.returncode:
        raise RuntimeError(f"{side} {label}: {status}, log {log}")
    out = parse(log)
    out.update(clean=status.startswith("clean"), status=status, log=log)
    return out


def median(rows, key):
    v = [r[key] for r in rows]
    return None if None in v else statistics.median(v)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("data_dir")
    a.add_argument("method", choices=("bh", "rpie"))
    a.add_argument("--name", default="fermat_clean")
    a.add_argument("--long", type=int, default=2000)
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--every", type=int, default=50, help="pty-chi long run: epochs per LLK")
    a.add_argument("--example", default=os.path.join(HERE, "..", "..", "target", "release",
                                                     "examples", "gpu_target"))
    a.add_argument("--gpu2", default=os.path.join(SCRATCH, "tools", "gpu2.sh"))
    a.add_argument("--out", default=os.path.join(SCRATCH, "py", "ttt"))
    a.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a.add_argument("--target", type=float, help=argparse.SUPPRESS)
    args = a.parse_args()
    if args.worker:
        return worker(args)

    with open(os.path.join(args.data_dir, args.name + ".json")) as f:
        meta = json.load(f)
    args.cxi, args.init = (os.path.join(args.data_dir, meta["files"][k]) for k in ("cxi", "init"))
    args.ptyrs = ptyrs_args(args.method, ptychi_options(args.method, meta["nframe"]))
    print(f"{args.name} {args.method}: gpu_target {' '.join(args.ptyrs)}", flush=True)

    sides = ("ptyrs", "pty-chi")
    final = {}
    for side in sides:
        r = run(args, side, "long", None)
        e, final[side] = r["llk"][-1]
        print(f"{side:8s} long run: Gaussian LLK {final[side]:.6f} at epoch {e}, "
              f"{r['status']}", flush=True)
    target = FACTOR * min(final.values())
    print(f"target {FACTOR:g} x {min(final.values()):.6f} = {target:.6f}", flush=True)

    summary = {"name": args.name, "method": args.method, "gpu_target": args.ptyrs,
               "long_epochs": args.long, "final_llk": final, "target": target, "runs": {}}
    for side in sides:
        clean = []
        for k in range(3 * args.runs):
            if len(clean) == args.runs:
                break
            r = run(args, side, f"run{k}", target)
            print(f"{side:8s} run {k}: {r['status']}, setup {r['setup_s']:.3f} s (files "
                  f"{r['read_s']:.3f} s), epochs "
                  f"{r['epochs']}, wall {r.get('wall_s', float('nan')):.3f} s"
                  + (f", imports {r['imports_s']:.3f} s" if "imports_s" in r else ""),
                  flush=True)
            r.setdefault("wall_s", None)
            del r["llk"]
            if r["clean"]:
                clean.append(r)
        summary["runs"][side] = clean
        if len(clean) < args.runs:
            sys.exit(f"{side}: {len(clean)} clean runs of {args.runs}")

    print(f"median of {args.runs} clean runs: epochs to target, wall to target, setup (files)")
    for side in sides:
        rows = summary["runs"][side]
        print(f"{side:8s} {median(rows, 'epochs')} epochs, {median(rows, 'wall_s')} s, "
              f"{median(rows, 'setup_s'):.3f} s ({median(rows, 'read_s'):.3f} s)")
    path = os.path.join(args.out, f"{args.name}_{args.method}.json")
    m3ref.dump(summary, path)
    print(path)


if __name__ == "__main__":
    main()
