"""pty-chi BH and Gaussian LSQML on a near-field case: epochs and GPU time to 1% of the truth LLK.

    run_near.py run <out> [--only RUN,...] [--epochs N]   pty-chi on the visible GPU, <out>/<run>.json
    run_near.py ptyrs <out> [--epochs N]                  ptyrs CPU f64, <out>/ptyrs_<spec>.txt
    run_near.py compare <out>                             epochs and seconds to the target

Run as micromamba run -n ptyrs-ref python -I refs/ptychi/run_near.py ..., `run` for timing
through the gpu2.sh lock wrapper, which pins GPU2.

Checks the BH paper's claim (Carlsson, Nikitin, Wendt 2025) of about 10x less compute time than
LSQML on near-field data, for which M3's far-field runs found no gain (DESIGN B7). The paper's
geometry and data are not available locally, so the case is refs/f64/lsqml_near.py's
angular-spectrum geometry (its case 0): 16 frames of 24x24 with two probe modes on a 40x40
object, wavelength 0.1 nm, pixel 50 nm, distance 0.5 mm, a start object of ones and a perturbed
probe, Poisson counts of the truth.

As in M3 (run_m3.py, m3ref.py) the target is the PyNX Gaussian LLK at TARGET times the truth's,
and a run's wall_s is a timing figure only if GpuWatch saw no other process on its GPU. Here the
LLK is evaluated in float64 from each epoch's state, outside the timed task.run(1); pty-chi runs
in its default float32 on the GPU. Every run takes the engine's random_batches(16, batch, SEED,
epoch) (refs/f64/random_batches.py) and pty-chi's options of the parity scripts
(lsqml_near.task with the Gaussian model, refs/f64/bh_near.task), so `ptyrs` runs the same
trajectory in float64 through the engine_near_speed example of ptyrs::engine; LSQML's a_far jitter
(torch.randn) returns zeros, as ptyrs has none. A pty-chi run stops at the target or at
--epochs.

Runs: LSQML at every batch in BATCHES; BH at every batch in BATCHES and probe rho in RHOS, with
CG at 16 (one batch per epoch) and steepest descent below (DESIGN B7). RHOS brackets the best
probe rho at every batch, 10 in a ptyrs scan from 0.001 to 1000; pty-chi's suggested near-field
value 1 reaches the target only at 16, in 78 epochs.
"""

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ref  # noqa: E402
import run_m3  # noqa: E402

import ptychi.api as api  # noqa: E402
import ptychi.utils as utils  # noqa: E402
from ptychi.api.task import PtychographyTask  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(m3ref.HERE))
SEED = 7
TARGET = 1.01
EPOCHS = 300
RHOS = (5, 7, 10, 14, 20)


def load_lsqml_near():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "lsqml_near", os.path.join(ROOT, "refs", "f64", "lsqml_near.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ln = load_lsqml_near()
NFRAME = ln.GRID**2
BATCHES = (1, 2, 4, 8, NFRAME)


def runs():
    """name -> (algorithm, batch, probe rho or None, ptyrs spec)."""
    out = {f"lsqml_b{b}": ("lsqml", b, None, f"lsqml:{b}") for b in BATCHES}
    for b in BATCHES:
        algo = "bh_cg" if b == NFRAME else "bh_gd"
        for r in RHOS:
            out[f"{algo}_b{b}_rho{r:g}"] = (algo, b, r, f"bh:{b}:{r:g}")
    return out


class Case:
    """The start, counts, float64 propagator and truth LLK of lsqml_near.py's case 0."""

    def __init__(self):
        torch.set_default_dtype(torch.float64)
        utils.set_default_complex_dtype(torch.complex128)
        truth_obj, truth_probe, self.probe, self.corners, rng = ln.make_state()
        self.obj = np.ones(ln.OBJ, complex)
        self.g = ln.Geometry(*ln.GEOMETRIES[0])
        self.counts = ln.counts_of(self.g, truth_obj, truth_probe, self.corners, rng)
        self.truth = self.llk(truth_obj, truth_probe)
        self.truth_state = truth_obj, truth_probe

    def llk(self, obj, probe):
        """m3ref.evaluate's PyNX LLK of every frame, in float64."""
        icalc = ln.icalc(self.g, obj.astype(complex), probe.astype(complex), self.corners)
        tot = m3ref.llk_sums(torch.from_numpy(icalc), torch.from_numpy(self.counts))
        tot = {k: v.item() for k, v in tot.items()}
        nb = tot["nb_obs"]
        out = {k: tot[k] / nb for k in ("poisson", "gaussian", "euclidian")}
        out["poisson_floor_share"] = tot["poisson_floor"] / tot["poisson"]
        return out


def build(case, algo, batch, rho):
    """pty-chi's task in float32 on the GPU, with lsqml_near.task's options (Gaussian model) or
    bh_near.task's."""
    ny, nx = ln.SHAPE
    g = case.g
    if algo == "lsqml":
        o = api.LSQMLOptions()
        o.object_options.build_preconditioner_with_all_modes = True
        r = o.reconstructor_options
        r.noise_model = api.NoiseModels.GAUSSIAN
        r.forward_model_options.subpixel_shift_all_probe_modes = True
        r.solve_step_sizes_only_using_first_probe_mode = False
        r.rescale_probe_intensity_in_first_epoch = False
    else:
        o = api.BHOptions()
        o.reconstructor_options.method = "CG" if algo == "bh_cg" else "GD"
        o.probe_options.rho = rho
    o.data_options.wavelength_m = g.wavelength
    o.data_options.fft_shift = False
    o.data_options.free_space_propagation_distance_m = g.distance
    o.data_options.save_data_on_device = True
    ob = o.object_options
    ob.optimizable = True
    ob.pixel_size_m = g.pixel
    ob.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    ob.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    ob.remove_object_probe_ambiguity.enabled = False
    pr = o.probe_options
    pr.optimizable = True
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    o.probe_position_options.optimizable = False
    r = o.reconstructor_options
    r.batch_size = batch
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    task = PtychographyTask(
        o,
        diffraction_data=case.counts.astype(np.float32),
        object_data=case.obj[None].astype(np.complex64),
        probe_data=(case.probe / np.sqrt(g.k))[None].astype(np.complex64),
        probe_position_y_px=case.corners[:, 0].astype(np.float64),
        probe_position_x_px=case.corners[:, 1].astype(np.float64),
    )
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset,
                                batch_sampler=ln.rb.Sampler(len(case.corners), batch, SEED))
    return task


def run(case, out, name, algo, batch, rho, epochs):
    task = build(case, algo, batch, rho)
    goal = TARGET * case.truth["gaussian"]
    curve = [dict(epoch=0, wall_s=0.0, **case.llk(*ln.state_of(task, case.g)))]
    wall = 0.0
    with run_m3.GpuWatch() as watch, ln.ZeroRandn() as zero:
        for ep in range(1, epochs + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            task.run(1)
            torch.cuda.synchronize()
            wall += time.perf_counter() - t0
            curve.append(dict(epoch=ep, wall_s=wall, **case.llk(*ln.state_of(task, case.g))))
            if curve[-1]["gaussian"] <= goal or not np.isfinite(curve[-1]["gaussian"]):
                break
    last = curve[-1]
    print(f"{name}: {last['epoch']} epochs {last['wall_s']:.3f} s gaussian "
          f"{last['gaussian'] / case.truth['gaussian']:.4f}x truth, randn {zero.calls}, "
          f"shared {sorted(watch.seen)}", flush=True)
    m3ref.dump({
        "run": name,
        "algorithm": algo,
        "batch_size": batch,
        "probe_rho": rho,
        "epochs": epochs,
        "target": TARGET,
        "options": json.loads(json.dumps(task.get_options_as_dict(), default=str)),
        "environment": run_m3.environment(),
        "gpu_shared_with": sorted(watch.seen),
        "gpu_watch_period_s": run_m3.GpuWatch.PERIOD_S,
        "randn_calls": zero.calls,
        "llk_definition": "PyNX normalised LLK (m3ref.py) in float64 of each epoch's state",
        "llk_truth": case.truth,
        "time_to_target": m3ref.time_to_target(curve, case.truth),
        "curve": curve,
    }, os.path.join(out, name + ".json"))
    del task
    torch.cuda.empty_cache()


def first(gaussian, goal):
    """The first epoch at or below `goal`, or None."""
    return next((e for e, v in enumerate(gaussian) if v <= goal), None)


def compare(out):
    truth = Case().truth["gaussian"]
    goal = TARGET * truth
    rows = {}
    for name, (algo, batch, rho, spec) in runs().items():
        path = os.path.join(out, name + ".json")
        mine = os.path.join(out, f"ptyrs_{spec.replace(':', '_')}.txt")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        if d["llk_truth"]["gaussian"] != truth:
            raise ValueError(f"{name}: truth LLK {d['llk_truth']['gaussian']} != {truth}")
        curve = d["curve"]
        hit = first([r["gaussian"] for r in curve], goal)
        row = {"algorithm": algo, "batch": batch, "probe_rho": rho,
               "epochs": hit, "wall_s": None if hit is None else curve[hit]["wall_s"],
               "wall_per_epoch_s": curve[-1]["wall_s"] / curve[-1]["epoch"],
               "clean": not d["gpu_shared_with"], "ptyrs_epochs": None}
        if os.path.exists(mine):
            t = np.loadtxt(mine)
            row["ptyrs_epochs"] = first(t[:, 2], goal)
            n = min(len(curve), len(t))
            ours = np.array([r["gaussian"] for r in curve[:n]])
            row["max_rel_llk_diff"] = float(np.max(np.abs(ours - t[:n, 2]) / t[:n, 2]))
        rows[name] = row
    print(f"truth Gaussian LLK {truth:.6f}, target {TARGET} x truth")
    print(f"{'run':22s} {'epochs':>6s} {'ptyrs':>6s} {'s':>8s} {'s/epoch':>8s} clean  llk diff")
    for name, r in rows.items():
        print(f"{name:22s} {str(r['epochs']):>6s} {str(r['ptyrs_epochs']):>6s} "
              f"{r['wall_s'] or float('nan'):8.3f} {r['wall_per_epoch_s']:8.4f} "
              f"{str(r['clean']):5s}  {r.get('max_rel_llk_diff', float('nan')):.2e}")
    best = {}
    for family, pick in (("lsqml", lambda r: r["algorithm"] == "lsqml"),
                         ("bh_minibatch", lambda r: r["algorithm"] == "bh_gd"),
                         ("bh_cg_full", lambda r: r["algorithm"] == "bh_cg")):
        hit = {n: r for n, r in rows.items() if pick(r) and r["epochs"] is not None}
        if hit:
            best[family] = {
                "by_epochs": min(hit, key=lambda n: (hit[n]["epochs"], hit[n]["wall_s"])),
                "by_wall_s": min(hit, key=lambda n: hit[n]["wall_s"]),
            }
    print("best (fewest epochs; least seconds):")
    for family, b in best.items():
        e, w = rows[b["by_epochs"]], rows[b["by_wall_s"]]
        print(f"  {family:13s} {b['by_epochs']} {e['epochs']} epochs {e['wall_s']:.3f} s; "
              f"{b['by_wall_s']} {w['epochs']} epochs {w['wall_s']:.3f} s")
    if "lsqml" in best:
        ref = best["lsqml"]
        for family in ("bh_minibatch", "bh_cg_full"):
            if family in best:
                le, be = rows[ref["by_epochs"]], rows[best[family]["by_epochs"]]
                lw, bw = rows[ref["by_wall_s"]], rows[best[family]["by_wall_s"]]
                print(f"  LSQML / {family}: epochs {le['epochs'] / be['epochs']:.2f}x, "
                      f"seconds {lw['wall_s'] / bw['wall_s']:.2f}x")
    m3ref.dump({"truth_gaussian": truth, "target": TARGET, "runs": rows, "best": best},
               os.path.join(out, "summary.json"))


def main():
    logging.getLogger().setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--only", default=None)
    r.add_argument("--epochs", type=int, default=EPOCHS)
    p = sub.add_parser("ptyrs")
    p.add_argument("out")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    c = sub.add_parser("compare")
    c.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    all_runs = runs()
    if a.cmd == "run":
        case = Case()
        for name in (all_runs if a.only is None else a.only.split(",")):
            algo, batch, rho, _ = all_runs[name]
            run(case, a.out, name, algo, batch, rho, a.epochs)
        if case.llk(*case.truth_state) != case.truth:
            raise RuntimeError("the float64 LLK changed under pty-chi's float32 defaults")
    elif a.cmd == "ptyrs":
        specs = sorted({spec for *_, spec in all_runs.values()})
        subprocess.run(["cargo", "run", "--release", "-p", "ptyrs", "--example",
                        "engine_near_speed", "--", os.path.abspath(a.out), str(a.epochs), *specs],
                       cwd=ROOT, check=True)
    else:
        compare(a.out)


if __name__ == "__main__":
    main()
