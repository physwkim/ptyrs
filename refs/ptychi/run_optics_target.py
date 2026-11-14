"""pty-chi LSQML (Poisson) epochs and GPU time to 1.01x the truth LLK on the cases of ptyrs'
engine_optics_target example: a 3-slice object to the far field (ms) and one slice to the near
field by angular spectrum (near).

    run_optics_target.py count <dir> <case> <nepoch> <batch>...
    run_optics_target.py time <dir> <case> <batch>:<epochs>...

Run as micromamba run -n ptyrs-ref python -I refs/ptychi/run_optics_target.py ..., on the GPU
CUDA_VISIBLE_DEVICES names, after `engine_optics_target gen <dir>`. The task is pty-chi's
default float32 on the GPU with the options of refs/f64/lsqml_multislice.task (ms) and
refs/f64/lsqml_near.task (near), the data on the device, the start and counts the example's
(<case>_start.h5, <case>.cxi), the probe over sqrt(k), k the example's exit scale, and each
view at the example's integer corner (<case>.txt). ms's exit is stock pty-chi's far field at
infinite distance (fft_shift True), which is ptyrs' Optics::free_space at that distance: plain
Fourier, no object-plane phase. Batches are the engine's random_batches(nframe, batch, SEED,
epoch) (refs/f64/random_batches.py) and LSQML's a_far jitter (torch.randn) returns zeros, as
ptyrs has none.

`count` runs each batch until the m3ref.evaluate Poisson LLK of the state (f32 forward, f64
sums) is at TARGET times the example's f64 truth LLK, or for <nepoch> epochs, and writes the
curve to <dir>/ptychi_<case>_b<batch>.json. It first evaluates the truth state, so the two
forward models are held to each other. `time` runs WARMUP epochs of each batch untimed, as the
example does, then times <epochs> epochs of a new task one epoch at a time, synchronised and
with no LLK, and evaluates the end state. It prints the first epoch of both apart from the rest:
the warm-up's holds pty-chi's setup, once per process and batch.
"""

import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ref  # noqa: E402
import run_m3  # noqa: E402

import ptychi.api as api  # noqa: E402
from ptychi.api.task import PtychographyTask  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

SEED = 7
TARGET = 1.01
WARMUP = 12


def sibling(name):
    import importlib.util
    sys.dont_write_bytecode = True
    path = os.path.join(os.path.dirname(m3ref.HERE), "f64", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ln = sibling("lsqml_near")


def load_average():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


class Case:
    """<dir>/<case>.{cxi,txt} and the start and truth states, slices as the object's modes."""

    def __init__(self, path, name):
        self.name = name
        meta = {}
        self.corners = []
        with open(os.path.join(path, name + ".txt")) as f:
            for line in f:
                key, *vals = line.split()
                if key == "corner":
                    self.corners.append([int(vals[1]), int(vals[2])])
                else:
                    meta[key] = [float(v) for v in vals]
        self.corners = np.array(self.corners)
        self.wavelength, = meta["wavelength"]
        self.pixel, = meta["pixel"]
        self.distance, = meta["distance"]
        self.spacings = meta["spacings"]
        self.k, = meta["scale"]
        self.truth = dict(zip(("poisson", "gaussian", "euclidian"), meta["truth_llk"]))
        self.start_llk = dict(zip(("poisson", "gaussian", "euclidian"), meta["start_llk"]))
        with h5py.File(os.path.join(path, name + ".cxi"), "r") as f:
            self.counts = f["entry_1/data_1/data"][()].astype(np.float32)
        self.start = m3ref.load_modes(os.path.join(path, name + "_start.h5"))
        self.truth_state = m3ref.load_modes(os.path.join(path, name + "_truth.h5"))

    def task(self, batch, state=None):
        obj, probe = self.start if state is None else state
        ny, nx = probe.shape[-2:]
        opts = api.LSQMLOptions()
        opts.data_options.wavelength_m = self.wavelength
        opts.data_options.fft_shift = np.isinf(self.distance)
        opts.data_options.free_space_propagation_distance_m = self.distance
        opts.data_options.save_data_on_device = True
        o = opts.object_options
        o.optimizable = True
        o.pixel_size_m = self.pixel
        if self.spacings:
            o.slice_spacings_m = list(self.spacings)
        o.build_preconditioner_with_all_modes = True
        o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
        o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
        o.remove_object_probe_ambiguity.enabled = False
        pr = opts.probe_options
        pr.optimizable = True
        pr.orthogonalize_incoherent_modes.enabled = False
        pr.orthogonalize_opr_modes.enabled = False
        opts.probe_position_options.optimizable = False
        r = opts.reconstructor_options
        r.noise_model = api.NoiseModels.POISSON
        r.forward_model_options.subpixel_shift_all_probe_modes = True
        r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
        r.solve_step_sizes_only_using_first_probe_mode = False
        if self.spacings:
            r.multislice_solve_obj_prb_step_size_jointly = True
        r.rescale_probe_intensity_in_first_epoch = False
        r.batch_size = batch
        r.allow_nondeterministic_algorithms = False
        t = PtychographyTask(
            opts,
            diffraction_data=self.counts,
            object_data=obj.astype(np.complex64),
            probe_data=(probe / np.sqrt(self.k))[None].astype(np.complex64),
            probe_position_y_px=self.corners[:, 0].astype(np.float64),
            probe_position_x_px=self.corners[:, 1].astype(np.float64),
        )
        rec = t.reconstructor
        rec.dataloader = DataLoader(rec.dataset,
                                    batch_sampler=ln.rb.Sampler(len(self.corners), batch, SEED))
        return t


def ratio(got, want):
    return {k: got[k] / want[k] for k in ("poisson", "gaussian", "euclidian")}


def count(case, out, nepoch, batches):
    t = case.task(batches[0], case.truth_state)
    print(f"{case.name}: pty-chi f32 / ptyrs f64 LLK, truth {ratio(m3ref.evaluate(t), case.truth)}",
          flush=True)
    del t
    goal = TARGET * case.truth["poisson"]
    for batch in batches:
        t = case.task(batch)
        curve = [dict(epoch=0, **m3ref.evaluate(t))]
        print(f"{case.name} b{batch}: start {ratio(curve[0], case.start_llk)} of ptyrs f64",
              flush=True)
        with run_m3.GpuWatch() as watch, ln.ZeroRandn() as zero:
            for ep in range(1, nepoch + 1):
                t.run(1)
                curve.append(dict(epoch=ep, **m3ref.evaluate(t)))
                if curve[-1]["poisson"] <= goal or not np.isfinite(curve[-1]["gaussian"]):
                    break
        last = curve[-1]
        hit = last["epoch"] if last["poisson"] <= goal else None
        print(f"count {case.name} batch {batch}: target epoch {hit}, end "
              f"{last['poisson'] / case.truth['poisson']:.4f}x truth, poisson floor share "
              f"{max(r['poisson_floor_share'] for r in curve):.2e}, randn {zero.calls}",
              flush=True)
        m3ref.dump({
            "case": case.name,
            "batch_size": batch,
            "target": TARGET,
            "target_epoch": hit,
            "llk_truth_f64": case.truth,
            "randn_calls": zero.calls,
            "gpu_shared_with": sorted(watch.seen),
            "environment": run_m3.environment(),
            "options": json.loads(json.dumps(t.get_options_as_dict(), default=str)),
            "curve": curve,
        }, os.path.join(out, f"ptychi_{case.name}_b{batch}.json"))
        del t
        torch.cuda.empty_cache()


def timed(case, runs):
    for batch, nepoch in runs:
        t = case.task(batch)
        with ln.ZeroRandn():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t.run(1)
            torch.cuda.synchronize()
            cold = time.perf_counter() - t0
            t.run(WARMUP - 1)
        del t
        torch.cuda.empty_cache()
        load = load_average()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t = case.task(batch)
        torch.cuda.synchronize()
        build = time.perf_counter() - t0
        walls = []
        with run_m3.GpuWatch() as watch, ln.ZeroRandn():
            for _ in range(nepoch):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                t.run(1)
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - t0)
        end = m3ref.evaluate(t)["poisson"] / case.truth["poisson"]
        rest = walls[1:]
        print(f"time {case.name} batch {batch}: warm-up epoch 1 {cold:.3f} s, build {build:.3f} s, "
              f"epoch 1 {walls[0]:.3f} s, "
              f"epochs 2..{nepoch} {1e3 * sum(rest) / max(len(rest), 1):.3f} ms/epoch, "
              f"{nepoch} epochs {sum(walls):.3f} s, end {end:.4f}x truth, load {load} to "
              f"{load_average()}, shared {sorted(watch.seen)}, cumulative "
              f"{' '.join(f'{w:.3f}' for w in np.cumsum(walls))}", flush=True)
        del t
        torch.cuda.empty_cache()


def main():
    usage = __doc__.split("\n\n")[1]
    args = sys.argv[1:]
    if len(args) < 4 + (args[:1] == ["count"]) or args[0] not in ("count", "time"):
        sys.exit(usage)
    case = Case(args[1], args[2])
    if args[0] == "count":
        count(case, args[1], int(args[3]), [int(b) for b in args[4:]])
    else:
        timed(case, [tuple(int(v) for v in r.split(":")) for r in args[3:]])


if __name__ == "__main__":
    main()
