"""pty-chi LSQML and BH on the M3 datasets, one feature per run, PyNX LLK per epoch to JSON.

    run_m3.py sweep --noise poisson [--batches 16,32,...]  LSQML batch-size sweep, fermat_clean
    run_m3.py bh_sweep [--rho R] [--only LABEL,...]        BH probe-rho sweep at full batch, or
                                                           with --rho, mini-batch CG/GD at rho R
    run_m3.py features --batch B --bh-rho R [--only ...] [--algo A]
                                                           one run per feature and algorithm
    run_m3.py one FEATURE ALGO [--set PATH=JSON ...]       trial run into out/scratch/

Run as CUDA_VISIBLE_DEVICES=GPU-<uuid> micromamba run -n ptyrs-ref python refs/ptychi/run_m3.py
...; select the GPU by UUID, since CUDA indices shift when a GPU drops off the bus.

Each run writes out/<run>.json: the pty-chi options, the generator's truth LLK (PyNX
definition), time_to_target, peak GPU memory, the other processes seen on the GPU during the run
(GpuWatch; wall_s is a timing figure only when that list is empty), and the curve (epoch,
wall_s, logged_loss, poisson, gaussian, euclidian, poisson_floor_share, native_*,
position_rms_px where the data has position errors, intensity_rms_rel where it has intensity
variation). Options that differ from pty-chi defaults are those of pty-chi's tests for the same
feature, except the LSQML batch size and BH probe rho, which the sweeps here choose, and LSQML
position correction (LSQML_POSITIONS). sweep and bh_sweep take --prefix: timing_ names the
timing pass of the chosen settings on GPU2 (via gpu2.sh).
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ref  # noqa: E402

import ptychi  # noqa: E402
import ptychi.api as api  # noqa: E402
from ptychi.api.task import PtychographyTask  # noqa: E402
from ptychi.utils import (  # noqa: E402
    add_additional_opr_probe_modes_to_probe,
    generate_initial_opr_mode_weights,
)

OUT = os.path.join(m3ref.HERE, "out")
FULL = 1024  # frames per dataset: BH-CG at one batch per epoch

# LSQML position correction for "positions". pty-chi's test settings (positions_test) move
# fermat_poserr's positions by ~1e-3 px/epoch although the update direction is right (cosine 0.8
# with the true error): the Fourier derivative of the object patch is dominated by its
# high-frequency content (50x the energy of the sigma-0.33 Gaussian derivative), and
# ProbePositions.step_optimizer takes the MAD clip over all positions, of which only the batch
# moved, so a 64-of-1024 batch is clipped near its mean step instead of 10 MADs. At 40 epochs these
# settings leave 1.00 px RMS of the initial 2.76 px; the test settings leave 2.72 px.
LSQML_POSITIONS = {
    "probe_position_options.step_size": 3.0,
    "probe_position_options.correction_options.differentiation_method":
        api.ImageGradientMethods.GAUSSIAN,
    "probe_position_options.correction_options.clip_update_magnitude_by_mad": False,
}

# Feature -> dataset and pty-chi switches. "*_off" runs the same data without the feature model,
# the floor the feature has to beat. BH has no OPR, so opr and intensity are LSQML-only.
# `bh_batch`: BH-CG at full batch needs 15.8 GiB (2 modes) to 19.5 GiB (with positions); with
# raster_fly's 6 modes it exceeds 31 GiB, and 512 needs 20.7 GiB.
FEATURES = {
    "clean_raster": dict(dataset="raster_clean"),
    "clean_fermat": dict(dataset="fermat_clean"),
    "positions": dict(dataset="fermat_poserr", positions=True, lsqml_overrides=LSQML_POSITIONS),
    "positions_test": dict(dataset="fermat_poserr", positions=True, algos=("lsqml",)),
    "positions_off": dict(dataset="fermat_poserr"),
    "fly_modes": dict(dataset="raster_fly", bh_batch=512),
    "opr": dict(dataset="fermat_opr", opr=True, algos=("lsqml",)),
    "opr_off": dict(dataset="fermat_opr"),
    "intensity": dict(dataset="fermat_intvar", intensity=True, algos=("lsqml",)),
    "intensity_off": dict(dataset="fermat_intvar"),
    # pty-chi has no background model.
    "background_off": dict(dataset="fermat_bg"),
}

# BH sweep on fermat_clean. pty-chi's tests use CG, probe rho 0.1, batch 96; bh.py recommends GD
# for mini-batches and CG for one batch.
BH_RHOS = (1.0, 0.1, 0.01, 0.001)
BH_MINIBATCH = (("bh_cg", 512), ("bh_gd", 512), ("bh_cg", 96), ("bh_gd", 96))


def bh_sweep_runs(rho):
    """label -> (algo, batch, probe rho): the rho sweep at full batch, or with `rho`, the
    mini-batch variants at that rho."""
    if rho is None:
        return {f"cg_rho{r:g}": ("bh_cg", FULL, r) for r in BH_RHOS}
    return {f"{a[3:]}_rho{rho:g}_b{b}": (a, b, rho) for a, b in BH_MINIBATCH}


def build_task(ds, algo, batch, state="init", positions=False, opr=False, intensity=False,
               noise="gaussian", seed=0, overrides=None, nmode=None):
    """`overrides` maps dotted option paths to values, applied last; `nmode` truncates the
    initial probe modes."""
    obj, probe = ds.modes(state)
    probe = ds.probe_to_ptychi(probe[:nmode])[None]  # [n_opr=1, nmode, ny, nx]
    weights = None
    torch.manual_seed(seed)  # add_additional_opr_probe_modes_to_probe draws random eigenmodes
    if algo == "lsqml":
        o = api.LSQMLOptions()
        o.reconstructor_options.noise_model = {
            "poisson": api.NoiseModels.POISSON, "gaussian": api.NoiseModels.GAUSSIAN}[noise]
        o.object_options.build_preconditioner_with_all_modes = True
        # Two incoherent modes carry power here; the defaults shift and solve steps on mode 0
        # only, and LSQML-Poisson then diverges on raster_fly's 6-mode init by epoch 2.
        o.reconstructor_options.forward_model_options.subpixel_shift_all_probe_modes = True
        o.reconstructor_options.solve_step_sizes_only_using_first_probe_mode = False
        if positions:
            # test_2d_ptycho_lsqml.py::test_2d_ptycho_lsqml_poscorr
            pc = o.probe_position_options.correction_options
            pc.update_magnitude_limit = 5.0
            pc.correction_type = api.PositionCorrectionTypes.GRADIENT
            pc.differentiation_method = api.ImageGradientMethods.FOURIER_DIFFERENTIATION
        if opr or intensity:
            # One eigenmode on top of the main mode (test_2d_ptycho_lsqml_opr uses 3). Intensity
            # variation needs it too: with one OPR mode ForwardModel.get_unique_probes ignores
            # the per-frame weights, so the intensity update is computed but never applied.
            # Without opr its weights stay at ~1e-6, as in test_2d_ptycho_opt_plan.py.
            probe = add_additional_opr_probe_modes_to_probe(torch.as_tensor(probe), 1).numpy()
            weights = generate_initial_opr_mode_weights(ds.nframe, 2).cpu().numpy()
            o.opr_mode_weight_options.optimizable = True
        if intensity:
            # Assignments are validated one at a time: one of the two must stay True.
            o.opr_mode_weight_options.optimize_intensity_variation = True
            o.opr_mode_weight_options.optimize_eigenmode_weights = opr
    elif algo in ("bh_cg", "bh_gd"):
        if opr or intensity:
            raise ValueError("pty-chi BH has no OPR (bh.py check_inputs)")
        if noise != "gaussian":
            raise ValueError("pty-chi BH has only the amplitude (Gaussian) model")
        o = api.BHOptions()
        o.reconstructor_options.method = "CG" if algo == "bh_cg" else "GD"
        # test_2d_ptycho_bh.py
        o.probe_options.rho = 0.1
        o.probe_position_options.rho = 2
        o.reconstructor_options.forward_model_options.pad_for_shift = 4
    else:
        raise ValueError(algo)
    o.data_options.wavelength_m = ds.geometry["wavelength_m"]
    o.data_options.save_data_on_device = True
    o.object_options.pixel_size_m = ds.pixel
    o.object_options.optimizable = True
    o.probe_options.optimizable = True
    o.probe_position_options.optimizable = positions
    o.reconstructor_options.batch_size = batch
    o.reconstructor_options.random_seed = seed
    o.reconstructor_options.allow_nondeterministic_algorithms = False
    for path, value in (overrides or {}).items():
        *parents, leaf = path.split(".")
        target = o
        for name in parents:
            target = getattr(target, name)
        if not hasattr(target, leaf):
            raise AttributeError(f"no option {path}")
        setattr(target, leaf, value)
    task = PtychographyTask(
        o,
        diffraction_data=ds.counts.astype(np.float32),
        object_data=obj.astype(np.complex64),
        probe_data=probe.astype(np.complex64),
        probe_position_x_px=ds.pos_px[:, 1],
        probe_position_y_px=ds.pos_px[:, 0],
        **({} if weights is None else {"opr_mode_weights_data": weights}),
    )
    return task, o


def position_error(ds):
    """Per-row column: RMS distance of pty-chi's positions ([y, x]) from the truth, after
    removing the mean offset (a global shift is not observable)."""
    truth = np.asarray(ds.meta["positions"]["true_in_nominal_frame_px"])

    def column(task):
        d = task.get_data_to_cpu("probe_positions", as_numpy=True) - truth
        d -= d.mean(axis=0)
        return {"position_rms_px": float(np.sqrt((d**2).sum(axis=1).mean()))}

    return column


def intensity_error(ds):
    """Per-row column: RMS of pty-chi's per-frame intensity (main OPR weight squared) minus the
    generator's flux, both scaled to mean 1 (the mean is degenerate with the probe norm). pty-chi
    scales incoherent mode 0 only; the generator scales all modes."""
    truth = np.asarray(ds.meta["intensity_scale"])
    truth = truth / truth.mean()

    def column(task):
        w = task.get_data_to_cpu("opr_mode_weights", as_numpy=True)[:, 0].astype(np.float64) ** 2
        return {"intensity_rms_rel": float(np.sqrt(((w / w.mean() - truth) ** 2).mean()))}

    return column


def gpu_uuid():
    return "GPU-" + str(torch.cuda.get_device_properties(0).uuid)


def other_gpu_processes(gpu):
    """Other compute processes on GPU `gpu` (UUID)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    rows = [[f.strip() for f in line.split(",")] for line in out.splitlines() if line.strip()]
    return [name for uuid, pid, name in rows if uuid == gpu and pid != str(os.getpid())]


class GpuWatch:
    """Other compute processes seen on this run's GPU, sampled every PERIOD_S from a thread for
    the whole run (a check at the start and end only misses jobs that come and go in between):
    wall_s is a timing figure only if none are seen."""

    PERIOD_S = 1.0

    def __enter__(self):
        self.gpu = gpu_uuid()
        self.seen = set(other_gpu_processes(self.gpu))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(self.PERIOD_S):
            self.seen.update(other_gpu_processes(self.gpu))

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.seen.update(other_gpu_processes(self.gpu))


def environment():
    return {
        "ptychi": getattr(ptychi, "__version__", None),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": gpu_uuid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": platform.node(),
    }


def run(run_name, dataset, algo, batch, epochs, **kw):
    ds = m3ref.Dataset(dataset)
    task, _ = build_task(ds, algo, batch, **kw)
    truth = ds.meta["llk_truth"]
    extra = (position_error(ds) if ds.meta["positions"]["error_sigma_px"] > 0
             else intensity_error(ds) if ds.meta["intensity_scale"] else None)
    torch.cuda.reset_peak_memory_stats()

    def log(r):
        if r["epoch"] > 5 and r["epoch"] % 25:
            return
        pos = f" pos_rms {r['position_rms_px']:.3f}" if "position_rms_px" in r else ""
        pos += f" int_rms {r['intensity_rms_rel']:.4f}" if "intensity_rms_rel" in r else ""
        print(f"{run_name} ep {r['epoch']:4d} t {r['wall_s']:7.2f}s poisson {r['poisson']:.5f} "
              f"gaussian {r['gaussian']:.5f} (truth {truth['poisson']:.5f} "
              f"{truth['gaussian']:.5f}) floor {r['poisson_floor_share']:.3f}{pos}", flush=True)

    with GpuWatch() as watch:
        curve = m3ref.run_epochs(task, epochs, log, extra)
    result = {
        "run": run_name,
        "dataset": dataset,
        "algorithm": algo,
        "batch_size": batch,
        "epochs": epochs,
        "settings": kw,
        "options": json.loads(json.dumps(task.get_options_as_dict(), default=str)),
        "environment": environment(),
        "gpu_shared_with": sorted(watch.seen),
        "gpu_watch_period_s": GpuWatch.PERIOD_S,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "llk_definition": "PyNX normalised LLK (see m3ref.py); icalc from pty-chi's forward model",
        "llk_truth": truth,
        "time_to_target": m3ref.time_to_target(curve, truth),
        "curve": curve,
    }
    m3ref.dump(result, os.path.join(OUT, run_name + ".json"))
    del task
    torch.cuda.empty_cache()
    return result


def main():
    logging.getLogger().setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--noise", required=True, choices=("poisson", "gaussian"))
    s.add_argument("--batches", default="16,32,64,128,256,512,1024")
    s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--prefix", default="", help="run-name prefix, e.g. timing_ for a GPU2 pass")
    b = sub.add_parser("bh_sweep")
    b.add_argument("--rho", type=float, default=None,
                   help="probe rho of the mini-batch runs; omit for the rho sweep")
    b.add_argument("--only", default=None)
    b.add_argument("--epochs", type=int, default=300)
    b.add_argument("--prefix", default="")
    f = sub.add_parser("features")
    f.add_argument("--batch", type=int, required=True, help="LSQML batch size")
    f.add_argument("--bh-rho", type=float, required=True, help="BH-CG probe rho")
    f.add_argument("--only", default=None)
    f.add_argument("--algo", default=None, choices=("lsqml", "bh_cg"))
    f.add_argument("--epochs", type=int, default=300)
    one = sub.add_parser("one")
    one.add_argument("feature")
    one.add_argument("algo")
    one.add_argument("--batch", type=int, default=64)
    one.add_argument("--epochs", type=int, default=10)
    one.add_argument("--noise", default="gaussian")
    one.add_argument("--set", action="append", default=[], metavar="PATH=JSON",
                     help="option override, e.g. probe_options.rho=0.01")
    a = ap.parse_args()

    if a.cmd == "sweep":
        for batch in map(int, a.batches.split(",")):
            run(f"{a.prefix}sweep_lsqml_{a.noise}_b{batch}", "fermat_clean", "lsqml", batch,
                a.epochs, noise=a.noise)
    elif a.cmd == "bh_sweep":
        runs = bh_sweep_runs(a.rho)
        for label in (runs if a.only is None else a.only.split(",")):
            algo, batch, rho = runs[label]
            run(f"{a.prefix}bh_sweep_{label}", "fermat_clean", algo, batch, a.epochs,
                overrides={"probe_options.rho": rho})
    elif a.cmd == "features":
        for name in (FEATURES if a.only is None else a.only.split(",")):
            spec = dict(FEATURES[name])
            dataset = spec.pop("dataset")
            lsqml_overrides = spec.pop("lsqml_overrides", {})
            bh_batch = spec.pop("bh_batch", FULL)
            for algo in spec.pop("algos", ("lsqml", "bh_cg")):
                if a.algo not in (None, algo):
                    continue
                if algo == "lsqml":
                    run(f"{name}_lsqml_poisson", dataset, algo, a.batch, a.epochs,
                        noise="poisson", overrides=lsqml_overrides, **spec)
                else:
                    run(f"{name}_bh_cg", dataset, algo, bh_batch, a.epochs,
                        overrides={"probe_options.rho": a.bh_rho}, **spec)
    else:
        spec = dict(FEATURES[a.feature])
        dataset = spec.pop("dataset")
        spec.pop("algos", None)
        spec.pop("bh_batch", None)
        overrides = dict(spec.pop("lsqml_overrides", {}) if a.algo == "lsqml" else {})
        overrides.update({k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)})
        run(f"scratch/{a.feature}_{a.algo}", dataset, a.algo, a.batch, a.epochs, noise=a.noise,
            overrides=overrides, **spec)


if __name__ == "__main__":
    main()
