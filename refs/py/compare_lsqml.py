"""pty-chi against ptyrs through the `ptyrs` module: CPU f64 LSQML on one M3 dataset from its
init as it is, both sides over the engine's random_batches order, PyNX LLK per epoch.

    python -I compare_lsqml.py DATA_DIR [--name fermat_clean] [--noise gaussian|poisson]
        [--batch 32] [--epochs 40] [--seed 7] [--module DIR]

DATA_DIR is gen_m3's output (ptyrs-refgen), DIR holds the module ptyrs-py/scripts/build.sh
installs. Prints the start's far-field intensity difference (Ptycho.forward against pty-chi's
forward model), each epoch's LLK of the noise model on both sides, and the first epoch each side
reaches 1.01, 1.02 and 1.05 times the truth; exits 1 if those epochs differ. pty-chi's Gaussian
noise model is the engine's Cost::Amplitude.

Matched config, as in the M3 parity curves: pty-chi features ptyrs does not implement are off
(object-probe ambiguity removal, mode orthogonalisation, the first-epoch probe intensity
rescale), and torch.randn returns zeros (LSQML Poisson's alpha jitter).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories
sys.path[:0] = [os.path.join(HERE, "..", "ptychi"), os.path.join(HERE, "..", "f64")]
import m3ref  # noqa: E402
from random_batches import Sampler  # noqa: E402

FACTORS = (1.01, 1.02, 1.05)


def ptychi_task(ds, obj, probe, noise, batch, epochs):
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = ds.frame
    opts = api.LSQMLOptions()
    r = opts.reconstructor_options
    r.noise_model = {"gaussian": api.NoiseModels.GAUSSIAN,
                     "poisson": api.NoiseModels.POISSON}[noise]
    opts.object_options.build_preconditioner_with_all_modes = True
    r.forward_model_options.subpixel_shift_all_probe_modes = True
    r.solve_step_sizes_only_using_first_probe_mode = False
    r.rescale_probe_intensity_in_first_epoch = False
    opts.data_options.wavelength_m = ds.geometry["wavelength_m"]
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = True
    o.pixel_size_m = ds.pixel
    # Corners as given: the view of corner c starts at c + origin - (n - 1) / 2.
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    pr = opts.probe_options
    pr.optimizable = True
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    opts.probe_position_options.optimizable = False
    r.batch_size = batch
    r.num_epochs = epochs
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    # PyNX view corners, the ones the engine preprocesses the positions to.
    corners = (np.array(obj.shape[-2:]) - ds.frame) // 2 + np.round(ds.pos_px)
    return PtychographyTask(
        opts,
        diffraction_data=ds.counts.astype(np.float64),
        object_data=obj.copy(),
        probe_data=ds.probe_to_ptychi(probe)[None],
        probe_position_y_px=corners[:, 0],
        probe_position_x_px=corners[:, 1],
    )


@torch.no_grad()
def ptychi_icalc(task, chunk=64):
    fm = task.reconstructor.forward_model
    saved = fm.intermediate_variables
    n = task.dataset.patterns.shape[0]
    icalc = [fm.forward(torch.arange(s, min(n, s + chunk))).numpy() for s in range(0, n, chunk)]
    fm.intermediate_variables = saved
    return np.concatenate(icalc)


def first_hits(curve, truth):
    return [next((e for e, v in enumerate(curve) if v <= f * truth), None) for f in FACTORS]


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("data_dir")
    a.add_argument("--name", default="fermat_clean")
    a.add_argument("--noise", choices=("gaussian", "poisson"), default="gaussian")
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--epochs", type=int, default=40)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--module", default=os.path.expanduser("~/work/ptyrs-scratch/py"))
    args = a.parse_args()
    sys.path.insert(0, args.module)
    import ptyrs
    from torch.utils.data import DataLoader

    ds = m3ref.Dataset(args.name, args.data_dir)
    obj, probe = ds.modes("init")
    g = ds.geometry
    scan = ptyrs.Scan(ds.counts, ds.y_m, ds.x_m, g["wavelength_m"], g["distance_m"],
                      g["detector_pixel_m"])
    p = ptyrs.Ptycho(scan, obj, probe)
    lsqml = ptyrs.Lsqml(args.batch, {"gaussian": "amplitude", "poisson": "poisson"}[args.noise],
                        args.seed)
    task = ptychi_task(ds, obj, probe, args.noise, args.batch, args.epochs)
    rec = task.reconstructor
    sampler = Sampler(ds.nframe, args.batch, args.seed)
    rec.dataloader = DataLoader(rec.dataset, batch_sampler=sampler)

    ours = (np.abs(p.forward(obj, probe)) ** 2).sum(axis=1)
    theirs = ptychi_icalc(task)
    print(f"start far-field intensity: max |ptyrs - pty-chi| / max pty-chi "
          f"{np.abs(ours - theirs).max() / theirs.max():.2e}")
    del ours, theirs

    key = args.noise
    curves = {"ptyrs": [p.llk(obj, probe)[key]], "pty-chi": [m3ref.evaluate(task)[key]]}
    wall = {"ptyrs": 0.0, "pty-chi": 0.0}
    randn, zeroed = torch.randn, []

    def zeros(*shape, **kw):
        zeroed.append(shape)
        return torch.zeros(*shape, **kw)

    print(f"{args.name} LSQML {key} b{args.batch} seed {args.seed}: epoch, {key} LLK ptyrs, "
          f"pty-chi, relative difference")
    o, pr = obj, probe
    for e in range(args.epochs + 1):
        if e:
            t = time.perf_counter()
            o, pr, _ = lsqml.epoch(p, o, pr)
            wall["ptyrs"] += time.perf_counter() - t
            t = time.perf_counter()
            torch.randn = zeros
            try:
                task.run(n_epochs=1)
            finally:
                torch.randn = randn
            wall["pty-chi"] += time.perf_counter() - t
            assert sampler.epoch == e, (sampler.epoch, e)
            curves["ptyrs"].append(p.llk(o, pr)[key])
            curves["pty-chi"].append(m3ref.evaluate(task)[key])
        x, y = curves["ptyrs"][e], curves["pty-chi"][e]
        print(f"{e:4d} {x:.10e} {y:.10e} {abs(x - y) / y:.1e}", flush=True)
        if not (np.isfinite(x) and np.isfinite(y)):
            break

    truth = ds.meta["llk_truth"][key]
    hits = {side: first_hits(c, truth) for side, c in curves.items()}
    print(f"torch.randn calls zeroed: {len(zeroed)}")
    for side, h in hits.items():
        per = wall[side] / (len(curves[side]) - 1)
        print(f"{side:8s} first epoch at " + ", ".join(
            f"{f:g}x truth: {e}" for f, e in zip(FACTORS, h)) + f"; {per:.2f} s/epoch")
    sys.exit(0 if hits["ptyrs"] == hits["pty-chi"] else 1)


if __name__ == "__main__":
    main()
