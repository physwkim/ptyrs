"""pty-chi against ptyrs through the `ptyrs` module: CPU f64 rPIE on one M3 dataset from its
init as it is, with pty-chi's own rPIE parameters on both sides, PyNX LLK per epoch.

    python -I compare_rpie.py DATA_DIR [--name fermat_clean] [--llk gaussian|poisson|euclidian]
        [--epochs 20] [--rtol 1e-4] [--module DIR]
    python -I compare_rpie.py DATA_DIR --ptychi-defaults [--seed 7] ...

DATA_DIR is refgen_gen_m3's output (ptyrs::refgen), DIR holds the module ptyrs-py/scripts/build.sh
installs. Prints the start's far-field intensity difference, each epoch's LLK on both sides and
their relative difference; exits 1 at the first epoch past rtol.

The rPIE parameters are read from pty-chi's RPIEOptions defaults (pty-chi 0605630) and handed to
Ptycho.rpie. The rest is matched as in compare_lsqml.py: pty-chi features ptyrs does not implement
are off (object-probe ambiguity removal, mode orthogonalisation), PATCH shifts at integer corners.
--ptychi-defaults runs pty-chi alone with its defaults otherwise untouched as well (shuffled
batches seeded by --seed, ambiguity removal, orthogonalisation, AUTO shifts).
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories
sys.path.insert(0, os.path.join(HERE, "..", "ptychi"))
import m3ref  # noqa: E402


def ptychi_task(ds, obj, probe, epochs, matched, seed):
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = ds.frame
    opts = api.RPIEOptions()
    r = opts.reconstructor_options
    opts.data_options.wavelength_m = ds.geometry["wavelength_m"]
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.pixel_size_m = ds.pixel
    # Corners as given: the view of corner c starts at c + origin - (n - 1) / 2.
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    pr = opts.probe_options
    r.num_epochs = epochs
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.random_seed = seed
    if matched:
        o.remove_object_probe_ambiguity.enabled = False
        pr.orthogonalize_incoherent_modes.enabled = False
        pr.orthogonalize_opr_modes.enabled = False
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


def rpie_params(opts):
    """Ptycho.rpie keywords equal to pty-chi's rPIE settings in `opts`."""
    import ptychi.api as api

    o, pr, r = opts.object_options, opts.probe_options, opts.reconstructor_options
    for p in (o, pr):
        # Both arrays from epoch 0 on every epoch, plain SGD: a step adds lr times the update.
        plan = p.optimization_plan
        assert p.optimizable and (plan.start, plan.stop, plan.stride) == (0, None, 1), plan
        assert p.optimizer == api.Optimizers.SGD and not p.optimizer_params, p.optimizer
    return {
        "obj_step": o.step_size,  # ObjectOptions.step_size: O += step * sum over the batch
        "probe_step": pr.step_size,  # ProbeOptions.step_size: P += step * mean over the batch
        "obj_alpha": o.alpha,  # PIEObjectOptions.alpha: (1-a)|P|^2 + a max_batch sum_m |P_m|^2
        "probe_alpha": pr.alpha,  # PIEProbeOptions.alpha: (1-a)|O|^2 + a max_view |O|^2
        "batch_size": r.batch_size,  # ReconstructorOptions.batch_size, the last batch shorter
        # probe update start: optimization_plan.start 0 (asserted), the engine's only schedule
        # batch order: the engine's consecutive scan order, pty-chi's RANDOM loader replaced
        # update rule: both from the batch-start state (pie.py compute_updates, apply_updates)
    }


@torch.no_grad()
def ptychi_icalc(task, chunk=64):
    fm = task.reconstructor.forward_model
    saved = fm.intermediate_variables
    n = task.dataset.patterns.shape[0]
    icalc = [fm.forward(torch.arange(s, min(n, s + chunk))).numpy() for s in range(0, n, chunk)]
    fm.intermediate_variables = saved
    return np.concatenate(icalc)


def ptychi_defaults(ds, obj, probe, args):
    task = ptychi_task(ds, obj, probe, args.epochs, False, args.seed)
    print(f"{args.name} pty-chi rPIE defaults seed {args.seed}: epoch, {args.llk} LLK")
    for e in range(args.epochs + 1):
        if e:
            task.run(n_epochs=1)
        v = m3ref.evaluate(task)[args.llk]
        print(f"{e:4d} {v:.10e}", flush=True)
        if not np.isfinite(v):
            break


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("data_dir")
    a.add_argument("--name", default="fermat_clean")
    a.add_argument("--llk", choices=("gaussian", "poisson", "euclidian"), default="gaussian")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--rtol", type=float, default=1e-4)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--ptychi-defaults", action="store_true")
    a.add_argument("--module", default=os.path.expanduser("~/work/ptyrs-scratch/py"))
    args = a.parse_args()

    ds = m3ref.Dataset(args.name, args.data_dir)
    obj, probe = ds.modes("init")
    if args.ptychi_defaults:
        return ptychi_defaults(ds, obj, probe, args)

    sys.path.insert(0, args.module)
    import ptyrs
    from torch.utils.data import DataLoader

    g = ds.geometry
    scan = ptyrs.Scan(ds.counts, ds.y_m, ds.x_m, g["wavelength_m"], g["distance_m"],
                      g["detector_pixel_m"])
    p = ptyrs.Ptycho(scan, obj, probe)
    task = ptychi_task(ds, obj, probe, args.epochs, True, args.seed)
    params = rpie_params(task.options)
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset, batch_size=params["batch_size"], shuffle=False)

    ours = (np.abs(p.forward(obj, probe)) ** 2).sum(axis=1)
    theirs = ptychi_icalc(task)
    print(f"start far-field intensity: max |ptyrs - pty-chi| / max pty-chi "
          f"{np.abs(ours - theirs).max() / theirs.max():.2e}")
    del ours, theirs

    key = args.llk
    print(f"{args.name} rPIE {params}: epoch, {key} LLK ptyrs, pty-chi, relative difference")
    parted = None
    for e in range(args.epochs + 1):
        if e:
            p.rpie(1, **params)
            task.run(n_epochs=1)
        x, y = p.llk(p.object, p.probe)[key], m3ref.evaluate(task)[key]
        d = abs(x - y) / abs(y)
        print(f"{e:4d} {x:.10e} {y:.10e} {d:.1e}", flush=True)
        if not d <= args.rtol:
            parted = e
            break
    if parted is not None:
        print(f"parted at epoch {parted} (rtol {args.rtol:g})")
    sys.exit(0 if parted is None else 1)


if __name__ == "__main__":
    main()
