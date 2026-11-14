"""pty-chi against ptyrs through the `ptyrs` module: CPU f64 LSQML (Poisson) over the module's
Optics, on the cases of refs/f64/lsqml_near.py (one slice, free space) and
refs/f64/lsqml_multislice.py (three slices before free space or the far field).

    python -I compare_optics.py [--model near|multislice|both] [--module DIR]

DIR holds the module ptyrs-py/scripts/build.sh installs. Each case is the f64 script's own state
and pty-chi task (its matched config, torch.randn zeroed), both sides over the engine's
random_batches order from one start. Prints each epoch's normalised Poisson LLK of both states
(the f64 script's llk) and, per case, the largest relative LLK difference, the final object and
probe differences over pty-chi's largest modulus and the epochs after which each side is at or
under the truth's LLK. Exits 1 if those epochs differ in a case, 2 if a side diverges
(m3ref.exit_unless_converging).
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories
sys.path[:0] = [os.path.join(HERE, "..", "ptychi")]
import m3ref  # noqa: E402


def f64(name):
    """Import refs/f64/<name>.py."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "f64",
                                                                     f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reached(curve, truth):
    """Epochs after which `curve` is first at or under `truth`, None if never."""
    return next((e for e, v in enumerate(curve) if v <= truth), None)


def near_cases(ptyrs, near):
    """lsqml_near.py's cases: (name, geometry, module optics, start, corners, counts, truth)."""
    truth_obj, truth_probe, probe, corners, rng = near.make_state()
    obj = np.ones((1, *near.OBJ), complex)
    for case, geom in enumerate(near.GEOMETRIES):
        g = near.Geometry(*geom)
        counts = near.counts_of(g, truth_obj, truth_probe, corners, rng)
        w = ptyrs.Wavefield(near.SHAPE, g.wavelength, [g.pixel, g.pixel], g.distance)
        yield (f"near {case}", g, ptyrs.Optics.free_space(w), obj, probe, corners, counts,
               near.llk(g, truth_obj, truth_probe, corners, counts))


def multislice_cases(ptyrs, ms):
    """lsqml_multislice.py's cases, as near_cases."""
    near = ms.near
    truth_obj, truth_probe, probe, corners, rng = ms.make_state()
    obj = np.ones((ms.NSLICE, *near.OBJ), complex)
    for case, geom in enumerate(ms.GEOMETRIES):
        if case == ms.FAR:
            g = ms.FarGeometry(*geom)
            lz = g.wavelength * g.distance
            optics = ptyrs.Optics.far_field(g.wavelength, g.distance,
                                            lz / (near.SHAPE[0] * geom[1]), near.SHAPE)
        else:
            g = ms.Geometry(*geom)
            w = ptyrs.Wavefield(near.SHAPE, g.wavelength, [g.pixel, g.pixel], g.distance)
            optics = ptyrs.Optics.free_space(w)
        gaps = [ptyrs.Wavefield(near.SHAPE, g.wavelength, [g.pixel, g.pixel], d)
                for d in ms.SPACINGS]
        counts = rng.poisson(ms.icalc(g, truth_obj, truth_probe, corners)).astype(np.float64)
        yield (f"multislice {case}", g, optics.with_slices(gaps), obj, probe, corners, counts,
               ms.llk(g, truth_obj, truth_probe, corners, counts))


def run(ptyrs, mod, near, case):
    """One case through both sides; its curves and the epochs each side reaches the truth."""
    from torch.utils.data import DataLoader

    name, g, optics, obj, probe, corners, counts, truth = case
    multislice = obj.shape[0] > 1
    # corner = (OBJ - SHAPE) / 2 + position at a pixel of 1
    positions = corners - (np.array(near.OBJ) - np.array(near.SHAPE)) // 2
    views = ptyrs.Views(counts, positions.astype(np.float64), [1.0, 1.0], near.OBJ, near.NMODE,
                        1, optics)
    lsqml = ptyrs.Lsqml(mod.BATCH, "poisson", mod.BATCH_SEED)
    t = mod.task(g, obj if multislice else obj[0], probe, corners, counts, mod.BATCH, mod.EPOCHS)
    rec = t.reconstructor
    sampler = near.rb.Sampler(len(corners), mod.BATCH, mod.BATCH_SEED)
    rec.dataloader = DataLoader(rec.dataset, batch_sampler=sampler)

    def llk(o, p):
        return mod.llk(g, o if multislice else o[0], p, corners, counts)

    def theirs():
        o, p = mod.state_of(t, g)
        return (o if multislice else o[None]), p

    start = llk(obj, probe)
    curves = {"ptyrs": [start], "pty-chi": [start]}
    print(f"--- {name}: kind {g.kind}, truth LLK {truth:.6f}; epoch, LLK ptyrs, pty-chi, "
          "relative difference")
    o, p = obj[:, None], probe
    with near.ZeroRandn() as z:
        for e in range(1, mod.EPOCHS + 1):
            o, p, _ = lsqml.epoch(views, o, p)
            t.run(1)
            assert sampler.epoch == e, (sampler.epoch, e)
            x, y = llk(o[:, 0], p), llk(*theirs())
            curves["ptyrs"].append(x)
            curves["pty-chi"].append(y)
            print(f"{e:4d} {x:.10e} {y:.10e} {abs(x - y) / y:.1e}", flush=True)
            if not (np.isfinite(x) and np.isfinite(y)):
                break
    m3ref.exit_unless_converging(curves)
    to, tp = theirs()
    a, b = (np.array(curves[s]) for s in ("ptyrs", "pty-chi"))
    hits = {s: reached(c, truth) for s, c in curves.items()}
    print(f"{name}: LLK max relative difference {np.abs(a / b - 1).max():.1e}, object "
          f"{np.abs(o[:, 0] - to).max() / np.abs(to).max():.1e}, probe "
          f"{np.abs(p - tp).max() / np.abs(tp).max():.1e}; truth reached after ptyrs "
          f"{hits['ptyrs']}, pty-chi {hits['pty-chi']} epochs; {z.calls} torch.randn calls "
          "zeroed", flush=True)
    return hits["ptyrs"] == hits["pty-chi"]


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--model", choices=("near", "multislice", "both"), default="both")
    a.add_argument("--module", default=os.path.expanduser("~/work/ptyrs-scratch/py"))
    args = a.parse_args()
    sys.path.insert(0, args.module)
    import ptyrs
    import torch
    import ptychi.utils as utils

    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)
    near, same = f64("lsqml_near"), True
    if args.model in ("near", "both"):
        for case in near_cases(ptyrs, near):
            same &= run(ptyrs, near, near, case)
    if args.model in ("multislice", "both"):
        ms = f64("lsqml_multislice")
        for case in multislice_cases(ptyrs, ms):
            same &= run(ptyrs, ms, ms.near, case)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
