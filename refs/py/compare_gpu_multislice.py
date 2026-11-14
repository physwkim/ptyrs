"""pty-chi against the GPU engine through the `ptyrs` module: LSQML (Poisson) through three
object slices, GpuPtycho in f32 against pty-chi CPU f64, under refs/f64/lsqml_multislice.py's
cases (an angular-spectrum exit, a Fresnel transform, the far field).

    python -I compare_gpu_multislice.py [--device 0] [--module DIR]

DIR holds the module ptyrs-py/scripts/build.sh installs. The GPU's fused passes take frames of
64 pixels or more, so the state is lsqml_multislice.py's at SHAPE on an OBJ object, its 16
corners mirrored through the centre as in refs/ptychi/run_near.py (their mean is the centre,
which the engine's PyNX position preprocessing then leaves alone); its geometries keep their
propagators at SHAPE. Each case otherwise is compare_optics.py's multislice one: the f64
script's pty-chi task (its matched config, torch.randn zeroed), both sides over the engine's
random_batches order from one start. GpuPtycho.lsqml starts a fresh LSQML per call, so the GPU
state after e epochs is a fresh run of e epochs; the script holds each run's last LLK record to
the longest run's record of that epoch, so the states are one run. Every state's LLK is the
f64 script's llk of it.

Before the runs the GPU's own LLK of the truth is held to the f64 script's (the frames'
placement and the forward model, in f32). Prints each epoch's LLK of both sides and, per case,
the largest relative LLK difference, the final slices' and probe's differences over pty-chi's
largest modulus, how far the GPU's own LLK records are from the f64 llk of its states, and the
epochs after which each side is at or under the truth's LLK. Exits 1 if the truth's LLKs part
or those epochs differ in a case, 2 if a side diverges (m3ref.exit_unless_converging).
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories

SHAPE, OBJ = (64, 64), (104, 104)
# the GPU's f32 LLK of the truth against the f64 one, 5x the worst measured (7.8e-8)
TRUTH_RTOL = 4e-7


def sibling(name):
    """Import refs/py/<name>.py."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


co = sibling("compare_optics")
m3ref = co.m3ref


def cases(ptyrs, ms):
    """lsqml_multislice.py's cases at SHAPE on OBJ: (name, geometry, module optics, start
    slices, start probe, corners, counts, truth slices and probe, truth LLK)."""
    near = ms.near
    near.SHAPE, near.OBJ = ms.SHAPE, ms.OBJ = SHAPE, OBJ
    truth_obj, truth_probe, probe, corners, rng = ms.make_state()
    half = corners[:len(corners) // 2]
    corners = np.concatenate([half, np.array(OBJ) - np.array(SHAPE) - half])
    assert len({tuple(c) for c in corners}) == len(corners)
    obj = np.ones((ms.NSLICE, *OBJ), complex)
    for case, geom in enumerate(ms.GEOMETRIES):
        if case == ms.FAR:
            g = ms.FarGeometry(*geom)
            lz = g.wavelength * g.distance
            optics = ptyrs.Optics.far_field(g.wavelength, g.distance, lz / (SHAPE[0] * geom[1]),
                                            SHAPE)
        else:
            g = ms.Geometry(*geom)
            w = ptyrs.Wavefield(SHAPE, g.wavelength, [g.pixel, g.pixel], g.distance)
            optics = ptyrs.Optics.free_space(w)
        gaps = [ptyrs.Wavefield(SHAPE, g.wavelength, [g.pixel, g.pixel], d)
                for d in ms.SPACINGS]
        counts = rng.poisson(ms.icalc(g, truth_obj, truth_probe, corners)).astype(np.float64)
        yield (f"multislice {case}", g, optics.with_slices(gaps), obj, probe, corners, counts,
               (truth_obj, truth_probe), ms.llk(g, truth_obj, truth_probe, corners, counts))


def gpu_runs(ptyrs, scan, optics, obj, probe, device, ms):
    """GpuPtycho LSQML from (obj, probe): per epoch count e in 1..EPOCHS a fresh run of e
    epochs, its slices, probe and own LLK record of e; and the longest run's records."""
    runs = []
    for e in range(1, ms.EPOCHS + 1):
        gpu = ptyrs.GpuPtycho(scan, obj[:, None], probe, device, optics=optics)
        gpu.lsqml(e, 1, ms.BATCH, "poisson", ms.BATCH_SEED)
        h = gpu.history["poisson"]
        assert len(h) == e, (len(h), e)
        runs.append((gpu.objects[:, 0].astype(complex), gpu.probe.astype(complex), h[-1]))
    return runs, h


def run(ptyrs, ms, case, device):
    """One case through both sides: whether the truth's LLKs agree and each side reaches the
    truth's LLK after the same epoch."""
    from torch.utils.data import DataLoader

    name, g, optics, obj, probe, corners, counts, truth, truth_llk = case
    # corner = (OBJ - SHAPE) / 2 + position
    pos = (corners - (np.array(OBJ) - np.array(SHAPE)) // 2) * g.pixel
    # of the scan's geometry the engine reads the frame alone under optics
    scan = ptyrs.Scan(counts, pos[:, 0].copy(), pos[:, 1].copy(), g.wavelength, 1.0, g.pixel)
    own_truth = ptyrs.GpuPtycho(scan, truth[0][:, None], truth[1], device,
                                optics=optics).llk()["poisson"]
    truth_err = abs(own_truth / truth_llk - 1)

    t = ms.task(g, obj, probe, corners, counts, ms.BATCH, ms.EPOCHS)
    sampler = ms.near.rb.Sampler(len(corners), ms.BATCH, ms.BATCH_SEED)
    t.reconstructor.dataloader = DataLoader(t.reconstructor.dataset, batch_sampler=sampler)
    start = ms.llk(g, obj, probe, corners, counts)
    curves = {"ptyrs": [start], "pty-chi": [start]}
    with ms.near.ZeroRandn() as z:
        for e in range(1, ms.EPOCHS + 1):
            t.run(1)
            assert sampler.epoch == e, (sampler.epoch, e)
            curves["pty-chi"].append(ms.llk(g, *ms.state_of(t, g), corners, counts))
    runs, longest = gpu_runs(ptyrs, scan, optics, obj, probe, device, ms)
    own = np.array([r[2] for r in runs])
    assert (own == longest).all(), "a shorter GPU run is not the longest run's prefix"
    curves["ptyrs"] += [ms.llk(g, o, p, corners, counts) for o, p, _ in runs]

    print(f"--- {name}: kind {g.kind}, truth LLK {truth_llk:.6f}, the GPU's {own_truth:.6f} "
          f"({truth_err:.1e}); epoch, LLK GPU, pty-chi, relative difference")
    for e, (x, y) in enumerate(zip(curves["ptyrs"], curves["pty-chi"])):
        print(f"{e:4d} {x:.10e} {y:.10e} {abs(x - y) / y:.1e}")
    m3ref.exit_unless_converging(curves)
    to, tp = ms.state_of(t, g)
    o, p, _ = runs[-1]
    a, b = (np.array(curves[s]) for s in ("ptyrs", "pty-chi"))
    hits = {s: co.reached(c, truth_llk) for s, c in curves.items()}
    print(f"{name}: LLK max relative difference {np.abs(a / b - 1).max():.1e}, slices "
          f"{np.abs(o - to).max() / np.abs(to).max():.1e}, probe "
          f"{np.abs(p - tp).max() / np.abs(tp).max():.1e}; the GPU's LLK records from the f64 "
          f"llk of its states {np.abs(own / a[1:] - 1).max():.1e}; truth reached after GPU "
          f"{hits['ptyrs']}, pty-chi {hits['pty-chi']} epochs; {z.calls} torch.randn calls "
          "zeroed", flush=True)
    return truth_err <= TRUTH_RTOL and hits["ptyrs"] == hits["pty-chi"]


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--device", type=int, default=0)
    a.add_argument("--module", default=os.path.expanduser("~/work/ptyrs-scratch/py"))
    args = a.parse_args()
    sys.path.insert(0, args.module)
    import ptyrs
    import torch
    import ptychi.utils as utils

    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)
    ms = co.f64("lsqml_multislice")
    same = True
    for case in cases(ptyrs, ms):
        same &= run(ptyrs, ms, case, args.device)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
