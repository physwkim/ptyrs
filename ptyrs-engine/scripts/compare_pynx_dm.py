"""PyNX DM on the scan and starting point written by `cargo run --example dm_llk`.

Usage: python -I compare_pynx_dm.py <dir> [--cuda]

Reads <dir>/scan.cxi, init.h5, llk_rust.txt and final_rust.h5. Runs DM(calc_llk=1) for
as many cycles as the Rust run with
  - f64: PyNX cpu_operator with every complex64/float32 promoted to complex128/float64,
  - f32: the stock PyNX cpu_operator,
  - cuda (with --cuda): the PyNX CUDA operator,
and prints per-cycle normalised LLK (Poisson, Gaussian, Euclidian, photon count) relative
differences against Rust, and the relative L2 difference of the final object and probe.
"""
import sys
import types

import h5py
import numpy as np
from scipy.fft import fftshift

from pynx.ptycho.ptycho import Ptycho, PtychoData
import pynx.ptycho.cpu_operator as cpu32

KEYS = ("llk_poisson", "llk_gaussian", "llk_euclidian", "nb_photons_calc")


def cpu_operator_f64():
    """cpu_operator with every complex64/float32 cast promoted to double precision."""
    with open(cpu32.__file__) as f:
        src = f.read()
    src = src.replace("np.complex64", "np.complex128").replace("np.float32", "np.float64")
    mod = types.ModuleType("pynx.ptycho.cpu_operator_f64")
    mod.__package__ = "pynx.ptycho"
    mod.__file__ = cpu32.__file__
    sys.modules[mod.__name__] = mod
    exec(compile(src, cpu32.__file__, "exec"), mod.__dict__)
    return mod


def load(d):
    with h5py.File(f"{d}/scan.cxi", "r") as f:
        raw = f["entry_1/data_1/data"][()]
        iobs = raw.astype(np.float64)
        if raw.dtype == np.uint16:
            iobs[raw == np.iinfo(np.uint16).max] = -1
        if "entry_1/instrument_1/detector_1/mask" in f:
            iobs[:, f["entry_1/instrument_1/detector_1/mask"][()] != 0] = -1
        t = f["entry_1/sample_1/geometry_1/translation"][()].astype(np.float64)
        energy = float(f["entry_1/instrument_1/source_1/energy"][()])
        distance = float(f["entry_1/instrument_1/detector_1/distance"][()])
        pixel = float(f["entry_1/instrument_1/detector_1/x_pixel_size"][()])
    wavelength = 12.3984e-10 / (energy / 1.60218e-16)
    with h5py.File(f"{d}/init.h5", "r") as f:
        obj, probe = f["object"][()], f["probe"][()]
    return iobs, (t[0], t[1]), distance, pixel, wavelength, obj, probe


def build(iobs, pos, distance, pixel, wavelength, obj, probe, f64):
    data = PtychoData(iobs=iobs, positions=pos, detector_distance=distance,
                      pixel_size_detector=pixel, wavelength=wavelength)
    p = Ptycho(probe=probe, obj=obj, data=data)
    if f64:
        # copies: PyNX updates these in place, and every run starts from the same arrays
        p._obj = np.array(obj, dtype=np.complex128, order="C", copy=True)
        p._probe = np.array(probe, dtype=np.complex128, order="C", copy=True)
        p.data.iobs = np.ascontiguousarray(fftshift(iobs, axes=(-2, -1)), dtype=np.float64)
        p.pixel_size_object = np.float64(data.pixel_size_object()[0])
    return p


def history(p, ncycle):
    h = p.history
    return np.array([[float(h[k][c]) for k in KEYS] for c in range(ncycle)])


def rel_l2(a, b):
    """min over complex k of |a - k b| / |a|, and k."""
    k = np.vdot(b, a) / np.vdot(b, b)
    return np.linalg.norm(a - k * b) / np.linalg.norm(a), k


def main():
    d = sys.argv[1]
    rust = np.loadtxt(f"{d}/llk_rust.txt", ndmin=2)
    ncycle = len(rust)
    assert (rust[:, 0] == np.arange(ncycle)).all(), "llk_rust.txt must hold every cycle"
    rust = rust[:, 1:]
    with h5py.File(f"{d}/final_rust.h5", "r") as f:
        obj_r, probe_r = f["object"][()], f["probe"][()]
    args = load(d)

    # (name, operator module, f64, lag). The CUDA DM computes the LLK inside the stack loop,
    # before the cycle's object/probe update, so its record at cycle c + 1 is the CPU's at c.
    # Its ScaleObjProbe(absolute=False) after each LLK only trades scale between object and
    # probe, which DM is covariant under; the final arrays are compared up to a complex scale.
    runs = [("f64", cpu_operator_f64(), True, 0), ("f32", cpu32, False, 0)]
    if "--cuda" in sys.argv[2:]:
        import pynx.ptycho.cu_operator as cuop
        runs.append(("cuda", cuop, False, 1))

    for name, op, f64, lag in runs:
        p = build(*args, f64=f64)
        p = op.DM(calc_llk=1) ** ncycle * p
        h = history(p, ncycle)
        np.savetxt(f"{d}/llk_pynx_{name}.txt",
                   np.column_stack([np.arange(ncycle), h]), fmt=["%d"] + ["%.17e"] * 4)
        h, ref = h[lag:], rust[:ncycle - lag]
        r = np.abs(h - ref) / np.abs(ref)
        print(f"--- {name}: max |rel diff| vs Rust over {len(h)} cycles (PyNX lag {lag})")
        for k, v in zip(KEYS, r.T):
            i = int(v.argmax())
            print(f"  {k:16s} {v.max():.3e} (cycle {i})")
        (eo, ko), (ep, kp) = rel_l2(obj_r, p.get_obj()), rel_l2(probe_r, p.get_probe())
        print(f"  final object rel L2 {eo:.3e} (scale {ko:.6f})"
              f"  probe rel L2 {ep:.3e} (scale {kp:.6f})")
        for c in sorted({0, 1, 2, 5, 10, len(h) // 2, len(h) - 1}):
            if c < len(h):
                print(f"  cycle {c:4d} poisson rust {ref[c, 0]:.10e} pynx {h[c, 0]:.10e}")


if __name__ == "__main__":
    main()
