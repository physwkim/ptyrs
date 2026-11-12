"""PyNX CUDA ML on the scan and starting point written by `cargo run --example engine_llk_trace -- <dir> ml`.

Usage: python -I compare_pynx_ml.py <dir> [--fix-gamma] [--noise] [--time N]

Reads <dir>/scan.cxi, init.h5 (the end of engine_llk_trace's AP warm-up), llk_rust.txt and
final_rust.h5. Runs ML(update_object=True, update_probe=True, calc_llk=1) of the PyNX CUDA
operator for as many cycles as the Rust run,
  - stock: as shipped, which Hann-filters the Fourier-space Poisson gradient,
  - nohann: with hann_filter forced to 0, the unfiltered gradient ptyrs uses,
and prints per-cycle normalised LLK relative differences against Rust and the relative L2
difference of the final object and probe. PyNX's files are untouched: both changes wrap what
it calls. --fix-gamma corrects the cross term `PdO.real() * dPO.imag()` of PyNX's
CG_Poisson_Gamma to `PdO.imag() * dPO.imag()` (as in its CG_Poisson_Gamma4) before the
kernels compile; it matters only when object and probe are both updated. --noise also prints
PyNX nohann against itself, rerun and with the starting object nudged by 1e-7, which bounds
how closely any other implementation can track it cycle by cycle. --time N then times N
cycles of stock ML without LLK or phase-ramp removal, after 5 untimed ones, as
`cargo run --example engine_gpu_time -- ml` times ptyrs. PyNX's CPU ML is no reference: its first
direction is +grad and the later ones beta*dir - grad.
"""
import sys

import h5py
import numpy as np

from pynx.ptycho.ptycho import Ptycho, PtychoData
import pynx.ptycho.cu_operator as cuop

KEYS = ("llk_poisson", "llk_gaussian", "llk_euclidian", "nb_photons_calc")
GAMMA_SRC = "ptycho/cuda/cg_gamma_red.cu"
GAMMA_BUG = "PdO.real() * dPO.real() + PdO.real() * dPO.imag()"
GAMMA_FIX = "PdO.real() * dPO.real() + PdO.imag() * dPO.imag()"


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


def build(iobs, pos, distance, pixel, wavelength, obj, probe):
    data = PtychoData(iobs=iobs, positions=pos, detector_distance=distance,
                      pixel_size_detector=pixel, wavelength=wavelength)
    return Ptycho(probe=probe.copy(), obj=obj.copy(), data=data)


def run_ml(args, ncycle):
    p = cuop.ML(update_object=True, update_probe=True, calc_llk=1) ** ncycle * build(*args)
    h = p.history
    return p, np.array([[float(h[k][c]) for k in KEYS] for c in range(ncycle)])


def rel_l2(a, b):
    """min over complex k of |a - k b| / |a|, and k."""
    k = np.vdot(b, a) / np.vdot(b, b)
    return np.linalg.norm(a - k * b) / np.linalg.norm(a), k


def set_hann(on):
    """Wraps (or unwraps) cu_grad_poisson_fourier so its hann_filter argument is 0."""
    pu = cuop.default_processing_unit
    orig = getattr(pu.cu_grad_poisson_fourier, "_orig", pu.cu_grad_poisson_fourier)
    if on:
        pu.cu_grad_poisson_fourier = orig
        return

    def nohann(*a, **kw):
        a = list(a)
        assert a[9] == 1, "argument 9 of grad_poisson_fourier is hann_filter"
        a[9] = np.int8(0)
        return orig(*a, **kw)

    nohann._orig = orig
    pu.cu_grad_poisson_fourier = nohann


def fix_gamma():
    """Makes cu_operator compile CG_Poisson_Gamma with the corrected cross term."""
    getks = cuop.getks

    def fixed(name):
        src = getks(name)
        if name == GAMMA_SRC:
            assert src.count(GAMMA_BUG) == 1, "CG_Poisson_Gamma cross term not found"
            src = src.replace(GAMMA_BUG, GAMMA_FIX)
        return src

    cuop.getks = fixed


def print_poisson(ref, h, ncycle):
    r = np.abs(h[:, 0] - ref) / np.abs(ref)
    for c in sorted({0, 1, 2, 3, 5, 10, 20, ncycle // 2, ncycle - 1}):
        if c < ncycle:
            print(f"  cycle {c:4d} poisson ref {ref[c]:.10e} pynx {h[c, 0]:.10e}  rel {r[c]:.2e}")


def main():
    d, flags = sys.argv[1], sys.argv[2:]
    opts = set(flags)
    tag = ""
    if "--fix-gamma" in opts:
        fix_gamma()
        tag = "_fixgamma"
    rust = np.loadtxt(f"{d}/llk_rust.txt", ndmin=2)
    ncycle = len(rust)
    assert (rust[:, 0] == np.arange(ncycle)).all(), "llk_rust.txt must hold every cycle"
    rust = rust[:, 1:]
    with h5py.File(f"{d}/final_rust.h5", "r") as f:
        obj_r, probe_r = f["object"][()], f["probe"][()]
    args = load(d)
    # initialise the processing unit and compile its kernels with a throwaway run
    cuop.ScaleObjProbe() * build(*args)
    for name, hann in ((f"stock{tag}", True), (f"nohann{tag}", False)):
        set_hann(hann)
        p, h = run_ml(args, ncycle)
        np.savetxt(f"{d}/llk_pynx_ml_{name}.txt", np.column_stack([np.arange(ncycle), h]),
                   fmt=["%d"] + ["%.17e"] * 4)
        r = np.abs(h - rust) / np.abs(rust)
        print(f"--- {name}: max |rel diff| vs Rust over {ncycle} cycles")
        for k, v in zip(KEYS, r.T):
            i = int(v.argmax())
            print(f"  {k:16s} {v.max():.3e} (cycle {i})")
        (eo, ko), (ep, kp) = rel_l2(obj_r, p.get_obj()), rel_l2(probe_r, p.get_probe())
        print(f"  final object rel L2 {eo:.3e} (scale {ko:.6f})"
              f"  probe rel L2 {ep:.3e} (scale {kp:.6f})")
        print_poisson(rust[:, 0], h, ncycle)
    if "--noise" in opts:
        rng = np.random.default_rng(1)
        obj = args[5]
        nudge = 1e-7 * (rng.standard_normal(obj.shape) + 1j * rng.standard_normal(obj.shape))
        nudged = args[:5] + ((obj * (1 + nudge)).astype(obj.dtype), args[6])
        _, ref = run_ml(args, ncycle)
        for name, a in (("rerun", args), ("object nudged by 1e-7", nudged)):
            _, h = run_ml(a, ncycle)
            r = np.abs(h[:, 0] - ref[:, 0]) / ref[:, 0]
            print(f"--- nohann{tag} against itself, {name}: max poisson rel {r.max():.2e}"
                  f" (cycle {r.argmax()})")
            print_poisson(ref[:, 0], h, ncycle)
    if "--time" in opts:
        import time
        ntime = int(flags[flags.index("--time") + 1])
        set_hann(True)

        def ml():
            return cuop.ML(update_object=True, update_probe=True, calc_llk=0,
                           zero_phase_ramp=False)

        p = ml() ** 5 * build(*args)
        cuop.default_processing_unit.finish()
        t = time.perf_counter()
        p = ml() ** ntime * p
        cuop.default_processing_unit.finish()
        dt = (time.perf_counter() - t) / ntime
        print(f"--- PyNX CUDA ML: {1e3 * dt:.3f} ms/cycle over {ntime} cycles")


if __name__ == "__main__":
    main()
