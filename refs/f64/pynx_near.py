"""Float64 reference of PyNX AP, DM, RAAR and ML in its near-field model.

    micromamba run -n ptyrs-ref python -I refs/f64/pynx_near.py <dir>

The state is rpie_near.py's (truth, start probe, object of ones, 16 whole-pixel corners whose
mean is the object's centre). GEOMETRY (wavelength, pixel, distance) is one where pty-chi's
free_space, and so the engine's, is angular spectrum; the counts are the truth's Poisson counts
through it, natural (centred) layout, which PtychoData fftshifts as PyNX's psi is.

PyNX's cpu_operator runs with every complex64/float32 promoted to complex128/float64, its
PtychoData(near_field=True), under two kernels in PropagateNearField:
  - paraxial: PyNX's own, exp(-i pi lambda z f^2), f the spatial frequency;
  - exact: the engine's angular spectrum exp(2 pi i z sqrt(1/lambda^2 - f^2)).
Up to a global phase, which no algorithm here sees, the two differ by at most
pi lambda^3 z f^4 / 4 over the frame, printed as `kernel`: the paraxial runs' differences to
the engine above what that phase produces are the algorithm's, the exact runs' are rounding.

Two defects of cpu_operator's ML are corrected for both kernels: its first direction is
+gradient (later ones beta d - gradient, so beta adds the opposite of the last direction), and
in near field it divides the object-plane gradient by the propagator's quadratic phase (the
far-field object-plane phase it divides by there is not applied in near field). ZeroPhaseRamp
is a no-op in near field.

Runs from the start, each with calc_llk=1, update_probe=True and PyNX's defaults otherwise:
AP and DM and RAAR for CYCLES cycles, ML for CYCLES cycles after AP_BEFORE_ML AP cycles (ML
from a flat object climbs, as the engine's ML docs say). Writes, every complex value `re im`,
every plane row-major, `run` the row of the runs table:
    <dir>/pynx_near_case_f64.txt     `wavelength pixel distance kind scale truth_llk kernel`
    <dir>/pynx_near_corners_f64.txt  `frame cy cx`
    <dir>/pynx_near_obj_f64.txt      `iy ix re im`, the start object
    <dir>/pynx_near_probe_f64.txt    `m iy ix re im`, the start probe
    <dir>/pynx_near_counts_f64.txt   `frame` and the frame's counts, natural layout
    <dir>/pynx_near_runs_f64.txt     `run exact algorithm cycles`, exact 1 for the exact
                                     kernel, algorithm 0 AP, 1 DM, 2 RAAR, 3 ML
    <dir>/pynx_near_llk_f64.txt      `run record poisson gaussian euclidian nb_photons_calc`,
                                     PyNX's history in order (ML's without AP's)
    <dir>/pynx_near_run_obj_f64.txt    `run iy ix re im`, the object at the end of the run
    <dir>/pynx_near_run_probe_f64.txt  `run m iy ix re im`, the probe at the end of the run
"""
import importlib.util
import pathlib
import sys
import types

import numpy as np
from scipy.fft import fftfreq, fftn, fftshift, ifftn

CYCLES = 30
AP_BEFORE_ML = 10
GEOMETRY = (1e-10, 50e-9, 1e-4)
ALGORITHMS = ("AP", "DM", "RAAR", "ML")
KEYS = ("llk_poisson", "llk_gaussian", "llk_euclidian", "nb_photons_calc")


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rpie = sibling("rpie_near")
near = rpie.near
ls = near.ls


def transfer(exact):
    """PropagateNearField's forward kernel over the frame, FFT order, and the largest phase
    difference between the two kernels beyond their global phase exp(2 pi i z / lambda)."""
    wavelength, pixel, z = GEOMETRY
    # the engine's AngularSpectrum::new operation by operation, lengths in wavelengths: the
    # phase is ~2 pi z / lambda, so a differently rounded one moves every pixel by ~1e-9 rad
    fy, fx = fftfreq(near.SHAPE[0])[:, None], fftfreq(near.SHAPE[1])
    width = pixel / wavelength
    ratio = (fx * fx + fy * fy) / (width * width)
    assert (ratio < 1).all()
    as_phase = 2.0 * np.pi * (z / wavelength) * np.sqrt(1.0 - ratio)
    paraxial = -np.pi * wavelength * z * (fx * fx + fy * fy) / pixel**2
    kernel = np.abs(as_phase - 2 * np.pi * z / wavelength - paraxial).max()
    return np.exp(1j * (as_phase if exact else paraxial)), kernel


def cpu_operator_f64(exact):
    """cpu_operator promoted to f64, ML's two defects corrected; with `exact`,
    PropagateNearField multiplies by the angular-spectrum kernel."""
    import pynx.ptycho.cpu_operator as cpu32
    with open(cpu32.__file__) as f:
        src = f.read()
    src = src.replace("np.complex64", "np.complex128").replace("np.float32", "np.float64")
    for old, new, n in (
            ("obj_dir = obj_grad.copy()", "obj_dir = -obj_grad", 1),
            ("probe_dir = probe_grad.copy()", "probe_dir = -probe_grad", 1),
            ("/ p._cpu_tmp_quad_phase[0])", "/ _grad_quad(p))", 3)):
        assert src.count(old) == n, old
        src = src.replace(old, new)
    src += ("\n\ndef _grad_quad(p):\n"
            "    return 1 if p.data.near_field else p._cpu_tmp_quad_phase[0]\n")
    name = f"pynx.ptycho.cpu_operator_f64_near{int(exact)}"
    mod = types.ModuleType(name)
    mod.__package__ = "pynx.ptycho"
    mod.__file__ = cpu32.__file__
    sys.modules[name] = mod
    exec(compile(src, cpu32.__file__, "exec"), mod.__dict__)
    if exact:
        t = transfer(True)[0]

        def op(self, p):
            k = t if self.forward else np.conj(t)
            p._psi = ifftn(fftn(p._psi, axes=(-2, -1)) * k, axes=(-2, -1))
            return p

        mod.PropagateNearField.op = op
    return mod


def build(counts, corners, obj, probe):
    """A PyNX near-field Ptycho in f64 on the state; positions (x, y) in metres."""
    from pynx.ptycho.ptycho import Ptycho, PtychoData
    wavelength, pixel, z = GEOMETRY
    pos = (corners - (np.array(near.OBJ) - np.array(near.SHAPE)) // 2) * pixel
    data = PtychoData(iobs=counts, positions=(pos[:, 1], pos[:, 0]), detector_distance=z,
                      pixel_size_detector=pixel, wavelength=wavelength, near_field=True)
    p = Ptycho(probe=probe, obj=obj[None], data=data)
    p._obj = np.array(obj[None], dtype=np.complex128, order="C", copy=True)
    p._probe = np.array(probe, dtype=np.complex128, order="C", copy=True)
    p.data.iobs = np.ascontiguousarray(fftshift(counts, axes=(-2, -1)), dtype=np.float64)
    p.pixel_size_object = np.float64(data.pixel_size_object()[0])
    return p


def run(op, algorithm, p):
    """`algorithm` on p; PyNX's LLK history of it, one row per record."""
    if algorithm == "AP":
        p = op.AP(update_object=True, update_probe=True, calc_llk=1) ** CYCLES * p
    elif algorithm == "DM":
        p = op.DM(calc_llk=1) ** CYCLES * p
    elif algorithm == "RAAR":
        p = op.RAAR(calc_llk=1) ** CYCLES * p
    else:
        p = op.AP(update_object=True, update_probe=True) ** AP_BEFORE_ML * p
        p = op.ML(update_object=True, update_probe=True, calc_llk=1) ** CYCLES * p
    h = p.history
    cycles = sorted(c for c in h[KEYS[0]].keys() if algorithm != "ML" or c >= AP_BEFORE_ML)
    assert len(cycles) == CYCLES, cycles
    return p, np.array([[float(h[k][c]) for k in KEYS] for c in cycles])


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    truth_obj, truth_probe, probe, corners, rng = rpie.make_state()
    obj = np.ones(near.OBJ, complex)
    nframe = len(corners)
    g = near.Geometry(*GEOMETRY)
    assert g.kind == 1 and g.k == 1, (g.kind, g.k)
    counts = near.counts_of(g, truth_obj, truth_probe, corners, rng)
    truth_llk = near.llk(g, truth_obj, truth_probe, corners, counts)
    kernel = transfer(False)[1]
    print(f"{GEOMETRY}: kind {g.kind}, {nframe} frames, counts max {counts.max():.0f} mean "
          f"{counts.mean():.1f}, LLK truth {truth_llk:.6f} start "
          f"{near.llk(g, obj, probe, corners, counts):.6f}; kernel difference "
          f"{kernel:.3e} rad")

    out = {k: [] for k in ("runs", "llk", "run_obj", "run_probe")}
    ends = {}
    for exact in (False, True):
        op = cpu_operator_f64(exact)
        for a, algorithm in enumerate(ALGORITHMS):
            p, h = run(op, algorithm, build(counts, corners, obj, probe))
            o, pr = p._obj[0], p._probe
            ends[exact, algorithm] = o, pr, h
            line = (f"{'exact' if exact else 'paraxial':8s} {algorithm:4s}: LLK {h[0, 0]:.6f} -> "
                    f"{h[-1, 0]:.6f} (truth {truth_llk:.6f})")
            if exact:
                o0, p0, h0 = ends[False, algorithm]
                line += ("; paraxial vs exact (max, L2) object %.1e %.1e, probe %.1e %.1e, "
                         % (*ls.mg.rel(o0, o), *ls.mg.rel(p0, pr))
                         + f"LLK {np.abs(h0[:, 0] / h[:, 0] - 1).max():.1e}")
            print(line, flush=True)
            assert np.isfinite(h).all() and h[-1, 0] < h[0, 0], (algorithm, h[:, 0])
            r = len(out["runs"])
            out["runs"].append([r, int(exact), a, CYCLES])
            out["llk"].append(np.column_stack([np.full(CYCLES, r), np.arange(CYCLES), h]))
            out["run_obj"].append(near.lead(r, ls.rows_obj(o)))
            out["run_probe"].append(near.lead(r, ls.rows_probe(pr)))

    def save(name, rows, fmt):
        path = f"{d}/pynx_near_{name}_f64.txt"
        np.savetxt(path, rows, fmt=fmt)
        print(f"wrote {path} ({len(rows)} rows)")

    f = "%.17e"
    save("case", np.array([[*GEOMETRY, g.kind, g.k, truth_llk, kernel]]),
         [f, f, f, "%d", f, f, f])
    save("corners", np.column_stack([np.arange(nframe), corners]), "%d")
    save("obj", *ls.rows_obj(obj))
    save("probe", *ls.rows_probe(probe))
    save("counts", np.column_stack([np.arange(nframe), counts.reshape(nframe, -1)]), "%d")
    save("runs", np.array(out["runs"]), "%d")
    save("llk", np.concatenate(out["llk"]), ["%d", "%d"] + [f] * 4)
    for name in ("run_obj", "run_probe"):
        save(name, np.concatenate([r for r, _ in out[name]]), out[name][0][1])


if __name__ == "__main__":
    main()
