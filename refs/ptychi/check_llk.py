"""Check m3ref's PyNX<->pty-chi conversion on the clean datasets at the truth and init states:
the same object/probe/positions through PyNX's CPU operators (ObjProbe2Psi, FT, LLK), through
pty-chi's forward model (m3ref.evaluate), and through a float64 numpy transcription of PyNX's
model give the same PyNX LLK, and match the generator's llk_truth at the truth state.

    CUDA_VISIBLE_DEVICES=GPU-<uuid> micromamba run -n ptyrs-ref python refs/ptychi/check_llk.py

The Poisson LLK at the init state is not asserted: it is dominated by float32 FFT rounding
(m3ref docstring; poisson_floor_share is recorded). Writes out/check_llk.json; exits non-zero if
an asserted relative difference exceeds TOL.
"""

import os
import sys

import numpy as np
from numpy.fft import fft2, fftshift, ifftshift
from scipy.special import xlogy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ref  # noqa: E402
from run_m3 import OUT, build_task, environment  # noqa: E402

TOL = 1e-5
KEYS = ("poisson", "gaussian", "euclidian")


def pynx_llk(ds, obj, probe):
    from pynx.ptycho import cpu_operator as cpuop
    from pynx.ptycho.ptycho import Ptycho, PtychoData

    g = ds.geometry
    data = PtychoData(
        iobs=ds.counts.astype(np.float32),
        positions=(ds.x_m, ds.y_m),
        detector_distance=g["distance_m"],
        pixel_size_detector=g["detector_pixel_m"],
        wavelength=g["wavelength_m"],
    )
    p = Ptycho(probe=probe, obj=obj, data=data)
    with np.errstate(divide="ignore"):
        p = cpuop.LLK() * cpuop.FT(scale=True) * cpuop.ObjProbe2Psi() * p
    d = p.get_llk()
    return {k: float(d[k]) for k in KEYS}


def f64_llk(ds, obj, probe):
    """PyNX's far-field model and LLK in float64 (cpu_operator.ObjProbe2Psi, FT, LLK)."""
    obj = obj[0].astype(np.complex128)
    probe = probe.astype(np.complex128)
    q = ds.quad_phase()
    (ny, nx), (nyo, nxo) = ds.frame, obj.shape
    tot = dict.fromkeys(KEYS, 0.0)
    for i in range(ds.nframe):
        cy = (nyo - ny) // 2 + int(round(ds.pos_px[i, 0]))
        cx = (nxo - nx) // 2 + int(round(ds.pos_px[i, 1]))
        view = obj[cy:cy + ny, cx:cx + nx] * q
        icalc = sum(np.abs(fft2(fftshift(p * view), norm="ortho")) ** 2 for p in probe)
        iobs = ifftshift(ds.counts[i].astype(np.float64))
        tot["poisson"] += (icalc - iobs + xlogy(iobs, iobs) - xlogy(iobs, icalc)).sum()
        tot["gaussian"] += ((iobs - icalc) ** 2 / (iobs + 1)).sum()
        tot["euclidian"] += 4 * ((np.sqrt(iobs) - np.sqrt(icalc)) ** 2).sum()
    nb = ds.counts.size
    return {k: v / nb for k, v in tot.items()}


def rel(a, b):
    return abs(a - b) / abs(b) if np.isfinite(a) and np.isfinite(b) else float("inf")


def main():
    rows = []
    worst = 0.0
    for name in ("fermat_clean", "raster_clean"):
        ds = m3ref.Dataset(name)
        for state in ("truth", "init"):
            obj, probe = ds.modes(state)
            pynx = pynx_llk(ds, obj, probe)
            f64 = f64_llk(ds, obj, probe)
            task, _ = build_task(ds, "lsqml", 64, state=state)
            ptychi = m3ref.evaluate(task)
            del task
            keys = KEYS if state == "truth" else ("gaussian", "euclidian")
            row = {
                "dataset": name,
                "state": state,
                "pynx_cpu_f32": pynx,
                "numpy_f64": f64,
                "ptychi_eval": ptychi,
                "asserted": list(keys),
                "rel_ptychi_vs_f64": {k: rel(ptychi[k], f64[k]) for k in KEYS},
                "rel_pynx_vs_f64": {k: rel(pynx[k], f64[k]) for k in KEYS},
                "rel_4mse_sqrt_vs_euclidian": rel(4 * ptychi["native_mse_sqrt"],
                                                  ptychi["euclidian"]),
            }
            checked = [row["rel_ptychi_vs_f64"][k] for k in keys]
            checked += [row["rel_pynx_vs_f64"][k] for k in keys]
            if state == "truth":
                t = ds.meta["llk_truth"]
                row["generator_llk_truth"] = t
                row["rel_f64_vs_generator"] = {k: rel(f64[k], t[k]) for k in KEYS}
                checked += list(row["rel_f64_vs_generator"].values())
            row["worst_asserted_rel"] = max(checked)
            worst = max(worst, row["worst_asserted_rel"])
            rows.append(row)
            print(name, state, "worst", f"{row['worst_asserted_rel']:.2e}", "poisson",
                  pynx["poisson"], ptychi["poisson"], f64["poisson"],
                  "floor share", f"{ptychi['poisson_floor_share']:.3f}",
                  "4mse/eucl", f"{row['rel_4mse_sqrt_vs_euclidian']:.2e}", flush=True)
    result = {"tol": TOL, "worst_rel": worst, "environment": environment(), "checks": rows}
    m3ref.dump(result, os.path.join(OUT, "check_llk.json"))
    print(f"worst asserted relative difference {worst:.3e} (tol {TOL:g})")
    sys.exit(0 if worst <= TOL else 1)


if __name__ == "__main__":
    main()
