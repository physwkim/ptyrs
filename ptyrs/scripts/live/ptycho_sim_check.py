"""Check the M5 scan simulator's ground-truth dump against an independent
numpy forward model: the ptycho_truth example writes object, probe,
positions, offsets and frames for a raster and a Fermat scan; every frame
is compared with Poisson statistics around the convention in meta.json,
computed here with numpy's FFT.

  cargo build --release -p ptyrs --examples
  micromamba run -n epicsrs-pva python ptyrs/scripts/live/ptycho_sim_check.py --out DIR
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def run(target, out, extra):
    p = subprocess.run([f"{target}/examples/ptycho_truth", "--model", "ptycho", "--truth", out]
                       + extra, capture_output=True, text=True)
    print(p.stdout, end="")
    if p.returncode != 0:
        print(p.stderr, end="")
        sys.exit(f"ptycho_truth exit {p.returncode}")


def check_dump(d, check):
    meta = json.load(open(f"{d}/meta.json"))
    n, npts, dx = meta["n"], meta["points"], meta["obj_pixel_m"]
    obj = np.load(f"{d}/object.npy")
    probe = np.load(f"{d}/probe.npy")
    pos = np.load(f"{d}/positions.npy")
    off = np.load(f"{d}/offsets.npy")
    frames = np.load(f"{d}/frames.npy")
    tag = meta["path"]
    check(f"{tag}-dtypes-shapes",
          (obj.dtype, probe.dtype, pos.dtype, off.dtype, frames.dtype)
          == (np.complex64, np.complex64, np.float64, np.int64, np.uint16)
          and list(obj.shape) == meta["object_shape"] and probe.shape == (2, n, n)
          and pos.shape == (npts, 2) and off.shape == (npts, 2) and frames.shape == (npts, n, n),
          f"{obj.shape} {probe.shape} {pos.shape} {off.shape} {frames.shape}")
    # Offsets are the whole-pixel positions less their minimum; windows
    # span the object exactly.
    pix = np.round(pos / dx).astype(np.int64)[:, ::-1]
    check(f"{tag}-positions-on-pixel-grid", np.allclose(pix * dx, pos[:, ::-1], rtol=0, atol=1e-18))
    check(f"{tag}-offsets-from-positions", (off == pix - pix.min(axis=0)).all())
    check(f"{tag}-windows-span-object",
          (off.min(axis=0) == 0).all() and (off.max(axis=0) + n == obj.shape).all())
    # Scan density: points inside the inner half of the scan radius against
    # one per step² (raster and Fermat alike).
    r = np.hypot(*(pos / dx).T)
    rho = 0.5 * r.max()
    inside, want = (r < rho).sum(), np.pi * rho ** 2 / meta["step_px"] ** 2
    print(f"ptycho_sim_check.py {tag} points={npts} object={obj.shape} inside_half_radius={inside} "
          f"one_per_step2={want:.0f}")
    check(f"{tag}-density", abs(inside - want) < 2 * np.pi * rho / meta["step_px"] + 2,
          f"{inside} vs {want:.0f}")
    power = (np.abs(probe.astype(np.complex128)) ** 2).sum(axis=(1, 2))
    check(f"{tag}-mode-power", np.allclose(power, np.array(meta["mode_power"]) * meta["photons"],
                                           rtol=1e-5), f"{power}")
    p0, p1 = probe.astype(np.complex128)
    overlap = abs(np.vdot(p0, p1)) / np.sqrt(power[0] * power[1])
    check(f"{tag}-modes-orthogonal", overlap < 1e-6, f"{overlap:.3g}")
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    w0 = np.abs(p0) ** 2 / power[0]
    rms = np.sqrt((w0 * (xx ** 2 + yy ** 2)).sum())
    print(f"ptycho_sim_check.py {tag} probe_mode0_rms_radius_px={rms:.1f} "
          f"centroid=({(w0 * yy).sum():.2e},{(w0 * xx).sum():.2e})")
    # Forward model in numpy, frame by frame.
    o = obj.astype(np.complex128)
    z_max, chi_num, chi_n, resid, total, peak = 0.0, 0.0, 0, 0.0, 0.0, 0
    for k in range(npts):
        oy, ox = off[k]
        exit_ = probe.astype(np.complex128) * o[oy:oy + n, ox:ox + n]
        inten = (np.abs(np.fft.fftshift(np.fft.fft2(exit_, norm="ortho"), axes=(-2, -1))) ** 2).sum(0)
        c = frames[k].astype(np.float64)
        z_max = max(z_max, abs(c.sum() - inten.sum()) / np.sqrt(inten.sum()))
        lit = inten > 10
        chi_num += ((c[lit] - inten[lit]) ** 2 / inten[lit]).sum()
        chi_n += lit.sum()
        resid += (c - inten).sum()
        total += inten.sum()
        peak = max(peak, int(frames[k].max()))
    chi = chi_num / chi_n
    print(f"ptycho_sim_check.py {tag} frame_total_z_max={z_max:.2f} chi2_per_pixel(I>10)={chi:.4f} "
          f"over={chi_n} rel_resid={resid / total:.2e} mean_photons={total / npts:.4g} peak={peak}")
    check(f"{tag}-frame-totals-poisson", z_max < 5, f"{z_max:.2f}")
    check(f"{tag}-chi2-per-pixel", abs(chi - 1) < 0.01, f"{chi:.4f}")
    check(f"{tag}-no-clipping", peak < 65535, f"{peak}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=f"{root}/target/release")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bad, ok = [], {}

    def check(what, good, detail=""):
        ok[what] = ok.get(what, 0) + bool(good)
        if not good:
            bad.append(f"{what} {detail}")

    scans = {
        "raster": ["--path", "raster", "--points", "150", "--cols", "13", "--step-px", "20"],
        "fermat": ["--path", "fermat", "--points", "200", "--step-px", "20"],
    }
    for tag, extra in scans.items():
        d = f"{a.out}/{tag}"
        run(a.target, d, extra)
        check_dump(d, check)
    for what, n in sorted(ok.items()):
        print(f"ptycho_sim_check.py {what} ok={n}")
    for b in bad:
        print(f"ptycho_sim_check.py FAIL {b}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
