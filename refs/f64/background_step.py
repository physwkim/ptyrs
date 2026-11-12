"""Float64 reference of the incoherent background: the costs' gradient and curvature with respect
to each detector pixel's background and PyNX's AP background update, at a fixed state.

    <engine>/target/release/examples/llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/background_step.py <dir> [--cuda]

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames, no
masked pixel) plus a background b(u), one detector plane added to every frame, as ptyrs::refgen
adds it (Background::plane, detector order, DC at [ny/2, nx/2]):

    b = 2 + 50 exp(-r^2 / (2 3^2)),  r the distance in pixels from [ny/2, nx/2]

In the engine's convention otherwise (ml_grad's P, q, corners and F = FFT_ortho fftshift):

    icalc_k = sum_m |F(P_m q V_k)|^2 + b
    L = sum (icalc - iobs + iobs ln(iobs / icalc))   (icalc where iobs = 0; ops::llk)
    F = sum (sqrt(icalc) - sqrt(iobs))^2

Both are sums of per-pixel terms in b, so the derivatives are per pixel:

    dL/db = sum_k (1 - iobs_k / icalc_k),        d2L/db2 = sum_k iobs_k / icalc_k^2
    dF/db = sum_k (1 - sqrt(iobs_k / icalc_k)),  d2F/db2 = sum_k sqrt(iobs_k / icalc_k)
                                                                / (2 icalc_k)

PyNX's AP background update (cu_operator ApplyAmplitude with update_background:
ProjectionAmplitudeBackgroundMode, then cu_background_update_mode, no smoothing) is the
multiplicative fixed point of L in b, over the N = 40 frames:

    b' = b sum_k iobs_k / max(icalc_k, 1e-12) / N  =  b - (b / N) dL/db   (no masked pixel)

Writes <dir>/background_f64.txt, one row per detector pixel in detector order (as the CXI file,
PyNX's set_background / get_background and refgen hold it), row-major, `iy ix b dL/db dF/db
d2L/db2 d2F/db2 b'_ap`, and <dir>/background_values_f64.txt, one row `nb_obs L F` at that
state. Printed checks: central differences of each pixel's L and F terms and of the gradient;
b'_ap against the gradient identity; L after the update. With --cuda (float32, the visible GPU):
PyNX's AP update itself, and PyNX's ML background gradient (GradPoissonFourier), which writes
-sum (1 - iobs/icalc) over the frames of its current stack only, so after a pass it holds minus
the last stack's part of dL/db (its source marks the sign "check sign"); ML then steps the
background along it with the object's gamma.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fftshift
from scipy.special import xlogy

AX = (-2, -1)
FLAT, PEAK, WIDTH = 2.0, 50.0, 3.0
STEPS = (1e-3, 1e-4, 1e-5)


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mg = sibling("ml_grad")


def plane(ny, nx):
    """ptyrs::refgen Background::plane, detector order."""
    y, x = np.arange(ny)[:, None] - ny // 2, np.arange(nx)[None, :] - nx // 2
    return FLAT + PEAK * np.exp(-(y**2 + x**2) / (2 * WIDTH**2))


def pixel_costs(m, i, b):
    """Per-pixel L and F terms summed over frames, corner order, [2, ny, nx]."""
    ic, t = i + b, m.iobs
    return np.stack([(ic - t + xlogy(t, t) - xlogy(t, ic)).sum(0),
                     ((np.sqrt(ic) - np.sqrt(t)) ** 2).sum(0)])


def derivs(m, i, b):
    """dL/db, dF/db, d2L/db2, d2F/db2, corner order, [4, ny, nx]."""
    ic, t = i + b, m.iobs
    r = np.sqrt(t / ic)
    return np.stack([(1 - t / ic).sum(0), (1 - r).sum(0), (t / ic**2).sum(0),
                     (r / (2 * ic)).sum(0)])


def ap_update(m, i, b):
    return b * (m.iobs / np.maximum(i + b, 1e-12)).sum(0) / len(m.iobs)


def pynx_cuda(args, b_det):
    """PyNX CUDA (float32): Grad's background gradient and AP's updated background, both in
    detector order, and the CUDA stack size."""
    import pycuda.gpuarray as cua
    import pynx.ptycho.cu_operator as cuop

    class BackgroundGrad(cuop.CUOperatorPtycho):
        def op(self, p):
            p._cu_background_grad = cua.zeros_like(p._cu_background)
            p = cuop.Grad(update_object=False, update_background=True, reg_fac_obj=None,
                          reg_fac_probe=None) * p
            p._background_grad = p._cu_background_grad.get()
            return p

    p = mg.build(*args, f64=False)
    p.set_background(b_det)
    p = BackgroundGrad() * p
    grad = fftshift(p._background_grad).astype(np.float64)
    p = mg.build(*args, f64=False)
    p.set_background(b_det)
    p = cuop.AP(update_object=False, update_probe=False, update_background=True,
                calc_llk=False) * p
    stack = int(cuop.default_processing_unit.cu_stack_size)
    return grad, p.get_background().astype(np.float64), stack


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, _, obj0, _ = args
    m = mg.Model(*args)
    assert m.valid.all(), "no masked pixel"
    nf, ny, nx = iobs.shape
    nb_obs = int(m.valid.sum())
    b_det = plane(ny, nx)
    b = fftshift(b_det)  # corner order, as m.iobs
    i = m.icalc(m.views(obj0[0]))
    lf = pixel_costs(m, i, b).sum(AX)
    g = derivs(m, i, b)
    print(f"{nf} frames, b in [{b.min():.4f}, {b.max():.4f}]; L {lf[0]:.17e}, F {lf[1]:.17e}; "
          f"max |dL/db| {np.abs(g[0]).max():.6e}, max |dF/db| {np.abs(g[1]).max():.6e}")
    print("central differences, (max, L2) rel diff (dL/db; dF/db; d2L/db2; d2F/db2):")
    for h in STEPS:
        c = (pixel_costs(m, i, b + h) - pixel_costs(m, i, b - h)) / (2 * h)
        cc = (derivs(m, i, b + h)[:2] - derivs(m, i, b - h)[:2]) / (2 * h)
        out = [*mg.rel(c[0], g[0]), *mg.rel(c[1], g[1]), *mg.rel(cc[0], g[2]),
               *mg.rel(cc[1], g[3])]
        print(f"  h {h:.0e}: " + "; ".join("%.3e %.3e" % tuple(out[j:j + 2])
                                         for j in range(0, 8, 2)))

    new = ap_update(m, i, b)
    print("AP update b' vs b - (b / N) dL/db (max, L2) %.3e %.3e" % mg.rel(new, b - b / nf * g[0]))
    after = pixel_costs(m, i, new).sum(AX)
    print(f"  b' in [{new.min():.4f}, {new.max():.4f}]; L {lf[0]:.10e} -> {after[0]:.10e}, "
          f"F {lf[1]:.10e} -> {after[1]:.10e}")

    if "--cuda" in sys.argv[2:]:
        grad, ap, stack = pynx_cuda(args, b_det)
        last = (nf - 1) // stack * stack
        part = fftshift((1 - m.iobs[last:] / (i[last:] + b)).sum(0))
        print(f"PyNX CUDA (float32, stack {stack}):")
        print("  AP updated background vs b' (max, L2) %.3e %.3e" % mg.rel(ap, fftshift(new)))
        k, res = mg.fit(grad, fftshift(g[0]))
        print(f"  Grad background_grad: fit to dL/db k = {k.real:.6f}, residual {res:.3e}; vs "
              f"minus frames {last}..{nf - 1} of dL/db (max, L2) %.3e %.3e" % mg.rel(grad, -part))

    rows = np.column_stack([np.repeat(np.arange(ny), nx), np.tile(np.arange(nx), ny),
                            b_det.ravel(), *(fftshift(a).ravel() for a in g),
                            fftshift(new).ravel()])
    np.savetxt(f"{d}/background_f64.txt", rows, fmt=["%d", "%d"] + ["%.17e"] * 6)
    with open(f"{d}/background_values_f64.txt", "w") as f:
        f.write(f"{nb_obs} {lf[0]:.17e} {lf[1]:.17e}\n")
    print(f"wrote {d}/background_f64.txt ({len(rows)} rows), {d}/background_values_f64.txt")


if __name__ == "__main__":
    main()
