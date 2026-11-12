"""Float64 reference of the Poisson and amplitude costs' gradients with respect to each frame's
position, at a fixed state with sub-pixel positions.

    <engine>/target/release/examples/llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/pos_grad.py <dir>

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames, no
masked pixel) with frame k's window start moved from its PyNX corner c_k (ml_grad's corners)
to s_k = c_k + delta_k, delta_k uniform in [0.25, 0.75]^2 from numpy default_rng(0), so that
central differences up to h = 1e-3 stay inside one pixel. The sub-pixel model is pty-chi's
PATCH object sampler with Fourier interpolation and pad 4, the model of its BH (object patches
shifted, apply_subpixel_shifts_on_probe=False; refs/ptychi/run_m3.py pad_for_shift = 4), in
the engine's convention otherwise (ml_grad's P, q and F = FFT_ortho fftshift):

    S_k = O[fl_k - 4 : fl_k + n + 4], fl_k = floor(s_k), zero outside O, shape (ny+8, nx+8)
    V_k(u) = DFT^-1(DFT(S_k) exp(2 pi i (f_y fr_ky + f_x fr_kx)))(u + 4),  fr_k = s_k - fl_k
    f_y, f_x = fftfreq(ny + 8), fftfreq(nx + 8) (Nyquist at -1/2), DFT unnormalised
    psi_km = F(P_m q V_k),  icalc_k = sum_m |psi_km|^2

V_k(u) interpolates O(u + s_k): a larger s_ky moves the window to larger iy. Costs over all
frames and pixels:

    L = sum (icalc - iobs + iobs ln(iobs / icalc))   (icalc where iobs = 0; ops::llk)
    F = sum (sqrt(icalc) - sqrt(iobs))^2               (BH's cost without its 1e-8)

and their gradients, with dV_k/ds_ky the same inverse DFT times 2 pi i f_y (x likewise):

    dC/ds_ky = sum_u Re(conj(b_k) dV_k/ds_ky),  b_k = sum_m conj(P_m q) g_km = 2 dC/dV_k*
    g_km = 2 fftshift(IFFT_ortho(r_k psi_km)),  r = 1 - iobs / icalc (L), 1 - sqrt(iobs / icalc) (F)

Writes <dir>/pos_grad_f64.txt, one row per frame, `k s_y s_x dL/ds_y dL/ds_x dF/ds_y dF/ds_x`
(s in object pixels, gradients per object pixel), and <dir>/pos_grad_values_f64.txt, one row
`nb_obs L F` at that state. Printed checks: central differences of each frame's terms of L and
F; pty-chi's BH forward model (L and F from its intensities) and its torch autograd gradient
with respect to the positions; pty-chi BH's own gradient_pos, which under PATCH Fourier-
differentiates the cropped n x n patch on its own periodic grid instead of dV_k/ds, against
the exact gradient and against that estimate computed here.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.special import xlogy

AX = (-2, -1)
PAD = 4
SEED = 0
STEPS = (1e-3, 1e-4, 1e-5)
EPS = 1e-8  # pty-chi BH's amplitude offset, in its gradient_pos only


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mg = sibling("ml_grad")


def freqs(shape):
    return np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]


def sample(obj, starts, n, axis=None):
    """V_k (axis None) or dV_k/ds_k along axis 0 (y) or 1 (x), [nf, ny, nx]."""
    ny, nx = n
    h, w = ny + 2 * PAD, nx + 2 * PAD
    fl = np.floor(starts).astype(int)
    fr = starts - fl
    big = np.pad(obj, ((h, h), (w, w)))
    s = np.stack([big[cy - PAD + h:cy - PAD + 2 * h, cx - PAD + w:cx - PAD + 2 * w]
                  for cy, cx in fl])
    fy, fx = freqs((h, w))
    mult = np.exp(2j * np.pi * (fy * fr[:, 0, None, None] + fx * fr[:, 1, None, None]))
    if axis is not None:
        mult = mult * 2j * np.pi * (fy, fx)[axis]
    return ifft2(fft2(s) * mult)[:, PAD:PAD + ny, PAD:PAD + nx]


def frame_costs(m, v):
    """Per-frame L and F terms, [2, nf]."""
    ic = m.icalc(v)
    t = m.iobs
    lk = (ic - t + xlogy(t, t) - xlogy(t, ic)).sum(AX)
    fk = ((np.sqrt(ic) - np.sqrt(t)) ** 2).sum(AX)
    return np.stack([lk, fk])


def view_grads(m, v, eps=0.0):
    """b_k = 2 dC/dV_k* for C = L and F, [2, nf, ny, nx]."""
    psi = m.psi(v)
    ic = (np.abs(psi) ** 2).sum(-3)
    out = []
    for r in (1 - m.iobs / ic, 1 - np.sqrt(m.iobs) / (np.sqrt(ic) + eps)):
        g = 2 * fftshift(ifft2(r[:, None] * psi, norm="ortho"), axes=AX)
        out.append((np.conj(m.pq) * g).sum(-3))
    return np.stack(out)


def pos_grads(m, obj, starts):
    """[nf, 4]: dL/ds_y, dL/ds_x, dF/ds_y, dF/ds_x."""
    b = view_grads(m, sample(obj, starts, m.n))
    d = [sample(obj, starts, m.n, axis) for axis in (0, 1)]
    return np.stack([np.real(np.conj(b[c]) * d[a]).sum(AX) for c in (0, 1) for a in (0, 1)], -1)


def central(m, obj, starts, h):
    """Per-frame central differences in the order of pos_grads: frame k's terms depend on s_k
    only, so every frame moves at once."""
    cols = []
    for a in (0, 1):
        e = np.zeros(2)
        e[a] = h
        cols.append((frame_costs(m, sample(obj, starts + e, m.n))
                     - frame_costs(m, sample(obj, starts - e, m.n))) / (2 * h))
    return np.stack([cols[0][0], cols[1][0], cols[0][1], cols[1][1]], -1)


def patch_estimate(m, obj, starts):
    """pty-chi BH's PATCH gradient_pos: the cropped V_k Fourier-differentiated on its own
    n x n grid, against b_k of F with the 1e-8 offset. [nf, 2]."""
    v = sample(obj, starts, m.n)
    b = view_grads(m, v, EPS)[1]
    fy, fx = freqs(m.n)
    spec = fft2(v)
    return np.stack([np.real(np.conj(b) * ifft2(spec * 2j * np.pi * f)).sum(AX)
                     for f in (fy, fx)], -1)


def ptychi(m, iobs, wavelength, obj, probe, starts):
    """pty-chi BHReconstructor's forward model at the positions s (origin (n - 1) / 2, so a
    position is the window start): L and F from its intensities, their autograd gradients with
    respect to the positions, and its gradient_pos."""
    import torch
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    nf, ny, nx = iobs.shape
    opts = api.BHOptions()
    opts.data_options.wavelength_m = wavelength
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = False
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    opts.probe_options.optimizable = False
    opts.probe_position_options.optimizable = True
    r = opts.reconstructor_options
    r.batch_size = nf
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    r.forward_model_options.pad_for_shift = PAD
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj.copy(),
        probe_data=(probe * m.q / np.sqrt(ny * nx))[None].copy(),
        probe_position_y_px=starts[:, 0].copy(),
        probe_position_x_px=starts[:, 1].copy(),
    )
    rec = task.reconstructor
    pos = rec.parameter_group.probe_positions.tensor
    t = task.dataset.patterns.to(pos.device)
    idx = torch.arange(nf)
    out = {}
    torch.set_grad_enabled(True)
    for name, cost in (
            ("L", lambda y: (y - t + torch.xlogy(t, t) - torch.xlogy(t, y)).sum()),
            ("F", lambda y: ((torch.sqrt(y) - torch.sqrt(t)) ** 2).sum())):
        pos.grad = None
        c = cost(rec.forward_model.forward(idx))
        c.backward()
        out[name] = float(c)
        out[f"d{name}"] = pos.grad.detach().cpu().numpy().copy()
    with torch.no_grad():
        rec.forward_model.forward(idx)
        iv = rec.forward_model.intermediate_variables
        rec.current_constrained_pixel_mask = rec.get_constrained_pixel_mask(t)[:, None]
        grad_f = rec.gradientF(iv["psi_far"], torch.sqrt(t)[:, None])
        p = rec.parameter_group.probe.get_opr_mode(0)
        out["gradient_pos"] = rec.gradient_pos(iv["obj_patches"], p, pos[idx], grad_f).numpy()
    return out


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    assert m.valid.all(), "no masked pixel"
    obj = obj0[0]
    nf = len(iobs)
    nb_obs = int(m.valid.sum())
    starts = m.corners + np.random.default_rng(SEED).uniform(0.25, 0.75, (nf, 2))
    lf = frame_costs(m, sample(obj, starts, m.n)).sum(1)
    g = pos_grads(m, obj, starts)
    print(f"{nf} frames, pad {PAD}, s - c in [{(starts - m.corners).min():.3f}, "
          f"{(starts - m.corners).max():.3f}]; L {lf[0]:.17e}, F {lf[1]:.17e}; "
          f"max |dL/ds| {np.abs(g[:, :2]).max():.6e}, max |dF/ds| {np.abs(g[:, 2:]).max():.6e}")
    print("central differences, max |num - g| / max |g| (dL/ds, dF/ds):")
    for h in STEPS:
        c = central(m, obj, starts, h)
        print(f"  h {h:.0e}: {rel(c[:, :2], g[:, :2]):.3e} {rel(c[:, 2:], g[:, 2:]):.3e}")

    pc = ptychi(m, iobs, wavelength, obj0, probe, starts)
    print(f"pty-chi BH forward (PATCH, Fourier, pad {PAD}): "
          f"L rel {abs(pc['L'] - lf[0]) / lf[0]:.3e}, F rel {abs(pc['F'] - lf[1]) / lf[1]:.3e}")
    print(f"pty-chi autograd d/d(positions), max rel: L {rel(pc['dL'], g[:, :2]):.3e}, "
          f"F {rel(pc['dF'], g[:, 2:]):.3e}")
    est = patch_estimate(m, obj, starts)
    gp = pc["gradient_pos"]
    cos = (gp * g[:, 2:]).sum() / np.linalg.norm(gp) / np.linalg.norm(g[:, 2:])
    print(f"pty-chi BH gradient_pos (PATCH estimate): vs the estimate here {rel(gp, est):.3e}; "
          f"vs exact dF/ds max rel {rel(gp, g[:, 2:]):.3e}, rel L2 "
          f"{np.linalg.norm(gp - g[:, 2:]) / np.linalg.norm(g[:, 2:]):.3e}, cosine {cos:.6f}")

    rows = np.column_stack([np.arange(nf), starts, g])
    np.savetxt(f"{d}/pos_grad_f64.txt", rows, fmt=["%d"] + ["%.17e"] * 6)
    with open(f"{d}/pos_grad_values_f64.txt", "w") as f:
        f.write(f"{nb_obs} {lf[0]:.17e} {lf[1]:.17e}\n")
    print(f"wrote {d}/pos_grad_f64.txt ({nf} rows), {d}/pos_grad_values_f64.txt")


if __name__ == "__main__":
    main()
