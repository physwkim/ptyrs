"""Float64 reference of the Poisson and amplitude costs' gradients with respect to each frame's
position, at a fixed state with sub-pixel positions, the sub-pixel part applied to the probe.

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/pos_grad.py <dir>

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames, no
masked pixel) with frame k's window start moved from its PyNX corner c_k (ml_grad's corners)
to s_k = c_k + delta_k, delta_k uniform in [-0.4, 0.4]^2 from numpy default_rng(0), so that
round(s_k) = c_k and central differences up to h = 1e-3 never cross a half pixel. The model is
pty-chi LSQML's (apply_subpixel_shifts_on_probe, subpixel_shift_all_probe_modes, Fourier
interpolation, pad_for_shift 0): the object window starts at the whole pixel n_k = round(s_k)
and every probe mode (P_m q, ml_grad's pq) is translated by the remainder fr_k = s_k - n_k on
its own periodic n x n grid; otherwise the engine's convention (F = FFT_ortho fftshift):

    V_k = O[n_k : n_k + n],  W_km = DFT^-1(DFT(P_m q) e_k),  e_k = exp(-2 pi i f . fr_k)
    f = (f_y, f_x) = fftfreq(ny), fftfreq(nx) (Nyquist at -1/2), DFT unnormalised
    psi_km = F(W_km V_k),  icalc_k = sum_m |psi_km|^2

W_km(u) interpolates (P_m q)(u - fr_k): exit(u) = (P q)(u) O(u + s_k) with u -> u - fr_k, up
to the wrap of P q on its grid, and a detector phase ramp that |psi|^2 drops. Costs over all
frames and pixels:

    L = sum (icalc - iobs + iobs ln(iobs / icalc))   (icalc where iobs = 0; ops::llk)
    F = sum (sqrt(icalc) - sqrt(iobs))^2

and their gradients, with dW_km/ds_ky = DFT^-1(DFT(P_m q) e_k (-2 pi i f_y)) (x likewise):

    dC/ds_ky = sum_m sum_u Re(conj(g_km) V_k dW_km/ds_ky),  g_km = 2 dC/d(W_km V_k)*
    g_km = 2 fftshift(IFFT_ortho(r_k psi_km)),  r = 1 - iobs / icalc (L), 1 - sqrt(iobs / icalc) (F)

Writes <dir>/pos_grad_f64.txt, one row per frame, `k s_y s_x dL/ds_y dL/ds_x dF/ds_y dF/ds_x`
(s in object pixels, gradients per object pixel), and <dir>/pos_grad_values_f64.txt, one row
`nb_obs L F` at that state. Printed checks: central differences of each frame's terms of L and
F; pty-chi LSQML's forward model (L and F from its intensities) and its torch autograd gradient
with respect to the positions; the object-shift model of the previous reference (pty-chi BH's
PATCH sampler: the object patch, padded by 4, Fourier-translated by s_k - floor(s_k)) at the
same positions. P q wraps by nothing measurable (its power outside the inner (n-2)^2 is ~1e-13),
but this object's phase is white over the grid band, so the two models differ by the aliasing
of their exit waves; with the object low-passed, exp(-|f|^2 / (2 0.1^2)) on its DFT, icalc and
dicalc/ds of the two models agree, and the same comparison with the probe translated by -fr_k
or not at all gives the size of a wrong model.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.special import xlogy

AX = (-2, -1)
PAD = 4  # the object-shift comparison only
LOWPASS = 0.1  # cycles per pixel, likewise
SEED = 0
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


def freqs(shape):
    return np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]


def split(starts):
    whole = np.rint(starts).astype(int)
    return whole, starts - whole


def windows(m, obj, whole):
    ny, nx = m.n
    return np.stack([obj[cy:cy + ny, cx:cx + nx] for cy, cx in whole])


def probes(m, frac, axis=None):
    """W_km (axis None) or dW_km/ds_k along axis 0 (y) or 1 (x), [nf, nm, ny, nx]."""
    fy, fx = freqs(m.n)
    e = np.exp(-2j * np.pi * (fy * frac[:, 0, None, None] + fx * frac[:, 1, None, None]))
    if axis is not None:
        e = e * -2j * np.pi * (fy, fx)[axis]
    return ifft2(fft2(m.pq)[None] * e[:, None])


def psi(w, v):
    return fft2(fftshift(w * v[:, None], axes=AX), norm="ortho")


def frame_costs(m, ic):
    """Per-frame L and F terms, [2, nf]."""
    t = m.iobs
    lk = (ic - t + xlogy(t, t) - xlogy(t, ic)).sum(AX)
    fk = ((np.sqrt(ic) - np.sqrt(t)) ** 2).sum(AX)
    return np.stack([lk, fk])


def icalc(m, obj, starts):
    whole, frac = split(starts)
    return (np.abs(psi(probes(m, frac), windows(m, obj, whole))) ** 2).sum(-3)


def exit_grads(m, p):
    """g_km = 2 dC/d(exit_km)* for C = L and F, [2, nf, nm, ny, nx]."""
    ic = (np.abs(p) ** 2).sum(-3)
    return np.stack([2 * fftshift(ifft2(r[:, None] * p, norm="ortho"), axes=AX)
                     for r in (1 - m.iobs / ic, 1 - np.sqrt(m.iobs / ic))])


def pos_grads(m, obj, starts):
    """[nf, 4]: dL/ds_y, dL/ds_x, dF/ds_y, dF/ds_x."""
    whole, frac = split(starts)
    v = windows(m, obj, whole)
    g = exit_grads(m, psi(probes(m, frac), v))
    d = [probes(m, frac, axis) * v[:, None] for axis in (0, 1)]
    return np.stack([np.real(np.conj(g[c]) * d[a]).sum((-3, -2, -1))
                     for c in (0, 1) for a in (0, 1)], -1)


def central(m, obj, starts, h):
    """Per-frame central differences in the order of pos_grads: frame k's terms depend on s_k
    only, so every frame moves at once."""
    cols = []
    for a in (0, 1):
        e = np.zeros(2)
        e[a] = h
        cols.append((frame_costs(m, icalc(m, obj, starts + e))
                     - frame_costs(m, icalc(m, obj, starts - e))) / (2 * h))
    return np.stack([cols[0][0], cols[1][0], cols[0][1], cols[1][1]], -1)


def object_views(m, obj, starts, axis=None):
    """The previous reference's V_k, the padded object patch Fourier-translated by
    s_k - floor(s_k) (axis None), or dV_k/ds_k along axis 0 (y) or 1 (x), [nf, ny, nx]."""
    ny, nx = m.n
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


def object_shift(m, obj, starts):
    """Per-frame L and F terms [2, nf] and the gradients in the order of pos_grads under the
    object-shift model."""
    p = m.psi(object_views(m, obj, starts))
    b = (np.conj(m.pq) * exit_grads(m, p)).sum(-3)
    d = [object_views(m, obj, starts, axis) for axis in (0, 1)]
    grads = np.stack([np.real(np.conj(b[c]) * d[a]).sum(AX) for c in (0, 1) for a in (0, 1)], -1)
    return frame_costs(m, (np.abs(p) ** 2).sum(-3)), grads


def model_gap(m, obj, starts, h=1e-4):
    """Relative L2 gaps to the object-shift model: icalc for the probe translated by +fr, -fr
    and not at all, and dicalc/ds (central differences, both axes) for +fr."""
    whole, frac = split(starts)
    v = windows(m, obj, whole)
    io = (np.abs(m.psi(object_views(m, obj, starts))) ** 2).sum(-3)
    out = [((np.abs(psi(probes(m, f), v)) ** 2).sum(-3) - io) for f in (frac, -frac, 0 * frac)]
    out = [np.linalg.norm(a) / np.linalg.norm(io) for a in out]
    dp, do = [], []
    for a in (0, 1):
        e = np.zeros(2)
        e[a] = h
        dp.append((icalc(m, obj, starts + e) - icalc(m, obj, starts - e)) / (2 * h))
        do.append(((np.abs(m.psi(object_views(m, obj, starts + e))) ** 2).sum(-3)
                   - (np.abs(m.psi(object_views(m, obj, starts - e))) ** 2).sum(-3)) / (2 * h))
    dp, do = np.stack(dp), np.stack(do)
    return out + [np.linalg.norm(dp - do) / np.linalg.norm(do)]


def ptychi(m, iobs, wavelength, obj, probe, starts):
    """pty-chi LSQMLReconstructor's forward model at the positions s (origin (n - 1) / 2, so a
    position is the window start): L and F from its intensities and their autograd gradients
    with respect to the positions."""
    import torch
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    nf, ny, nx = iobs.shape
    opts = api.LSQMLOptions()
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
    fm = r.forward_model_options
    fm.shift_operator = api.ShiftOperators.PATCH
    fm.pad_for_shift = 0
    fm.subpixel_shift_all_probe_modes = True
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj.copy(),
        probe_data=(probe * m.q / np.sqrt(ny * nx))[None].copy(),
        probe_position_y_px=starts[:, 0].copy(),
        probe_position_x_px=starts[:, 1].copy(),
    )
    rec = task.reconstructor
    assert rec.forward_model.apply_subpixel_shifts_on_probe
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
    starts = m.corners + np.random.default_rng(SEED).uniform(-0.4, 0.4, (nf, 2))
    assert (split(starts)[0] == m.corners).all()
    lk = frame_costs(m, icalc(m, obj, starts))
    lf = lk.sum(1)
    g = pos_grads(m, obj, starts)
    _, frac = split(starts)
    print(f"{nf} frames, s - round(s) in [{frac.min():.3f}, {frac.max():.3f}]; "
          f"L {lf[0]:.17e}, F {lf[1]:.17e}; "
          f"max |dL/ds| {np.abs(g[:, :2]).max():.6e}, max |dF/ds| {np.abs(g[:, 2:]).max():.6e}")
    print("central differences, max |num - g| / max |g| (dL/ds, dF/ds):")
    for h in STEPS:
        c = central(m, obj, starts, h)
        print(f"  h {h:.0e}: {rel(c[:, :2], g[:, :2]):.3e} {rel(c[:, 2:], g[:, 2:]):.3e}")

    pc = ptychi(m, iobs, wavelength, obj0, probe, starts)
    print("pty-chi LSQML forward (probe shift, all modes, Fourier, pad 0): "
          f"L rel {abs(pc['L'] - lf[0]) / lf[0]:.3e}, F rel {abs(pc['F'] - lf[1]) / lf[1]:.3e}")
    print(f"pty-chi autograd d/d(positions), max rel: L {rel(pc['dL'], g[:, :2]):.3e}, "
          f"F {rel(pc['dF'], g[:, 2:]):.3e}")

    ok, og = object_shift(m, obj, starts)
    of = ok.sum(1)
    print(f"object-shift model (pad {PAD}) at the same s, this minus it: "
          f"L rel {(lf[0] - of[0]) / of[0]:.3e}, F rel {(lf[1] - of[1]) / of[1]:.3e}; "
          f"per frame max rel L {rel(lk[0], ok[0]):.3e}, F {rel(lk[1], ok[1]):.3e}")
    for c, name in ((slice(0, 2), "dL/ds"), (slice(2, 4), "dF/ds")):
        a, b = g[:, c], og[:, c]
        print(f"  {name}: max rel {rel(a, b):.3e}, rel L2 "
              f"{np.linalg.norm(a - b) / np.linalg.norm(b):.3e}")
    e = (np.abs(m.pq) ** 2).sum(0)
    print(f"  P q power outside the inner (n-2)^2: {1 - e[1:-1, 1:-1].sum() / e.sum():.3e}")
    fy, fx = freqs(obj.shape)
    for cut in (None, LOWPASS):
        o = obj if cut is None else ifft2(fft2(obj) * np.exp(-(fy**2 + fx**2) / (2 * cut**2)))
        gap = model_gap(m, o, starts)
        print(f"  object {'as is' if cut is None else f'low-passed {cut}'}: icalc rel L2 gap, "
              "probe by +fr %.3e, -fr %.3e, unshifted %.3e; dicalc/ds +fr %.3e" % tuple(gap))

    rows = np.column_stack([np.arange(nf), starts, g])
    np.savetxt(f"{d}/pos_grad_f64.txt", rows, fmt=["%d"] + ["%.17e"] * 6)
    with open(f"{d}/pos_grad_values_f64.txt", "w") as f:
        f.write(f"{nb_obs} {lf[0]:.17e} {lf[1]:.17e}\n")
    print(f"wrote {d}/pos_grad_f64.txt ({nf} rows), {d}/pos_grad_values_f64.txt")


if __name__ == "__main__":
    main()
