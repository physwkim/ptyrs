"""Float64 reference of one pty-chi LSQML minibatch step (object and probe) at a fixed state.

    <engine>/target/release/examples/llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_step.py <dir>

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames). The
step is pty-chi's LSQMLReconstructor.run_minibatch on frames 0-19 (BATCH, scan order) with the
M3 options of refs/ptychi/run_m3.py, written here in the engine's convention (PyNX probe P,
psi_m = FFT_ortho(fftshift(P_m q O_view)), F = FFT_ortho fftshift, corners and q as ml_grad):

    c = 2 w (1 - sqrt(iobs) / (sqrt(icalc) + 1e-9)), w = 1/(2 sigma)^2 = 1, a_far = 0.5  (Gaussian)
    c = 1 - iobs / (icalc + 1e-6), a_far from PtychoShelves' xi solver: two passes from 0.5 of
        a = clip(a_prev/2 + sum(xi (icalc - iobs/(1 - a_prev xi))) / sum(xi^2 icalc) / 2, 0, 1),
        xi = 1 - iobs/(icalc + 1e-5), a_prev = the batch mean of the last pass      (Poisson)
    chi_m = F^-1(-a_far c psi_m)
    dO_hat = scatter(sum_m conj(P_m q) chi_m) / sqrt(I^2 + (0.1 max I)^2),
        I = sum over all 40 windows of sum_m |P_m|^2 (pty-chi's illumination map)
    dP_hat = conj(q) mean_batch(conj(O_view) chi_m)
    per frame, A [a_o, a_p] = b with dOP = view(dO_hat) P q, dPO = q dP_hat O_view, sums over
        modes and pixels: a11 = sum |dOP|^2 + 1.2e-7 n, a22 = sum |dPO|^2 + 1.2e-7 n
        (n = modes x pixels), each then + 0.1 x its batch mean, a12 = conj(a21) = sum dOP conj(dPO),
        b1 = sum Re(conj(dOP) chi), b2 = sum Re(conj(dPO) chi); a = 0.9 clip(Re a, 0) and
        a_o /= n_modes
    O += trim_mean(a_o, 0.1) dO_hat (MATLAB trimmean: k = ceil(n 0.05 - 0.5) from each end),
    P += mean(a_p) dP_hat.

Conversions from pty-chi, whose probe is P q / sqrt(ny nx) under an unnormalised FFT: its
exit waves and chi are the ones above / sqrt(ny nx), its illumination map I / (ny nx), its
probe direction q dP_hat / sqrt(ny nx); the object, dO_hat, every step size and every eps are
the same. Its lambda_0 = 1.2e-7 / (ny nx) per pixel and mode is the 1.2e-7 above. Departures
from a pty-chi epoch, stated: rescale_probe_intensity_in_first_epoch is off (otherwise epoch 0
only rescales the probe); the preconditioner is set to I before the step, where pty-chi's first
epoch averages I with I of a probe renormalised from the data (Object.update_preconditioner);
the Poisson a_far has no jitter (pty-chi adds torch.randn * 1e-2 per frame, unseedable here);
no post-epoch hook runs, so the object-probe ambiguity removal and mode orthogonalisation that
run_m3 leaves at pty-chi's defaults are not part of the step.

Writes, per noise model N in gaussian, poisson, in the engine's convention:
    <dir>/lsqml_N_obj_f64.txt    `iy ix re im`, the object change, every pixel, row-major
    <dir>/lsqml_N_probe_f64.txt  `m iy ix re im`, the probe change
    <dir>/lsqml_N_steps_f64.txt  `frame a_far a_o a_p`, the batch frames
Printed checks: with the eps set to 0, -2 dO_comb and -2 BATCH dP_hat (dO_comb the scatter
before preconditioning) against central differences of sum_i w_i L_i along a random direction,
w_i = a_far,i (Poisson) or 2 a_far,i (Gaussian), L_i the Poisson L of ml_grad.py or
sum (sqrt(icalc) - sqrt(iobs))^2; pty-chi's own step in f64 on CPU; the LLK before and after.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.special import xlogy

AX = (-2, -1)
BATCH = 20
NOISES = ("gaussian", "poisson")
LAMBDA0 = 1.2e-7
STEPS = (1e-4, 1e-5, 1e-6)


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mg = sibling("ml_grad")


def views(a, corners, n):
    ny, nx = n
    return np.stack([a[cy:cy + ny, cx:cx + nx] for cy, cx in corners])


def scatter(per_view, corners, shape):
    ny, nx = per_view.shape[-2:]
    out = np.zeros(shape, per_view.dtype)
    for (cy, cx), a in zip(corners, per_view, strict=True):
        out[cy:cy + ny, cx:cx + nx] += a
    return out


def far(pq, v):
    return fft2(fftshift(pq * v[:, None], axes=AX), norm="ortho")


def frame_llk(noise, yp, yt, ok):
    if noise == "poisson":
        t = yp - xlogy(np.maximum(yt, 0), yp)
    else:
        t = (np.sqrt(yp) - np.sqrt(np.maximum(yt, 0))) ** 2
    return np.where(ok, t, 0).sum(AX)


def trim_mean(x, fraction=0.1):
    """pty-chi maths.trim_mean (MATLAB trimmean rounding) of a 1-D array without NaN."""
    k = int(np.ceil(len(x) * fraction / 2 - 0.5))
    return np.sort(x)[k:len(x) - k].mean()


def step(m, obj, probe, idx, noise, eps=True):
    """One LSQML minibatch step in the engine's convention (module docstring)."""
    nm = probe.shape[0]
    pq = probe * m.q
    corners = m.corners[idx]
    v = views(obj, corners, m.n)
    psi = far(pq, v)
    yp = (np.abs(psi) ** 2).sum(1)
    yt, ok = m.iobs[idx], m.valid[idx]
    a_far = np.full(len(idx), 0.5)
    if noise == "gaussian":
        e = 1e-9 if eps else 0.0
        c = np.where(ok, 2 * (1 - np.sqrt(np.maximum(yt, 0)) / (np.sqrt(yp) + e)), 0)
        weight = 2 * a_far
    else:
        e1, e2 = (1e-6, 1e-5) if eps else (0.0, 0.0)
        c = np.where(ok, 1 - yt / (yp + e1), 0)
        xi = np.where(ok, 1 - yt / (yp + e2), 0)
        for _ in range(2):
            prev = a_far.mean()
            num = (xi * (yp - yt / (1 - prev * xi))).sum(AX)
            den = (xi**2 * yp).sum(AX)
            a_far = np.clip(0.5 * prev + 0.5 * np.where(den > 0, num / den, 0), 0, 1)
        weight = a_far
    chi = fftshift(ifft2(-(a_far[:, None, None] * c)[:, None] * psi, norm="ortho"), axes=AX)

    do_comb = scatter((chi * np.conj(pq)).sum(1), corners, obj.shape)
    illum = scatter(np.broadcast_to((np.abs(probe) ** 2).sum(0), (len(m.corners), *m.n)),
                    m.corners, obj.shape)
    do_hat = do_comb / np.sqrt(illum**2 + (0.1 * illum.max()) ** 2)
    dp_q = (chi * np.conj(v)[:, None]).mean(0)

    dop = views(do_hat, corners, m.n)[:, None] * pq
    dpo = dp_q[None] * v[:, None]
    s = (1, 2, 3)
    a11 = (np.abs(dop) ** 2 + LAMBDA0).sum(s)
    a22 = (np.abs(dpo) ** 2 + LAMBDA0).sum(s)
    a11 = a11 + 0.1 * a11.mean()
    a22 = a22 + 0.1 * a22.mean()
    a12 = (dop * np.conj(dpo)).sum(s)
    b = np.stack([np.real(np.conj(dop) * chi).sum(s), np.real(np.conj(dpo) * chi).sum(s)], 1)
    a = np.stack([a11, a12, np.conj(a12), a22], 1).reshape(-1, 2, 2)
    x = np.linalg.solve(a, b[..., None].astype(complex))[..., 0].real.clip(0, None)
    a_o = 0.9 * x[:, 0] / nm
    a_p = 0.9 * x[:, 1]
    return dict(a_far=a_far, a_o=a_o, a_p=a_p, weight=weight, illum=illum,
                do_comb=do_comb, dp_hat=np.conj(m.q) * dp_q,
                d_obj=trim_mean(a_o) * do_hat, d_probe=a_p.mean() * np.conj(m.q) * dp_q)


def weighted_llk(m, obj, probe, idx, noise, w):
    psi = far(probe * m.q, views(obj, m.corners[idx], m.n))
    return (w * frame_llk(noise, (np.abs(psi) ** 2).sum(1), m.iobs[idx], m.valid[idx])).sum()


def directional(f, x, d, h):
    return (f(x + h * d) - f(x - h * d)) / (2 * h)


def ptychi_step(m, iobs, wavelength, obj, probe, noise, nbatch):
    """pty-chi LSQMLReconstructor.run_minibatch in f64 on CPU on the first `nbatch` frames,
    mapped as in ptyrs/scripts/engine/compare_ptychi_rpie.py, with the preconditioner set to
    the illumination map and torch.randn returning zeros during the step."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
    opts = api.LSQMLOptions()
    opts.data_options.wavelength_m = wavelength
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = True
    o.build_preconditioner_with_all_modes = True
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    pr = opts.probe_options
    pr.optimizable = True
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    opts.probe_position_options.optimizable = False
    r = opts.reconstructor_options
    r.noise_model = {"poisson": api.NoiseModels.POISSON,
                     "gaussian": api.NoiseModels.GAUSSIAN}[noise]
    r.forward_model_options.subpixel_shift_all_probe_modes = True
    r.solve_step_sizes_only_using_first_probe_mode = False
    r.rescale_probe_intensity_in_first_epoch = False
    r.batch_size = nbatch
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj.copy(),
        probe_data=(probe * m.q / root_n)[None].copy(),
        probe_position_y_px=m.corners[:, 0].astype(np.float64),
        probe_position_x_px=m.corners[:, 1].astype(np.float64),
    )
    rec = task.reconstructor
    pg = rec.parameter_group
    calls = []
    randn = torch.randn

    def zeros(*args, **kwargs):
        calls.append(args)
        return torch.zeros(*args, **kwargs)

    with torch.no_grad():
        rec.run_pre_run_hooks()
        rec.run_pre_epoch_hooks()
        pg.object.preconditioner = pg.object.calculate_illumination_map(
            pg.probe, pg.probe_positions, use_all_modes=True)
        batch = next(iter(DataLoader(rec.dataset, batch_size=nbatch, shuffle=False)))
        input_data, y_true = rec.prepare_batch_data(batch)
        torch.randn = zeros
        try:
            rec.run_minibatch(input_data, y_true)
        finally:
            torch.randn = randn
    idx = input_data[0].numpy()
    buf = rec.reconstructor_buffers
    a_far = rec.alpha_psi_far.cpu().numpy().reshape(-1)
    return dict(
        idx=idx, randn_calls=len(calls),
        a_far=np.broadcast_to(a_far, idx.shape),
        a_o=buf.alpha_object_all_pos_all_slices[idx, 0].cpu().numpy(),
        a_p=buf.alpha_probe_all_pos[idx].cpu().numpy(),
        illum=pg.object.preconditioner.cpu().numpy() * ny * nx,
        d_obj=task.get_data_to_cpu("object", as_numpy=True)[0] - obj[0],
        d_probe=task.get_data_to_cpu("probe", as_numpy=True)[0] * np.conj(m.q) * root_n
        - probe,
    )


def rows_obj(a):
    ny, nx = a.shape
    return np.column_stack([np.repeat(np.arange(ny), nx), np.tile(np.arange(nx), ny),
                            a.real.ravel(), a.imag.ravel()]), ["%d", "%d", "%.17e", "%.17e"]


def rows_probe(a):
    g = np.indices(a.shape).reshape(3, -1)
    return (np.column_stack([*g, a.real.ravel(), a.imag.ravel()]),
            ["%d", "%d", "%d", "%.17e", "%.17e"])


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    obj = obj0[0]
    idx = np.arange(BATCH)
    rng = np.random.default_rng(0)
    d_obj = rng.standard_normal(obj.shape) + 1j * rng.standard_normal(obj.shape)
    d_probe = rng.standard_normal(probe.shape) + 1j * rng.standard_normal(probe.shape)
    d_obj *= np.abs(obj).max() / np.abs(d_obj).max()
    d_probe *= np.abs(probe).max() / np.abs(d_probe).max()
    print(f"object {obj.shape}, probe {probe.shape}, {len(m.corners)} frames, batch {idx}")
    for noise in NOISES:
        print(f"--- {noise}")
        s0 = step(m, obj, probe, idx, noise, eps=False)
        w = s0["weight"]
        go = -2 * s0["do_comb"]
        gp = -2 * BATCH * s0["dp_hat"]
        print("  directions vs central differences of sum_i w_i L_i (eps 0), rel diff:")
        for h in STEPS:
            fo = directional(lambda x: weighted_llk(m, x, probe, idx, noise, w), obj, d_obj, h)
            fp = directional(lambda x: weighted_llk(m, obj, x, idx, noise, w), probe, d_probe, h)
            eo, ep = np.vdot(go, d_obj).real, np.vdot(gp, d_probe).real
            print(f"    h {h:.0e}: object {abs(fo - eo) / abs(eo):.3e}  "
                  f"probe {abs(fp - ep) / abs(ep):.3e}")

        s = step(m, obj, probe, idx, noise)
        print("  eps 0 vs pty-chi eps (max, L2): object %.3e %.3e" % mg.rel(s0["d_obj"],
                                                                           s["d_obj"]),
              " probe %.3e %.3e" % mg.rel(s0["d_probe"], s["d_probe"]))
        print(f"  a_far {s['a_far'].min():.6f}..{s['a_far'].max():.6f}, "
              f"a_o {s['a_o'].min():.6e}..{s['a_o'].max():.6e} trim_mean "
              f"{trim_mean(s['a_o']):.6e}, a_p {s['a_p'].min():.6e}..{s['a_p'].max():.6e} "
              f"mean {s['a_p'].mean():.6e}")

        pc = ptychi_step(m, iobs, wavelength, obj0, probe, noise, BATCH)
        assert (pc["idx"] == idx).all(), pc["idx"]
        print(f"  pty-chi LSQML f64 ({pc['randn_calls']} torch.randn call(s) zeroed):")
        print("    illumination map (max, L2) %.3e %.3e" % mg.rel(pc["illum"], s["illum"]))
        for k in ("a_far", "a_o", "a_p", "d_obj", "d_probe"):
            print(f"    {k:8s} %.3e %.3e" % mg.rel(pc[k], s[k]))

        before = weighted_llk(m, obj, probe, idx, noise, 1.0)
        after = weighted_llk(m, obj + s["d_obj"], probe + s["d_probe"], idx, noise, 1.0)
        full = np.arange(len(m.corners))
        b_all = weighted_llk(m, obj, probe, full, noise, 1.0)
        a_all = weighted_llk(m, obj + s["d_obj"], probe + s["d_probe"], full, noise, 1.0)
        print(f"  {noise} LLK sum, batch {before:.10e} -> {after:.10e}, "
              f"all frames {b_all:.10e} -> {a_all:.10e}")

        for name, (rows, fmt) in (
                ("obj", rows_obj(s["d_obj"])), ("probe", rows_probe(s["d_probe"])),
                ("steps", (np.column_stack([idx, s["a_far"], s["a_o"], s["a_p"]]),
                           ["%d"] + ["%.17e"] * 3))):
            path = f"{d}/lsqml_{noise}_{name}_f64.txt"
            np.savetxt(path, rows, fmt=fmt)
            print(f"  wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
