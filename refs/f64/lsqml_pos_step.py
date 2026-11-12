"""Float64 reference of pty-chi LSQML with probe position correction: one minibatch step at a
fixed state, without and with orthogonal probe relaxation, and three epochs.

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_pos_step.py <dir>

The state is the one lsqml_step.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames),
each frame at its corner plus an offset d_k (dy, dx), drawn uniform in [-1, 1) by numpy
default_rng(1) and scaled by OFFSET for the step and by EPOCH_OFFSET for the epochs, so that
every |d| stays below 1/2, where pty-chi's PATCH model keeps the window at the corner and moves
every probe mode by d with a Fourier shift. In the engine's convention (lsqml_step.py), with
T_d(x) = ifft2(fft2(x) exp(-2 pi i f.d)), f numpy's fftfreq, on a natural-layout plane:

    W_km = T_dk(P_km q),  psi_km = F(W_km V_k)
    c, a_far and chi_km = F^-1(-a_far c psi_km) as lsqml_step.py
    dO_hat = scatter(sum_m conj(W_km) chi_km) / sqrt(I^2 + (0.1 max I)^2), I as lsqml_step.py
        (pty-chi's illumination map takes the unmoved probe at the corners)
    m_km = T_-dk(conj(V_k) chi_km), pty-chi's adjoint-shifted delta_p_i,  dP_hat = conj(q)
        mean_batch m_k
    step sizes as lsqml_step.py with dOP = view(dO_hat) W_km and dPO = q dP_hat V_k
    O += trim_mean(a_o) dO_hat,  P += mean(a_p) dP_hat

and from the same state, pty-chi's gradient position correction (FOURIER_DIFFERENTIATION,
slice 0 of 1, SGD at lr POS_STEP, update_magnitude_limit POS_LIMIT, MAD clip on):

    dV = ifft(fft(V_k) 2 pi i f) along each axis,  g = dV W_k0
    delta_k = sum Re(conj(g) chi_k0) / (sum |g|^2 + max(max_batch sum |g|^2, 1e-6 N))
    dx = POS_STEP delta at the batch frames, 0 at the others (all 40)
    d += clip(dx, -L, L) per axis, L = min(10 mean |dx - mean dx|, POS_LIMIT) over all frames

(1e-6 N is pty-chi's 1e-6 on its probe, N = ny nx.) With OPR (K eigenmodes and weights drawn
as lsqml_opr_step.py draws them, on its batch), P_k0 = P_0 + sum_i w_ki U_i is moved as above
and OPRModeWeights takes lsqml_opr_step.py's step with its residue from the moved directions,
r_k = conj(q) m_k0 - mean_batch(conj(q) m_k0), and its weight numerator from the unmoved
eigenmode, sum Re(chi_k0 conj(U_i'' q V_k)).

The step files are this script's step, checked against pty-chi's own step in f64 on CPU
(lsqml_step.ptychi_step's options, positions optimizable, its preconditioner set to its
illumination map, torch.randn returning zeros). The epoch files are pty-chi's EPOCHS epochs of
random_batches(40, EPOCH_BATCH, SEED, epoch) as lsqml_epochs.py runs them, positions
optimizable.

Writes, in the engine's convention, `_g` Gaussian and `_p` Poisson:
    <dir>/lsqml_pos_obj_f64.txt         `iy ix re_g im_g re_p im_p`, the object change
    <dir>/lsqml_pos_probe_f64.txt       `m iy ix re_g im_g re_p im_p`, the probe change
    <dir>/lsqml_pos_steps_f64.txt       `k a_far_g a_o_g a_p_g a_far_p a_o_p a_p_p`, the batch
    <dir>/lsqml_pos_offsets_f64.txt     `k dy dx dy'_g dx'_g dy'_p dx'_p`, every frame
    <dir>/lsqml_pos_opr_*_f64.txt       the same four for the OPR step on lsqml_opr_step.py's
                                        batch, and `eigen`, `weights` as lsqml_opr_step.py's
                                        without its gradient columns
    <dir>/lsqml_pos_epochs_*_f64.txt    obj, probe and offsets as above over the epochs, and
                                        steps `epoch k a_far_g a_o_g a_p_g a_far_p a_o_p a_p_p`
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2

AX = (-2, -1)
BATCH = 20
NOISES = ("gaussian", "poisson")
LAMBDA0 = 1.2e-7
EPS_POS = 1e-6
POS_STEP = 0.3
POS_LIMIT = 0.1
OFFSET = 0.4
EPOCH_OFFSET = 0.15
EPOCHS = 3
EPOCH_BATCH = 16
SEED = 7


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mg = sibling("ml_grad")
ls = sibling("lsqml_step")
lo = sibling("lsqml_opr_step")
rb = sibling("random_batches")


def move(x, d):
    """T_d of every plane of x [nf, nm, ny, nx] (natural layout), frame k by d[k]."""
    ny, nx = x.shape[-2:]
    fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
    ph = np.exp(-2j * np.pi * (fy[None, :, None] * d[:, 0, None, None]
                               + fx[None, None, :] * d[:, 1, None, None]))
    return ifft2(fft2(x, axes=AX) * ph[:, None], axes=AX)


def fourier_gradient(v):
    """pty-chi image_proc.fourier_gradient: (dV/dy, dV/dx) of every plane, taken as periodic."""
    ny, nx = v.shape[-2:]
    fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
    gy = np.fft.ifft(np.fft.fft(v, axis=-2) * (2j * np.pi * fy)[:, None], axis=-2)
    gx = np.fft.ifft(np.fft.fft(v, axis=-1) * (2j * np.pi * fx), axis=-1)
    return gy, gx


def step(m, obj, probe, u, w, offsets, idx, noise):
    """One LSQML minibatch with position correction, and OPR when u holds eigenmodes."""
    nm, n = probe.shape[0], probe[0].size
    corners = m.corners[idx]
    v = ls.views(obj, corners, m.n)
    d = offsets[idx]
    pw = move(lo.probes(probe, u, w[idx]) * m.q, d)
    psi = ls.far(pw, v)
    yp = (np.abs(psi) ** 2).sum(1)
    yt, ok = m.iobs[idx], m.valid[idx]
    a_far = np.full(len(idx), 0.5)
    if noise == "gaussian":
        c = np.where(ok, 2 * (1 - np.sqrt(np.maximum(yt, 0)) / (np.sqrt(yp) + 1e-9)), 0)
    else:
        c = np.where(ok, 1 - yt / (yp + 1e-6), 0)
        xi = np.where(ok, 1 - yt / (yp + 1e-5), 0)
        for _ in range(2):
            prev = a_far.mean()
            num = (xi * (yp - yt / (1 - prev * xi))).sum(AX)
            den = (xi**2 * yp).sum(AX)
            a_far = np.clip(0.5 * prev + 0.5 * np.where(den > 0, num / den, 0), 0, 1)
    chi = fftshift(ifft2(-(a_far[:, None, None] * c)[:, None] * psi, norm="ortho"), axes=AX)

    do_comb = ls.scatter((chi * np.conj(pw)).sum(1), corners, obj.shape)
    illum = ls.scatter(np.broadcast_to((np.abs(probe) ** 2).sum(0), (len(m.corners), *m.n)),
                       m.corners, obj.shape)
    do_hat = do_comb / np.sqrt(illum**2 + (0.1 * illum.max()) ** 2)
    moved = move(chi * np.conj(v)[:, None], -d)
    dp_q = moved.mean(0)

    dop = ls.views(do_hat, corners, m.n)[:, None] * pw
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

    delta = []
    for g in fourier_gradient(v):
        gw = g * pw[:, 0]
        num = np.real(np.conj(gw) * chi[:, 0]).sum(AX)
        den = (np.abs(gw) ** 2).sum(AX)
        delta.append(num / (den + max(den.max(), EPS_POS * n)))
    dx = np.zeros_like(offsets)
    dx[idx] = POS_STEP * np.stack(delta, 1)
    limit = np.minimum(10 * np.abs(dx - dx.mean(0)).mean(0), POS_LIMIT)
    new_offsets = offsets + np.clip(dx, -limit, limit)

    u2, w2 = u.copy(), w.copy()
    if len(u):
        d0 = np.conj(m.q) * moved[:, 0]
        r = d0 - d0.mean(0)
        wb = w[idx]
        a_u = min(0.1, len(idx) / len(w))
        for i in range(len(u)):
            p = (np.real(np.conj(r) * u[i]).sum(AX) / n**2 + wb[:, i]) / (wb[:, i] ** 2).sum()
            e = (r * p[:, None, None]).mean(0)
            u1 = u[i] + a_u * np.sqrt(n) * e / lo.mnorm(e)
            u2[i] = np.sqrt(n) * u1 / lo.mnorm(u1)
            uqv = u2[i] * m.q * v
            num = np.real(chi[:, 0] * np.conj(uqv)).sum(AX)
            den = (np.abs(uqv) ** 2).sum(AX)
            w2[idx, i] = wb[:, i] + num / (den + 0.1 * den.mean())
            if i < len(u) - 1:
                r = r - ((r * np.conj(u2[i])).sum(AX, keepdims=True)
                         / (np.abs(u2[i]) ** 2).sum()) * u2[i]
    return dict(a_far=a_far, a_o=a_o, a_p=a_p, illum=illum, u=u2, w=w2, offsets=new_offsets,
                limit=limit, d_obj=ls.trim_mean(a_o) * do_hat,
                d_probe=a_p.mean() * np.conj(m.q) * dp_q)


def options(api, wavelength, ny, nx, noise, batch, opr):
    """lsqml_step.ptychi_step's options with the positions optimizable."""
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
    opts.probe_position_options.optimizable = True
    if opr:
        ow = opts.opr_mode_weight_options
        ow.optimizable = True
        ow.optimize_intensity_variation = False
        ow.optimize_eigenmode_weights = True
    r = opts.reconstructor_options
    r.noise_model = {"poisson": api.NoiseModels.POISSON,
                     "gaussian": api.NoiseModels.GAUSSIAN}[noise]
    r.forward_model_options.subpixel_shift_all_probe_modes = True
    r.solve_step_sizes_only_using_first_probe_mode = False
    r.rescale_probe_intensity_in_first_epoch = False
    r.batch_size = batch
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    return opts


def task_of(api, m, iobs, wavelength, obj, u, w, offsets, noise, batch):
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
    p = (m.probe * m.q / root_n)[None]
    extra = {}
    if len(u):
        eig = np.zeros((len(u), *p.shape[1:]), complex)
        eig[:, 0] = u * m.q / root_n
        p = np.concatenate([p, eig])
        extra["opr_mode_weights_data"] = np.column_stack([np.ones(len(w)), w])
    pos = m.corners.astype(np.float64) + offsets
    return PtychographyTask(
        options(api, wavelength, ny, nx, noise, batch, len(u) > 0),
        diffraction_data=iobs,
        object_data=obj.copy(),
        probe_data=p.copy(),
        probe_position_y_px=pos[:, 0].copy(),
        probe_position_x_px=pos[:, 1].copy(),
        **extra,
    )


def ptychi_step(m, iobs, wavelength, obj, u, w, offsets, idx, noise):
    """pty-chi's step on the frames idx, in the engine's convention."""
    import torch
    from torch.utils.data import DataLoader, Subset
    import ptychi.api as api

    _, ny, nx = iobs.shape
    back = np.conj(m.q) * np.sqrt(ny * nx)
    task = task_of(api, m, iobs, wavelength, obj, u, w, offsets, noise, len(idx))
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
        u0 = pg.probe.data[1:, 0].detach().cpu().numpy() * back
        loader = DataLoader(Subset(rec.dataset, idx.tolist()), batch_size=len(idx),
                            shuffle=False)
        input_data, y_true = rec.prepare_batch_data(next(iter(loader)))
        torch.randn = zeros
        try:
            rec.run_minibatch(input_data, y_true)
        finally:
            torch.randn = randn
    got = input_data[0].numpy()
    buf = rec.reconstructor_buffers
    probe = pg.probe.data.detach().cpu().numpy()
    out = dict(
        idx=got, u0=u0, calls=len(calls),
        a_far=np.broadcast_to(rec.alpha_psi_far.cpu().numpy().reshape(-1), got.shape),
        a_o=buf.alpha_object_all_pos_all_slices[got, 0].cpu().numpy(),
        a_p=buf.alpha_probe_all_pos[got].cpu().numpy(),
        illum=pg.object.preconditioner.cpu().numpy() * ny * nx,
        d_obj=pg.object.data.detach().cpu().numpy()[0] - obj[0],
        d_probe=probe[0] * back - m.probe,
        offsets=task.get_data_to_cpu("probe_positions", as_numpy=True) - m.corners)
    if len(u):
        out["u"] = probe[1:, 0] * back
        out["w"] = pg.opr_mode_weights.data.detach().cpu().numpy()[:, 1:]
    return out


def ptychi_epochs(m, iobs, wavelength, obj, offsets, noise):
    """pty-chi LSQML for EPOCHS epochs as lsqml_epochs.ptychi_epochs, positions optimizable."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api

    _, ny, nx = iobs.shape
    none = np.zeros((0, ny, nx), complex)
    task = task_of(api, m, iobs, wavelength, obj, none, np.zeros((len(m.corners), 0)),
                   offsets, noise, EPOCH_BATCH)
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset,
                                batch_sampler=rb.Sampler(len(m.corners), EPOCH_BATCH, SEED))
    cls = type(rec)
    buf = rec.reconstructor_buffers
    steps, peak = [], []

    def minibatch(input_data, y_true):
        out = cls.run_minibatch(rec, input_data, y_true)
        idx = input_data[0].numpy().copy()
        a_far = rec.alpha_psi_far.cpu().numpy().reshape(-1)
        steps.append(dict(
            epoch=rec.current_epoch, idx=idx, a_far=np.broadcast_to(a_far, idx.shape).copy(),
            a_o=buf.alpha_object_all_pos_all_slices[idx, 0].cpu().numpy().copy(),
            a_p=buf.alpha_probe_all_pos[idx].cpu().numpy().copy()))
        pos = task.get_data_to_cpu("probe_positions", as_numpy=True)
        peak.append(np.abs(pos - m.corners).max())
        return out

    calls = []
    randn = torch.randn

    def zeros(*args, **kwargs):
        calls.append(args)
        return torch.zeros(*args, **kwargs)

    rec.run_minibatch = minibatch
    torch.randn = zeros
    try:
        task.run(EPOCHS)
    finally:
        torch.randn = randn
    back = np.conj(m.q) * np.sqrt(ny * nx)
    return dict(
        steps=steps, randn_calls=len(calls), peak=max(peak),
        d_obj=task.get_data_to_cpu("object", as_numpy=True)[0] - obj[0],
        d_probe=task.get_data_to_cpu("probe", as_numpy=True)[0] * back - m.probe,
        offsets=task.get_data_to_cpu("probe_positions", as_numpy=True) - m.corners)


def cplx_rows(shape, *arrays):
    return (np.column_stack([*[a.ravel() for a in np.indices(shape)],
                             *[f(x).ravel() for x in arrays for f in (np.real, np.imag)]]),
            ["%d"] * len(shape) + ["%.17e"] * 2 * len(arrays))


def save(d, name, rows):
    a, fmt = rows
    path = f"{d}/lsqml_pos_{name}_f64.txt"
    np.savetxt(path, a, fmt=fmt)
    print(f"  wrote {path} ({len(a)} rows)")


def write_step(d, prefix, obj, probe, idx, offsets, s):
    g, p = s["gaussian"], s["poisson"]
    save(d, f"{prefix}obj", cplx_rows(obj.shape, g["d_obj"], p["d_obj"]))
    save(d, f"{prefix}probe", cplx_rows(probe.shape, g["d_probe"], p["d_probe"]))
    save(d, f"{prefix}steps", (np.column_stack([idx, *[x[k] for x in (g, p)
                                                       for k in ("a_far", "a_o", "a_p")]]),
                               ["%d"] + ["%.17e"] * 6))
    save(d, f"{prefix}offsets", (np.column_stack([np.arange(len(offsets)), offsets,
                                                  g["offsets"], p["offsets"]]),
                                 ["%d"] + ["%.17e"] * 6))


def compare(name, pc, s, idx, keys):
    assert (pc["idx"] == idx).all(), pc["idx"]
    print(f"  pty-chi {name} f64 ({pc['calls']} torch.randn call(s) zeroed), (max, L2):")
    for k in keys:
        a, b = (pc[k][idx], s[k][idx]) if k == "w" else (pc[k], s[k])
        print(f"    {k:8s} %.3e %.3e" % mg.rel(a, b))
    print("    offset change %.3e %.3e" % mg.rel(pc["offsets"] - s["start"],
                                                s["offsets"] - s["start"]))


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    obj = obj0[0]
    nf, ny, nx = iobs.shape
    unit = np.random.default_rng(1).uniform(-1, 1, (nf, 2))
    offsets = OFFSET * unit
    idx = np.arange(BATCH)
    none_u, none_w = np.zeros((0, ny, nx), complex), np.zeros((nf, 0))
    print(f"object {obj.shape}, probe {probe.shape}, {nf} frames, offsets within "
          f"{np.abs(offsets).max():.3f}")

    print(f"--- step on frames 0..{BATCH}")
    steps = {}
    for noise in NOISES:
        s = step(m, obj, probe, none_u, none_w, offsets, idx, noise)
        s["start"] = offsets
        print(f"  {noise}: a_far {s['a_far'].min():.6f}..{s['a_far'].max():.6f}, a_o trim_mean "
              f"{ls.trim_mean(s['a_o']):.6e}, a_p mean {s['a_p'].mean():.6e}, limit "
              f"{s['limit']}, max |offset change| {np.abs(s['offsets'] - offsets).max():.4f}")
        pc = ptychi_step(m, iobs, wavelength, obj0, none_u, none_w, offsets, idx, noise)
        compare(noise, pc, s, idx, ("illum", "a_far", "a_o", "a_p", "d_obj", "d_probe"))
        steps[noise] = s
    write_step(d, "", obj, probe, idx, offsets, steps)

    rng = np.random.default_rng(lo.SEED)
    r = rng.standard_normal((lo.K, ny, nx)) + 1j * rng.standard_normal((lo.K, ny, nx))
    u = np.stack([np.sqrt(ny * nx) * ri / lo.mnorm(ri) for ri in r])
    w = 1e-6 * rng.standard_normal((nf, lo.K))
    oidx = np.sort(rng.permutation(nf)[:lo.BATCH])
    print(f"--- OPR K {lo.K} step on frames {oidx}")
    steps = {}
    for noise in NOISES:
        pc = ptychi_step(m, iobs, wavelength, obj0, u, w, offsets, oidx, noise)
        print(f"  its start eigenmodes vs U (max, L2) %.3e %.3e" % mg.rel(pc["u0"], u))
        s = step(m, obj, probe, pc["u0"], w, offsets, oidx, noise)
        s["start"] = offsets
        compare(noise, pc, s, oidx, ("illum", "a_far", "a_o", "a_p", "d_obj", "d_probe", "u",
                                     "w"))
        s = step(m, obj, probe, u, w, offsets, oidx, noise)
        print(f"  {noise}: max |w' - w| {np.abs(s['w'] - w).max():.6e}, max |offset change| "
              f"{np.abs(s['offsets'] - offsets).max():.4f}")
        steps[noise] = s
    write_step(d, "opr_", obj, probe, oidx, offsets, steps)
    g, p = steps["gaussian"], steps["poisson"]
    save(d, "opr_eigen", cplx_rows(u.shape, u, g["u"], p["u"]))
    save(d, "opr_weights", (np.column_stack([np.arange(nf), w, g["w"], p["w"]]),
                            ["%d"] + ["%.17e"] * 3 * lo.K))

    offsets = EPOCH_OFFSET * unit
    print(f"--- {EPOCHS} epochs of batches of {EPOCH_BATCH}, seed {SEED}, offsets within "
          f"{np.abs(offsets).max():.3f}")
    runs = {}
    for noise in NOISES:
        pc = ptychi_epochs(m, iobs, wavelength, obj0, offsets, noise)
        for e in range(EPOCHS):
            got = [s["idx"].tolist() for s in pc["steps"] if s["epoch"] == e]
            want = [b.tolist() for b in rb.random_batches(nf, EPOCH_BATCH, SEED, e)]
            assert got == want, (e, got, want)
        assert pc["peak"] < 0.5, pc["peak"]
        print(f"  {noise}: {len(pc['steps'])} steps, {pc['randn_calls']} torch.randn call(s) "
              f"zeroed, max |offset| over the run {pc['peak']:.4f}, max |offset change| "
              f"{np.abs(pc['offsets'] - offsets).max():.4f}")
        runs[noise] = pc
    g, p = runs["gaussian"], runs["poisson"]
    save(d, "epochs_obj", cplx_rows(obj.shape, g["d_obj"], p["d_obj"]))
    save(d, "epochs_probe", cplx_rows(probe.shape, g["d_probe"], p["d_probe"]))
    rows = np.column_stack([
        [s["epoch"] for s in g["steps"] for _ in s["idx"]],
        np.concatenate([s["idx"] for s in g["steps"]]),
        *[np.concatenate([s[k] for s in x["steps"]]) for x in (g, p)
          for k in ("a_far", "a_o", "a_p")]])
    save(d, "epochs_steps", (rows, ["%d", "%d"] + ["%.17e"] * 6))
    save(d, "epochs_offsets", (np.column_stack([np.arange(nf), offsets, g["offsets"],
                                                p["offsets"]]), ["%d"] + ["%.17e"] * 6))


if __name__ == "__main__":
    main()
