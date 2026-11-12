"""Float64 reference of one pty-chi LSQML minibatch with orthogonal probe relaxation (OPR) of
K = 3 eigenmodes of probe mode 0: the object, probe, eigenmode and weight step at a fixed state.

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_opr_step.py <dir>

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames, no
masked pixel) plus K eigenmodes U_i and their weights w_ki per frame, drawn as pty-chi's
test_2d_ptycho_lsqml_opr draws them (additional OPR probe modes with real and imaginary parts
standard normal, normalised to mnorm 1 in pty-chi's units; generate_initial_opr_mode_weights,
1e-6 x standard normal):

    U_i = sqrt(N) R_i / mnorm(R_i),  w_ki = 1e-6 x standard normal

R_i complex normal; R, then w [40, K], then the batch from numpy default_rng(0); N = ny nx,
mnorm(x) = sqrt(mean |x|^2); U_i maps to pty-chi as the probe does, U_i q / sqrt(N). Frame k
sees, as Probe.get_unique_probes(mode_to_apply=0) with the main mode's weight held at 1,

    P_k0 = P_0 + sum_i w_ki U_i,  P_km = P_m (m >= 1)

The batch is BATCH frames drawn without replacement, in scan order. The step is
LSQMLReconstructor.run_minibatch as lsqml_step.py, with each frame's own probe where pty-chi
takes the unique probes:

    psi_km = F(P_km q V_k), c and a_far as lsqml_step.py, chi_km = F^-1(-a_far c psi_km)
    dO_hat = scatter(sum_m conj(P_km q) chi_km) / sqrt(I^2 + (0.1 max I)^2), I the
        illumination of P (opr mode 0) over all 40 windows
    d_km = conj(q V_k) chi_km,  dP_hat = mean_batch d_k
    step sizes as lsqml_step.py with dOP = view(dO_hat) P_km q and dPO = q dP_hat V_k
    O += trim_mean(a_o) dO_hat,  P += mean(a_p) dP_hat

and OPRModeWeights.update_opr_probe_modes_and_weights from the same chi (relaxations at their
default 1), each eigenmode with its own weights on entry:

    r_k = d_k0 - dP_hat_0
    for i in 0..K:
        p_k = (sum_u Re(conj(r_k) U_i) / N^2 + w_ki) / sum_batch w_i^2
        e = mean_batch(r_k p_k)
        U_i' = U_i + a_u sqrt(N) e / mnorm(e),  a_u = min(0.1, BATCH / 40)
        U_i'' = sqrt(N) U_i' / mnorm(U_i')
        w_ki' = w_ki + num_k / (den_k + 0.1 mean_batch den),
            num_k = sum_u Re(chi_k0 conj(U_i'' q V_k)),  den_k = sum_u |U_i'' V_k|^2
        r_k -= (sum_u r_k conj(U_i'') / sum_u |U_i''|^2) U_i''      (i < K - 1)

The weight gradients are opr_step.py's per eigenmode: dL/dw_ki = sum_u (1 - iobs / icalc)
2 Re(conj(psi_k0) F(U_i q V_k)), dF/dw_ki likewise with 1 - sqrt(iobs / icalc).

Writes, in the engine's convention, `_g` Gaussian and `_p` Poisson:
    <dir>/lsqml_opr_eigen_f64.txt    `i iy ix re(U) im(U) re(U''_g) im(U''_g) re(U''_p) im(U''_p)`
    <dir>/lsqml_opr_weights_f64.txt  `k w_0.. dL/dw_0.. dF/dw_0.. w'_g,0.. w'_p,0..`, K columns
        each, every frame (w' = w outside the batch)
    <dir>/lsqml_opr_obj_f64.txt      `iy ix re_g im_g re_p im_p`, the object change
    <dir>/lsqml_opr_probe_f64.txt    `m iy ix re_g im_g re_p im_p`, the probe change
    <dir>/lsqml_opr_steps_f64.txt    `k a_far_g a_o_g a_p_g a_far_p a_o_p a_p_p`, the batch
    <dir>/lsqml_opr_values_f64.txt   `nb_obs L F` at the state, over every frame
Printed checks: central differences of each frame's L and F in each w_ki; mnorm and mutual
overlaps of U''; L and F over every frame before and after the step; pty-chi's LSQML in f64 on
CPU on the same minibatch (mapped as lsqml_step.ptychi_step), object, probe and eigenmode
weights optimizable, intensity variation and both orthogonalisations off, its preconditioner
set to its illumination map, torch.randn returning zeros: its step against the step here taken
from its eigenmodes after normalize_eigenmodes (whose + 1e-8 moves U by about 1e-8).
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2

AX = (-2, -1)
K = 3
BATCH = 20
NOISES = ("gaussian", "poisson")
LAMBDA0 = 1.2e-7
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
it = sibling("intensity_step")
ls = sibling("lsqml_step")


def mnorm(a):
    return np.sqrt(np.mean(np.abs(a) ** 2))


def probes(probe, u, w):
    """P_km per frame, [len(w), nm, ny, nx], P_0 + w_k0 U_0 + w_k1 U_1 + ... in that order."""
    p = np.repeat(probe[None], len(w), axis=0)
    for i in range(len(u)):
        p[:, 0] += w[:, i, None, None] * u[i]
    return p


def far(pq, v):
    return fft2(fftshift(pq * v[:, None], axes=AX), norm="ortho")


def icalc(m, v, probe, u, w):
    return (np.abs(far(probes(probe, u, w) * m.q, v)) ** 2).sum(1)


def w_grads(m, v, u, w):
    """dL/dw_ki and dF/dw_ki, [2, nf, K]."""
    psi = far(probes(m.probe, u, w) * m.q, v)
    ic, t = (np.abs(psi) ** 2).sum(1), m.iobs
    g = np.zeros((2, len(w), len(u)))
    for i, ui in enumerate(u):
        di = 2 * np.real(np.conj(psi[:, 0]) * fft2(fftshift(ui * m.q * v, axes=AX), norm="ortho"))
        g[0, :, i] = ((1 - t / ic) * di).sum(AX)
        g[1, :, i] = ((1 - np.sqrt(t / ic)) * di).sum(AX)
    return g


def step(m, obj, probe, u, w, idx, noise):
    """One LSQML minibatch with OPR (module docstring)."""
    nm, n = probe.shape[0], u[0].size
    corners = m.corners[idx]
    v = ls.views(obj, corners, m.n)
    pq = probes(probe, u, w[idx]) * m.q
    psi = far(pq, v)
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

    do_comb = ls.scatter((chi * np.conj(pq)).sum(1), corners, obj.shape)
    illum = ls.scatter(np.broadcast_to((np.abs(probe) ** 2).sum(0), (len(m.corners), *m.n)),
                       m.corners, obj.shape)
    do_hat = do_comb / np.sqrt(illum**2 + (0.1 * illum.max()) ** 2)
    dp_q = (chi * np.conj(v)[:, None]).mean(0)

    dop = ls.views(do_hat, corners, m.n)[:, None] * pq
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

    d0 = np.conj(m.q * v) * chi[:, 0]
    r = d0 - d0.mean(0)
    wb = w[idx]
    u2, w2 = u.copy(), w.copy()
    a_u = min(0.1, len(idx) / len(w))
    for i in range(len(u)):
        p = (np.real(np.conj(r) * u[i]).sum(AX) / n**2 + wb[:, i]) / (wb[:, i] ** 2).sum()
        e = (r * p[:, None, None]).mean(0)
        u1 = u[i] + a_u * np.sqrt(n) * e / mnorm(e)
        u2[i] = np.sqrt(n) * u1 / mnorm(u1)
        uqv = u2[i] * m.q * v
        num = np.real(chi[:, 0] * np.conj(uqv)).sum(AX)
        den = (np.abs(uqv) ** 2).sum(AX)
        w2[idx, i] = wb[:, i] + num / (den + 0.1 * den.mean())
        if i < len(u) - 1:
            r = r - ((r * np.conj(u2[i])).sum(AX, keepdims=True)
                     / (np.abs(u2[i]) ** 2).sum()) * u2[i]
    return dict(a_far=a_far, a_o=a_o, a_p=a_p, illum=illum, u=u2, w=w2,
                d_obj=ls.trim_mean(a_o) * do_hat, d_probe=a_p.mean() * np.conj(m.q) * dp_q)


def ptychi(m, iobs, wavelength, obj, u, w, idx):
    """pty-chi's step per noise model, in the engine's convention."""
    import torch
    from torch.utils.data import DataLoader, Subset
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
    p = m.probe * m.q / root_n
    eig = np.zeros((len(u), *p.shape), complex)
    eig[:, 0] = u * m.q / root_n
    back = np.conj(m.q) * root_n
    out = {}
    randn = torch.randn
    for noise in NOISES:
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
        r.batch_size = len(idx)
        r.default_dtype = api.Dtypes.FLOAT64
        r.default_device = api.Devices.CPU
        r.allow_nondeterministic_algorithms = False
        r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
        task = PtychographyTask(
            opts,
            diffraction_data=iobs,
            object_data=obj.copy(),
            probe_data=np.concatenate([p[None], eig]),
            probe_position_y_px=m.corners[:, 0].astype(np.float64),
            probe_position_x_px=m.corners[:, 1].astype(np.float64),
            opr_mode_weights_data=np.column_stack([np.ones(len(w)), w]),
        )
        rec = task.reconstructor
        pg = rec.parameter_group
        calls = []

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
        wt = pg.opr_mode_weights.data.detach().cpu().numpy()
        probe = pg.probe.data.detach().cpu().numpy()
        out[noise] = dict(
            idx=got, u0=u0, calls=len(calls),
            a_far=np.broadcast_to(rec.alpha_psi_far.cpu().numpy().reshape(-1), got.shape),
            a_o=buf.alpha_object_all_pos_all_slices[got, 0].cpu().numpy(),
            a_p=buf.alpha_probe_all_pos[got].cpu().numpy(),
            illum=pg.object.preconditioner.cpu().numpy() * ny * nx,
            d_obj=pg.object.data.detach().cpu().numpy()[0] - obj[0],
            d_probe=probe[0] * back - m.probe,
            u=probe[1:, 0] * back, w=wt[:, 1:], main=np.abs(wt[:, 0] - 1).max(),
            other=np.abs(probe[1:, 1:]).max())
    return out


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    assert m.valid.all(), "no masked pixel"
    obj = obj0[0]
    nf, ny, nx = iobs.shape
    n = ny * nx
    rng = np.random.default_rng(SEED)
    r = rng.standard_normal((K, ny, nx)) + 1j * rng.standard_normal((K, ny, nx))
    u = np.stack([np.sqrt(n) * ri / mnorm(ri) for ri in r])
    w = 1e-6 * rng.standard_normal((nf, K))
    idx = np.sort(rng.permutation(nf)[:BATCH])
    v = m.views(obj)
    print(f"{nf} frames, K {K}, batch {idx}")
    print(f"mnorm(P_0) {mnorm(probe[0]):.6e}, max |w| {np.abs(w).max():.6e}")

    lf = it.costs(m, icalc(m, v, probe, u, w))
    g = w_grads(m, v, u, w)
    print(f"L {lf[0].sum():.17e}, F {lf[1].sum():.17e}")
    print("central differences in w_ki, (max, L2) rel diff (dL/dw; dF/dw):")
    for h in STEPS:
        c = np.zeros_like(g)
        for i in range(K):
            e = np.zeros_like(w)
            e[:, i] = h
            c[:, :, i] = (it.costs(m, icalc(m, v, probe, u, w + e))
                          - it.costs(m, icalc(m, v, probe, u, w - e))) / (2 * h)
        print(f"  h {h:.0e}: %.3e %.3e; %.3e %.3e" % (*mg.rel(c[0], g[0]), *mg.rel(c[1], g[1])))

    new = {}
    for noise in NOISES:
        s = step(m, obj, probe, u, w, idx, noise)
        gram = np.abs(np.einsum("iyx,jyx->ij", np.conj(s["u"]), s["u"])) / n**2
        overlap = (gram - np.diag(np.diag(gram))).max()
        norms = " ".join(f"{mnorm(x) / np.sqrt(n):.15f}" for x in s["u"])
        moved = " ".join(f"{np.linalg.norm(a - b) / np.linalg.norm(b):.3e}"
                         for a, b in zip(s["u"], u))
        after = it.costs(m, icalc(m, v, probe + s["d_probe"], s["u"], s["w"])).sum(1)
        print(f"--- {noise}: a_far {s['a_far'].min():.6f}..{s['a_far'].max():.6f}, a_o "
              f"{s['a_o'].min():.6e}..{s['a_o'].max():.6e}, a_p {s['a_p'].min():.6e}.."
              f"{s['a_p'].max():.6e}")
        print(f"  mnorm(U''_i) / sqrt(N) {norms}; max |<U''_i, U''_j>| / (|U''_i| |U''_j|), "
              f"i != j, {overlap:.3e}; |U''_i - U_i| / |U_i| {moved}; max |w' - w| "
              f"{np.abs(s['w'] - w).max():.6e}")
        print(f"  L {lf[0].sum():.10e} -> {after[0]:.10e}, F {lf[1].sum():.10e} -> "
              f"{after[1]:.10e} (object step not applied in either)")
        new[noise] = s

    pc = ptychi(m, iobs, wavelength, obj0, u, w, idx)
    print("pty-chi LSQML f64; its step vs the step here from its normalised eigenmodes "
          "(max, L2):")
    for noise in NOISES:
        q = pc[noise]
        assert (q["idx"] == idx).all(), q["idx"]
        s = step(m, obj, probe, q["u0"], w, idx, noise)
        out = [i for i in range(nf) if i not in idx]
        print(f"  {noise}: {q['calls']} torch.randn call(s) zeroed; its start eigenmodes vs U "
              "%.3e %.3e; w outside the batch moved %.1e; main-mode weights moved %.1e; "
              "eigenmodes of mode 1 %.1e"
              % (*mg.rel(q["u0"], u), np.abs(q["w"][out] - w[out]).max(), q["main"],
                 q["other"]))
        for k in ("illum", "a_far", "a_o", "a_p", "d_obj", "d_probe", "u"):
            print(f"    {k:8s} %.3e %.3e" % mg.rel(q[k], s[k]))
        print("    w (batch) %.3e %.3e" % mg.rel(q["w"][idx], s["w"][idx]))

    ng, npo = new["gaussian"], new["poisson"]
    eig = np.column_stack([
        *[a.ravel() for a in np.indices(u.shape)],
        *[f(a).ravel() for a in (u, ng["u"], npo["u"]) for f in (np.real, np.imag)]])
    rows = {
        "eigen": (eig, ["%d"] * 3 + ["%.17e"] * 6),
        "weights": (np.column_stack([np.arange(nf), w, g[0], g[1], ng["w"], npo["w"]]),
                    ["%d"] + ["%.17e"] * 5 * K),
        "obj": (np.column_stack([*[a.ravel() for a in np.indices(obj.shape)],
                                 *[f(x["d_obj"]).ravel() for x in (ng, npo)
                                   for f in (np.real, np.imag)]]),
                ["%d"] * 2 + ["%.17e"] * 4),
        "probe": (np.column_stack([*[a.ravel() for a in np.indices(probe.shape)],
                                   *[f(x["d_probe"]).ravel() for x in (ng, npo)
                                     for f in (np.real, np.imag)]]),
                  ["%d"] * 3 + ["%.17e"] * 4),
        "steps": (np.column_stack([idx, *[x[k] for x in (ng, npo)
                                          for k in ("a_far", "a_o", "a_p")]]),
                  ["%d"] + ["%.17e"] * 6),
    }
    for name, (a, fmt) in rows.items():
        path = f"{d}/lsqml_opr_{name}_f64.txt"
        np.savetxt(path, a, fmt=fmt)
        print(f"wrote {path} ({len(a)} rows)")
    with open(f"{d}/lsqml_opr_values_f64.txt", "w") as f:
        f.write(f"{int(m.valid.sum())} {lf[0].sum():.17e} {lf[1].sum():.17e}\n")
    print(f"wrote {d}/lsqml_opr_values_f64.txt")


if __name__ == "__main__":
    main()
