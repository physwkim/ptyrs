"""Float64 reference of pty-chi LSQML's probe-variation (OPR) step: one minibatch's update of the
first eigenmode of probe mode 0 and of its per-frame weights, at a fixed state.

    <engine>/target/release/examples/llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/opr_step.py <dir>

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames, no
masked pixel) plus one eigenmode U of probe mode 0 and its weight w_k per frame, applied as
pty-chi's Probe.get_unique_probes(mode_to_apply=0) does, the main mode's weight (the intensity
variation) held at 1:

    P_k0 = P_0 + w_k U,  P_km = P_m (m >= 1)
    U = sqrt(N) R |P_0| / mnorm(R |P_0|),  w_k = 0.2 mnorm(P_0) / sqrt(N) x uniform(-1, 1)

R complex normal, R then w drawn from numpy default_rng(0), N = ny nx, mnorm(x) = sqrt(mean
|x|^2). U maps to pty-chi as the probe does, U q / sqrt(N), so mnorm(U) = sqrt(N) is pty-chi's
normalize_eigenmodes and w_k U perturbs P_0 by up to 20 % of mnorm(P_0); the weights are the
same in both. Otherwise the engine's convention (ml_grad's P, q, corners, F = FFT_ortho
fftshift), with L, F, c and a_far as intensity_step.py:

    psi_km = F(P_km q V_k),  icalc_k = sum_m |psi_km|^2,  chi_k0 = F^-1(-a_k c_k psi_k0)
    dL/dw_k = sum_u (1 - iobs / icalc) 2 Re(conj(psi_k0) F(U q V_k)),  dF/dw_k likewise with
        1 - sqrt(iobs / icalc)

The step is OPRModeWeights.update_opr_probe_modes_and_weights (LSQML, eigenmode_update_relaxation
and update_relaxation at their default 1) on frames 0-19 (BATCH, scan order):

    d_k = conj(q V_k) chi_k0          (pty-chi's delta_p_i, mode 0; its adjoint shift is 1 here)
    r_k = d_k - mean_batch d
    p_k = (sum_u Re(conj(r_k) U) / N^2 + w_k) / sum_batch w^2
    e = mean_batch(r_k p_k)
    U' = U + a_u sqrt(N) e / mnorm(e),  a_u = min(0.1, BATCH / 40) = 0.1
    U'' = sqrt(N) U' / mnorm(U')
    w_k' = w_k + num_k / (den_k + 0.1 mean_batch den),
        num_k = sum_u Re(chi_k0 conj(U'' q V_k)),  den_k = sum_u |U'' V_k|^2

With the eps set to 0, by Parseval, d_k is the gradient of frame k's cost in P_k0: along
P_k0 -> P_k0 + t D,

    sum_u Re(conj(D) d_k) = -a_k / 2 dL_k/dt  (Poisson),  -a_k dF_k/dt  (Gaussian),

and D = U turns the left side into num_k with U in place of U'' and dt into dw_k.

Writes <dir>/opr_eigen_f64.txt, one row per probe pixel, row-major, `iy ix re(U) im(U)
re(U''_gaussian) im(U''_gaussian) re(U''_poisson) im(U''_poisson)`; <dir>/opr_weights_f64.txt,
one row per frame, `k w dL/dw dF/dw w'_gaussian w'_poisson a_far_poisson` (w' = w and a_far nan
outside the batch); <dir>/opr_values_f64.txt, one row `nb_obs L F` at that state. Printed
checks: central differences of each frame's L and F in w_k and of the batch frames' costs along
a random D (default_rng(0) after w); the identities above; L and F after each step; pty-chi's
LSQML in f64 on CPU on the same minibatch (mapped as lsqml_step.ptychi_step), probe and
eigenmode weights optimizable (its eigenmode update needs the probe optimizable), intensity
variation off, object and positions fixed, torch.randn returning zeros: its eigenmode and
weights after the step against the step here taken from its eigenmode after normalize_eigenmodes
(whose + 1e-8 moves U by about 1e-8). Its main mode also takes the LSQML probe step there
(lsqml_step.py), which is not compared.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2

AX = (-2, -1)
BATCH = 20
NOISES = ("gaussian", "poisson")
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


def mnorm(a):
    return np.sqrt(np.mean(np.abs(a) ** 2))


def far(pq, v):
    return fft2(fftshift(pq * v[:, None], axes=AX), norm="ortho")


def probes(m, u, w, d=None, t=0.0):
    """P_km q per frame, [len(w), nm, ny, nx], with P_0 moved to P_0 + t d."""
    p = np.repeat(m.probe[None], len(w), axis=0)
    p[:, 0] += w[:, None, None] * u + (0 if d is None else t * d)
    return p * m.q


def icalc(m, v, pq):
    return (np.abs(far(pq, v)) ** 2).sum(1)


def w_grads(m, v, u, w):
    """dL/dw_k and dF/dw_k, [2, nf]."""
    psi = far(probes(m, u, w), v)
    ic, t = (np.abs(psi) ** 2).sum(1), m.iobs
    di = 2 * np.real(np.conj(psi[:, 0]) * fft2(fftshift(u * m.q * v, axes=AX), norm="ortho"))
    return np.stack([((1 - t / ic) * di).sum(AX), ((1 - np.sqrt(t / ic)) * di).sum(AX)])


def step(m, v, u, w, idx, noise, eps=True):
    """The OPR step on the frames idx (module docstring)."""
    vb, wb, n = v[idx], w[idx], u.size
    psi = far(probes(m, u, wb), vb)
    yp, yt = (np.abs(psi) ** 2).sum(1), m.iobs[idx]
    if noise == "gaussian":
        c = 2 * (1 - np.sqrt(yt) / (np.sqrt(yp) + (1e-9 if eps else 0.0)))
        a = np.full(len(idx), 0.5)
    else:
        e1, e2 = (1e-6, 1e-5) if eps else (0.0, 0.0)
        c = 1 - yt / (yp + e1)
        a = it.a_far_poisson(yp, yt, e2)
    chi = fftshift(ifft2(-a[:, None, None] * c * psi[:, 0], norm="ortho"), axes=AX)
    d = np.conj(m.q * vb) * chi
    r = d - d.mean(0)
    p = (np.real(np.conj(r) * u).sum(AX) / n**2 + wb) / (wb**2).sum()
    e = (r * p[:, None, None]).mean(0)
    u1 = u + min(0.1, len(idx) / len(w)) * np.sqrt(n) * e / mnorm(e)
    u2 = np.sqrt(n) * u1 / mnorm(u1)
    uqv = u2 * m.q * vb
    num = np.real(chi * np.conj(uqv)).sum(AX)
    den = (np.abs(uqv) ** 2).sum(AX)
    w2 = w.copy()
    w2[idx] = wb + num / (den + 0.1 * den.mean())
    num_u = np.real(chi * np.conj(u * m.q * vb)).sum(AX)
    return dict(u=u2, w=w2, a=a, d=d, num_u=num_u)


def ptychi(m, iobs, wavelength, obj, u, w):
    """pty-chi's eigenmode and weights before and after one LSQML minibatch, per noise model,
    in the engine's convention."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
    p = m.probe * m.q / root_n
    eig = np.zeros_like(p)
    eig[0] = u * m.q / root_n
    back = np.conj(m.q) * root_n
    out = {}
    randn = torch.randn
    for noise in NOISES:
        opts = api.LSQMLOptions()
        opts.data_options.wavelength_m = wavelength
        opts.data_options.fft_shift = True
        o = opts.object_options
        o.optimizable = False
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
        r.rescale_probe_intensity_in_first_epoch = False
        r.batch_size = BATCH
        r.default_dtype = api.Dtypes.FLOAT64
        r.default_device = api.Devices.CPU
        r.allow_nondeterministic_algorithms = False
        r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
        task = PtychographyTask(
            opts,
            diffraction_data=iobs,
            object_data=obj.copy(),
            probe_data=np.stack([p, eig]),
            probe_position_y_px=m.corners[:, 0].astype(np.float64),
            probe_position_x_px=m.corners[:, 1].astype(np.float64),
            opr_mode_weights_data=np.column_stack([np.ones_like(w), w]),
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
            u0 = pg.probe.data[1, 0].detach().cpu().numpy() * back
            batch = next(iter(DataLoader(rec.dataset, batch_size=BATCH, shuffle=False)))
            input_data, y_true = rec.prepare_batch_data(batch)
            torch.randn = zeros
            try:
                rec.run_minibatch(input_data, y_true)
            finally:
                torch.randn = randn
        wt = pg.opr_mode_weights.data.detach().cpu().numpy()
        u1 = pg.probe.data[1, 0].detach().cpu().numpy() * back
        out[noise] = dict(idx=input_data[0].numpy(), u0=u0, u1=u1, w1=wt[:, 1],
                          main=np.abs(wt[:, 0] - 1).max(), calls=len(calls))
    return out


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, _ = args
    m = mg.Model(*args)
    assert m.valid.all(), "no masked pixel"
    nf, ny, nx = iobs.shape
    n = ny * nx
    nb_obs = int(m.valid.sum())
    rng = np.random.default_rng(SEED)
    p0 = m.probe[0]
    r = (rng.standard_normal(p0.shape) + 1j * rng.standard_normal(p0.shape)) * np.abs(p0)
    u = np.sqrt(n) * r / mnorm(r)
    w = 0.2 * mnorm(p0) / np.sqrt(n) * rng.uniform(-1, 1, nf)
    dvec = rng.standard_normal(p0.shape) + 1j * rng.standard_normal(p0.shape)
    v = m.views(obj0[0])
    idx = np.arange(BATCH)

    lf = it.costs(m, icalc(m, v, probes(m, u, w)))
    g = w_grads(m, v, u, w)
    print(f"{nf} frames, mnorm(P_0) {mnorm(p0):.6e}, w in [{w.min():.6e}, {w.max():.6e}], "
          f"max mnorm(w U) / mnorm(P_0) {np.abs(w).max() * np.sqrt(n) / mnorm(p0):.4f}")
    print(f"L {lf[0].sum():.17e}, F {lf[1].sum():.17e}; max |dL/dw| {np.abs(g[0]).max():.6e}, "
          f"max |dF/dw| {np.abs(g[1]).max():.6e}")
    print("central differences, (max, L2) rel diff (dL/dw; dF/dw):")
    for h in STEPS:
        c = (it.costs(m, icalc(m, v, probes(m, u, w + h)))
             - it.costs(m, icalc(m, v, probes(m, u, w - h)))) / (2 * h)
        print(f"  h {h:.0e}: %.3e %.3e; %.3e %.3e" % (*mg.rel(c[0], g[0]), *mg.rel(c[1], g[1])))

    new = {}
    for noise in NOISES:
        pois = noise == "poisson"
        s = step(m, v, u, w, idx, noise)
        s0 = step(m, v, u, w, idx, noise, eps=False)
        a0 = s0["a"]
        k = a0 / 2 if pois else a0
        ident = -k * g[0 if pois else 1][idx]
        dirs = np.real(np.conj(dvec) * s0["d"]).sum(AX)
        print(f"--- {noise}: a_far in [{s['a'].min():.6f}, {s['a'].max():.6f}]")
        print("  eps 0, num_k with U vs -%s d%s/dw_k (max, L2) %.3e %.3e"
              % ("a/2" if pois else "a", "L" if pois else "F", *mg.rel(s0["num_u"], ident)))
        print("  eps 0, sum Re(conj(D) d_k) vs central differences along D, (max, L2):")
        for h in STEPS:
            c = (it.costs(m, icalc(m, v, probes(m, u, w, dvec, h)))
                 - it.costs(m, icalc(m, v, probes(m, u, w, dvec, -h)))) / (2 * h)
            print(f"    h {h:.0e}: %.3e %.3e" % mg.rel(dirs, -k * c[0 if pois else 1][idx]))
        print("  eps vs eps 0 (max, L2): U'' %.3e %.3e; w' %.3e %.3e"
              % (*mg.rel(s["u"], s0["u"]), *mg.rel(s["w"][idx], s0["w"][idx])))
        after = it.costs(m, icalc(m, v, probes(m, s["u"], s["w"]))).sum(1)
        print(f"  mnorm(U'') / sqrt(N) {mnorm(s['u']) / np.sqrt(n):.15f}, |U'' - U| / |U| "
              f"{np.linalg.norm(s['u'] - u) / np.linalg.norm(u):.6e}, max |w' - w| "
              f"{np.abs(s['w'] - w).max():.6e}")
        print(f"  L {lf[0].sum():.10e} -> {after[0]:.10e}, F {lf[1].sum():.10e} -> "
              f"{after[1]:.10e}")
        new[noise] = s

    pc = ptychi(m, iobs, wavelength, obj0, u, w)
    print("pty-chi LSQML f64; its step vs the step here from its normalised eigenmode (max, L2):")
    for noise in NOISES:
        q = pc[noise]
        assert (q["idx"] == idx).all()
        s = step(m, v, q["u0"], w, idx, noise)
        print(f"  {noise}: U'' %.3e %.3e; w' (batch) %.3e %.3e; w outside the batch moved "
              f"%.1e; main-mode weights moved %.1e; its start eigenmode vs U %.3e %.3e; "
              f"{q['calls']} torch.randn call(s) zeroed"
              % (*mg.rel(q["u1"], s["u"]), *mg.rel(q["w1"][idx], s["w"][idx]),
                 np.abs(q["w1"][BATCH:] - w[BATCH:]).max(), q["main"], *mg.rel(q["u0"], u)))

    a_far = np.full(nf, np.nan)
    a_far[idx] = new["poisson"]["a"]
    rows = np.column_stack([np.arange(nf), w, g.T, new["gaussian"]["w"], new["poisson"]["w"],
                            a_far])
    np.savetxt(f"{d}/opr_weights_f64.txt", rows, fmt=["%d"] + ["%.17e"] * 6)
    cols = [f(a).ravel() for a in (u, new["gaussian"]["u"], new["poisson"]["u"])
            for f in (np.real, np.imag)]
    eig = np.column_stack([np.repeat(np.arange(ny), nx), np.tile(np.arange(nx), ny), *cols])
    np.savetxt(f"{d}/opr_eigen_f64.txt", eig, fmt=["%d", "%d"] + ["%.17e"] * 6)
    with open(f"{d}/opr_values_f64.txt", "w") as f:
        f.write(f"{nb_obs} {lf[0].sum():.17e} {lf[1].sum():.17e}\n")
    print(f"wrote {d}/opr_weights_f64.txt ({nf} rows), {d}/opr_eigen_f64.txt ({len(eig)} rows), "
          f"{d}/opr_values_f64.txt")


if __name__ == "__main__":
    main()
