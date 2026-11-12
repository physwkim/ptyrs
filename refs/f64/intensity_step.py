"""Float64 reference of the per-frame intensity scale: the costs' gradients with respect to each
frame's scale and pty-chi's LSQML intensity-variation step, at a fixed state.

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/intensity_step.py <dir>

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames, no
masked pixel) with frame k's intensity multiplied by s_k, uniform in [0.8, 1.2] from numpy
default_rng(0). s_k scales every probe mode, as the flux scale of ptyrs::refgen does
(`intensity_scale` in its meta); otherwise the engine's convention (ml_grad's P, q, corners and
F = FFT_ortho fftshift):

    Phi_km = F(P_m q V_k),  I_k = sum_m |Phi_km|^2,  icalc_k = s_k I_k
    L = sum (icalc - iobs + iobs ln(iobs / icalc))   (icalc where iobs = 0; ops::llk)
    F = sum (sqrt(icalc) - sqrt(iobs))^2
    dL/ds_k = sum_u (1 - iobs / icalc) I_k,  dF/ds_k = sum_u (1 - sqrt(iobs / icalc)) I_k

pty-chi's step (OPRModeWeights._calculate_intensity_variation_update_direction under LSQML,
applied as weights += 0.1 delta by its SGD with step size 1) acts on the amplitude weight
w_k = sqrt(s_k). Here it is summed over all modes; pty-chi scales and sums mode 0 only:

    psi_km = w_k Phi_km,  chi_km = fftshift(IFFT_ortho(-a_far,k c_k psi_km))
    delta_k = sum_m sum_u Re(conj(P_m q V_k) chi_km) / sum_m sum_u |P_m q V_k|^2
    s_k' = (w_k + 0.1 delta_k)^2

c and a_far as lsqml_step.py: Gaussian c = 2 (1 - sqrt(iobs) / (sqrt(icalc) + 1e-9)), a_far =
0.5; Poisson c = 1 - iobs / (icalc + 1e-6), a_far from PtychoShelves' xi solver starting at 0.5
for every frame, batch means over frames 0-19 and 20-39 (BATCH, scan order), no jitter. With
the eps set to 0, by Parseval,

    delta_k = -w_k dF/ds_k / sum_u I_k           (Gaussian; w_k + delta_k is the minimiser
                                                  sum_u sqrt(iobs I_k) / sum_u I_k of F over w_k)
    delta_k = -a_far,k w_k dL/ds_k / sum_u I_k   (Poisson)

Writes <dir>/intensity_f64.txt, one row per frame, `k s dL/ds dF/ds s'_gaussian s'_poisson
a_far_poisson`, and <dir>/intensity_values_f64.txt, one row `nb_obs L F` at that state.
Printed checks: central differences of each frame's L and F terms in s_k; delta (eps 0) against
the identities above; L and F after each step; pty-chi's LSQML in f64 on CPU with intensity
variation on, whose OPR needs a second (eigen)mode with nonzero weights
(LSQMLReconstructor.check_inputs): the eigenmode is zero and its weights 1, object, probe and
eigenmode weights are fixed, torch.randn returns zeros; it is compared with the step here
with mode 0 alone scaled and summed.
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


def model(m, obj, s, scaled):
    """I_k over the modes `scaled` (bool per mode) and icalc_k, [nf, ny, nx] each."""
    a = np.abs(m.psi(m.views(obj))) ** 2
    i_s = a[:, scaled].sum(1)
    return i_s, s[:, None, None] * i_s + a[:, ~scaled].sum(1)


def costs(m, ic):
    """Per-frame L and F terms, [2, nf]."""
    t = m.iobs
    return np.stack([(ic - t + xlogy(t, t) - xlogy(t, ic)).sum(AX),
                     ((np.sqrt(ic) - np.sqrt(t)) ** 2).sum(AX)])


def grads(m, i_s, ic):
    """dL/ds_k and dF/ds_k, [2, nf]."""
    t = m.iobs
    return np.stack([((1 - t / ic) * i_s).sum(AX), ((1 - np.sqrt(t / ic)) * i_s).sum(AX)])


def a_far_poisson(yp, yt, eps):
    """pty-chi LSQMLReconstructor.get_psi_far_step_size (Poisson) per batch, without jitter."""
    xi = 1 - yt / (yp + eps)
    a = np.full(len(yp), 0.5)
    for b in range(0, len(yp), BATCH):
        sl = slice(b, b + BATCH)
        for _ in range(2):
            prev = a[sl].mean()
            num = (xi[sl] * (yp[sl] - yt[sl] / (1 - prev * xi[sl]))).sum(AX)
            den = (xi[sl] ** 2 * yp[sl]).sum(AX)
            a[sl] = np.clip(0.5 * prev + 0.5 * np.where(den > 0, num / den, 0), 0, 1)
    return a


def step(m, obj, s, scaled, noise, eps=True):
    """0.1 delta_k and a_far,k, [nf] each (module docstring), over the modes `scaled`."""
    pqv = m.pq * m.views(obj)[:, None]
    mult = np.where(scaled, np.sqrt(s)[:, None], 1.0)
    psi = fft2(fftshift(pqv, axes=AX), norm="ortho") * mult[..., None, None]
    yp = (np.abs(psi) ** 2).sum(1)
    yt = m.iobs
    if noise == "gaussian":
        c = 2 * (1 - np.sqrt(yt) / (np.sqrt(yp) + (1e-9 if eps else 0.0)))
        a = np.full(len(s), 0.5)
    else:
        e1, e2 = (1e-6, 1e-5) if eps else (0.0, 0.0)
        c = 1 - yt / (yp + e1)
        a = a_far_poisson(yp, yt, e2)
    chi = fftshift(ifft2(-(a[:, None, None] * c)[:, None] * psi, norm="ortho"), axes=AX)
    op = pqv[:, scaled]
    num = np.real(np.conj(op) * chi[:, scaled]).sum((1, 2, 3))
    return 0.1 * num / (np.abs(op) ** 2).sum((1, 2, 3)), a


def ptychi(m, iobs, wavelength, obj, probe, s):
    """pty-chi's change of the mode-0 weights over one pass of unshuffled BATCH minibatches,
    mapped as lsqml_step.ptychi_step, per noise model, and the torch.randn calls zeroed."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    p = probe * m.q / np.sqrt(ny * nx)
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
        pr.optimizable = False
        pr.orthogonalize_incoherent_modes.enabled = False
        pr.orthogonalize_opr_modes.enabled = False
        opts.probe_position_options.optimizable = False
        ow = opts.opr_mode_weight_options
        ow.optimizable = True
        ow.optimize_intensity_variation = True
        ow.optimize_eigenmode_weights = False
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
            probe_data=np.stack([p, np.zeros_like(p)]),
            probe_position_y_px=m.corners[:, 0].astype(np.float64),
            probe_position_x_px=m.corners[:, 1].astype(np.float64),
            opr_mode_weights_data=np.column_stack([np.sqrt(s), np.ones_like(s)]),
        )
        rec = task.reconstructor
        weights = rec.parameter_group.opr_mode_weights
        calls = []

        def zeros(*args, **kwargs):
            calls.append(args)
            return torch.zeros(*args, **kwargs)

        with torch.no_grad():
            rec.run_pre_run_hooks()
            rec.run_pre_epoch_hooks()
            w0 = weights.data.detach().cpu().numpy().copy()
            torch.randn = zeros
            try:
                for batch in DataLoader(rec.dataset, batch_size=BATCH, shuffle=False):
                    rec.run_minibatch(*rec.prepare_batch_data(batch))
            finally:
                torch.randn = randn
        w1 = weights.data.detach().cpu().numpy()
        out[noise] = dict(dw=w1[:, 0] - w0[:, 0], w0=w0[:, 0], eigen=np.abs(w1[:, 1] - 1).max(),
                          calls=len(calls))
    return out


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    assert m.valid.all(), "no masked pixel"
    obj = obj0[0]
    nf, nm = len(iobs), len(probe)
    nb_obs = int(m.valid.sum())
    s = np.random.default_rng(SEED).uniform(0.8, 1.2, nf)
    w = np.sqrt(s)
    every = np.ones(nm, bool)
    i_s, ic = model(m, obj, s, every)
    lf = costs(m, ic)
    g = grads(m, i_s, ic)
    print(f"{nf} frames, s in [{s.min():.4f}, {s.max():.4f}]; L {lf[0].sum():.17e}, "
          f"F {lf[1].sum():.17e}; max |dL/ds| {np.abs(g[0]).max():.6e}, "
          f"max |dF/ds| {np.abs(g[1]).max():.6e}")
    print("central differences, (max, L2) rel diff (dL/ds; dF/ds):")
    for h in STEPS:
        c = (costs(m, model(m, obj, s + h, every)[1])
             - costs(m, model(m, obj, s - h, every)[1])) / (2 * h)
        print(f"  h {h:.0e}: %.3e %.3e; %.3e %.3e" % (*mg.rel(c[0], g[0]), *mg.rel(c[1], g[1])))

    total = i_s.sum(AX)
    new = {}
    for noise in NOISES:
        st, a = step(m, obj, s, every, noise)
        st0, a0 = step(m, obj, s, every, noise, eps=False)
        ident = -0.1 * w * (g[1] if noise == "gaussian" else a0 * g[0]) / total
        print(f"--- {noise}: 0.1 delta in [{st.min():.6e}, {st.max():.6e}], "
              f"a_far in [{a.min():.6f}, {a.max():.6f}]")
        print("  eps 0 vs identity (max, L2) %.3e %.3e; eps 0 vs eps %.3e %.3e"
              % (*mg.rel(st0, ident), *mg.rel(st0, st)))
        if noise == "gaussian":
            best = np.sqrt(m.iobs * i_s).sum(AX) / total
            print("  w + 10 x 0.1 delta (eps 0) vs the F minimiser over w (max, L2) %.3e %.3e"
                  % mg.rel(w + 10 * st0, best))
        new[noise] = (w + st) ** 2, a
        after = costs(m, model(m, obj, new[noise][0], every)[1]).sum(1)
        print(f"  L {lf[0].sum():.10e} -> {after[0]:.10e}, F {lf[1].sum():.10e} -> "
              f"{after[1]:.10e}")

    mode0 = np.arange(nm) == 0
    pc = ptychi(m, iobs, wavelength, obj0, probe, s)
    print("pty-chi LSQML f64, mode 0 scaled; its weight change vs 0.1 delta here (max, L2):")
    for noise in NOISES:
        q = pc[noise]
        st = step(m, obj, s, mode0, noise)[0]
        print(f"  {noise}: %.3e %.3e (start weights %.1e, eigenmode weights moved %.1e, "
              f"{q['calls']} torch.randn call(s) zeroed)"
              % (*mg.rel(q["dw"], st), np.abs(q["w0"] - w).max(), q["eigen"]))

    rows = np.column_stack([np.arange(nf), s, g.T, new["gaussian"][0], new["poisson"][0],
                            new["poisson"][1]])
    np.savetxt(f"{d}/intensity_f64.txt", rows, fmt=["%d"] + ["%.17e"] * 6)
    with open(f"{d}/intensity_values_f64.txt", "w") as f:
        f.write(f"{nb_obs} {lf[0].sum():.17e} {lf[1].sum():.17e}\n")
    print(f"wrote {d}/intensity_f64.txt ({nf} rows), {d}/intensity_values_f64.txt")


if __name__ == "__main__":
    main()
