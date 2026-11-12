"""Float64 reference of pty-chi LSQML (Poisson) in its planar near-field model.

    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_near.py <dir>

The state is generated here from numpy default_rng(SEED): a truth object [OBJ] of smooth
amplitude and phase, a truth probe [2, 24, 24] (a diffuser-like mode 0 and a weaker mode 1),
16 frames on a jittered 4x4 grid of whole-pixel corners, and per geometry the Poisson counts
of the truth through that geometry's propagator, natural (centred) layout, as pty-chi reads
them with fft_shift False. The start is an object of ones and the truth probe plus a smooth
complex perturbation. GEOMETRIES (wavelength, pixel, distance) give an angular-spectrum
propagator, a Fresnel transform and an angular spectrum with an evanescent band.

In the engine's convention the exit wave is psi = U(P O_view), U pty-chi's propagate_forward
over sqrt(k) (k its get_intensity_scaling_factor), so pty-chi's probe is P / sqrt(k), its
psi_far is psi and its chi is chi / sqrt(k). The LSQML step of lsqml_step.py then reads

    c = 1 - iobs / (icalc + 1e-6), a_far by PtychoShelves' xi solver (lsqml_step.py)
    chi_m = U^H((1 - a_far c) U psi_m) - psi_m                       (pty-chi Eq. 19)
    dO_hat = scatter(sum_m conj(P_m) chi_m) / sqrt(I^2 + (0.1 max I)^2)
    dP_hat = mean_batch(conj(O_view) chi_m)
    a11 = sum |dOP|^2 + lambda0 n, a22 = sum |dPO|^2 + lambda0 n, lambda0 = 1.2e-7 k / (ny nx)

and the rest as lsqml_step.py with q = 1. With an evanescent band U^H U is a projection, so
chi carries -(1 - U^H U) psi_m on top of the gradient's U^H(-a_far c U psi_m).

Writes, every complex value `re im`, every plane row-major:
    <dir>/lsqml_near_cases_f64.txt    `case wavelength pixel distance kind scale truth_llk`
    <dir>/lsqml_near_corners_f64.txt  `frame cy cx`
    <dir>/lsqml_near_obj_f64.txt      `iy ix re im`, the start object
    <dir>/lsqml_near_probe_f64.txt    `m iy ix re im`, the start probe
    <dir>/lsqml_near_counts_f64.txt   `case frame` and the frame's counts, natural layout
    <dir>/lsqml_near_icalc_f64.txt    `case frame` and pty-chi's y_pred at the start, frames
                                      0..ICALC_FRAMES
    <dir>/lsqml_near_step_{obj,probe,steps}_f64.txt    pty-chi's step on frames 0..BATCH
        with the illumination map as preconditioner: `case iy ix re im`, `case m iy ix re im`
        (the changes) and `case frame a_far a_o a_p`
    <dir>/lsqml_near_epochs_{obj,probe,steps}_f64.txt  pty-chi over EPOCHS epochs of
        random_batches(16, BATCH, BATCH_SEED, epoch), cases EPOCH_CASES, its own
        preconditioner schedule: the changes and `case epoch frame a_far a_o a_p`
    <dir>/lsqml_near_llk_f64.txt      `case epoch llk`, the normalised Poisson LLK after each
                                      epoch of that run, epoch -1 the start
torch.randn returns zeros in every pty-chi run (its 1e-2 a_far jitter); the object-probe
ambiguity removal and the mode orthogonalisation are off, as in lsqml_epochs.py.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import xlogy

AX = (-2, -1)
SEED = 11
SHAPE = (24, 24)
OBJ = (40, 40)
NMODE = 2
GRID = 4
BATCH = 8
BATCH_SEED = 7
EPOCHS = 30
EPOCH_CASES = (0, 1)
ICALC_FRAMES = 4
LAMBDA0 = 1.2e-7
STEPS = (1e-4, 1e-5, 1e-6)
# (wavelength, pixel, distance): angular spectrum, Fresnel transform, angular spectrum with an
# evanescent band (lambda^2 (fy^2 + fx^2) > 1 in the frequency corners).
GEOMETRIES = [(1e-10, 50e-9, 5e-4), (1e-10, 50e-9, 2e-3), (5e-7, 3e-7, 3e-6)]


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ls = sibling("lsqml_step")
rb = sibling("random_batches")
pp = sibling("propagators")


def smooth(rng, shape, sigma):
    """A periodic Gaussian-filtered normal field scaled to unit standard deviation."""
    f = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return f / f.std()


def make_state():
    rng = np.random.default_rng(SEED)
    obj = (0.9 + 0.05 * smooth(rng, OBJ, 2.0)) * np.exp(0.5j * smooth(rng, OBJ, 2.0))
    env = 12.0 * (1 + 0.3 * smooth(rng, SHAPE, 1.5))
    probe = np.stack([env * np.exp(1j * np.pi * smooth(rng, SHAPE, 1.0)),
                      0.3 * env * np.exp(1j * np.pi * smooth(rng, SHAPE, 1.0))])
    shift = 0.15 * np.abs(probe).max() * np.stack(
        [smooth(rng, SHAPE, 2.0) + 1j * smooth(rng, SHAPE, 2.0) for _ in range(NMODE)])
    span = np.array(OBJ) - np.array(SHAPE)
    base = np.linspace(1, span[0] - 1, GRID).round().astype(int)
    corners = np.array([[y, x] for y in base for x in base])
    corners = np.clip(corners + rng.integers(-1, 2, corners.shape), 0, span)
    return obj, probe, probe + shift, corners, rng


class Geometry:
    """One planar near-field geometry: pty-chi's free-space propagator and its unitary U."""

    def __init__(self, wavelength, pixel, distance):
        import torch
        self.torch = torch
        self.wavelength, self.pixel, self.distance = wavelength, pixel, distance
        self.prop = pp.free_space(SHAPE, wavelength, pixel, pixel, distance)
        self.kind = pp.kind(self.prop)
        self.k = float(pp.get_intensity_scaling_factor(self.prop, SHAPE))

    def fwd(self, x):
        """U x."""
        return self.prop.propagate_forward(self.torch.from_numpy(x)).numpy() / np.sqrt(self.k)

    def adj(self, y):
        """U^H y."""
        return self.prop.propagate_backward(self.torch.from_numpy(y)).numpy() * np.sqrt(self.k)


def views(a, corners):
    return ls.views(a, corners, SHAPE)


def icalc(g, obj, probe, corners):
    return (np.abs(g.fwd(probe[None] * views(obj, corners)[:, None])) ** 2).sum(1)


def llk(g, obj, probe, corners, counts):
    """The engine's normalised Poisson LLK (ops::llk) over every frame."""
    yp = icalc(g, obj, probe, corners)
    t = np.where(counts > 0, yp - counts + xlogy(counts, counts) - xlogy(counts, yp), yp)
    return t.sum() / counts.size


def step(g, obj, probe, corners, counts, idx, eps=True):
    """One LSQML Poisson step in the engine's convention (module docstring)."""
    nm = probe.shape[0]
    c_idx = corners[idx]
    v = views(obj, c_idx)
    psi0 = probe[None] * v[:, None]
    psi = g.fwd(psi0)
    yp = (np.abs(psi) ** 2).sum(1)
    yt = counts[idx]
    e1, e2 = (1e-6, 1e-5) if eps else (0.0, 0.0)
    c = 1 - yt / (yp + e1)
    xi = 1 - yt / (yp + e2)
    a_far = np.full(len(idx), 0.5)
    for _ in range(2):
        prev = a_far.mean()
        num = (xi * (yp - yt / (1 - prev * xi))).sum(AX)
        den = (xi**2 * yp).sum(AX)
        a_far = np.clip(0.5 * prev + 0.5 * np.where(den > 0, num / den, 0), 0, 1)
    rescaled = (1 - (a_far[:, None, None] * c)[:, None]) * psi
    chi = g.adj(rescaled) - psi0
    grad_chi = g.adj(-(a_far[:, None, None] * c)[:, None] * psi)

    do_comb = ls.scatter((chi * np.conj(probe)).sum(1), c_idx, OBJ)
    grad_comb = ls.scatter((grad_chi * np.conj(probe)).sum(1), c_idx, OBJ)
    illum = ls.scatter(np.broadcast_to((np.abs(probe) ** 2).sum(0), (len(corners), *SHAPE)),
                       corners, OBJ)
    do_hat = do_comb / np.sqrt(illum**2 + (0.1 * illum.max()) ** 2)
    dp_hat = (chi * np.conj(v)[:, None]).mean(0)

    dop = views(do_hat, c_idx)[:, None] * probe
    dpo = dp_hat[None] * v[:, None]
    s = (1, 2, 3)
    lambda0 = LAMBDA0 * g.k / (SHAPE[0] * SHAPE[1])
    a11 = (np.abs(dop) ** 2 + lambda0).sum(s)
    a22 = (np.abs(dpo) ** 2 + lambda0).sum(s)
    a11 = a11 + 0.1 * a11.mean()
    a22 = a22 + 0.1 * a22.mean()
    a12 = (dop * np.conj(dpo)).sum(s)
    b = np.stack([np.real(np.conj(dop) * chi).sum(s), np.real(np.conj(dpo) * chi).sum(s)], 1)
    a = np.stack([a11, a12, np.conj(a12), a22], 1).reshape(-1, 2, 2)
    x = np.linalg.solve(a, b[..., None].astype(complex))[..., 0].real.clip(0, None)
    a_o = 0.9 * x[:, 0] / nm
    a_p = 0.9 * x[:, 1]
    return dict(a_far=a_far, a_o=a_o, a_p=a_p, illum=illum, do_comb=do_comb,
                grad_comb=grad_comb, dp_hat=dp_hat,
                grad_dp=(grad_chi * np.conj(v)[:, None]).mean(0),
                d_obj=ls.trim_mean(a_o) * do_hat, d_probe=a_p.mean() * dp_hat)


def weighted_cost(g, obj, probe, corners, counts, idx, w):
    """sum_i w_i sum (icalc - iobs ln icalc), the Poisson L whose icalc derivative is c (eps 0)."""
    yp = icalc(g, obj, probe, corners[idx])
    return (w * (yp - xlogy(counts[idx], yp)).sum(AX)).sum()


def task(g, obj, probe, corners, counts, batch, epochs):
    """pty-chi's LSQML task on the case, as lsqml_epochs.ptychi_epochs in near field."""
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = SHAPE
    opts = api.LSQMLOptions()
    opts.data_options.wavelength_m = g.wavelength
    opts.data_options.fft_shift = False
    opts.data_options.free_space_propagation_distance_m = g.distance
    o = opts.object_options
    o.optimizable = True
    o.pixel_size_m = g.pixel
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
    r.noise_model = api.NoiseModels.POISSON
    r.forward_model_options.subpixel_shift_all_probe_modes = True
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    r.solve_step_sizes_only_using_first_probe_mode = False
    r.rescale_probe_intensity_in_first_epoch = False
    r.batch_size = batch
    r.num_epochs = epochs
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    return PtychographyTask(
        opts,
        diffraction_data=counts.copy(),
        object_data=obj[None].copy(),
        probe_data=(probe / np.sqrt(g.k))[None].copy(),
        probe_position_y_px=corners[:, 0].astype(np.float64),
        probe_position_x_px=corners[:, 1].astype(np.float64),
    )


class ZeroRandn:
    """torch.randn returning zeros while active, counting its calls."""

    def __init__(self):
        import torch
        self.torch, self.calls = torch, 0

    def __enter__(self):
        self.randn = self.torch.randn

        def zeros(*args, **kwargs):
            self.calls += 1
            return self.torch.zeros(*args, **kwargs)

        self.torch.randn = zeros
        return self

    def __exit__(self, *exc):
        self.torch.randn = self.randn


def state_of(t, g):
    """pty-chi's object and probe in the engine's convention."""
    return (t.get_data_to_cpu("object", as_numpy=True)[0],
            t.get_data_to_cpu("probe", as_numpy=True)[0] * np.sqrt(g.k))


def ptychi_step(g, obj, probe, corners, counts):
    """pty-chi run_minibatch on frames 0..BATCH with the illumination map as preconditioner,
    and its forward model's y_pred at the start."""
    import torch
    from torch.utils.data import DataLoader

    t = task(g, obj, probe, corners, counts, BATCH, 1)
    rec = t.reconstructor
    pg = rec.parameter_group
    with torch.no_grad(), ZeroRandn() as z:
        rec.run_pre_run_hooks()
        rec.run_pre_epoch_hooks()
        pg.object.preconditioner = pg.object.calculate_illumination_map(
            pg.probe, pg.probe_positions, use_all_modes=True)
        batch = next(iter(DataLoader(rec.dataset, batch_size=BATCH, shuffle=False)))
        input_data, y_true = rec.prepare_batch_data(batch)
        y_pred = rec.forward_model(*input_data).numpy().copy()
        rec.run_minibatch(input_data, y_true)
    idx = input_data[0].numpy()
    buf = rec.reconstructor_buffers
    o, p = state_of(t, g)
    return dict(
        idx=idx, randn_calls=z.calls, y_pred=y_pred,
        a_far=rec.alpha_psi_far.cpu().numpy().reshape(-1),
        a_o=buf.alpha_object_all_pos_all_slices[idx, 0].cpu().numpy(),
        a_p=buf.alpha_probe_all_pos[idx].cpu().numpy(),
        illum=pg.object.preconditioner.cpu().numpy() * g.k,
        d_obj=o - obj, d_probe=p - probe)


def ptychi_epochs(g, obj, probe, corners, counts):
    """pty-chi over EPOCHS epochs of the engine's batches, one task.run(1) per epoch: every
    step's frames and step sizes, the LLK after each epoch and the changes over the run."""
    from torch.utils.data import DataLoader

    t = task(g, obj, probe, corners, counts, BATCH, EPOCHS)
    rec = t.reconstructor
    rec.dataloader = DataLoader(rec.dataset,
                                batch_sampler=rb.Sampler(len(corners), BATCH, BATCH_SEED))
    cls = type(rec)
    buf = rec.reconstructor_buffers
    steps = []

    def minibatch(input_data, y_true):
        out = cls.run_minibatch(rec, input_data, y_true)
        idx = input_data[0].numpy().copy()
        steps.append(dict(
            epoch=rec.current_epoch, idx=idx,
            a_far=rec.alpha_psi_far.cpu().numpy().reshape(-1).copy(),
            a_o=buf.alpha_object_all_pos_all_slices[idx, 0].cpu().numpy().copy(),
            a_p=buf.alpha_probe_all_pos[idx].cpu().numpy().copy()))
        return out

    rec.run_minibatch = minibatch
    trace = [llk(g, obj, probe, corners, counts)]
    with ZeroRandn() as z:
        for _ in range(EPOCHS):
            t.run(1)
            trace.append(llk(g, *state_of(t, g), corners, counts))
    o, p = state_of(t, g)
    return dict(steps=steps, randn_calls=z.calls, llk=np.array(trace),
                d_obj=o - obj, d_probe=p - probe)


def counts_of(g, obj, probe, corners, rng):
    return rng.poisson(icalc(g, obj, probe, corners)).astype(np.float64)


def lead(case, rows_fmt):
    rows, fmt = rows_fmt
    return np.column_stack([np.full(len(rows), case), rows]), ["%d"] + fmt


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    truth_obj, truth_probe, probe, corners, rng = make_state()
    obj = np.ones(OBJ, complex)
    nframe = len(corners)
    print(f"object {OBJ}, probe {probe.shape}, {nframe} frames, corners {corners.tolist()}")
    out = {k: [] for k in ("cases", "counts", "icalc", "step_obj", "step_probe", "step_steps",
                           "epochs_obj", "epochs_probe", "epochs_steps", "llk")}
    idx = np.arange(BATCH)
    for case, geom in enumerate(GEOMETRIES):
        g = Geometry(*geom)
        counts = counts_of(g, truth_obj, truth_probe, corners, rng)
        assert counts.max() < 65535, counts.max()
        truth_llk = llk(g, truth_obj, truth_probe, corners, counts)
        print(f"--- case {case}: {geom}, kind {g.kind}, k {g.k:.6e}, counts max "
              f"{counts.max():.0f} mean {counts.mean():.1f}, LLK truth {truth_llk:.6f} start "
              f"{llk(g, obj, probe, corners, counts):.6f}")
        out["cases"].append([case, *geom, g.kind, g.k, truth_llk])
        out["counts"].append(np.column_stack([np.full(nframe, case), np.arange(nframe),
                                              counts.reshape(nframe, -1)]))

        s0 = step(g, obj, probe, corners, counts, idx, eps=False)
        w = s0["a_far"]
        print("  gradient U^H(-a_far c U psi) and the step's chi vs central differences of "
              "sum_i a_far,i L_i (eps 0), rel diff:")
        d_obj, d_probe = direction(np.random.default_rng(case), obj, probe)
        for h in STEPS:
            fo = ls.directional(lambda x: weighted_cost(g, x, probe, corners, counts, idx, w),
                                obj, d_obj, h)
            fp = ls.directional(lambda x: weighted_cost(g, obj, x, corners, counts, idx, w),
                                probe, d_probe, h)
            msg = []
            for name, go, gp in (("gradient", s0["grad_comb"], s0["grad_dp"]),
                                 ("chi", s0["do_comb"], s0["dp_hat"])):
                eo = np.vdot(-2 * go, d_obj).real
                ep = np.vdot(-2 * BATCH * gp, d_probe).real
                msg.append(f"{name} object {abs(fo - eo) / abs(fo):.3e} "
                           f"probe {abs(fp - ep) / abs(fp):.3e}")
            print(f"    h {h:.0e}: " + "; ".join(msg))

        s = step(g, obj, probe, corners, counts, idx)
        pc = ptychi_step(g, obj, probe, corners, counts)
        assert (pc["idx"] == idx).all(), pc["idx"]
        print(f"  pty-chi step ({pc['randn_calls']} torch.randn call(s) zeroed), "
              "port vs pty-chi (max, L2):")
        yp = icalc(g, obj, probe, corners[idx])
        print("    y_pred   %.3e %.3e" % ls.mg.rel(pc["y_pred"], yp))
        print("    illum    %.3e %.3e" % ls.mg.rel(pc["illum"], s["illum"]))
        for k in ("a_far", "a_o", "a_p", "d_obj", "d_probe"):
            print(f"    {k:8s} %.3e %.3e" % ls.mg.rel(s[k], pc[k]))
        out["icalc"].append(np.column_stack([np.full(ICALC_FRAMES, case),
                                             np.arange(ICALC_FRAMES),
                                             pc["y_pred"][:ICALC_FRAMES].reshape(
                                                 ICALC_FRAMES, -1)]))
        for name, rows_fmt in (
                ("obj", ls.rows_obj(pc["d_obj"])), ("probe", ls.rows_probe(pc["d_probe"])),
                ("steps", (np.column_stack([idx, pc["a_far"], pc["a_o"], pc["a_p"]]),
                           ["%d"] + ["%.17e"] * 3))):
            out[f"step_{name}"].append(lead(case, rows_fmt))

        if case not in EPOCH_CASES:
            continue
        pe = ptychi_epochs(g, obj, probe, corners, counts)
        for e in range(EPOCHS):
            got = [st["idx"].tolist() for st in pe["steps"] if st["epoch"] == e]
            want = [b.tolist() for b in rb.random_batches(nframe, BATCH, BATCH_SEED, e)]
            assert got == want, (e, got, want)
        print(f"  pty-chi {EPOCHS} epochs ({pe['randn_calls']} torch.randn call(s) zeroed), "
              f"LLK by epoch:")
        print("    " + " ".join(f"{v:.4f}" for v in pe["llk"]))
        rows = np.concatenate([np.column_stack([np.full(len(st["idx"]), st["epoch"]),
                                                st["idx"], st["a_far"], st["a_o"],
                                                st["a_p"]])
                               for st in pe["steps"]])
        for name, rows_fmt in (
                ("obj", ls.rows_obj(pe["d_obj"])), ("probe", ls.rows_probe(pe["d_probe"])),
                ("steps", (rows, ["%d", "%d"] + ["%.17e"] * 3))):
            out[f"epochs_{name}"].append(lead(case, rows_fmt))
        out["llk"].append(np.column_stack([np.full(EPOCHS + 1, case),
                                           np.arange(-1, EPOCHS), pe["llk"]]))

    def save(name, rows, fmt):
        path = f"{d}/lsqml_near_{name}_f64.txt"
        np.savetxt(path, rows, fmt=fmt)
        print(f"wrote {path} ({len(rows)} rows)")

    f = "%.17e"
    save("cases", np.array(out["cases"]), ["%d", f, f, f, "%d", f, f])
    save("corners", np.column_stack([np.arange(nframe), corners]), "%d")
    save("obj", *ls.rows_obj(obj))
    save("probe", *ls.rows_probe(probe))
    save("counts", np.concatenate(out["counts"]), "%d")
    save("icalc", np.concatenate(out["icalc"]), ["%d", "%d"] + [f] * (SHAPE[0] * SHAPE[1]))
    for name in ("step_obj", "step_probe", "step_steps", "epochs_obj", "epochs_probe",
                 "epochs_steps"):
        rows = [r for r, _ in out[name]]
        save(name, np.concatenate(rows), out[name][0][1])
    save("llk", np.concatenate(out["llk"]), ["%d", "%d", f])


def direction(rng, obj, probe):
    """Random object and probe directions scaled to each one's largest modulus."""
    d_obj = rng.standard_normal(obj.shape) + 1j * rng.standard_normal(obj.shape)
    d_probe = rng.standard_normal(probe.shape) + 1j * rng.standard_normal(probe.shape)
    return (d_obj * np.abs(obj).max() / np.abs(d_obj).max(),
            d_probe * np.abs(probe).max() / np.abs(d_probe).max())


if __name__ == "__main__":
    main()
