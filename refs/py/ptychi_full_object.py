"""Float64 references of pty-chi rPIE and single-slice BH refining the probe positions under its
FULL_OBJECT shift operator, on the state and offsets of refs/f64/rpie_pos.py and bh_pos.py.

    <engine>/target/release/examples/engine_llk_trace <state> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/py/ptychi_full_object.py <state> <out>

FULL_OBJECT keeps the probe unmoved and samples frame k's window at its corner plus d_k from the
whole object's periodic Fourier interpolant; the object's update is placed by that sampling's
adjoint, and the positions' gradient is the interpolant's derivative. rPIE runs rpie_pos.py's
cases (`step`, `epochs`) through its `ptychi`, BH bh_pos.py's (`full`: CG on one batch of every
frame, `mini`: GD on batches of 16, both in the engine's batch order) through `bh` here, each
with FULL_OBJECT in place of PATCH and from that script's offsets. rPIE's `odd` is `step` on the
76 x 80 object plane less its first row and column: both sides odd, so the interpolant has a
middle frequency (n - 1) / (2 n) on each, and every window one pixel nearer the origin, so on
the same pixels.

At zero offsets the two operators sample the same windows. The script runs one batch of every
frame under each there, on both planes, FULL_OBJECT's position derivatives taken PATCH's way
(from the window's own interpolant), and exits non-zero unless every change, and BH's alpha, top
and bottom, agree to ZERO_BOUND; it prints how far the full-object derivatives move the offsets'
change.

Writes, every complex value `re im`, in the engine's convention:
    <out>/rpie_full_object_<case>_{obj,probe,offsets}_f64.txt     as rpie_pos.py's files
    <out>/bh_full_object_<case>_{obj,probe,offsets,steps}_f64.txt as bh_pos.py's files
"""
import importlib.util
import pathlib
import sys

import numpy as np

# 5x the worst measured, rPIE's offsets change (4.6e-13)
ZERO_BOUND = 2.5e-12


def f64_script(name):
    """Import a refs/f64 script (python -I keeps every script directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).parents[1] / "f64" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mg = f64_script("ml_grad")
lp = f64_script("lsqml_pos_step")
rb = f64_script("random_batches")
rpie_pos = f64_script("rpie_pos")
bh_pos = f64_script("bh_pos")


class Operator:
    """Within the block, every PtychographyTask runs `operator` (a ShiftOperators name), its
    position derivatives taken PATCH's way with `window`; `tasks` collects them."""

    def __init__(self, operator, window=False):
        self.operator, self.window, self.tasks = operator, window, []

    def __enter__(self):
        import ptychi.api as api
        import ptychi.api.task as at

        self.real = real = at.PtychographyTask

        def task(opts, **kw):
            opts.reconstructor_options.forward_model_options.shift_operator = (
                api.ShiftOperators[self.operator])
            t = real(opts, **kw)
            fm = t.reconstructor.forward_model
            assert fm.resolved_shift_operator.name == self.operator, fm.resolved_shift_operator
            if self.window:
                window(t.reconstructor)
            self.tasks.append(t)
            return t

        at.PtychographyTask = task
        return self

    def __exit__(self, *exc):
        import ptychi.api.task as at

        at.PtychographyTask = self.real


def window(rec):
    """rec's position derivatives from each window's own interpolant, as PATCH takes them:
    rPIE's fall back to its differentiation method, BH's take PATCH's branch."""
    import ptychi.api as api

    fm = rec.forward_model
    fm.get_object_position_gradients = lambda *_, **__: None
    if not hasattr(rec, "position_directional_derivative"):
        return
    cls = type(rec)

    def patch(*args, **kwargs):
        operator = fm.shift_operator
        fm.shift_operator = api.ShiftOperators.PATCH
        try:
            return cls.position_directional_derivative(rec, *args, **kwargs)
        finally:
            fm.shift_operator = operator

    rec.position_directional_derivative = patch


def bh(m, iobs, wavelength, obj, probe, offsets, size, cg, nepoch):
    """pty-chi BHReconstructor with the positions optimizable (bh_pos.ptychi_step's options,
    CG or GD, batches of `size` in the engine's order) for nepoch epochs from corners + offsets:
    (obj, probe, offsets, steps), each step a dict of its frames, epoch, beta, alpha (scaled by
    its share of the batch size, as applied), top and bottom."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask
    from ptychi.maths import redot

    nframe, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
    opts = api.BHOptions()
    opts.data_options.wavelength_m = wavelength
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = True
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    pr = opts.probe_options
    pr.optimizable = True
    pr.rho = bh_pos.RHO
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    pp = opts.probe_position_options
    pp.optimizable = True
    pp.rho = bh_pos.POS_RHO
    r = opts.reconstructor_options
    r.method = "CG" if cg else "GD"
    r.batch_size = size
    r.num_epochs = nepoch
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.pad_for_shift = 0
    pos = m.corners + offsets
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj[None].copy(),
        probe_data=(probe * m.q / root_n)[None].copy(),
        probe_position_y_px=pos[:, 0].copy(),
        probe_position_x_px=pos[:, 1].copy(),
    )
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset, batch_sampler=rb.Sampler(nframe, size, bh_pos.SEED))
    cls, steps = type(rec), []

    def minibatch(input_data, y_true):
        steps.append(dict(frames=input_data[0].numpy().copy(), epoch=rec.current_epoch,
                          beta=0.0))
        return cls.run_minibatch(rec, input_data, y_true)

    def beta(*args, **kwargs):
        b = cls.calc_beta_object_probe_positions(rec, *args, **kwargs)
        steps[-1]["beta"] = float(b)
        return b

    def alpha(p, do1, dp1, dpos1, do2, dp2, dpos2, *rest):
        a = cls.calc_alpha_object_probe_positions(rec, p, do1, dp1, dpos1, do2, dp2, dpos2, *rest)
        top = -redot(do1, do2) - redot(dp1, dp2) - torch.sum(dpos1 * dpos2)
        s = steps[-1]
        s.update(top=float(top), bottom=float(top / a), alpha=float(a) * len(s["frames"]) / size)
        return a

    rec.run_minibatch = minibatch
    rec.calc_beta_object_probe_positions = beta
    rec.calc_alpha_object_probe_positions = alpha
    with torch.no_grad():
        task.run(nepoch)
    back = np.conj(m.q) * root_n
    return (task.get_data_to_cpu("object", as_numpy=True)[0],
            task.get_data_to_cpu("probe", as_numpy=True)[0] * back,
            task.get_data_to_cpu("probe_positions", as_numpy=True) - m.corners, steps)


def rel(a, b):
    """max |a - b| / max |b|."""
    return np.abs(a - b).max() / np.abs(b).max()


def at_zero(m, iobs, wavelength, obj, probe):
    """One batch of every frame of rPIE and BH from zero offsets under PATCH and FULL_OBJECT:
    the worst relative difference."""
    nframe = len(m.corners)
    zero = np.zeros((nframe, 2))
    prm = dict(rpie_pos.STEP_PARAMS, batch_size=nframe)
    runs = dict(
        rpie=lambda: rpie_pos.ptychi(m, iobs, wavelength, obj[None], zero, prm, 1)[:3],
        bh=lambda: bh(m, iobs, wavelength, obj, probe, zero, nframe, False, 1),
    )
    worst = 0.0
    for name, run in runs.items():
        out = {}
        for key, operator, win in (("patch", "PATCH", False), ("full", "FULL_OBJECT", False),
                                   ("window", "FULL_OBJECT", True)):
            with Operator(operator, win):
                out[key] = run()
        (o1, p1, d1, *s1), (o2, p2, d2, *s2) = out["patch"], out["window"]
        diffs = dict(obj=rel(o2 - obj, o1 - obj), probe=rel(p2 - probe, p1 - probe),
                     offsets=rel(d2, d1))
        if s1:
            diffs.update({k: abs(s2[0][0][k] - s1[0][0][k]) / abs(s1[0][0][k])
                          for k in ("alpha", "top", "bottom")})
        print(f"{name} at zero offsets, FULL_OBJECT with window derivatives vs PATCH: "
              + ", ".join(f"{k} {v:.2e}" for k, v in diffs.items())
              + "; the full-object derivatives move the offsets' change by "
              + f"{rel(out['full'][2], d1):.2e}")
        worst = max(worst, *diffs.values())
    return worst


def write(out, name, obj, probe, d0, d1, o1, p1):
    rows = {
        "obj": lp.cplx_rows(obj.shape, o1 - obj),
        "probe": lp.cplx_rows(probe.shape, p1 - probe),
        "offsets": (np.column_stack([np.arange(len(d0)), d0, d1]), ["%d"] + ["%.17e"] * 4),
    }
    for kind, (a, fmt) in rows.items():
        path = f"{out}/{name}_{kind}_f64.txt"
        np.savetxt(path, a, fmt=fmt)
        print(f"  wrote {path} ({len(a)} rows)")


def main():
    state, out = sys.argv[1:]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    args = mg.load(state)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    obj = obj0[0]
    odd0 = obj0[:, 1:, 1:]
    m_odd = mg.Model(*args[:5], odd0, probe)
    assert (m_odd.corners == m.corners - 1).all() and odd0.shape[1] % 2 and odd0.shape[2] % 2
    nframe = len(m.corners)
    worst = max(at_zero(m, iobs, wavelength, obj, probe),
                at_zero(m_odd, iobs, wavelength, odd0[0], probe))
    if not worst <= ZERO_BOUND:
        sys.exit(f"FULL_OBJECT departs from PATCH at zero offsets: {worst:.3e} > {ZERO_BOUND:.0e}")

    unit = np.random.default_rng(1).uniform(-1, 1, (nframe, 2))
    for case, mc, o0, scale, prm, nepoch in (
            ("step", m, obj0, rpie_pos.OFFSET, rpie_pos.STEP_PARAMS, 1),
            ("epochs", m, obj0, rpie_pos.EPOCH_OFFSET, rpie_pos.EPOCH_PARAMS, rpie_pos.EPOCHS),
            ("odd", m_odd, odd0, rpie_pos.OFFSET, rpie_pos.STEP_PARAMS, 1)):
        d0 = scale * unit
        with Operator("FULL_OBJECT"):
            o1, p1, d1, peak, nbatch = rpie_pos.ptychi(mc, iobs, wavelength, o0, d0, prm, nepoch)
        print(f"--- rPIE {case}: {o0.shape[1:]} object, {nepoch} epoch(s), {nbatch} batches, "
              f"{prm}, max |offset| over the run {peak:.4f}, max |offset change| "
              f"{np.abs(d1 - d0).max():.4f}")
        write(out, f"rpie_full_object_{case}", o0[0], probe, d0, d1, o1, p1)

    d0 = 0.5 + bh_pos.OFFSET * unit
    for case, size, cg in bh_pos.CASES:
        with Operator("FULL_OBJECT"):
            o1, p1, d1, steps = bh(m, iobs, wavelength, obj, probe, d0, size, cg, bh_pos.EPOCHS)
        print(f"--- BH {case}: batch {size}, {'CG' if cg else 'GD'}, {bh_pos.EPOCHS} epochs, max "
              f"|offset change| {np.abs(d1 - d0).max():.4f}")
        rows = []
        for k, s in enumerate(steps):
            print(f"  step {k} epoch {s['epoch']} count {len(s['frames'])}: beta {s['beta']: .6e} "
                  f"alpha {s['alpha']: .6e} top {s['top']: .6e} bottom {s['bottom']: .6e}")
            rows.append(f"{k} {s['epoch']} {len(s['frames'])} "
                        + " ".join(f"{s[x]:.17e}" for x in ("beta", "alpha", "top", "bottom"))
                        + " " + " ".join(str(f) for f in s["frames"]))
        name = f"bh_full_object_{case}"
        write(out, name, obj, probe, d0, d1, o1, p1)
        path = f"{out}/{name}_steps_f64.txt"
        pathlib.Path(path).write_text("".join(r + "\n" for r in rows))
        print(f"  wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
