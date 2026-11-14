"""Float64 reference of pty-chi LSQML (Poisson) in lsqml_near.py's planar near-field model with
the probe positions optimizable, on its angular-spectrum cases (0, and 2 with an evanescent
band): its step on frames 0..BATCH from that script's start probe and its truth object (the
start object, ones, has no gradient to move a position by), each frame at its corner plus an
offset d_k (dy, dx) uniform in [-OFFSET, OFFSET) from numpy default_rng(SEED), under each of
pty-chi's two shift operators.

    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_near_pos.py <dir>

<dir> holds lsqml_near.py's files (refs/f64 or ptyrs/tests/engine/data), whose start probe,
corners, cases and counts are read here; its truth object is made again by its make_state.
pty-chi's PATCH model keeps the window at the corner and moves the probe by d (|d| < 1/2), and
takes the position gradient from the window's own periodic interpolant. Its FULL_OBJECT model
keeps the probe and samples the object at corner + d + u, u over the frame, from the whole
object's periodic Fourier interpolant; its illumination map is placed by that sampling's
adjoint, and its position gradient is that interpolant's derivative. The run is
lsqml_near.ptychi_step's with the positions optimizable at pty-chi's defaults.

At zero offsets the two operators sample the same windows. The script runs both there and
exits non-zero unless every step size and every change agrees to ZERO_BOUND, the offsets'
change once FULL_OBJECT takes PATCH's window gradient; it prints how far the full-object
gradient moves that change.

Writes, PATCH as `pos` and FULL_OBJECT as `pos_full`, every complex value `re im`:
    <dir>/lsqml_near_{pos,pos_full}_obj_f64.txt      `case iy ix re im`, the object's change
    <dir>/lsqml_near_{pos,pos_full}_probe_f64.txt    `case m iy ix re im`, the probe's change
    <dir>/lsqml_near_{pos,pos_full}_steps_f64.txt    `case frame a_far a_o a_p`
    <dir>/lsqml_near_{pos,pos_full}_offsets_f64.txt  `case frame dy dx dy' dx'`, the offsets
                                                     before and after the step
    <dir>/lsqml_near_pos_start_f64.txt               `iy ix re im`, the object
"""
import importlib.util
import pathlib
import sys

import numpy as np

SEED = 2
OFFSET = 0.3
# lsqml_near.py's angular-spectrum cases
CASES = (0, 2)
# 300x the worst measured, case 0's object change (3.4e-15)
ZERO_BOUND = 1e-12


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


near = sibling("lsqml_near")
ls = near.ls


def task(g, obj, probe, corners, counts, offsets, operator):
    """lsqml_near.task at corners + offsets, the positions optimizable, under the shift operator
    named `operator`."""
    import ptychi.api as api
    import ptychi.api.task as at

    real = at.PtychographyTask

    def moved(opts, **kw):
        opts.probe_position_options.optimizable = True
        opts.reconstructor_options.forward_model_options.shift_operator = (
            api.ShiftOperators[operator])
        kw["probe_position_y_px"] = kw["probe_position_y_px"] + offsets[:, 0]
        kw["probe_position_x_px"] = kw["probe_position_x_px"] + offsets[:, 1]
        return real(opts, **kw)

    at.PtychographyTask = moved
    try:
        return near.task(g, obj, probe, corners, counts, near.BATCH, 1)
    finally:
        at.PtychographyTask = real


def ptychi_step(g, obj, probe, corners, counts, offsets, operator, window_gradient=False):
    """lsqml_near.ptychi_step with the positions, and the offsets after it; with
    `window_gradient`, FULL_OBJECT's position gradient is PATCH's, the window's own."""
    import torch
    from torch.utils.data import DataLoader

    t = task(g, obj, probe, corners, counts, offsets, operator)
    rec = t.reconstructor
    pg = rec.parameter_group
    fm = rec.forward_model
    assert pg.probe_positions.optimization_enabled(0)
    assert fm.resolved_shift_operator.name == operator, fm.resolved_shift_operator
    if window_gradient:
        fm.get_object_position_gradients = lambda *_, **__: None
    with torch.no_grad(), near.ZeroRandn() as z:
        rec.run_pre_run_hooks()
        rec.run_pre_epoch_hooks()
        pg.object.preconditioner = pg.object.calculate_illumination_map(
            pg.probe, pg.probe_positions, use_all_modes=True,
            sampling=fm.get_object_sampling_parameters())
        batch = next(iter(DataLoader(rec.dataset, batch_size=near.BATCH, shuffle=False)))
        input_data, y_true = rec.prepare_batch_data(batch)
        rec.run_minibatch(input_data, y_true)
    idx = input_data[0].numpy()
    buf = rec.reconstructor_buffers
    o, p = near.state_of(t, g)
    return dict(
        idx=idx, randn_calls=z.calls,
        a_far=rec.alpha_psi_far.cpu().numpy().reshape(-1),
        a_o=buf.alpha_object_all_pos_all_slices[idx, 0].cpu().numpy(),
        a_p=buf.alpha_probe_all_pos[idx].cpu().numpy(),
        d_obj=o - obj, d_probe=p - probe,
        offsets=t.get_data_to_cpu("probe_positions", as_numpy=True) - corners)


def rel(a, b):
    """max |a - b| / max |b|."""
    return np.abs(a - b).max() / np.abs(b).max()


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    load = lambda name: np.loadtxt(f"{d}/lsqml_near_{name}_f64.txt", ndmin=2)
    corners = load("corners")[:, 1:].astype(int)
    obj, _, start, made, _ = near.make_state()
    rows = load("probe")
    probe = (rows[:, 3] + 1j * rows[:, 4]).reshape(near.NMODE, *near.SHAPE)
    assert (made == corners).all() and (start == probe).all()
    cases, counts = load("cases"), load("counts")
    nframe = len(corners)
    offsets = OFFSET * np.random.default_rng(SEED).uniform(-1, 1, (nframe, 2))
    idx = np.arange(near.BATCH)
    zero = np.zeros_like(offsets)
    runs = (("pos", "PATCH"), ("pos_full", "FULL_OBJECT"))
    out = {(key, k): [] for key, _ in runs for k in ("obj", "probe", "steps", "offsets")}
    f = "%.17e"
    worst = 0.0
    for case in CASES:
        geom = near.GEOMETRIES[case]
        assert np.allclose(cases[case, 1:4], geom, rtol=1e-15, atol=0), cases[case]
        g = near.Geometry(*geom)
        assert g.kind == 1 == cases[case, 4], (case, g.kind)  # angular spectrum
        c = counts[counts[:, 0] == case][:, 2:].reshape(nframe, *near.SHAPE)
        step = lambda at, *op, **kw: ptychi_step(g, obj, probe, corners, c, at, *op, **kw)
        patch, full = step(zero, "PATCH"), step(zero, "FULL_OBJECT")
        window = step(zero, "FULL_OBJECT", window_gradient=True)
        head = lambda r: r["offsets"][:near.BATCH]
        diffs = {k: rel(full[k], patch[k]) for k in ("a_far", "a_o", "a_p", "d_obj", "d_probe")}
        diffs["offsets"] = rel(head(window), head(patch))
        gradient = rel(head(full), head(patch))
        print(f"case {case} at zero offsets, FULL_OBJECT vs PATCH: "
              + ", ".join(f"{k} {v:.2e}" for k, v in diffs.items())
              + f"; the full-object position gradient moves the offsets' change by {gradient:.2e}")
        worst = max(worst, *diffs.values())
        for key, operator in runs:
            pc = step(offsets, operator)
            assert (pc["idx"] == idx).all(), pc["idx"]
            change = pc["offsets"] - offsets
            assert np.abs(change[near.BATCH:]).max() < 1e-12
            print(f"case {case} {operator} ({pc['randn_calls']} torch.randn call(s) zeroed): a_o "
                  f"(frame 0) {pc['a_o'][0]}, max |offset change| {np.abs(change).max():.4e}, "
                  f"by axis {np.abs(change).max(0)}")
            out[key, "obj"].append(near.lead(case, ls.rows_obj(pc["d_obj"])))
            out[key, "probe"].append(near.lead(case, ls.rows_probe(pc["d_probe"])))
            out[key, "steps"].append(near.lead(case, (np.column_stack(
                [pc["idx"], pc["a_far"], pc["a_o"], pc["a_p"]]), ["%d"] + [f] * 3)))
            out[key, "offsets"].append(near.lead(case, (np.column_stack(
                [np.arange(nframe), offsets, pc["offsets"]]), ["%d"] + [f] * 4)))
    if not worst <= ZERO_BOUND:
        sys.exit(f"FULL_OBJECT departs from PATCH at zero offsets: {worst:.3e} > {ZERO_BOUND:.0e}")
    out["pos", "start"] = [ls.rows_obj(obj)]
    for (key, name), parts in out.items():
        path = f"{d}/lsqml_near_{key}_{name}_f64.txt"
        np.savetxt(path, np.concatenate([r for r, _ in parts]), fmt=parts[0][1])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
