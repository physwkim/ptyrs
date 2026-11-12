"""Float64 reference of pty-chi LSQML (Poisson) through lsqml_multislice.py's multislice object
with the probe positions optimizable: its step on frames 0..BATCH from that script's start probe
and its truth object (the start object, every slice ones, has no gradient to move a position
by), each frame at its corner plus an offset d_k (dy, dx) uniform in [-OFFSET, OFFSET) from
numpy default_rng(SEED), every case.

    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_multislice_pos.py <dir>

<dir> holds lsqml_multislice.py's files (refs/f64 or ptyrs/tests/engine/data), whose start
probe, corners, cases and counts are read here; its truth object is made again by its
make_state. pty-chi's PATCH model keeps the windows at the corners
and moves the probe entering slice 0 by d (|d| < 1/2), and corrects the positions on slice
NSLICE // 2 from the wave entering it (lsqml_pos_step.py's formulas on that slice). The run is
lsqml_multislice.ptychi_step's with the positions optimizable at pty-chi's defaults.

Writes `<dir>/lsqml_multislice_pos_{obj,probe,steps}_f64.txt` as lsqml_multislice.py's step
files, `<dir>/lsqml_multislice_pos_offsets_f64.txt`, `case frame dy dx dy' dx'`, and
`<dir>/lsqml_multislice_pos_start_f64.txt`, the object, `slice iy ix re im`.
"""
import importlib.util
import pathlib
import sys

import numpy as np

SEED = 2
OFFSET = 0.3


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ms = sibling("lsqml_multislice")
near = ms.near


def task(g, obj, probe, corners, counts, offsets):
    """lsqml_multislice.task at corners + offsets, the positions optimizable."""
    import ptychi.api.task as at

    real = at.PtychographyTask

    def moved(opts, **kw):
        opts.probe_position_options.optimizable = True
        kw["probe_position_y_px"] = kw["probe_position_y_px"] + offsets[:, 0]
        kw["probe_position_x_px"] = kw["probe_position_x_px"] + offsets[:, 1]
        return real(opts, **kw)

    at.PtychographyTask = moved
    try:
        return ms.task(g, obj, probe, corners, counts, ms.BATCH, 1)
    finally:
        at.PtychographyTask = real


def ptychi_step(g, obj, probe, corners, counts, offsets):
    """lsqml_multislice.ptychi_step with the positions, and the offsets after it."""
    import torch
    from torch.utils.data import DataLoader

    t = task(g, obj, probe, corners, counts, offsets)
    rec = t.reconstructor
    pg = rec.parameter_group
    assert pg.probe_positions.optimization_enabled(0)
    with torch.no_grad(), near.ZeroRandn() as z:
        rec.run_pre_run_hooks()
        rec.run_pre_epoch_hooks()
        pg.object.preconditioner = pg.object.calculate_illumination_map(
            pg.probe, pg.probe_positions, use_all_modes=True)
        batch = next(iter(DataLoader(rec.dataset, batch_size=ms.BATCH, shuffle=False)))
        input_data, y_true = rec.prepare_batch_data(batch)
        rec.run_minibatch(input_data, y_true)
    idx = input_data[0].numpy()
    buf = rec.reconstructor_buffers
    o, p = ms.state_of(t, g)
    return dict(
        idx=idx, randn_calls=z.calls,
        a_far=rec.alpha_psi_far.cpu().numpy().reshape(-1),
        a_o=buf.alpha_object_all_pos_all_slices[idx].cpu().numpy(),
        a_p=buf.alpha_probe_all_pos[idx].cpu().numpy(),
        d_obj=o - obj, d_probe=p - probe,
        offsets=t.get_data_to_cpu("probe_positions", as_numpy=True) - corners)


def planes(rows, n, shape):
    """`lead iy ix re im` rows as n planes."""
    return (rows[:, 3] + 1j * rows[:, 4]).reshape(n, *shape)


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    load = lambda name: np.loadtxt(f"{d}/lsqml_multislice_{name}_f64.txt", ndmin=2)
    corners = load("corners")[:, 1:].astype(int)
    obj, _, start, made, _ = ms.make_state()
    probe = planes(load("probe"), ms.NMODE, ms.SHAPE)
    assert (made == corners).all() and (start == probe).all()
    cases, counts = load("cases"), load("counts")
    nframe = len(corners)
    offsets = OFFSET * np.random.default_rng(SEED).uniform(-1, 1, (nframe, 2))
    idx = np.arange(ms.BATCH)
    out = {k: [] for k in ("obj", "probe", "steps", "offsets")}
    f = "%.17e"
    for case, geom in enumerate(ms.GEOMETRIES):
        assert np.allclose(cases[case, 1:4], geom, rtol=1e-15, atol=0), cases[case]
        g = (ms.FarGeometry if case == ms.FAR else ms.Geometry)(*geom)
        c = counts[counts[:, 0] == case][:, 2:].reshape(nframe, *ms.SHAPE)
        pc = ptychi_step(g, obj, probe, corners, c, offsets)
        assert (pc["idx"] == idx).all(), pc["idx"]
        change = pc["offsets"] - offsets
        assert np.abs(change[ms.BATCH:]).max() < 1e-12
        print(f"case {case} ({pc['randn_calls']} torch.randn call(s) zeroed): a_o (frame 0) "
              f"{pc['a_o'][0]}, max |offset change| {np.abs(change).max():.4e}, by axis "
              f"{np.abs(change).max(0)}")
        out["obj"].append(near.lead(case, ms.rows_slices(pc["d_obj"])))
        out["probe"].append(near.lead(case, ms.ls.rows_probe(pc["d_probe"])))
        out["steps"].append(near.lead(case, (ms.step_rows([], pc),
                                             ["%d"] + [f] * (ms.NSLICE + 2))))
        out["offsets"].append(near.lead(case, (np.column_stack(
            [np.arange(nframe), offsets, pc["offsets"]]), ["%d"] + [f] * 4)))
    out["start"] = [ms.rows_slices(obj)]
    for name, parts in out.items():
        path = f"{d}/lsqml_multislice_pos_{name}_f64.txt"
        np.savetxt(path, np.concatenate([r for r, _ in parts]), fmt=parts[0][1])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
