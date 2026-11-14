"""Float64 reference of pty-chi rPIE in its planar near-field model, against the model here.

    micromamba run -n ptyrs-ref python -I refs/f64/rpie_near.py <dir>

The state is lsqml_near.py's (truth, start probe, object of ones) on 16 whole-pixel corners:
its first 8 and their mirror images through the centre, so their mean is the centre and the
engine's PyNX position preprocessing leaves them alone. GEOMETRIES (wavelength, pixel,
distance) give an angular-spectrum propagator and a Fresnel transform forward and back; the
counts are the truth's Poisson counts through each, natural (centred) layout.

The batches are consecutive in scan order. Per batch, all from the batch-start state, in
lsqml_near.py's convention (U pty-chi's propagate_forward over sqrt(k), pty-chi's probe
P / sqrt(k)):

    psi_km = P_m V_k,  far = U psi
    Delta_km = U^H(far_km sqrt(iobs + 1e-7) / (sqrt(sum_m |far_km|^2) + 1e-7)) - psi_km
    O += obj_step scatter(sum_m conj(P_m) Delta_km / ((1 - a_o)|P_m|^2 + a_o Pmax)),
        Pmax = max over the pixels of sum_m |P_m|^2, the scatter a sum over the batch's frames
    P_m += probe_step mean_batch(conj(V_k) Delta_km / ((1 - a_p)|V_k|^2 + a_p max |V_k|^2))

pty-chi's RPIEReconstructor (pie.py compute_updates and apply_updates) in f64 on CPU runs the
same epochs with the object-probe ambiguity removal and the mode orthogonalisation off and the
positions fixed. Every batch size of BATCHES at every object step of OBJ_STEPS runs EPOCHS
epochs on both; printed per run: the model here against pty-chi (max, L2 relative difference
of the object and probe changes and the largest relative LLK difference) and the epochs after
which pty-chi is at or under the truth's LLK.

Writes, every complex value `re im`, every plane row-major, `run` the row of the runs table:
    <dir>/rpie_near_cases_f64.txt    `case wavelength pixel distance kind scale truth_llk`
    <dir>/rpie_near_corners_f64.txt  `frame cy cx`
    <dir>/rpie_near_obj_f64.txt      `iy ix re im`, the start object
    <dir>/rpie_near_probe_f64.txt    `m iy ix re im`, the start probe
    <dir>/rpie_near_counts_f64.txt   `case frame` and the frame's counts, natural layout
    <dir>/rpie_near_runs_f64.txt     `run case batch epochs obj_step probe_step obj_alpha
                                     probe_alpha`, the runs SAVED of pty-chi
    <dir>/rpie_near_run_obj_f64.txt    `run iy ix re im`, the object change over the run
    <dir>/rpie_near_run_probe_f64.txt  `run m iy ix re im`, the probe change over the run
    <dir>/rpie_near_llk_f64.txt      `run epoch llk`, the normalised Poisson LLK after each
                                     epoch, epoch -1 the start
"""
import importlib.util
import pathlib
import sys

import numpy as np

EPOCHS = 30
BATCHES = (1, 2, 4, 8, 16)
# pty-chi's defaults but for the object step: the object change is a sum over the batch's
# frames, so at pty-chi's 1 it diverges on both sides once a batch's frames overlap
PARAMS = dict(probe_step=1.0, obj_alpha=0.1, probe_alpha=0.1)
OBJ_STEPS = (1.0, 0.25, 0.05)
# (batch, obj_step) saved of every case
SAVED = ((1, 0.25), (4, 0.25), (16, 0.05))
# (wavelength, pixel, distance): angular spectrum, Fresnel transform, Fresnel transform back
GEOMETRIES = [(1e-10, 50e-9, 5e-4), (1e-10, 50e-9, 2e-3), (1e-10, 50e-9, -2e-3)]


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
AX = near.AX


def make_state():
    truth_obj, truth_probe, probe, corners, rng = near.make_state()
    span = np.array(near.OBJ) - np.array(near.SHAPE)
    half = corners[:len(corners) // 2]
    corners = np.concatenate([half, span - half[::-1]])
    assert (2 * corners.sum(0) == len(corners) * span).all(), corners.mean(0)
    return truth_obj, truth_probe, probe, corners, rng


def batch_step(g, obj, probe, corners, counts, idx, prm):
    """One rPIE batch on the frames idx: (obj, probe)."""
    c_idx = corners[idx]
    v = near.views(obj, c_idx)
    psi = probe[None] * v[:, None]
    far = g.fwd(psi)
    r = np.sqrt(counts[idx] + 1e-7) / (np.sqrt((np.abs(far) ** 2).sum(1)) + 1e-7)
    delta = g.adj(far * r[:, None]) - psi

    a = prm["obj_alpha"]
    p2 = np.abs(probe) ** 2
    w = np.conj(probe) / ((1 - a) * p2 + a * p2.sum(0).max())
    d_obj = ls.scatter((w[None] * delta).sum(1), c_idx, obj.shape)
    a = prm["probe_alpha"]
    v2 = np.abs(v) ** 2
    w = np.conj(v) / ((1 - a) * v2 + a * v2.max(AX, keepdims=True))
    d_probe = (w[:, None] * delta).mean(0)
    return obj + prm["obj_step"] * d_obj, probe + prm["probe_step"] * d_probe


def model(g, obj, probe, corners, counts, batch, prm):
    """EPOCHS epochs of the model: (obj, probe, LLK trace)."""
    trace = [near.llk(g, obj, probe, corners, counts)]
    for _ in range(EPOCHS):
        for b0 in range(0, len(corners), batch):
            idx = np.arange(b0, min(b0 + batch, len(corners)))
            obj, probe = batch_step(g, obj, probe, corners, counts, idx, prm)
        trace.append(near.llk(g, obj, probe, corners, counts))
    return obj, probe, np.array(trace)


def ptychi(g, obj, probe, corners, counts, batch, prm):
    """pty-chi RPIE for EPOCHS epochs of consecutive batches: (obj, probe, LLK trace)."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = near.SHAPE
    opts = api.RPIEOptions()
    opts.data_options.wavelength_m = g.wavelength
    opts.data_options.fft_shift = False
    opts.data_options.free_space_propagation_distance_m = g.distance
    o = opts.object_options
    o.optimizable = True
    o.pixel_size_m = g.pixel
    o.step_size = prm["obj_step"]
    o.alpha = prm["obj_alpha"]
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    pr = opts.probe_options
    pr.optimizable = True
    pr.step_size = prm["probe_step"]
    pr.alpha = prm["probe_alpha"]
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    opts.probe_position_options.optimizable = False
    r = opts.reconstructor_options
    r.batch_size = batch
    r.num_epochs = EPOCHS
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    r.forward_model_options.subpixel_shift_all_probe_modes = True
    t = PtychographyTask(
        opts,
        diffraction_data=counts.copy(),
        object_data=obj[None].copy(),
        probe_data=(probe / np.sqrt(g.k))[None].copy(),
        probe_position_y_px=corners[:, 0].astype(np.float64),
        probe_position_x_px=corners[:, 1].astype(np.float64),
    )
    rec = t.reconstructor
    rec.dataloader = DataLoader(rec.dataset, batch_size=batch, shuffle=False)
    trace = [near.llk(g, obj, probe, corners, counts)]
    with torch.no_grad():
        for _ in range(EPOCHS):
            t.run(1)
            trace.append(near.llk(g, *near.state_of(t, g), corners, counts))
    return (*near.state_of(t, g), np.array(trace))


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    truth_obj, truth_probe, probe, corners, rng = make_state()
    obj = np.ones(near.OBJ, complex)
    nframe = len(corners)
    print(f"object {near.OBJ}, probe {probe.shape}, {nframe} frames, corners {corners.tolist()}")
    out = {k: [] for k in ("cases", "counts", "runs", "run_obj", "run_probe", "llk")}
    for case, geom in enumerate(GEOMETRIES):
        g = near.Geometry(*geom)
        counts = near.counts_of(g, truth_obj, truth_probe, corners, rng)
        assert counts.max() < 65535, counts.max()
        truth_llk = near.llk(g, truth_obj, truth_probe, corners, counts)
        print(f"--- case {case}: {geom}, kind {g.kind}, k {g.k:.6e}, counts max "
              f"{counts.max():.0f} mean {counts.mean():.1f}, LLK truth {truth_llk:.6f} start "
              f"{near.llk(g, obj, probe, corners, counts):.6f}")
        out["cases"].append([case, *geom, g.kind, g.k, truth_llk])
        out["counts"].append(np.column_stack([np.full(nframe, case), np.arange(nframe),
                                              counts.reshape(nframe, -1)]))
        for batch, step in ((b, s) for s in OBJ_STEPS for b in BATCHES):
            prm = dict(PARAMS, obj_step=step)
            o1, p1, l1 = model(g, obj, probe, corners, counts, batch, prm)
            o2, p2, l2 = ptychi(g, obj, probe, corners, counts, batch, prm)
            hit = next((e for e, v in enumerate(l2) if v <= truth_llk), None)
            fine = np.isfinite(l1) & np.isfinite(l2)
            print(f"  batch {batch:2d} obj_step {step}: here vs pty-chi (max, L2) "
                  "object change %.1e %.1e, probe change %.1e %.1e, " % (
                      *ls.mg.rel(o1 - obj, o2 - obj), *ls.mg.rel(p1 - probe, p2 - probe))
                  + f"LLK {np.abs(l1[fine] / l2[fine] - 1).max():.1e} over {fine.sum() - 1} "
                  f"finite epochs; pty-chi LLK {l2[0]:.4f} -> {l2[-1]:.6g} (lowest "
                  f"{np.nanmin(l2):.4f}), truth after {hit} epochs", flush=True)
            if (batch, step) not in SAVED:
                continue
            assert l2[-1] < l2[0], (case, batch, l2)
            run = len(out["runs"])
            out["runs"].append([run, case, batch, EPOCHS, step, prm["probe_step"],
                                prm["obj_alpha"], prm["probe_alpha"]])
            out["run_obj"].append(near.lead(run, ls.rows_obj(o2 - obj)))
            out["run_probe"].append(near.lead(run, ls.rows_probe(p2 - probe)))
            out["llk"].append(np.column_stack([np.full(EPOCHS + 1, run),
                                               np.arange(-1, EPOCHS), l2]))

    def save(name, rows, fmt):
        path = f"{d}/rpie_near_{name}_f64.txt"
        np.savetxt(path, rows, fmt=fmt)
        print(f"wrote {path} ({len(rows)} rows)")

    f = "%.17e"
    save("cases", np.array(out["cases"]), ["%d", f, f, f, "%d", f, f])
    save("corners", np.column_stack([np.arange(nframe), corners]), "%d")
    save("obj", *ls.rows_obj(obj))
    save("probe", *ls.rows_probe(probe))
    save("counts", np.concatenate(out["counts"]), "%d")
    save("runs", np.array(out["runs"]), ["%d"] * 4 + [f] * 4)
    for name in ("run_obj", "run_probe"):
        save(name, np.concatenate([r for r, _ in out[name]]), out[name][0][1])
    save("llk", np.concatenate(out["llk"]), ["%d", "%d", f])


if __name__ == "__main__":
    main()
