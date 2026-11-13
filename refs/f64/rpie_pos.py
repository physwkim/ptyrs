"""Float64 reference of pty-chi rPIE with probe position correction, against the model here.

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/rpie_pos.py <dir>

The state is the one lsqml_pos_step.py reads (object [1, 76, 80], probe [2, 32, 32], 40
frames) with its offsets, d_k = OFFSET or EPOCH_OFFSET times numpy default_rng(1) uniform in
[-1, 1), every |d| below 1/2, where pty-chi's PATCH model keeps the window at the corner and
moves the probe by d with a Fourier shift (all modes: subpixel_shift_all_probe_modes, as the
engine moves them). The batches are consecutive in scan order. Per batch, in the engine's
convention (lsqml_pos_step.py's T_d, F and q), all from the batch-start state:

    W_km = T_dk(P_m q),  psi_km = F(W_km V_k)
    Delta_km = F^-1(F(psi_km) sqrt(iobs + 1e-7) / (sqrt(sum_m |F psi_km|^2) + 1e-7)) - psi_km
    O += obj_step scatter(sum_m conj(W_km) Delta_km / ((1 - a_o)|W_km|^2 + a_o Wmax)),
        Wmax = max over the batch's frames and pixels of sum_m |W_km|^2
    P += probe_step conj(q) mean_batch T_-dk(conj(V_k) Delta_km / ((1 - a_p)|V_k|^2
        + a_p max |V_k|^2))
    dV = ifft(fft(V_k) 2 pi i f) along each axis,  g = dV W_k0
    delta_k = sum Re(conj(g) Delta_k0) / (sum |g|^2 + max(max_batch sum |g|^2, 1e-6 N))
    dx = 0.3 delta at the batch frames, 0 at the others (all 40)
    d += clip(dx, -L, L) per axis, L = min(10 mean |dx - mean dx|, 0.1) over all frames

pty-chi's RPIEReconstructor (pie.py compute_updates and apply_updates, PositionCorrection's
gradient update, ProbePositions.step_optimizer) in f64 on CPU runs the same epochs with the
object-probe ambiguity removal and the mode orthogonalisation off (compare_rpie.py's matched
run) and the positions optimizable at their defaults.

Writes pty-chi's run, `<case>` `step` (one epoch of two batches of 20 at STEP_PARAMS from the
OFFSET offsets) and `epochs` (EPOCHS epochs of EPOCH_PARAMS from the EPOCH_OFFSET offsets):
    <dir>/rpie_pos_<case>_obj_f64.txt      `iy ix re im`, the object change
    <dir>/rpie_pos_<case>_probe_f64.txt    `m iy ix re im`, the probe change
    <dir>/rpie_pos_<case>_offsets_f64.txt  `k dy dx dy' dx'`, every frame
Printed: the model here against pty-chi (max, L2 relative difference) for each.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fftshift, ifft2

AX = (-2, -1)
EPS_POS = 1e-6
POS_STEP = 0.3
POS_LIMIT = 0.1
OFFSET = 0.4
EPOCH_OFFSET = 0.15
EPOCHS = 3
STEP_PARAMS = dict(obj_step=0.9, probe_step=0.8, obj_alpha=0.3, probe_alpha=0.2, batch_size=20)
EPOCH_PARAMS = dict(obj_step=1.0, probe_step=1.0, obj_alpha=0.5, probe_alpha=0.5, batch_size=4)


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
lp = sibling("lsqml_pos_step")


def batch_step(m, obj, probe, offsets, idx, prm):
    """One rPIE batch on the frames idx with position correction: (obj, probe, offsets)."""
    n = probe[0].size
    corners = m.corners[idx]
    v = ls.views(obj, corners, m.n)
    d = offsets[idx]
    pw = lp.move(np.broadcast_to(probe * m.q, (len(idx), *probe.shape)), d)
    psi = pw * v[:, None]
    far = ls.far(pw, v)
    calc = np.sqrt((np.abs(far) ** 2).sum(1))
    ok = m.valid[idx]
    r = np.where(ok, np.sqrt(np.maximum(m.iobs[idx], 0) + 1e-7) / (calc + 1e-7), 1)
    delta = fftshift(ifft2(far * r[:, None], axes=AX, norm="ortho"), axes=AX) - psi

    a = prm["obj_alpha"]
    w2 = np.abs(pw) ** 2
    wmax = w2.sum(1).max()
    d_obj = ls.scatter((np.conj(pw) * delta / ((1 - a) * w2 + a * wmax)).sum(1), corners,
                       obj.shape)
    a = prm["probe_alpha"]
    v2 = np.abs(v) ** 2
    vmax = v2.max(AX, keepdims=True)
    x = (np.conj(v) / ((1 - a) * v2 + a * vmax))[:, None] * delta
    d_probe = np.conj(m.q) * lp.move(x, -d).mean(0)

    delta_pos = []
    for g in lp.fourier_gradient(v):
        gw = g * pw[:, 0]
        num = np.real(np.conj(gw) * delta[:, 0]).sum(AX)
        den = (np.abs(gw) ** 2).sum(AX)
        delta_pos.append(num / (den + max(den.max(), EPS_POS * n)))
    dx = np.zeros_like(offsets)
    dx[idx] = POS_STEP * np.stack(delta_pos, 1)
    limit = np.minimum(10 * np.abs(dx - dx.mean(0)).mean(0), POS_LIMIT)
    return (obj + prm["obj_step"] * d_obj, probe + prm["probe_step"] * d_probe,
            offsets + np.clip(dx, -limit, limit))


def model(m, obj, probe, offsets, prm, nepoch):
    nf = len(offsets)
    for _ in range(nepoch):
        for b0 in range(0, nf, prm["batch_size"]):
            idx = np.arange(b0, min(b0 + prm["batch_size"], nf))
            obj, probe, offsets = batch_step(m, obj, probe, offsets, idx, prm)
    return obj, probe, offsets


def ptychi(m, iobs, wavelength, obj0, offsets, prm, nepoch):
    """pty-chi RPIE for nepoch epochs of consecutive batches: (obj, probe, offsets)."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
    opts = api.RPIEOptions()
    opts.data_options.wavelength_m = wavelength
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = True
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
    opts.probe_position_options.optimizable = True
    r = opts.reconstructor_options
    r.batch_size = prm["batch_size"]
    r.num_epochs = nepoch
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    r.forward_model_options.subpixel_shift_all_probe_modes = True
    pos = m.corners.astype(np.float64) + offsets
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj0.copy(),
        probe_data=(m.probe * m.q / root_n)[None].copy(),
        probe_position_y_px=pos[:, 0].copy(),
        probe_position_x_px=pos[:, 1].copy(),
    )
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset, batch_size=prm["batch_size"], shuffle=False)
    peak = []
    cls = type(rec)

    def minibatch(input_data, y_true):
        out = cls.run_minibatch(rec, input_data, y_true)
        p = task.get_data_to_cpu("probe_positions", as_numpy=True)
        peak.append(np.abs(p - m.corners).max())
        return out

    rec.run_minibatch = minibatch
    with torch.no_grad():
        task.run(nepoch)
    back = np.conj(m.q) * root_n
    return (task.get_data_to_cpu("object", as_numpy=True)[0],
            task.get_data_to_cpu("probe", as_numpy=True)[0] * back,
            task.get_data_to_cpu("probe_positions", as_numpy=True) - m.corners, max(peak),
            len(peak))


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    obj = obj0[0]
    nf = len(iobs)
    unit = np.random.default_rng(1).uniform(-1, 1, (nf, 2))
    for case, scale, prm, nepoch in (("step", OFFSET, STEP_PARAMS, 1),
                                      ("epochs", EPOCH_OFFSET, EPOCH_PARAMS, EPOCHS)):
        offsets = scale * unit
        o1, p1, d1 = model(m, obj, probe, offsets, prm, nepoch)
        o2, p2, d2, peak, nbatch = ptychi(m, iobs, wavelength, obj0, offsets, prm, nepoch)
        assert peak < 0.5, peak
        print(f"--- {case}: {nepoch} epoch(s), {nbatch} batches, {prm}, offsets within "
              f"{np.abs(offsets).max():.3f}, max |offset| over the run {peak:.4f}, max |offset "
              f"change| {np.abs(d2 - offsets).max():.4f}")
        print("  here vs pty-chi (max, L2): object change %.3e %.3e, probe change %.3e %.3e, "
              "offset change %.3e %.3e" % (*mg.rel(o1 - obj, o2 - obj),
                                           *mg.rel(p1 - probe, p2 - probe),
                                           *mg.rel(d1 - offsets, d2 - offsets)))
        rows = {
            "obj": lp.cplx_rows(obj.shape, o2 - obj),
            "probe": lp.cplx_rows(probe.shape, p2 - probe),
            "offsets": (np.column_stack([np.arange(nf), offsets, d2]), ["%d"] + ["%.17e"] * 4),
        }
        for name, (a, fmt) in rows.items():
            path = f"{d}/rpie_pos_{case}_{name}_f64.txt"
            np.savetxt(path, a, fmt=fmt)
            print(f"  wrote {path} ({len(a)} rows)")


if __name__ == "__main__":
    main()
