"""Float64 reference of pty-chi LSQML over EPOCHS whole epochs at a fixed start, for the
per-frame state an epoch carries: the Poisson a_far and the preconditioner schedule.

    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_epochs.py <dir>

<dir> is the state lsqml_step.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames).
pty-chi's LSQMLReconstructor runs EPOCHS epochs of the engine's random_batches(40, BATCH, SEED,
epoch) with the options of lsqml_step.ptychi_step and its own preconditioner schedule
(Object.update_preconditioner before every epoch). torch.randn returns zeros, so the Poisson
a_far of a frame is the clamped xi solve of its last step: 1/2 on its first visit, epoch 0
here, and its stored value on every later one. No post-epoch hook moves anything: the
defaults that would (object-probe ambiguity removal, mode orthogonalisation) are off.

Writes, per noise model N in gaussian, poisson, in the engine's convention:
    <dir>/lsqml_epochs_N_steps_f64.txt  `epoch frame a_far a_o a_p`, every step's frames
    <dir>/lsqml_epochs_N_obj_f64.txt    `iy ix re im`, the object change over all epochs
    <dir>/lsqml_epochs_N_probe_f64.txt  `m iy ix re im`, the probe change
"""
import importlib.util
import pathlib
import sys

import numpy as np

BATCH = 16
EPOCHS = 3
SEED = 7
NOISES = ("gaussian", "poisson")


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
rb = sibling("random_batches")


def ptychi_epochs(m, iobs, wavelength, obj, probe, noise):
    """pty-chi LSQML f64 on CPU for EPOCHS epochs; per step its frames, a_far, a_o and a_p,
    and the object and probe changes over the run."""
    import torch
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
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
    r = opts.reconstructor_options
    r.noise_model = {"poisson": api.NoiseModels.POISSON,
                     "gaussian": api.NoiseModels.GAUSSIAN}[noise]
    r.forward_model_options.subpixel_shift_all_probe_modes = True
    r.solve_step_sizes_only_using_first_probe_mode = False
    r.rescale_probe_intensity_in_first_epoch = False
    r.batch_size = BATCH
    r.num_epochs = EPOCHS
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj.copy(),
        probe_data=(probe * m.q / root_n)[None].copy(),
        probe_position_y_px=m.corners[:, 0].astype(np.float64),
        probe_position_x_px=m.corners[:, 1].astype(np.float64),
    )
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset,
                                batch_sampler=rb.Sampler(len(m.corners), BATCH, SEED))
    cls = type(rec)
    buf = rec.reconstructor_buffers
    steps = []

    def minibatch(input_data, y_true):
        out = cls.run_minibatch(rec, input_data, y_true)
        idx = input_data[0].numpy().copy()
        a_far = rec.alpha_psi_far.cpu().numpy().reshape(-1)
        steps.append(dict(
            epoch=rec.current_epoch, idx=idx, a_far=np.broadcast_to(a_far, idx.shape).copy(),
            a_o=buf.alpha_object_all_pos_all_slices[idx, 0].cpu().numpy().copy(),
            a_p=buf.alpha_probe_all_pos[idx].cpu().numpy().copy()))
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
    return dict(
        steps=steps, randn_calls=len(calls),
        d_obj=task.get_data_to_cpu("object", as_numpy=True)[0] - obj[0],
        d_probe=task.get_data_to_cpu("probe", as_numpy=True)[0] * np.conj(m.q) * root_n
        - probe,
    )


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    nframe = len(m.corners)
    print(f"object {obj0.shape}, probe {probe.shape}, {nframe} frames, "
          f"{EPOCHS} epochs of batches of {BATCH}, seed {SEED}")
    for noise in NOISES:
        pc = ptychi_epochs(m, iobs, wavelength, obj0, probe, noise)
        steps = pc["steps"]
        for e in range(EPOCHS):
            got = [s["idx"].tolist() for s in steps if s["epoch"] == e]
            want = [b.tolist() for b in rb.random_batches(nframe, BATCH, SEED, e)]
            assert got == want, (e, got, want)
        print(f"--- {noise}: {len(steps)} steps, {pc['randn_calls']} torch.randn call(s) zeroed")
        for s in steps:
            print(f"  epoch {s['epoch']} frames {len(s['idx']):2d}: a_far "
                  f"{s['a_far'].min():.6f}..{s['a_far'].max():.6f}, a_o trim_mean "
                  f"{ls.trim_mean(s['a_o']):.6e}, a_p mean {s['a_p'].mean():.6e}")
        rows = np.concatenate([np.column_stack([np.full(len(s["idx"]), s["epoch"]), s["idx"],
                                                s["a_far"], s["a_o"], s["a_p"]])
                               for s in steps])
        for name, (r, fmt) in (
                ("obj", ls.rows_obj(pc["d_obj"])), ("probe", ls.rows_probe(pc["d_probe"])),
                ("steps", (rows, ["%d", "%d"] + ["%.17e"] * 3))):
            path = f"{d}/lsqml_epochs_{noise}_{name}_f64.txt"
            np.savetxt(path, r, fmt=fmt)
            print(f"  wrote {path} ({len(r)} rows)")


if __name__ == "__main__":
    main()
