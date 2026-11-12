"""Float64 reference of pty-chi LSQML (Poisson) through a multislice object.

    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_multislice.py <dir>

The state is generated here from numpy default_rng(SEED) as in lsqml_near.py, with a truth
object of NSLICE slices [OBJ] of smooth amplitude and phase. The wave entering slice 0 is the
probe; slice s multiplies it by its window, and the wave reaches slice s + 1 by pty-chi's
AngularSpectrumPropagator over SPACINGS[s] at the object pixel (propagate_through_object).
The last slice's wave leaves through the case's free-space propagator. The counts are the
truth's Poisson counts through that model, natural layout. The start is every slice ones and
the truth probe plus a smooth complex perturbation.

The far-field case is the engine's Optics::far_field: the last slice's wave times PyNX's
object-plane phase q (ops::quad_phase) through pty-chi's FourierPropagator. pty-chi has no q,
and with several slices q cannot move into the probe as it does for one, so the script swaps
pty-chi's free-space propagator for FourierPropagator behind q (QuadFourier). Its geometry is
the ptyrs Geometry (wavelength, distance, detector pixel wavelength distance / (24 pixel)),
whose pixel_size_object is the pixel of q and of the gaps; pty-chi reads the data with
fft_shift True and its far field is in corner layout, the counts and y_pred are written
natural.

The in-object propagator has intensity scale 1, so pty-chi's waves are the engine's over
sqrt(k) at every slice, k the exit propagator's scale, as in lsqml_near.py. pty-chi runs with
multislice_solve_obj_prb_step_size_jointly True: the joint object/probe step sizes at slice 0,
the object-only Eq. 23b at the others, every object step size over NSLICE * NMODE.

Writes, every complex value `re im`, every plane row-major:
    <dir>/lsqml_multislice_cases_f64.txt    `case wavelength pixel distance kind scale
                                            truth_llk` and SPACINGS, kind 0 (Fourier) the
                                            far field with distance the detector's
    <dir>/lsqml_multislice_corners_f64.txt  `frame cy cx`
    <dir>/lsqml_multislice_obj_f64.txt      `slice iy ix re im`, the start object
    <dir>/lsqml_multislice_probe_f64.txt    `m iy ix re im`, the start probe
    <dir>/lsqml_multislice_counts_f64.txt   `case frame` and the frame's counts
    <dir>/lsqml_multislice_icalc_f64.txt    `case frame` and pty-chi's y_pred at the start,
                                            frames 0..ICALC_FRAMES
    <dir>/lsqml_multislice_step_{obj,probe,steps}_f64.txt  pty-chi's step on frames 0..BATCH
        with the illumination map as preconditioner: `case slice iy ix re im`,
        `case m iy ix re im` (the changes) and `case frame a_far a_o(slice 0..) a_p`
    <dir>/lsqml_multislice_epochs_{obj,probe,steps}_f64.txt  pty-chi over EPOCHS epochs of
        random_batches(16, BATCH, BATCH_SEED, epoch), its own preconditioner schedule: the
        changes and `case epoch frame a_far a_o(slice 0..) a_p`
    <dir>/lsqml_multislice_llk_f64.txt      `case epoch llk`, the normalised Poisson LLK after
                                            each epoch, epoch -1 the start
torch.randn returns zeros in every pty-chi run; the object-probe ambiguity removal and the mode
orthogonalisation are off, as in lsqml_near.py.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.special import xlogy

SEED = 13
NSLICE = 3
SPACINGS = (4e-5, 6e-5)
BATCH = 8
BATCH_SEED = 7
EPOCHS = 30
ICALC_FRAMES = 4
# (wavelength, pixel, distance) of the exit propagation: angular spectrum, Fresnel transform,
# far field (FarGeometry).
GEOMETRIES = [(1e-10, 50e-9, 5e-4), (1e-10, 50e-9, 2e-3), (1e-10, 50e-9, 5e-3)]
FAR = 2


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
rb = near.rb
pp = near.pp
SHAPE, OBJ, NMODE, GRID, AX = near.SHAPE, near.OBJ, near.NMODE, near.GRID, near.AX


def make_state():
    rng = np.random.default_rng(SEED)
    obj = np.stack([(0.95 + 0.03 * near.smooth(rng, OBJ, 2.0))
                    * np.exp(0.1j * near.smooth(rng, OBJ, 2.0)) for _ in range(NSLICE)])
    env = 12.0 * (1 + 0.3 * near.smooth(rng, SHAPE, 1.5))
    probe = np.stack([env * np.exp(1j * np.pi * near.smooth(rng, SHAPE, 1.0)),
                      0.3 * env * np.exp(1j * np.pi * near.smooth(rng, SHAPE, 1.0))])
    shift = 0.15 * np.abs(probe).max() * np.stack(
        [near.smooth(rng, SHAPE, 2.0) + 1j * near.smooth(rng, SHAPE, 2.0)
         for _ in range(NMODE)])
    span = np.array(OBJ) - np.array(SHAPE)
    base = np.linspace(1, span[0] - 1, GRID).round().astype(int)
    corners = np.array([[y, x] for y in base for x in base])
    corners = np.clip(corners + rng.integers(-1, 2, corners.shape), 0, span)
    return obj, probe, probe + shift, corners, rng


class Slices:
    """pty-chi's in-object propagator over each gap at `self.pixel`."""

    far = False

    def build_gaps(self):
        self.gaps = [pp.AngularSpectrumPropagator(
            pp.params(SHAPE, self.wavelength, self.pixel, self.pixel, d)) for d in SPACINGS]

    def natural(self, y):
        """pty-chi's far field (y_pred) in natural layout."""
        return np.fft.fftshift(y, axes=AX) if self.far else y

    def through(self, slices, probe, corners):
        """The exit waves [frame, mode] of the probe through the slices' windows."""
        psi = np.broadcast_to(probe, (len(corners), *probe.shape))
        for s in range(NSLICE):
            psi = psi * near.views(slices[s], corners)[:, None]
            if s < NSLICE - 1:
                psi = self.gaps[s].propagate_forward(self.torch.from_numpy(psi)).numpy()
        return psi

    def install(self, rec):
        """Fit pty-chi's reconstructor to this exit propagator."""


class Geometry(Slices, near.Geometry):
    """lsqml_near.Geometry through the slices."""

    def __init__(self, wavelength, pixel, distance):
        near.Geometry.__init__(self, wavelength, pixel, distance)
        self.ptychi_distance = distance
        self.build_gaps()


def quad_fourier():
    """pty-chi's FourierPropagator behind q, defined once ptychi is importable."""
    from ptychi.propagate import FourierPropagator

    class QuadFourier(FourierPropagator):
        """FourierPropagator of q x; propagate_backward stays its inverse."""

        def __init__(self, q):
            super().__init__()
            self.register_buffer("_q", q)

        def propagate_forward(self, wavefield):
            return super().propagate_forward(wavefield * self._q)

        def propagate_backward(self, wavefield):
            return super().propagate_backward(wavefield) * self._q.conj()

    return QuadFourier


class FarGeometry(Slices):
    """ptyrs Optics::far_field: q (ops::quad_phase) on the exit wave, then pty-chi's unnormalised
    FFT, k = ny nx. `pixel` sets the detector pixel; the gaps and q use pixel_size_object."""

    far = True
    kind = 0

    def __init__(self, wavelength, pixel, distance):
        import torch
        self.torch = torch
        ny, nx = SHAPE
        lz = wavelength * distance
        detector = lz / (ny * pixel)
        py, px = lz / (ny * detector), lz / (nx * detector)
        assert py == px
        self.wavelength, self.pixel, self.distance = wavelength, px, distance
        self.ptychi_distance = np.inf
        f = np.pi / lz
        dy = (np.arange(ny) - ny // 2)[:, None].astype(np.float64)
        dx = (np.arange(nx) - nx // 2)[None, :].astype(np.float64)
        t = f * (dx * dx * px * px + dy * dy * py * py)
        self.q = np.cos(t) + 1j * np.sin(t)
        self.prop = quad_fourier()(torch.from_numpy(self.q))
        self.k = float(pp.get_intensity_scaling_factor(self.prop, SHAPE))
        assert self.k == ny * nx
        self.build_gaps()

    def fwd(self, x):
        """U x, natural layout."""
        y = self.prop.propagate_forward(self.torch.from_numpy(x)).numpy()
        return np.fft.fftshift(y, axes=AX) / np.sqrt(self.k)

    def install(self, rec):
        # The script's patch to pty-chi: the far field behind the engine's q.
        fm = rec.forward_model
        fm.build_free_space_propagator = lambda: setattr(fm, "free_space_propagator", self.prop)
        fm.build_free_space_propagator()


def icalc(g, slices, probe, corners):
    return (np.abs(g.fwd(g.through(slices, probe, corners))) ** 2).sum(1)


def llk(g, slices, probe, corners, counts):
    """The engine's normalised Poisson LLK (ops::llk) over every frame."""
    yp = icalc(g, slices, probe, corners)
    t = np.where(counts > 0, yp - counts + xlogy(counts, counts) - xlogy(counts, yp), yp)
    return t.sum() / counts.size


def task(g, obj, probe, corners, counts, batch, epochs):
    """lsqml_near.task with a multislice object."""
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = SHAPE
    opts = api.LSQMLOptions()
    opts.data_options.wavelength_m = g.wavelength
    opts.data_options.fft_shift = g.far
    opts.data_options.free_space_propagation_distance_m = g.ptychi_distance
    o = opts.object_options
    o.optimizable = True
    o.pixel_size_m = g.pixel
    o.slice_spacings_m = list(SPACINGS)
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
    r.multislice_solve_obj_prb_step_size_jointly = True
    r.rescale_probe_intensity_in_first_epoch = False
    r.batch_size = batch
    r.num_epochs = epochs
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    t = PtychographyTask(
        opts,
        diffraction_data=counts.copy(),
        object_data=obj.copy(),
        probe_data=(probe / np.sqrt(g.k))[None].copy(),
        probe_position_y_px=corners[:, 0].astype(np.float64),
        probe_position_x_px=corners[:, 1].astype(np.float64),
    )
    g.install(t.reconstructor)
    return t


def state_of(t, g):
    """pty-chi's object slices and probe in the engine's convention."""
    return (t.get_data_to_cpu("object", as_numpy=True),
            t.get_data_to_cpu("probe", as_numpy=True)[0] * np.sqrt(g.k))


def ptychi_step(g, obj, probe, corners, counts):
    """pty-chi run_minibatch on frames 0..BATCH with the illumination map as preconditioner,
    and its forward model's y_pred at the start."""
    import torch
    from torch.utils.data import DataLoader

    t = task(g, obj, probe, corners, counts, BATCH, 1)
    rec = t.reconstructor
    pg = rec.parameter_group
    with torch.no_grad(), near.ZeroRandn() as z:
        rec.run_pre_run_hooks()
        rec.run_pre_epoch_hooks()
        pg.object.preconditioner = pg.object.calculate_illumination_map(
            pg.probe, pg.probe_positions, use_all_modes=True)
        batch = next(iter(DataLoader(rec.dataset, batch_size=BATCH, shuffle=False)))
        input_data, y_true = rec.prepare_batch_data(batch)
        y_pred = g.natural(rec.forward_model(*input_data).numpy().copy())
        rec.run_minibatch(input_data, y_true)
    idx = input_data[0].numpy()
    buf = rec.reconstructor_buffers
    o, p = state_of(t, g)
    return dict(
        idx=idx, randn_calls=z.calls, y_pred=y_pred,
        a_far=rec.alpha_psi_far.cpu().numpy().reshape(-1),
        a_o=buf.alpha_object_all_pos_all_slices[idx].cpu().numpy(),
        a_p=buf.alpha_probe_all_pos[idx].cpu().numpy(),
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
            a_o=buf.alpha_object_all_pos_all_slices[idx].cpu().numpy().copy(),
            a_p=buf.alpha_probe_all_pos[idx].cpu().numpy().copy()))
        return out

    rec.run_minibatch = minibatch
    trace = [llk(g, obj, probe, corners, counts)]
    with near.ZeroRandn() as z:
        for _ in range(EPOCHS):
            t.run(1)
            trace.append(llk(g, *state_of(t, g), corners, counts))
    o, p = state_of(t, g)
    return dict(steps=steps, randn_calls=z.calls, llk=np.array(trace),
                d_obj=o - obj, d_probe=p - probe)


def rows_slices(a):
    s, y, x = np.indices(a.shape)
    rows = np.column_stack([s.ravel(), y.ravel(), x.ravel(), a.real.ravel(), a.imag.ravel()])
    return rows, ["%d"] * 3 + ["%.17e"] * 2


def step_rows(lead_cols, st):
    return np.column_stack([*lead_cols, st["idx"], st["a_far"], st["a_o"], st["a_p"]])


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    truth_obj, truth_probe, probe, corners, rng = make_state()
    obj = np.ones((NSLICE, *OBJ), complex)
    nframe = len(corners)
    print(f"object {obj.shape}, spacings {SPACINGS}, probe {probe.shape}, {nframe} frames, "
          f"corners {corners.tolist()}")
    out = {k: [] for k in ("cases", "counts", "icalc", "step_obj", "step_probe", "step_steps",
                           "epochs_obj", "epochs_probe", "epochs_steps", "llk")}
    idx = np.arange(BATCH)
    f = "%.17e"
    steps_fmt = ["%d"] + [f] * (NSLICE + 2)
    for case, geom in enumerate(GEOMETRIES):
        g = (FarGeometry if case == FAR else Geometry)(*geom)
        counts = rng.poisson(icalc(g, truth_obj, truth_probe, corners)).astype(np.float64)
        assert counts.max() < 65535, counts.max()
        truth_llk = llk(g, truth_obj, truth_probe, corners, counts)
        print(f"--- case {case}: {geom}, kind {g.kind}, k {g.k:.6e}, counts max "
              f"{counts.max():.0f} mean {counts.mean():.1f}, LLK truth {truth_llk:.6f} start "
              f"{llk(g, obj, probe, corners, counts):.6f}")
        out["cases"].append([case, *geom, g.kind, g.k, truth_llk, *SPACINGS])
        out["counts"].append(np.column_stack([np.full(nframe, case), np.arange(nframe),
                                              counts.reshape(nframe, -1)]))

        pc = ptychi_step(g, obj, probe, corners, counts)
        assert (pc["idx"] == idx).all(), pc["idx"]
        yp = icalc(g, obj, probe, corners[idx])
        print(f"  pty-chi step ({pc['randn_calls']} torch.randn call(s) zeroed), y_pred vs "
              "numpy forward (max, L2): %.3e %.3e" % ls.mg.rel(pc["y_pred"], yp))
        print("  a_o by slice (frame 0):", pc["a_o"][0], "a_p", pc["a_p"][0])
        out["icalc"].append(np.column_stack([np.full(ICALC_FRAMES, case),
                                             np.arange(ICALC_FRAMES),
                                             pc["y_pred"][:ICALC_FRAMES].reshape(
                                                 ICALC_FRAMES, -1)]))
        for name, rows_fmt in (
                ("obj", rows_slices(pc["d_obj"])), ("probe", ls.rows_probe(pc["d_probe"])),
                ("steps", (step_rows([], pc), steps_fmt))):
            out[f"step_{name}"].append(near.lead(case, rows_fmt))

        pe = ptychi_epochs(g, obj, probe, corners, counts)
        for e in range(EPOCHS):
            got = [st["idx"].tolist() for st in pe["steps"] if st["epoch"] == e]
            want = [b.tolist() for b in rb.random_batches(nframe, BATCH, BATCH_SEED, e)]
            assert got == want, (e, got, want)
        reached = np.flatnonzero(pe["llk"] <= truth_llk)
        print(f"  pty-chi {EPOCHS} epochs ({pe['randn_calls']} torch.randn call(s) zeroed), "
              f"truth reached after {reached[0] if len(reached) else None} epochs, LLK by epoch:")
        print("    " + " ".join(f"{v:.4f}" for v in pe["llk"]))
        rows = np.concatenate([step_rows([np.full(len(st["idx"]), st["epoch"])], st)
                               for st in pe["steps"]])
        for name, rows_fmt in (
                ("obj", rows_slices(pe["d_obj"])), ("probe", ls.rows_probe(pe["d_probe"])),
                ("steps", (rows, ["%d"] + steps_fmt))):
            out[f"epochs_{name}"].append(near.lead(case, rows_fmt))
        out["llk"].append(np.column_stack([np.full(EPOCHS + 1, case),
                                           np.arange(-1, EPOCHS), pe["llk"]]))

    def save(name, rows, fmt):
        path = f"{d}/lsqml_multislice_{name}_f64.txt"
        np.savetxt(path, rows, fmt=fmt)
        print(f"wrote {path} ({len(rows)} rows)")

    save("cases", np.array(out["cases"]), ["%d", f, f, f, "%d", f, f] + [f] * len(SPACINGS))
    save("corners", np.column_stack([np.arange(nframe), corners]), "%d")
    save("obj", *rows_slices(obj))
    save("probe", *ls.rows_probe(probe))
    save("counts", np.concatenate(out["counts"]), "%d")
    save("icalc", np.concatenate(out["icalc"]), ["%d", "%d"] + [f] * (SHAPE[0] * SHAPE[1]))
    for name in ("step_obj", "step_probe", "step_steps", "epochs_obj", "epochs_probe",
                 "epochs_steps"):
        rows = [r for r, _ in out[name]]
        save(name, np.concatenate(rows), out[name][0][1])
    save("llk", np.concatenate(out["llk"]), ["%d", "%d", f])


if __name__ == "__main__":
    main()
