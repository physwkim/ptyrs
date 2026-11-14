"""pty-chi DM and RAAR under its FULL_OBJECT shift operator against the engine's CPU DM and RAAR
under Sampling::FullObject, on the near-field cases of ptyrs/tests/common/near.rs, in f64.

    micromamba run -n ptyrs-ref python -I refs/py/ptychi_full_object_dm.py <state>

<state> holds one directory per case (`as`, `fresnel`, `fresnel_back`: near.rs's angular
spectrum at 1e-4 m, Fresnel transform at 5e-3 m and back at -1e-2 m), each with the case and the
engine's runs from it, little-endian f64, complex as `re im` pairs:
    meta.txt          `key value`: wavelength, pixel, distance, nframe, nmode, ny, nx, oy, ox,
                      nb_obs
    counts.f64        [nframe, ny, nx], natural (centred) layout
    corners.f64       [nframe, 2], each frame's window corner (oy, ox)
    obj0.c128         [oy, ox], the start object; truth.c128 the object the counts are of
    probe0.c128       [nmode, ny, nx], the start probe
    offsets0.f64      [nframe, 2], near.rs's drawn_offsets
    <alg>_<label>_<n>_llk.f64     the engine's LLK history over n cycles (each cycle's start),
                                  then the normalised Poisson LLK of the arrays it ends on
    <alg>_<label>_<n>_{obj,probe}.c128, _offsets.f64   what the run ends on
for alg `dm` (near.rs's DM: inertia 0.01 and 0.001, alpha 0.02) and `raar` (beta 0.9), label
`fixed` (no features, every offset 0) and `position` (the drawn offsets, refined), and n the
cycles of each run, the largest of them taken as the long run. The engine runs without its
closing zero_phase_ramp, which under `position` changes the LLK of the state it ends on
(Ptycho::zero_phase_ramp), so the 8-cycle run must end on the long run's first 9 records.

pty-chi runs its own DM (Thibault 2009 Eq. 9, exit-wave relaxation 1) and RAAR (Marchesini 2016
Eq. 16), both in their FULL_OBJECT form: frame k samples the whole object's periodic Fourier
interpolant at its corner plus d_k, the probe stays unmoved, the object update is placed by that
sampling's adjoint with its weights floored at 5% of their maximum (base.py
stabilize_object_projection_denominator), and the positions refine by its gradient step (SGD at
0.3, clipped at min(0.1, 10 MADs), no mean removed). It takes the engine's inertias and beta in
its own forms (a mix with the old array, Marchesini's beta), from the same counts, start object,
probe (over sqrt(k), lsqml_near.py's convention) and corners plus offsets, mode orthogonalisation
and the object-probe ambiguity removal off. Its normalised Poisson LLK is the engine's (ops::llk)
of its forward model's intensities, after each epoch.

At the start both sides must agree on the LLK of the fixed and the drawn-offsets estimate (same
data, sampling and convention) to START_BOUND, or the script exits non-zero. Then, per case,
algorithm and label, it prints the LLK after MARKS cycles (7 is the last the engine records in
an 8-cycle run, engine_gpu_near.rs's) and at the end of the long run, its minimum and where, and
pty-chi's alone from the drawn offsets held (not refined); after 8 cycles and the long run, the
object of each side against the truth and against each other over the illuminated pixels (start
probe's illumination at the corners at least ILLUM of its maximum): |c a e^(i g.r) - b| / |b|
with c and the phase ramp g fitted (a ramp on O with the opposite on P changes no frame's
intensity), and the offsets' rms about their mean and the mean. Writes
<state>/ptychi_full_object_dm.json with the curves of every run.
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
from scipy.special import xlogy

OBJ_INERTIA = 0.01
PROBE_INERTIA = 0.001
BETA = 0.9
CASES = ("as", "fresnel", "fresnel_back")
SHORT = 8
MARKS = (0, 1, 2, 4, 7, 8)
ILLUM = 0.1
# Start LLK agreement, engine against pty-chi's forward model.
START_BOUND = 1e-10


def f64_script(name):
    """Import a refs/f64 script (python -I keeps every script directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).parents[1] / "f64" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = f64_script("propagators")


def read(path, shape, cplx=False):
    a = np.fromfile(path, dtype="<f8")
    return (a[0::2] + 1j * a[1::2] if cplx else a).reshape(shape)


class Case:
    def __init__(self, d):
        meta = dict(line.split() for line in (d / "meta.txt").read_text().splitlines())
        self.dir, self.name = d, d.name
        self.wavelength, self.pixel, self.distance = (float(meta[k]) for k in
                                                      ("wavelength", "pixel", "distance"))
        nf, nm, ny, nx, oy, ox = (int(meta[k]) for k in ("nframe", "nmode", "ny", "nx", "oy",
                                                          "ox"))
        self.nb_obs = int(meta["nb_obs"])
        self.counts = read(d / "counts.f64", (nf, ny, nx))
        self.corners = read(d / "corners.f64", (nf, 2))
        self.obj0 = read(d / "obj0.c128", (oy, ox), True)
        self.truth = read(d / "truth.c128", (oy, ox), True)
        self.probe0 = read(d / "probe0.c128", (nm, ny, nx), True)
        self.offsets0 = read(d / "offsets0.f64", (nf, 2))
        prop = pp.free_space((ny, nx), self.wavelength, self.pixel, self.pixel, self.distance)
        self.kind = type(prop).__name__
        self.k = float(pp.get_intensity_scaling_factor(prop, (ny, nx)))
        illum = np.zeros((oy, ox))
        p2 = (np.abs(self.probe0) ** 2).sum(0)
        for cy, cx in self.corners.astype(int):
            illum[cy:cy + ny, cx:cx + nx] += p2
        self.region = illum >= ILLUM * illum.max()
        runs = sorted({int(p.name.split("_")[2]) for p in d.glob("dm_fixed_*_llk.f64")})
        self.long = runs[-1]
        assert SHORT in runs, runs

    def engine(self, alg, label, n):
        stem = self.dir / f"{alg}_{label}_{n}"
        nf, (nm, ny, nx) = len(self.corners), self.probe0.shape
        return dict(llk=read(f"{stem}_llk.f64", (n + 1,)),
                    obj=read(f"{stem}_obj.c128", self.obj0.shape, True),
                    probe=read(f"{stem}_probe.c128", (nm, ny, nx), True),
                    offsets=read(f"{stem}_offsets.f64", (nf, 2)))


def task(c, alg, offsets, moved):
    """pty-chi's DM or RAAR task on case c from the corners plus offsets."""
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = c.counts.shape[1:]
    opts = api.DMOptions() if alg == "dm" else api.RAAROptions()
    opts.data_options.wavelength_m = c.wavelength
    opts.data_options.fft_shift = False
    opts.data_options.free_space_propagation_distance_m = c.distance
    o = opts.object_options
    o.optimizable = True
    o.pixel_size_m = c.pixel
    o.inertia = OBJ_INERTIA
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    pr = opts.probe_options
    pr.optimizable = True
    pr.inertia = PROBE_INERTIA
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    opts.probe_position_options.optimizable = moved
    r = opts.reconstructor_options
    r.forward_model_options.shift_operator = api.ShiftOperators.FULL_OBJECT
    if alg == "raar":
        r.beta = BETA
    r.num_epochs = c.long
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    pos = c.corners + offsets
    t = PtychographyTask(
        opts,
        diffraction_data=c.counts.copy(),
        object_data=c.obj0[None].copy(),
        probe_data=(c.probe0 / np.sqrt(c.k))[None].copy(),
        probe_position_y_px=pos[:, 0].copy(),
        probe_position_x_px=pos[:, 1].copy(),
    )
    fm = t.reconstructor.forward_model
    assert fm.resolved_shift_operator.name == "FULL_OBJECT", fm.resolved_shift_operator
    assert type(fm.free_space_propagator).__name__ == c.kind
    return t


def llk(t, c):
    """The engine's normalised Poisson LLK of pty-chi's forward-model intensities."""
    import torch
    with torch.no_grad():
        y = t.reconstructor.forward_model.forward(torch.arange(len(c.counts))).numpy()
    n = c.counts
    return np.where(n > 0, y - n + xlogy(n, n) - xlogy(n, y), y).sum() / c.nb_obs


def ended(t, c):
    return dict(obj=t.get_data_to_cpu("object", as_numpy=True)[0],
                probe=t.get_data_to_cpu("probe", as_numpy=True)[0] * np.sqrt(c.k),
                offsets=t.get_data_to_cpu("probe_positions", as_numpy=True) - c.corners)


def ptychi(c, alg, offsets, moved):
    """pty-chi over the long run: the LLK curve and the arrays after SHORT and after all."""
    t = task(c, alg, offsets, moved)
    curve, short = [llk(t, c)], None
    for e in range(c.long):
        t.run(1)
        curve.append(llk(t, c))
        if e + 1 == SHORT:
            short = ended(t, c)
    return np.array(curve), short, ended(t, c)


def obj_err(a, b, region):
    """|c a e^(i g.r) - b| / |b| over region, c and the ramp g fitted to b."""
    a, b = a.copy(), b.copy()
    r = np.where(region, a * np.conj(b), 0)
    gy = np.angle((r[1:] * np.conj(r[:-1])).sum())
    gx = np.angle((r[:, 1:] * np.conj(r[:, :-1])).sum())
    yy, xx = np.indices(a.shape)
    a = a * np.exp(-1j * (gy * yy + gx * xx))
    a, b = a[region], b[region]
    cfit = (np.conj(a) * b).sum() / (np.abs(a) ** 2).sum()
    return np.linalg.norm(cfit * a - b) / np.linalg.norm(b)


def off(d):
    m = d.mean(0)
    return np.sqrt(((d - m) ** 2).sum(1).mean()), np.abs(m).max()


def compare(c, tag, eng, chi):
    e_t, p_t = obj_err(eng["obj"], c.truth, c.region), obj_err(chi["obj"], c.truth, c.region)
    e_p = obj_err(chi["obj"], eng["obj"], c.region)
    (er, em), (pr, pm) = off(eng["offsets"]), off(chi["offsets"])
    print(f"    {tag}: object vs truth engine {e_t:.4f} pty-chi {p_t:.4f}, pty-chi vs engine "
          f"{e_p:.4f}; offsets rms engine {er:.4f} pty-chi {pr:.4f}, |mean| {em:.2e} {pm:.2e}")
    return dict(obj_truth_engine=e_t, obj_truth_ptychi=p_t, obj_ptychi_engine=e_p,
                offsets_rms_engine=er, offsets_rms_ptychi=pr)


def marks(curve):
    i = int(np.nanargmin(curve))
    return (" ".join(f"{curve[m]:9.4g}" for m in MARKS) + f" | end {curve[-1]:9.4g}, min "
            f"{curve[i]:.4g} at {i}")


def main():
    state = pathlib.Path(sys.argv[1])
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    cases = [Case(state / name) for name in CASES]
    worst = 0.0
    for c in cases:
        for label, d in (("fixed", 0 * c.offsets0), ("position", c.offsets0)):
            want = c.engine("dm", label, SHORT)["llk"][0]
            got = llk(task(c, "dm", d, False), c)
            err = abs(got - want) / want
            print(f"{c.name} {label}: start LLK engine {want:.12e} pty-chi {got:.12e}, {err:.1e}")
            worst = max(worst, err)
    if not worst <= START_BOUND:
        sys.exit(f"start LLK differs: {worst:.2e} > {START_BOUND:.0e}")

    out = {}
    for c in cases:
        print(f"--- {c.name}: {c.kind}, distance {c.distance:g} m, long run {c.long} cycles; "
              f"LLK after cycles {MARKS}")
        for alg in ("dm", "raar"):
            for label in ("fixed", "position"):
                eng_short, eng_long = c.engine(alg, label, SHORT), c.engine(alg, label, c.long)
                assert np.array_equal(eng_short["llk"], eng_long["llk"][:SHORT + 1]), label
                moved = label == "position"
                curve, chi_short, chi_long = ptychi(c, alg, c.offsets0 * moved, moved)
                print(f"  {alg} {label}\n    engine  {marks(eng_long['llk'])}\n"
                      f"    pty-chi {marks(curve)}")
                out[f"{c.name}/{alg}/{label}"] = dict(
                    engine=eng_long["llk"].tolist(), ptychi=curve.tolist(),
                    short=compare(c, f"after {SHORT}", eng_short, chi_short),
                    long=compare(c, f"after {c.long}", eng_long, chi_long))
            curve = ptychi(c, alg, c.offsets0, False)[0]
            print(f"  {alg} held\n    pty-chi {marks(curve)}")
            out[f"{c.name}/{alg}/held"] = dict(ptychi=curve.tolist())
    path = state / "ptychi_full_object_dm.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
