"""Float64 reference of pty-chi BH in its planar near-field model.

    micromamba run -n ptyrs-ref python -I refs/f64/bh_near.py <dir>

The state and counts are lsqml_near.py's (start object of ones, start probe, 16 whole-pixel
corners, per geometry the Poisson counts of the truth), regenerated from its seed, and its
GEOMETRIES: an angular-spectrum propagator, a Fresnel transform and an angular spectrum with an
evanescent band. Per geometry pty-chi's BHReconstructor runs EPOCHS epochs of the engine's
random_batches(16, size, SEED, epoch) with probe rho RHO and positions fixed, in two methods:
"full", CG with one batch of every frame per epoch, and "mini", GD with batches of 6, 6 and 4.

In the engine's convention (lsqml_near.py) pty-chi's probe is P / sqrt(k), k its
get_intensity_scaling_factor, and its psi_far is U psi with U unitary, so d, A and the
residual are the engine's; its probe gradient and direction are sqrt(k) times the engine's
probe-variable ones at rho_e = rho sqrt(k), and the object, alpha, beta, top = -<G, eta> and
bottom = H(eta, eta) are the same, once gradientF carries k on the residual. pty-chi applies k
there only for FourierPropagator, so the Fresnel case runs with that factor added.

Writes, per case C (geometry) and method M (0 full, 1 mini), every complex value `re im`:
    <dir>/bh_near_steps_f64.txt  `C M step epoch count beta alpha top bottom f_0 ..`, one row
                                 per batch, alpha after the batch-size factor
    <dir>/bh_near_obj_f64.txt    `C M iy ix re im`, the object change over the run
    <dir>/bh_near_probe_f64.txt  `C M m iy ix re im`, the probe change over the run, x sqrt(k)
"""
import importlib.util
import pathlib
import sys

import numpy as np

RHO = 1.0
EPOCHS = 2
SEED = 7
METHODS = (("full", 16, True), ("mini", 6, False))  # name, batch size, CG


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ln = sibling("lsqml_near")
rb = ln.rb


def cost(g, obj, probe, corners, counts):
    """F = sum (A - sqrt(iobs))^2 over every frame."""
    return ((np.sqrt(ln.icalc(g, obj, probe, corners)) - np.sqrt(counts)) ** 2).sum()


def task(g, obj, probe, corners, counts, size, cg):
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = ln.SHAPE
    opts = api.BHOptions()
    opts.data_options.wavelength_m = g.wavelength
    opts.data_options.fft_shift = False
    opts.data_options.free_space_propagation_distance_m = g.distance
    o = opts.object_options
    o.optimizable = True
    o.pixel_size_m = g.pixel
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    pr = opts.probe_options
    pr.optimizable = True
    pr.rho = RHO
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    opts.probe_position_options.optimizable = False
    r = opts.reconstructor_options
    r.method = "CG" if cg else "GD"
    r.batch_size = size
    r.num_epochs = EPOCHS
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    return PtychographyTask(
        opts,
        diffraction_data=counts.copy(),
        object_data=obj[None].copy(),
        probe_data=(probe / np.sqrt(g.k))[None].copy(),
        probe_position_y_px=corners[:, 0].astype(np.float64),
        probe_position_x_px=corners[:, 1].astype(np.float64),
    )


def ptychi_run(g, obj, probe, corners, counts, size, cg):
    """pty-chi BH over EPOCHS epochs of the engine's batches: per batch its frames, beta,
    alpha, top and bottom, and the object and probe changes over the run."""
    import torch
    from ptychi.propagate import FresnelTransformPropagator
    from torch.utils.data import DataLoader

    t = task(g, obj, probe, corners, counts, size, cg)
    rec = t.reconstructor
    rec.dataloader = DataLoader(rec.dataset,
                                batch_sampler=rb.Sampler(len(corners), size, SEED))
    cls = type(rec)
    steps = []
    redot = lambda a, b: float(torch.sum((torch.conj(a) * b).real))

    def minibatch(input_data, y_true):
        steps.append({"beta": 0.0, "epoch": rec.current_epoch,
                      "idx": input_data[0].numpy().copy()})
        return cls.run_minibatch(rec, input_data, y_true)

    def beta(*args, **kwargs):
        b = cls.calc_beta_object_probe(rec, *args, **kwargs)
        steps[-1]["beta"] = float(b)
        return b

    def alpha(p, do1, dp1, do2, dp2, *args, **kwargs):
        a = float(cls.calc_alpha_object_probe(rec, p, do1, dp1, do2, dp2, *args, **kwargs))
        top = -redot(do1, do2) - redot(dp1, dp2)
        steps[-1].update(alpha=a * len(steps[-1]["idx"]) / size, top=top, bottom=top / a)
        return a

    rec.run_minibatch = minibatch
    rec.calc_beta_object_probe = beta
    rec.calc_alpha_object_probe = alpha
    if isinstance(g.prop, FresnelTransformPropagator):
        # pty-chi scales gradientF's residual by k only for FourierPropagator; Fresnel needs it too.
        rec.gradientF = lambda psi_far, d: g.k * cls.gradientF(rec, psi_far, d)
    t.run(EPOCHS)
    o, p = ln.state_of(t, g)
    return steps, o - obj, p - probe


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    truth_obj, truth_probe, probe, corners, rng = ln.make_state()
    obj = np.ones(ln.OBJ, complex)
    nframe = len(corners)
    out = {"steps": [], "obj": [], "probe": []}
    for case, geom in enumerate(ln.GEOMETRIES):
        g = ln.Geometry(*geom)
        counts = ln.counts_of(g, truth_obj, truth_probe, corners, rng)
        f0 = cost(g, obj, probe, corners, counts)
        print(f"--- case {case}: {geom}, kind {type(g.prop).__name__}, k {g.k:.6e}, rho_e "
              f"{RHO * np.sqrt(g.k):.6e}, F start {f0:.10e}")
        for method, (name, size, cg) in enumerate(METHODS):
            steps, d_obj, d_probe = ptychi_run(g, obj, probe, corners, counts, size, cg)
            for e in range(EPOCHS):
                got = [s["idx"].tolist() for s in steps if s["epoch"] == e]
                want = [b.tolist() for b in rb.random_batches(nframe, size, SEED, e)]
                assert got == want, (e, got, want)
            f1 = cost(g, obj + d_obj, probe + d_probe, corners, counts)
            print(f"  {name} ({'CG' if cg else 'GD'}, batch {size}): F {f0:.10e} -> {f1:.10e}")
            print("    step epoch count  beta          alpha         top           bottom")
            for k, s in enumerate(steps):
                print(f"    {k:4d} {s['epoch']:5d} {len(s['idx']):5d}  {s['beta']: .6e} "
                      f"{s['alpha']: .6e} {s['top']: .6e} {s['bottom']: .6e}")
                out["steps"].append(" ".join(
                    [f"{case} {method} {k} {s['epoch']} {len(s['idx'])}"]
                    + [f"{s[n]:.17e}" for n in ("beta", "alpha", "top", "bottom")]
                    + [str(i) for i in s["idx"]]))
            lead = [np.full(d_obj.size, case), np.full(d_obj.size, method)]
            rows, fmt = ln.ls.rows_obj(d_obj)
            out["obj"].append((np.column_stack(lead + [rows]), ["%d", "%d"] + fmt))
            lead = [np.full(d_probe.size, case), np.full(d_probe.size, method)]
            rows, fmt = ln.ls.rows_probe(d_probe)
            out["probe"].append((np.column_stack(lead + [rows]), ["%d", "%d"] + fmt))

    path = f"{d}/bh_near_steps_f64.txt"
    pathlib.Path(path).write_text("".join(r + "\n" for r in out["steps"]))
    print(f"wrote {path} ({len(out['steps'])} rows)")
    for name in ("obj", "probe"):
        path = f"{d}/bh_near_{name}_f64.txt"
        rows = np.concatenate([r for r, _ in out[name]])
        np.savetxt(path, rows, fmt=out[name][0][1])
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
