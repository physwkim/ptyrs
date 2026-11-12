"""Float64 reference of one pty-chi LSQML minibatch with OPR and the main mode's weights refined
(OPRModeWeightsOptions.optimize_intensity_variation): lsqml_opr_step.py's state, eigenmodes,
weights and batch, plus main-mode weights c_k = 1 + 0.1 x standard normal drawn after them
from the same generator. Frame k sees

    P_k0 = c_k P_0 + sum_i w_ki U_i,  P_km = P_m (m >= 1)

and the step is lsqml_opr_step.step at those probes, then, from the same chi,
OPRModeWeights._calculate_intensity_variation_update_direction and pty-chi's SGD step of 0.1 of
it, with V_k the frame's object window and P_0 the shared probe's mode 0:

    c_k' = c_k + 0.1 sum_u Re(conj(V_k P_0 q) chi_k0) / sum_u |V_k P_0|^2

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/lsqml_opr_int_step.py <dir>

Writes lsqml_opr_step.py's eigen, obj, probe, steps and values tables at these probes as
`<dir>/lsqml_opr_int_{eigen,obj,probe,steps,values}_f64.txt`, and
`<dir>/lsqml_opr_int_weights_f64.txt`, `k c w_0.. dL/dw_0.. dF/dw_0.. c'_g w'_g,0.. c'_p
w'_p,0..` (K eigenmode columns each; c', w' = c, w outside the batch). Printed: central
differences of each frame's L and F in each w_ki; pty-chi's LSQML in f64 on CPU as
lsqml_opr_step.ptychi with the intensity variation on, against the step here from its
normalised eigenmodes.
"""
import importlib.util
import pathlib
import sys

import numpy as np


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lo = sibling("lsqml_opr_step")
mg, it = lo.mg, lo.it


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    assert m.valid.all(), "no masked pixel"
    obj = obj0[0]
    nf, ny, nx = iobs.shape
    n = ny * nx
    rng = np.random.default_rng(lo.SEED)
    r = rng.standard_normal((lo.K, ny, nx)) + 1j * rng.standard_normal((lo.K, ny, nx))
    u = np.stack([np.sqrt(n) * ri / lo.mnorm(ri) for ri in r])
    w = 1e-6 * rng.standard_normal((nf, lo.K))
    idx = np.sort(rng.permutation(nf)[: lo.BATCH])
    c = 1 + 0.1 * rng.standard_normal(nf)
    v = m.views(obj)
    print(f"{nf} frames, K {lo.K}, batch {idx}, c {c.min():.6f}..{c.max():.6f}")

    lf = it.costs(m, lo.icalc(m, v, probe, u, w, c))
    g = lo.w_grads(m, v, u, w, c)
    print(f"L {lf[0].sum():.17e}, F {lf[1].sum():.17e}")
    print("central differences in w_ki, (max, L2) rel diff (dL/dw; dF/dw):")
    for h in lo.STEPS:
        fd = np.zeros_like(g)
        for i in range(lo.K):
            e = np.zeros_like(w)
            e[:, i] = h
            fd[:, :, i] = (it.costs(m, lo.icalc(m, v, probe, u, w + e, c))
                           - it.costs(m, lo.icalc(m, v, probe, u, w - e, c))) / (2 * h)
        print(f"  h {h:.0e}: %.3e %.3e; %.3e %.3e" % (*mg.rel(fd[0], g[0]), *mg.rel(fd[1], g[1])))

    new = {}
    for noise in lo.NOISES:
        s = lo.step(m, obj, probe, u, w, idx, noise, c)
        print(f"--- {noise}: a_far {s['a_far'].min():.6f}..{s['a_far'].max():.6f}, max |c' - c| "
              f"{np.abs(s['w0'] - c).max():.6e}, max |w' - w| {np.abs(s['w'] - w).max():.6e}")
        new[noise] = s

    pc = lo.ptychi(m, iobs, wavelength, obj0, u, w, idx, c)
    print("pty-chi LSQML f64, intensity variation on; its step vs the step here from its "
          "normalised eigenmodes (max, L2):")
    out = [i for i in range(nf) if i not in idx]
    for noise in lo.NOISES:
        q = pc[noise]
        assert (q["idx"] == idx).all(), q["idx"]
        s = lo.step(m, obj, probe, q["u0"], w, idx, noise, c)
        print(f"  {noise}: {q['calls']} torch.randn call(s) zeroed; c outside the batch moved "
              "%.1e, w outside %.1e" % (np.abs(q["w0"][out] - c[out]).max(),
                                        np.abs(q["w"][out] - w[out]).max()))
        for k in ("illum", "a_far", "a_o", "a_p", "d_obj", "d_probe", "u"):
            print(f"    {k:8s} %.3e %.3e" % mg.rel(q[k], s[k]))
        print("    w (batch) %.3e %.3e" % mg.rel(q["w"][idx], s["w"][idx]))
        print("    c change  %.3e %.3e" % mg.rel(q["w0"][idx] - c[idx], s["w0"][idx] - c[idx]))

    ng, npo = new["gaussian"], new["poisson"]
    f = "%.17e"
    eig = np.column_stack([
        *[a.ravel() for a in np.indices(u.shape)],
        *[part(a).ravel() for a in (u, ng["u"], npo["u"]) for part in (np.real, np.imag)]])
    rows = {
        "eigen": (eig, ["%d"] * 3 + [f] * 6),
        "weights": (np.column_stack([np.arange(nf), c, w, g[0], g[1], ng["w0"], ng["w"],
                                     npo["w0"], npo["w"]]), ["%d"] + [f] * (5 * lo.K + 3)),
        "obj": (np.column_stack([*[a.ravel() for a in np.indices(obj.shape)],
                                 *[part(x["d_obj"]).ravel() for x in (ng, npo)
                                   for part in (np.real, np.imag)]]),
                ["%d"] * 2 + [f] * 4),
        "probe": (np.column_stack([*[a.ravel() for a in np.indices(probe.shape)],
                                   *[part(x["d_probe"]).ravel() for x in (ng, npo)
                                     for part in (np.real, np.imag)]]),
                  ["%d"] * 3 + [f] * 4),
        "steps": (np.column_stack([idx, *[x[k] for x in (ng, npo)
                                          for k in ("a_far", "a_o", "a_p")]]),
                  ["%d"] + [f] * 6),
    }
    for name, (a, fmt) in rows.items():
        path = f"{d}/lsqml_opr_int_{name}_f64.txt"
        np.savetxt(path, a, fmt=fmt)
        print(f"wrote {path} ({len(a)} rows)")
    path = f"{d}/lsqml_opr_int_values_f64.txt"
    with open(path, "w") as out_file:
        out_file.write(f"{int(m.valid.sum())} {lf[0].sum():.17e} {lf[1].sum():.17e}\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
