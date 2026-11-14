"""Float64 reference of the engine's BH through a multislice object, by torch autograd.

    micromamba run -n ptyrs-ref python -I refs/f64/bh_multislice.py <dir>

pty-chi's BH rejects several slices, so the reference is the step rule of engine/bh.rs's
Bh::epoch doc on lsqml_multislice.py's forward model, its derivatives by autograd. The state,
the three cases (angular spectrum, Fresnel transform, far field) and their counts are
lsqml_multislice.py's, regenerated from its seed and checked against its files in <dir>: NSLICE
slices of ones at the start, its start probe and 16 whole-pixel corners. A batch's cost is

    G = sum_frames sum_pixels (A - d)^2 + 2 EPS d ln(A + EPS),   d = sqrt(counts), EPS = 1e-8,

A = sqrt(sum_m |U psi_m|^2) with psi_m probe mode m through the slices and U the case's exit
over sqrt(k), lsqml_multislice.icalc in torch. G is the engine's amplitude cost with the EPS
pty-chi puts on A: its gradient has the engine's residual U psi (1 - d / (A + EPS)). Its
gradient g = 2 dG/dx* over every slice and the probe is torch's backward, its Hessian form
H_G(u, v) the derivative of <g, u> along v by a second backward, <a, b> = sum Re(conj(a) b).
The engine's H (engine/bh.rs BhBatch::hessian) is not G's: with c_u = sum_m Re(conj(U psi_m)
L_u), L_u = U J u, it has 2 d c_u c_v over (A + EPS)^3 where G's has it over A (A + EPS)^2.
The steps take the engine's, H_G plus that difference (the EPS term, closed form, L_u by
forward mode), so they are the engine's rule; each step saves its EPS term of H(D, D), whose
share of bottom is how far the engine's form is from G's Hessian.

Per case, METHODS: "full", one batch of all 16 frames per epoch, CG; "mini", batches of 6, 6
and 4, steepest descent; EPOCHS epochs of the engine's random_batches(16, size, SEED, epoch),
probe rho RHO. Per batch, Pg is g with its probe part times RHO^2 k, k the exit's intensity
scale; D = -Pg, or in "full" after the first epoch D = beta D' - Pg with
beta = H(Pg, D') / H(D', D') and D' the previous direction; top = -<g, D>, bottom = H(D, D),
alpha = top / bottom * count / size, every slice and the probe moved by alpha D.

Writes, per case C and method M (0 full, 1 mini), every complex value `re im`:
    <dir>/bh_multislice_steps_f64.txt  `C M step epoch count beta alpha top bottom eps f_0 ..`,
                                       one row per batch, eps the EPS term of bottom
    <dir>/bh_multislice_obj_f64.txt    `C M slice iy ix re im`, the slices' change over the run
    <dir>/bh_multislice_probe_f64.txt  `C M m iy ix re im`, the probe's change over the run
"""
import importlib.util
import pathlib
import sys

import numpy as np

RHO = 1.0
EPOCHS = 3
SEED = 7
EPS = 1e-8
METHODS = (("full", 16, True), ("mini", 6, False))  # name, batch size, CG


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
rb = near.rb
SHAPE, NSLICE = ms.SHAPE, ms.NSLICE


class Model:
    """lsqml_multislice's forward model of one case in torch, on frames `idx`."""

    def __init__(self, g, corners, counts):
        import torch
        self.torch, self.g, self.corners = torch, g, corners
        self.root = torch.from_numpy(np.sqrt(counts))

    def far_field(self, slices, probe, idx):
        """U psi [frame, mode], natural layout."""
        torch, g = self.torch, self.g
        ny, nx = SHAPE
        psi = probe[None].expand(len(idx), *probe.shape)
        for s in range(NSLICE):
            win = torch.stack([slices[s, cy:cy + ny, cx:cx + nx]
                               for cy, cx in self.corners[idx]])
            psi = psi * win[:, None]
            if s < NSLICE - 1:
                psi = g.gaps[s].propagate_forward(psi)
        y = g.prop.propagate_forward(psi) / np.sqrt(g.k)
        return torch.fft.fftshift(y, dim=(-2, -1)) if g.far else y

    def cost(self, slices, probe, idx):
        a = self.far_field(slices, probe, idx).abs().square().sum(1).sqrt()
        d = self.root[self.torch.as_tensor(idx)]
        return ((a - d).square() + 2 * EPS * d * (a + EPS).log()).sum()

    def eps_term(self, slices, probe, idx, u, v):
        """The engine's H(u, v) less G's: sum 2 d c_u c_v (1 / (A + EPS)^3 - 1 / (A (A + EPS)^2)),
        L_u by forward mode."""
        import torch.autograd.forward_ad as fw

        def tangent(w):
            with fw.dual_level():
                y = self.far_field(fw.make_dual(slices, w[0]), fw.make_dual(probe, w[1]), idx)
                return fw.unpack_dual(y)

        (y, lu), (_, lv) = tangent(u), tangent(v)
        a = y.abs().square().sum(1).sqrt()
        cu, cv = ((y.conj() * l).real.sum(1) for l in (lu, lv))
        e = a + EPS
        d = self.root[self.torch.as_tensor(idx)]
        return float((2 * d * cu * cv * (1 / e**3 - 1 / (a * e * e))).sum())


def dot(a, b):
    return sum(float((x.conj() * y).real.sum()) for x, y in zip(a, b))


def run(model, obj, probe, size, cg, rho2):
    """EPOCHS epochs of the step rule from (obj, probe): per batch its frames, beta, alpha,
    top, bottom and the EPS term of bottom; the slices and the probe at the end."""
    torch = model.torch
    o, p = torch.from_numpy(obj.copy()), torch.from_numpy(probe.copy())
    nframe = len(model.corners)
    prev, steps = None, []
    for epoch in range(EPOCHS):
        for idx in rb.random_batches(nframe, size, SEED, epoch):
            xs = [o.clone().requires_grad_(), p.clone().requires_grad_()]
            g = torch.autograd.grad(model.cost(*xs, idx), xs, create_graph=True)

            def hessian(u, v):
                """The engine's H(u, v): G's, <H_G u, v> by a second backward, and the EPS
                term."""
                s = sum((gi.conj() * ui).real.sum() for gi, ui in zip(g, u))
                hu = torch.autograd.grad(s, xs, retain_graph=True)
                return dot(hu, v) + model.eps_term(o, p, idx, u, v)

            gd = [gi.detach() for gi in g]
            pg = [gd[0], gd[1] * rho2]
            if cg and prev is not None:
                beta = hessian(prev, pg) / hessian(prev, prev)
                d = [beta * u - v for u, v in zip(prev, pg)]
            else:
                beta, d = 0.0, [-v for v in pg]
            top = -dot(gd, d)
            bottom = hessian(d, d)
            alpha = top / bottom * len(idx) / size
            eps = model.eps_term(o, p, idx, d, d)
            o, p = o + alpha * d[0], p + alpha * d[1]
            if cg:
                prev = d
            steps.append(dict(epoch=epoch, idx=idx, beta=beta, alpha=alpha, top=top,
                              bottom=bottom, eps=eps))
    return steps, o.numpy(), p.numpy()


def main():
    d = sys.argv[1]
    import torch
    import ptychi.utils as utils
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)

    truth_obj, truth_probe, probe, corners, rng = ms.make_state()
    obj = np.ones((NSLICE, *ms.OBJ), complex)
    nframe = len(corners)
    saved = np.loadtxt(f"{d}/lsqml_multislice_counts_f64.txt")
    saved_probe = np.loadtxt(f"{d}/lsqml_multislice_probe_f64.txt")
    assert (saved_probe[:, 3] + 1j * saved_probe[:, 4] == probe.ravel()).all()
    out = {"steps": [], "obj": [], "probe": []}
    for case, geom in enumerate(ms.GEOMETRIES):
        g = (ms.FarGeometry if case == ms.FAR else ms.Geometry)(*geom)
        counts = rng.poisson(ms.icalc(g, truth_obj, truth_probe, corners)).astype(np.float64)
        assert (saved[saved[:, 0] == case, 2:] == counts.reshape(nframe, -1)).all(), case
        model = Model(g, corners, counts)
        rho2 = RHO**2 * g.k
        f0 = float(model.cost(torch.from_numpy(obj), torch.from_numpy(probe), np.arange(nframe)))
        print(f"--- case {case}: {geom}, kind {g.kind}, k {g.k:.6e}, rho^2 k {rho2:.6e}, "
              f"G start {f0:.10e}")
        for method, (name, size, cg) in enumerate(METHODS):
            steps, o, p = run(model, obj, probe, size, cg, rho2)
            f1 = float(model.cost(torch.from_numpy(o), torch.from_numpy(p), np.arange(nframe)))
            print(f"  {name} ({'CG' if cg else 'GD'}, batch {size}): G {f0:.10e} -> {f1:.10e}")
            print("    step epoch count  beta          alpha         top           bottom"
                  "        eps / bottom")
            for k, s in enumerate(steps):
                print(f"    {k:4d} {s['epoch']:5d} {len(s['idx']):5d}  {s['beta']: .6e} "
                      f"{s['alpha']: .6e} {s['top']: .6e} {s['bottom']: .6e} "
                      f"{s['eps'] / s['bottom']: .1e}")
                out["steps"].append(" ".join(
                    [f"{case} {method} {k} {s['epoch']} {len(s['idx'])}"]
                    + [f"{s[n]:.17e}" for n in ("beta", "alpha", "top", "bottom", "eps")]
                    + [str(i) for i in s["idx"]]))
            assert np.isfinite(f1) and f1 < f0, (case, name, f0, f1)
            for key, (rows, fmt) in (("obj", ms.rows_slices(o - obj)),
                                     ("probe", near.ls.rows_probe(p - probe))):
                lead = [np.full(len(rows), case), np.full(len(rows), method)]
                out[key].append((np.column_stack(lead + [rows]), ["%d", "%d"] + fmt))

    path = f"{d}/bh_multislice_steps_f64.txt"
    pathlib.Path(path).write_text("".join(r + "\n" for r in out["steps"]))
    print(f"wrote {path} ({len(out['steps'])} rows)")
    for name in ("obj", "probe"):
        path = f"{d}/bh_multislice_{name}_f64.txt"
        rows = np.concatenate([r for r, _ in out[name]])
        np.savetxt(path, rows, fmt=out[name][0][1])
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
