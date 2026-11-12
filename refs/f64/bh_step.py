"""Float64 reference of pty-chi BH steps (object and probe) from a fixed state: CG over one
batch per epoch, steepest descent over mini-batches.

    <engine>/target/release/examples/llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/bh_step.py <dir>

The state is the one ml_grad.py reads (object [1, 76, 80], probe [2, 32, 32], 40 frames). The
steps are pty-chi's BHReconstructor with probe rho 0.01 (refs/ptychi/run_m3.py bh_sweep, the
runs behind DESIGN B7), positions fixed, over two epochs of the engine's random_batches(40,
size, SEED, epoch) (random_batches.py), fed to pty-chi as its batch sampler, in two cases:
"full", method CG with one batch of all 40 frames per epoch, and "mini", method GD (steepest
descent, no direction kept across batches) with batches of 16, 16 and 8 frames.
Written in the engine's convention (PyNX probe P, psi_m = F(P_m q O_view),
F = FFT_ortho fftshift, corners and q as ml_grad), with the batch frames B:

    F(O, P) = sum_B sum_pixels (A - sqrt(iobs))^2, A = sqrt(sum_m |psi_m|^2)
    r_m = F^-1(psi_m (1 - sqrt(iobs) / (A + 1e-8)))                      (gradF / 2)
    g_o = 2 scatter(sum_m conj(P_m q) r_m) = 2 dF/dO*,  g_P = 2 conj(q) sum_B conj(O_view) r_m
    probe variable P = rho_e P~ with rho_e = 0.01 sqrt(ny nx) = 0.32, so G = (g_o, rho_e g_P),
        a direction (u_o, u_p) displaces (O, P) by (u_o, rho_e u_p), and <a, b> = sum Re(conj a b)
        over object and probe
    H(u, v) = sum_B,m Re(conj(2 r_m) d2m) + 2 sum_B,pixels [(1 - d/A) sum_m Re(conj(F dm_u) F dm_v)
              + (d/A) c_u c_v],  dm_u = rho_e u_p q O_view + P q u_op,
              d2m = rho_e q (u_p v_op + v_p u_op), c_u = sum_m Re(conj(psi_m / A) F dm_u),
              d = sqrt(iobs), A + 1e-8 in place of A; u_op = view(u_o), the object direction
              at the batch's windows
    full (CG): eta = -G in the first epoch; later beta = H(G, eta') / H(eta', eta'),
        eta = -G + beta eta', eta' the previous epoch's direction
    mini (GD): eta = -G of the batch at every step
    alpha = -<G, eta> / H(eta, eta) x (frames in the batch / batch size)
    O += alpha eta_o, P += rho_e alpha eta_p

pty-chi's CG under PATCH caches eta_op per frame instead of taking view(eta_o). With one batch
per epoch the two are equal; with several the cache holds patches from when those frames were
last in a batch, so H and alpha belong to another direction than the one the object moves by.
The engine therefore takes GD for mini-batches, which needs no cache.

Conversions from pty-chi, whose probe is P q / sqrt(ny nx) under an unnormalised FFT: its gradF
is 2 sqrt(ny nx) r_m, its probe gradient and direction are q times the ones above, rho_e is its
rho times sqrt(ny nx); the object, its gradient and direction, alpha, beta, both inner products
and H are the same. No post-epoch hook acts (constraints off, as in lsqml_step.py), so the
object-probe ambiguity removal and mode orthogonalisation (which pty-chi's BH also applies to
eta_p) that run_m3 leaves at pty-chi's defaults are not part of the steps.

Writes, per case C in full, mini, in the engine's convention:
    <dir>/bh_C_obj_f64.txt    `iy ix re0 im0 re1 im1`, the object change over epoch 0 and over
                              epoch 1, every pixel, row-major
    <dir>/bh_C_probe_f64.txt  `m iy ix re0 im0 re1 im1`, the probe change likewise
    <dir>/bh_C_steps_f64.txt  `step epoch count beta alpha top bottom f_0 .. f_count-1`, one row
                              per batch, beta 0 in epoch 0 and in GD, alpha after the
                              batch-size factor, top = -<G, eta>, bottom = H(eta, eta), f the
                              batch's frames
Printed checks, with the eps at 0: g_o and g_P against central differences of F along a random
direction; H(eta, eta) and H(G, eta') against central differences of <G(x + h eta'), u>; the
quadratic model -alpha top + alpha^2 bottom / 2 against the change of F; pty-chi's own steps
(method CG for full, GD for mini) in f64 on CPU.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2

AX = (-2, -1)
RHO = 0.01
EPS = 1e-8
EPOCHS = 2
SEED = 7
CASES = (("full", 40, True), ("mini", 16, False))  # name, batch size, CG
STEPS = (1e-4, 1e-5, 1e-6)


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


def redot(a, b):
    return np.real(np.vdot(a, b))


class BH:
    """pty-chi BHReconstructor.compute_updates_object_probe in the engine's convention."""

    def __init__(self, m):
        self.m = m
        self.rho = RHO * np.sqrt(np.prod(m.n))
        self.eta_o = self.eta_p = None

    def state(self, obj, probe, idx, eps):
        m = self.m
        pq = probe * m.q
        v = ls.views(obj, m.corners[idx], m.n)
        psi = ls.far(pq, v)
        ok = m.valid[idx]
        d = np.sqrt(np.maximum(m.iobs[idx], 0))
        a = np.sqrt((np.abs(psi) ** 2).sum(1)) + eps
        r = fftshift(ifft2(psi * np.where(ok, 1 - d / a, 0)[:, None], norm="ortho"), axes=AX)
        return dict(idx=idx, pq=pq, v=v, psi=psi, ok=ok, d=d, a=a, r=r)

    def grads(self, s, shape):
        """(2 dF/dO*, 2 dF/dP*)."""
        m = self.m
        g_o = 2 * ls.scatter((np.conj(s["pq"]) * s["r"]).sum(1), m.corners[s["idx"]], shape)
        g_p = 2 * np.conj(m.q) * (np.conj(s["v"])[:, None] * s["r"]).sum(0)
        return g_o, g_p

    def hess(self, s, u_op, u_p, v_op, v_p):
        q, rho = self.m.q, self.rho

        def dm(w_op, w_p):
            return rho * w_p[None] * q * s["v"][:, None] + s["pq"] * w_op[:, None]

        def prop(x):
            return fft2(fftshift(x, axes=AX), norm="ortho")

        lu, lv = prop(dm(u_op, u_p)), prop(dm(v_op, v_p))
        d2m = rho * q * (u_p[None] * v_op[:, None] + v_p[None] * u_op[:, None])
        l0 = s["psi"] / s["a"][:, None]
        d0 = np.where(s["ok"], s["d"] / s["a"], 0)
        ok = s["ok"]
        v1 = (np.where(ok, 1 - d0, 0)[:, None] * np.real(np.conj(lu) * lv)).sum()
        c1 = np.real(np.conj(l0) * lu).sum(1)
        c2 = np.real(np.conj(l0) * lv).sum(1)
        return redot(2 * s["r"], d2m) + 2 * (v1 + (d0 * c1 * c2).sum())

    def step(self, obj, probe, idx, cg, batch_size, eps=EPS):
        m = self.m
        s = self.state(obj, probe, idx, eps)
        g_o, g_p = self.grads(s, obj.shape)
        go, gp = g_o, self.rho * g_p
        if cg:
            gop = ls.views(go, m.corners[idx], m.n)
            pop = ls.views(self.eta_o, m.corners[idx], m.n)
            beta = (self.hess(s, gop, gp, pop, self.eta_p)
                    / self.hess(s, pop, self.eta_p, pop, self.eta_p))
            prev = (self.eta_o, self.eta_p)
            eta_o, eta_p = -go + beta * self.eta_o, -gp + beta * self.eta_p
        else:
            beta, prev = 0.0, None
            eta_o, eta_p = -go, -gp
        self.eta_o, self.eta_p = eta_o, eta_p
        eta_op = ls.views(eta_o, m.corners[idx], m.n)
        top = -(redot(go, eta_o) + redot(gp, eta_p))
        bottom = self.hess(s, eta_op, eta_p, eta_op, eta_p)
        alpha = top / bottom * len(idx) / batch_size
        return dict(beta=beta, alpha=alpha, top=top, bottom=bottom, g_o=g_o, g_p=g_p, prev=prev,
                    d_obj=alpha * eta_o, d_probe=self.rho * alpha * eta_p)


def loss(m, obj, probe, idx):
    """F of the module docstring without eps."""
    psi = ls.far(probe * m.q, ls.views(obj, m.corners[idx], m.n))
    return ls.frame_llk("gaussian", (np.abs(psi) ** 2).sum(1), m.iobs[idx], m.valid[idx]).sum()


def run(m, obj, probe, size, cg):
    """Numpy BH (CG or GD) over EPOCHS epochs: per-step rows, per-epoch changes, per-step
    results."""
    bh = BH(m)
    rows, changes, steps = [], [], []
    k = 0
    for epoch in range(EPOCHS):
        o0, p0 = obj, probe
        for idx in rb.random_batches(len(m.corners), size, SEED, epoch):
            s = bh.step(obj, probe, idx, cg and epoch > 0, size)
            s.update(obj=obj, probe=probe, idx=idx, epoch=epoch)
            steps.append(s)
            rows.append(f"{k} {epoch} {len(idx)} "
                        + " ".join(f"{s[x]:.17e}" for x in ("beta", "alpha", "top", "bottom"))
                        + " " + " ".join(str(f) for f in idx))
            obj, probe = obj + s["d_obj"], probe + s["d_probe"]
            k += 1
        changes.append((obj - o0, probe - p0))
    return rows, changes, steps


def check(m, obj, probe, idx, rng):
    """Gradient and Hessian of the eps-free F against central differences."""
    bh = BH(m)
    s = bh.state(obj, probe, idx, 0.0)
    g_o, g_p = bh.grads(s, obj.shape)
    d_o = rng.standard_normal(obj.shape) + 1j * rng.standard_normal(obj.shape)
    d_p = rng.standard_normal(probe.shape) + 1j * rng.standard_normal(probe.shape)
    d_o *= np.abs(obj).max() / np.abs(d_o).max()
    d_p *= np.abs(probe).max() / np.abs(d_p).max()
    print("  g_o, g_P vs central differences of F (eps 0), rel diff:")
    for h in STEPS:
        fo = ls.directional(lambda x: loss(m, x, probe, idx), obj, d_o, h)
        fp = ls.directional(lambda x: loss(m, obj, x, idx), probe, d_p, h)
        eo, ep = redot(g_o, d_o), redot(g_p, d_p)
        print(f"    h {h:.0e}: object {abs(fo - eo) / abs(eo):.3e}  "
              f"probe {abs(fp - ep) / abs(ep):.3e}")


def check_hess(m, bh, step, obj, probe):
    """H(u, v) = d/dt <G(x + t v), u> along directions in the scaled variables, eps 0."""
    idx = step["idx"]
    s = bh.state(obj, probe, idx, 0.0)
    g_o, g_p = bh.grads(s, obj.shape)
    go, gp = g_o, bh.rho * g_p
    eta_o, eta_p = step["d_obj"] / step["alpha"], step["d_probe"] / (bh.rho * step["alpha"])
    pairs = [("H(eta, eta)", (eta_o, eta_p), (eta_o, eta_p))]
    if step["prev"] is not None:
        pairs.append(("H(G, eta')", (go, gp), step["prev"]))

    def dot(x_o, x_p, u):
        t = bh.state(x_o, x_p, idx, 0.0)
        a, b = bh.grads(t, obj.shape)
        return redot(a, u[0]) + redot(bh.rho * b, u[1])

    for name, u, v in pairs:
        h_an = bh.hess(s, *[ls.views(u[0], m.corners[idx], m.n), u[1],
                            ls.views(v[0], m.corners[idx], m.n), v[1]])
        out = []
        for h in (1e-3, 1e-4, 1e-5):
            hh = h * abs(step["alpha"])
            fd = (dot(obj + hh * v[0], probe + hh * bh.rho * v[1], u)
                  - dot(obj - hh * v[0], probe - hh * bh.rho * v[1], u)) / (2 * hh)
            out.append(f"h {h:.0e} alpha {abs(fd - h_an) / abs(h_an):.3e}")
        print(f"    {name}: " + "  ".join(out))


def ptychi_run(m, iobs, wavelength, obj, probe, size, cg):
    """pty-chi BHReconstructor (method CG or GD) in f64 on CPU over EPOCHS epochs of the
    engine's batches, mapped as in ptyrs/scripts/engine/compare_ptychi_rpie.py; per batch beta,
    alpha and the object and probe changes in the engine's convention."""
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    _, ny, nx = iobs.shape
    root_n = np.sqrt(ny * nx)
    opts = api.BHOptions()
    opts.data_options.wavelength_m = wavelength
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = True
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
    r.forward_model_options.pad_for_shift = 4
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
                                batch_sampler=rb.Sampler(len(m.corners), size, SEED))
    out = []
    cls = type(rec)

    def beta(*args, **kwargs):
        b = cls.calc_beta_object_probe(rec, *args, **kwargs)
        out[-1]["beta"] = float(b)
        return b

    def alpha(*args, **kwargs):
        a = cls.calc_alpha_object_probe(rec, *args, **kwargs)
        out[-1]["alpha_raw"] = float(a)
        return a

    def apply(delta_o, delta_p, delta_pos):
        out[-1].update(
            idx=rec.indices.numpy().copy(),
            d_obj=delta_o[0].cpu().numpy().copy(),
            d_probe=delta_p[0].cpu().numpy() * np.conj(m.q) * root_n)
        return cls.apply_updates(rec, delta_o, delta_p, delta_pos)

    def minibatch(input_data, y_true):
        out.append({"beta": 0.0, "epoch": rec.current_epoch})
        return cls.run_minibatch(rec, input_data, y_true)

    rec.calc_beta_object_probe = beta
    rec.calc_alpha_object_probe = alpha
    rec.apply_updates = apply
    rec.run_minibatch = minibatch
    task.run(EPOCHS)
    for o_ in out:
        o_["alpha"] = o_["alpha_raw"] * len(o_["idx"]) / size
    return out


def rows_obj(changes):
    ny, nx = changes[0].shape
    cols = [np.repeat(np.arange(ny), nx), np.tile(np.arange(nx), ny)]
    for c in changes:
        cols += [c.real.ravel(), c.imag.ravel()]
    return np.column_stack(cols), ["%d"] * 2 + ["%.17e"] * (2 * len(changes))


def rows_probe(changes):
    g = np.indices(changes[0].shape).reshape(3, -1)
    cols = list(g)
    for c in changes:
        cols += [c.real.ravel(), c.imag.ravel()]
    return np.column_stack(cols), ["%d"] * 3 + ["%.17e"] * (2 * len(changes))


def main():
    d = sys.argv[1]
    args = mg.load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    obj = obj0[0]
    nframe = len(m.corners)
    print(f"object {obj.shape}, probe {probe.shape}, {nframe} frames, probe rho {RHO} "
          f"(rho_e {RHO * np.sqrt(np.prod(m.n)):g})")
    rng = np.random.default_rng(0)
    for name, size, cg in CASES:
        print(f"--- {name}: batch {size}, method {'CG' if cg else 'GD'}, {EPOCHS} epochs")
        check(m, obj, probe, rb.random_batches(nframe, size, SEED, 0)[0], rng)
        rows, changes, steps = run(m, obj, probe, size, cg)
        print("  H vs central differences of <G(x + h eta'), u> (eps 0), rel diff:")
        for k, s in enumerate(steps):
            print(f"   step {k}:")
            check_hess(m, BH(m), s, s["obj"], s["probe"])
        print("  step  epoch  count  beta          alpha         top           bottom"
              "        F before -> after (batch)")
        for k, s in enumerate(steps):
            f0 = loss(m, s["obj"], s["probe"], s["idx"])
            f1 = loss(m, s["obj"] + s["d_obj"], s["probe"] + s["d_probe"], s["idx"])
            print(f"  {k:4d} {s['epoch']:5d}  {len(s['idx']):5d} {s['beta']: .6e} "
                  f"{s['alpha']: .6e} {s['top']: .6e} {s['bottom']: .6e}  "
                  f"{f0:.6e} -> {f1:.6e}")
        ratio = []
        for s in steps:
            df = (loss(m, s["obj"] + s["d_obj"], s["probe"] + s["d_probe"], s["idx"])
                  - loss(m, s["obj"], s["probe"], s["idx"]))
            a = s["alpha"]
            ratio.append(df / (-a * s["top"] + a * a * s["bottom"] / 2))
        print("  F change / its quadratic model -alpha top + alpha^2 bottom / 2: "
              + " ".join(f"{x:.4f}" for x in ratio))
        eps0 = BH(m)
        s0 = steps[0]
        e = eps0.step(s0["obj"], s0["probe"], s0["idx"], False, size, eps=0.0)
        print("  step 0, eps 0 vs 1e-8 (max, L2): object %.3e %.3e" % mg.rel(e["d_obj"],
                                                                            s0["d_obj"]),
              " probe %.3e %.3e" % mg.rel(e["d_probe"], s0["d_probe"]))

        pc = ptychi_run(m, iobs, wavelength, obj0, probe, size, cg)
        assert len(pc) == len(steps), (len(pc), len(steps))
        print("  pty-chi BH f64 vs numpy, per step (max, L2) / rel:")
        for k, (p, s) in enumerate(zip(pc, steps)):
            assert (p["idx"] == s["idx"]).all() and p["epoch"] == s["epoch"]
            db = abs(p["beta"] - s["beta"]) / abs(s["beta"]) if s["beta"] else abs(p["beta"])
            print(f"    step {k}: beta {db:.3e}  alpha "
                  f"{abs(p['alpha'] - s['alpha']) / abs(s['alpha']):.3e}  "
                  "object %.3e %.3e" % mg.rel(p["d_obj"], s["d_obj"])
                  + "  probe %.3e %.3e" % mg.rel(p["d_probe"], s["d_probe"]))

        full = np.arange(nframe)
        obj_e, probe_e = obj, probe
        f_all = [loss(m, obj_e, probe_e, full)]
        for c_o, c_p in changes:
            obj_e, probe_e = obj_e + c_o, probe_e + c_p
            f_all.append(loss(m, obj_e, probe_e, full))
        print("  F over all frames per epoch: " + " -> ".join(f"{f:.10e}" for f in f_all))

        for kind, (data, fmt) in (
                ("obj", rows_obj([c[0] for c in changes])),
                ("probe", rows_probe([c[1] for c in changes]))):
            path = f"{d}/bh_{name}_{kind}_f64.txt"
            np.savetxt(path, data, fmt=fmt)
            print(f"  wrote {path} ({len(data)} rows)")
        path = f"{d}/bh_{name}_steps_f64.txt"
        pathlib.Path(path).write_text("".join(r + "\n" for r in rows))
        print(f"  wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
