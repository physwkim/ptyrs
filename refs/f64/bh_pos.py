"""Float64 model of BH (pty-chi's bilinear-Hessian CG) refining the engine's positions.

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/bh_pos.py <dir>

pty-chi's BH with positions optimizable (compute_updates_object_probe_positions) on the
engine's position model: the probe of frame k is moved by its offset d_k (lsqml_pos_step.py's
T_d, F, q) where pty-chi translates the object patch. A Fourier translation or derivative
factors over a product only when the product is unaliased, which this state's views are not, so
the two costs and their steps differ (pty-chi's by 30-80% here) and no package computes these.
pty-chi checks signs and units where the two agree: at d = 0 on band_limited's state, where its
first step's directions, top and alpha are the model's. The state is the one bh_step.py reads,
with offsets d_k = 0.5 + OFFSET (numpy default_rng(1) uniform in [-1, 1)), inside the cell
where pty-chi's patch corner is the engine's view corner. In displacements (bh_step.py's
directions times rho_e for the probe, times rho_pos for the positions), over the batch B:

    W_km = T_dk(P_m q),  psi_km = F(W_km V_k),  A = sqrt(sum_m |psi_km|^2) + 1e-8
    r_km = F^-1(psi_km (1 - sqrt(iobs) / A)),  g_W = 2 conj(V_k) r_km
    g_o = 2 scatter(sum_m conj(W_km) r_km),  g_P = conj(q) sum_k T_-dk(g_W)
    g_dka = sum_m Re<g_W, d_a W_km>,  d_a W = F_grid^-1(F_grid(P q) e^(-2 pi i f.d) (-2 pi i f_a))
    Pg = (g_o, rho_e^2 g_P, rho_pos^2 g_d), rho_e = RHO sqrt(ny nx), rho_pos = POS_RHO
    H(u, v) = Re<2 r, d2psi> + 2 sum_B,pixels [(1 - d/A) sum_m Re(conj(F dpsi_u) F dpsi_v)
              + (d/A) c_u c_v]  (bh_step.py's), with
        dW_u = T_d(u_P q) + sum_a u_da d_a W,  dpsi_u = dW_u V + W view(u_o)
        d2psi = (sum_a (v_da d_a T_d(u_P q) + u_da d_a T_d(v_P q))
                 + sum_ab u_da v_db d_a d_b W) V + dW_u view(v_o) + dW_v view(u_o)
    full (CG): D = -Pg in the first epoch; later beta = H(Pg, D') / H(D', D'), D = -Pg + beta D'
    mini (GD): D = -Pg of the batch at every step
    alpha = -<g, D> / H(D, D) x (frames in the batch / batch size)
    O += alpha D_o,  P += alpha D_P,  d_B += POS_STEP alpha D_d

pty-chi's grad_pos is rho_pos times the cost's derivative over its positions, with no sqrt(ny nx)
(its gradF and probe carry reciprocal factors), so rho_pos maps as is; its positions take the
SGD step size of ProbePositionOptions, 0.3, unclipped (clip_update=False), while alpha comes
from the whole direction. Positive d moves the probe as a positive pty-chi position does.

Writes, per case C in full, mini, in the engine's convention, over EPOCHS epochs:
    <dir>/bh_pos_C_obj_f64.txt      `iy ix re im`, the object change
    <dir>/bh_pos_C_probe_f64.txt    `m iy ix re im`, the probe change
    <dir>/bh_pos_C_offsets_f64.txt  `k dy dx dy' dx'`, every frame's offsets before and after
    <dir>/bh_pos_C_steps_f64.txt    `step epoch count beta alpha top bottom f_0 .. f_count-1`
Printed checks, with the eps at 0: g_o, g_P and g_d against central differences of F along
random directions; H(D, D) and H(Pg, D') against central differences of <g(x + t v), u> with
u, v over object, probe and positions; at_corners.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2

AX = (-2, -1)
RHO = 0.01
POS_RHO = 2.0
POS_STEP = 0.3
EPS = 1e-8
EPOCHS = 2
SEED = 7
OFFSET = 0.4
CASES = (("full", 40, True), ("mini", 16, False))  # name, batch size, CG


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
rb = sibling("random_batches")


def redot(a, b):
    return np.real(np.vdot(a, b))


def moved(x, d):
    """T_d(x) of the planes x [nm, ny, nx] per frame of d [nf, 2], its first derivatives
    [a] and second derivatives [a][b] over d_a, d_b, each [nf, nm, ny, nx]."""
    ny, nx = x.shape[-2:]
    spec = fft2(x, axes=AX)[None]
    fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
    ph = np.exp(-2j * np.pi * (fy[None, :, None] * d[:, 0, None, None]
                               + fx[None, None, :] * d[:, 1, None, None]))[:, None] * spec
    k = [-2j * np.pi * fy[:, None] + 0 * fx[None], -2j * np.pi * fx[None] + 0 * fy[:, None]]
    return (ifft2(ph, axes=AX), [ifft2(ph * k[a], axes=AX) for a in range(2)],
            [[ifft2(ph * k[a] * k[b], axes=AX) for b in range(2)] for a in range(2)])


def prop(x):
    return fft2(fftshift(x, axes=AX), norm="ortho")


class BH:
    """The model's state, derivatives and steps."""

    def __init__(self, m):
        self.m = m
        self.rho = RHO * np.sqrt(np.prod(m.n))
        self.dir = None  # (D_o, D_P, D_d over every frame)

    def state(self, obj, probe, d, idx, eps):
        m = self.m
        w, dw, d2w = moved(probe * m.q, d[idx])
        v = ls.views(obj, m.corners[idx], m.n)
        psi = ls.far(w, v)
        ok = m.valid[idx]
        root = np.sqrt(np.maximum(m.iobs[idx], 0))
        a = np.sqrt((np.abs(psi) ** 2).sum(1)) + eps
        r = fftshift(ifft2(psi * np.where(ok, 1 - root / a, 0)[:, None], norm="ortho"), axes=AX)
        return dict(idx=idx, d=d[idx], w=w, dw=dw, d2w=d2w, v=v, psi=psi, ok=ok, root=root,
                    a=a, r=r)

    def grads(self, s, shape):
        """(g_o, g_P, g_d) with <g, u> the derivative of F along the displacement u."""
        m = self.m
        g_o = 2 * ls.scatter((np.conj(s["w"]) * s["r"]).sum(1), m.corners[s["idx"]], shape)
        gw = 2 * np.conj(s["v"])[:, None] * s["r"]
        g_p = np.conj(m.q) * lp.move(gw, -s["d"]).sum(0)
        g_d = np.stack([np.real(np.conj(gw) * s["dw"][a]).sum((1, 2, 3)) for a in range(2)], 1)
        return g_o, g_p, g_d

    def hess(self, s, u, v):
        """H(u, v) for displacements u, v = (object, probe, offsets of the batch)."""
        m = self.m
        idx = s["idx"]
        (uo, up, ud), (vo, vp, vd) = u, v
        uop, vop = (ls.views(x, m.corners[idx], m.n) for x in (uo, vo))
        tu, dtu, _ = moved(up * m.q, s["d"])
        tv, dtv, _ = moved(vp * m.q, s["d"])

        def dw(t, x):
            return t + sum(x[:, a, None, None, None] * s["dw"][a] for a in range(2))

        dwu, dwv = dw(tu, ud), dw(tv, vd)
        d2w = sum(vd[:, a, None, None, None] * dtu[a] + ud[:, a, None, None, None] * dtv[a]
                  for a in range(2))
        d2w = d2w + sum(ud[:, a, None, None, None] * vd[:, b, None, None, None] * s["d2w"][a][b]
                        for a in range(2) for b in range(2))
        lu = prop(dwu * s["v"][:, None] + s["w"] * uop[:, None])
        lv = prop(dwv * s["v"][:, None] + s["w"] * vop[:, None])
        d2psi = d2w * s["v"][:, None] + dwu * vop[:, None] + dwv * uop[:, None]
        l0 = s["psi"] / s["a"][:, None]
        d0 = np.where(s["ok"], s["root"] / s["a"], 0)
        v1 = (np.where(s["ok"], 1 - d0, 0)[:, None] * np.real(np.conj(lu) * lv)).sum()
        c1 = np.real(np.conj(l0) * lu).sum(1)
        c2 = np.real(np.conj(l0) * lv).sum(1)
        return redot(2 * s["r"], d2psi) + 2 * (v1 + (d0 * c1 * c2).sum())

    def step(self, obj, probe, d, idx, cg, batch_size, eps=EPS):
        s = self.state(obj, probe, d, idx, eps)
        g = self.grads(s, obj.shape)
        pg = (g[0], self.rho**2 * g[1], POS_RHO**2 * g[2])
        if cg:
            prev = (self.dir[0], self.dir[1], self.dir[2][idx])
            beta = self.hess(s, pg, prev) / self.hess(s, prev, prev)
            dirs = tuple(beta * p - x for p, x in zip(prev, pg))
        else:
            beta, prev = 0.0, None
            dirs = tuple(-x for x in pg)
        if self.dir is None:
            self.dir = (None, None, np.zeros_like(d))
        self.dir = (dirs[0], dirs[1], self.dir[2].copy())
        self.dir[2][idx] = dirs[2]
        top = -sum(redot(x, y) for x, y in zip(g, dirs))
        bottom = self.hess(s, dirs, dirs)
        alpha = top / bottom * len(idx) / batch_size
        d_pos = np.zeros_like(d)
        d_pos[idx] = POS_STEP * alpha * dirs[2]
        return dict(beta=beta, alpha=alpha, top=top, bottom=bottom, g=g, prev=prev, dirs=dirs,
                    d_obj=alpha * dirs[0], d_probe=alpha * dirs[1], d_pos=d_pos)


def loss(m, obj, probe, d, idx):
    """F without eps."""
    w = lp.move(np.broadcast_to(probe * m.q, (len(idx), *probe.shape)), d[idx])
    psi = ls.far(w, ls.views(obj, m.corners[idx], m.n))
    return ls.frame_llk("gaussian", (np.abs(psi) ** 2).sum(1), m.iobs[idx], m.valid[idx]).sum()


def run(m, obj, probe, d, size, cg):
    """EPOCHS epochs of the model: per-step rows, per-step records, final state."""
    bh = BH(m)
    rows, steps = [], []
    k = 0
    for epoch in range(EPOCHS):
        for idx in rb.random_batches(len(m.corners), size, SEED, epoch):
            s = bh.step(obj, probe, d, idx, cg and epoch > 0, size)
            s.update(obj=obj, probe=probe, d=d, idx=idx, epoch=epoch)
            steps.append(s)
            rows.append(f"{k} {epoch} {len(idx)} "
                        + " ".join(f"{s[x]:.17e}" for x in ("beta", "alpha", "top", "bottom"))
                        + " " + " ".join(str(f) for f in idx))
            obj, probe, d = obj + s["d_obj"], probe + s["d_probe"], d + s["d_pos"]
            k += 1
    return rows, steps, (obj, probe, d)


def random_dirs(rng, obj, probe, nb):
    def c(shape, scale):
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return x * scale / np.abs(x).max()

    return (c(obj.shape, np.abs(obj).max()), c(probe.shape, np.abs(probe).max()),
            rng.uniform(-1, 1, (nb, 2)))


def check(m, obj, probe, d, idx, rng):
    """g_o, g_P and g_d of the eps-free F against central differences."""
    bh = BH(m)
    s = bh.state(obj, probe, d, idx, 0.0)
    g = bh.grads(s, obj.shape)
    u = random_dirs(rng, obj, probe, len(idx))
    print("  g_o, g_P, g_d vs central differences of F (eps 0), rel diff:")
    for h in (1e-4, 1e-5, 1e-6):
        def along(k):
            def f(t):
                x = [obj, probe, d.copy()]
                if k == 2:
                    x[2][idx] += t * u[2]
                else:
                    x[k] = x[k] + t * u[k]
                return loss(m, *x, idx)
            return (f(h) - f(-h)) / (2 * h)
        out = []
        for k, name in enumerate(("object", "probe", "offsets")):
            e = redot(g[k], u[k])
            out.append(f"{name} {abs(along(k) - e) / abs(e):.3e}")
        print(f"    h {h:.0e}: " + "  ".join(out))


def check_hess(m, step):
    """H(u, v) = d/dt <g(x + t v), u> along joint displacements, eps 0."""
    bh = BH(m)
    obj, probe, d, idx = step["obj"], step["probe"], step["d"], step["idx"]
    s = bh.state(obj, probe, d, idx, 0.0)
    dirs = step["dirs"]
    pairs = [("H(D, D)", dirs, dirs)]
    if step["prev"] is not None:
        g = bh.grads(s, obj.shape)
        pg = (g[0], bh.rho**2 * g[1], POS_RHO**2 * g[2])
        pairs.append(("H(Pg, D')", pg, step["prev"]))

    def dot(t, u, v):
        dd = d.copy()
        dd[idx] += t * v[2]
        a = bh.grads(bh.state(obj + t * v[0], probe + t * v[1], dd, idx, 0.0), obj.shape)
        return sum(redot(x, y) for x, y in zip(a, u))

    out = []
    for name, u, v in pairs:
        h_an = bh.hess(s, u, v)
        sym = abs(bh.hess(s, v, u) - h_an) / abs(h_an)
        fd = []
        for h in (1e-3, 1e-4, 1e-5):
            hh = h * abs(step["alpha"])
            fd.append(f"{abs((dot(hh, u, v) - dot(-hh, u, v)) / (2 * hh) - h_an) / abs(h_an):.2e}")
        out.append(f"{name} " + " ".join(fd) + f" (sym {sym:.1e})")
    print("    " + "  ".join(out))


def ptychi_step(m, iobs, wavelength, obj, probe):
    """pty-chi BHReconstructor's first step with positions optimizable (GD, one batch of every
    frame, PATCH, Fourier, pad_for_shift 0) in f64 on CPU from positions at the view corners:
    alpha and the changes in the engine's convention."""
    from torch.utils.data import DataLoader
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    nframe, ny, nx = iobs.shape
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
    pp = opts.probe_position_options
    pp.optimizable = True
    pp.rho = POS_RHO
    r = opts.reconstructor_options
    r.method = "GD"
    r.batch_size = nframe
    r.num_epochs = 1
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    r.forward_model_options.pad_for_shift = 0
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj[None].copy(),
        probe_data=(probe * m.q / root_n)[None].copy(),
        probe_position_y_px=m.corners[:, 0].astype(np.float64),
        probe_position_x_px=m.corners[:, 1].astype(np.float64),
    )
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset, batch_sampler=rb.Sampler(nframe, nframe, SEED))
    cls, out = type(rec), {}

    def alpha(*args, **kwargs):
        out["alpha"] = cls.calc_alpha_object_probe_positions(rec, *args, **kwargs)
        return out["alpha"]

    def apply(delta_o, delta_p, delta_pos):
        # The object and probe deltas as given: their sums with O and P lose their digits.
        out.update(d_obj=delta_o[0].numpy().copy(),
                   d_probe=delta_p[0].numpy() * root_n * np.conj(m.q))
        return cls.apply_updates(rec, delta_o, delta_p, delta_pos)

    rec.calc_alpha_object_probe_positions = alpha
    rec.apply_updates = apply
    task.run(1)
    out["alpha"] = float(out["alpha"])
    out["d_pos"] = task.get_data_to_cpu("probe_positions", as_numpy=True) - m.corners
    return out


def band_limited(m, shape, probe):
    """An object of frequencies j / n, |j| <= 7, on the n-pixel view grid (so every view is
    periodic) and the probe with P q low-passed to the same band: their product is unaliased."""
    ny, nx = m.n
    rng = np.random.default_rng(3)
    j = np.arange(-7, 8)
    amp = rng.standard_normal((15, 15)) + 1j * rng.standard_normal((15, 15))
    amp *= np.exp(-(j[:, None] ** 2 + j[None] ** 2) / 20)
    ey = np.exp(2j * np.pi * np.outer(np.arange(shape[0]), j) / ny)
    ex = np.exp(2j * np.pi * np.outer(j, np.arange(shape[1])) / nx)
    o = ey @ amp @ ex
    keep = (np.abs(np.fft.fftfreq(ny)) < 0.25)[:, None] & (np.abs(np.fft.fftfreq(nx)) < 0.25)
    return 1 + 0.5 * o / np.abs(o).max(), ifft2(fft2(probe * m.q, axes=AX) * keep, axes=AX) / m.q


def at_corners(m, iobs, wavelength, shape, probe):
    """pty-chi's first GD step at d = 0 from band_limited's state, where the two models' exit
    waves agree with their first derivatives (the Fourier derivative obeys the product rule on
    an unaliased product, and translating psi leaves F unchanged): its directions, top and alpha
    against the model's, to the eps's effect on the far field's zeros (about 1e-7)."""
    obj, probe = band_limited(m, shape, probe)
    d = np.zeros((len(m.corners), 2))
    size = len(m.corners)
    s = BH(m).step(obj, probe, d, np.arange(size), False, size)
    p = ptychi_step(m, iobs, wavelength, obj, probe)
    a = p["alpha"]
    got = (p["d_obj"] / a, p["d_probe"] / a, p["d_pos"] / (POS_STEP * a))
    top = -sum(redot(x, y) for x, y in zip(s["g"], got))
    print("  pty-chi's first GD step at d = 0 vs the model, rel (max, L2): "
          + "  ".join(f"{n} %.2e %.2e" % mg.rel(x, y)
                      for n, x, y in zip(("D_o", "D_P", "D_d"), got, s["dirs"]))
          + f"  top {abs(top - s['top']) / s['top']:.2e}"
          + f"  alpha {abs(a - s['alpha']) / abs(s['alpha']):.2e}")


def main():
    out = sys.argv[1]
    args = mg.load(out)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = mg.Model(*args)
    obj = obj0[0]
    nframe = len(m.corners)
    d0 = 0.5 + OFFSET * np.random.default_rng(1).uniform(-1, 1, (nframe, 2))
    print(f"object {obj.shape}, probe {probe.shape}, {nframe} frames, probe rho {RHO} "
          f"(rho_e {RHO * np.sqrt(np.prod(m.n)):g}), position rho {POS_RHO}, step {POS_STEP}")
    at_corners(m, iobs, wavelength, obj.shape, probe)
    rng = np.random.default_rng(0)
    for name, size, cg in CASES:
        print(f"--- {name}: batch {size}, method {'CG' if cg else 'GD'}, {EPOCHS} epochs")
        check(m, obj, probe, d0, rb.random_batches(nframe, size, SEED, 0)[0], rng)
        rows, steps, (o1, p1, d1) = run(m, obj, probe, d0, size, cg)
        print("  H vs central differences of <g(x + t v), u> (eps 0), rel diff at h = "
              "1e-3, 1e-4, 1e-5 alpha:")
        for s in steps:
            check_hess(m, s)
        print("  step  epoch  count  beta          alpha         top           bottom"
              "        F before -> after (batch)  max |dd|")
        for k, s in enumerate(steps):
            f0 = loss(m, s["obj"], s["probe"], s["d"], s["idx"])
            f1 = loss(m, s["obj"] + s["d_obj"], s["probe"] + s["d_probe"], s["d"] + s["d_pos"],
                      s["idx"])
            print(f"  {k:4d} {s['epoch']:5d}  {len(s['idx']):5d} {s['beta']: .6e} "
                  f"{s['alpha']: .6e} {s['top']: .6e} {s['bottom']: .6e}  "
                  f"{f0:.6e} -> {f1:.6e}  {np.abs(s['d_pos']).max():.3e}")
        full = np.arange(nframe)
        print(f"  F over all frames: {loss(m, obj, probe, d0, full):.10e} -> "
              f"{loss(m, o1, p1, d1, full):.10e}; max |d - d0| {np.abs(d1 - d0).max():.4f}")

        rows_out = {
            "obj": lp.cplx_rows(obj.shape, o1 - obj),
            "probe": lp.cplx_rows(probe.shape, p1 - probe),
            "offsets": (np.column_stack([np.arange(nframe), d0, d1]), ["%d"] + ["%.17e"] * 4),
        }
        for kind, (a, fmt) in rows_out.items():
            path = f"{out}/bh_pos_{name}_{kind}_f64.txt"
            np.savetxt(path, a, fmt=fmt)
            print(f"  wrote {path} ({len(a)} rows)")
        path = f"{out}/bh_pos_{name}_steps_f64.txt"
        pathlib.Path(path).write_text("".join(r + "\n" for r in rows))
        print(f"  wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
