"""Float64 model of ML (Poisson conjugate gradient) with the AP/DM position step.

    <engine>/target/release/examples/engine_llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/ml_pos.py <dir>

No package refines positions this way (PyNX ML's update_pos moves the scan positions), so the
engine's step is checked against this model of it. The state is the one rpie_pos.py reads
(object [1, 76, 80], probe [2, 32, 32], 40 frames) with its offsets d_k = OFFSET times numpy
default_rng(1) uniform in [-1, 1). In the engine's convention (lsqml_pos_step.py's T_d, F,
q): ScaleObjProbe(absolute=True) under the offsets, then per cycle, all from the cycle-start
state,

    W_km = T_dk(P_m q),  psi_km = F(W_km V_k),  icalc = sum_m |psi_km|^2
    x_km = F^-1(psi_km sqrt(iobs) / max(sqrt(icalc), 1e-12))     (the magnitude projection)
    num_ka = Re sum_m conj(x_km - W_km V_k) V_k dW_kma,  den_ka = sum_m |V_k dW_kma|^2,
        dW_kma = d W_km / d d_ka = F_grid^-1(F_grid(P_m q) e^(-2 pi i f.d_k) (-2 pi i f_a))
    g_O = scatter(sum_m conj(W_km) b_km),  g_P = conj(q) sum_k T_-dk(conj(V_k) b_km),
        b_km = F^-1(2 (1 - iobs / max(icalc, 1e-12)) psi_km)
    beta = max(0, Re<g, g - g_last> / |g_last|^2) (0 on the first cycle), D = beta D - g
    gamma = num / den, with A = F(W dO + T_d(dP q) V), B = F(T_d(dP q) dO),
        r = sum_m Re(conj(psi) A), s = sum_m |A|^2 + 2 Re(conj(psi) B), f = 1 - iobs / icalc,
        num = -sum 2 r f,  den = sum 2 s f + iobs (2 r / icalc)^2
    O += gamma D_O, P += gamma D_P   (or D = -g where gamma is not finite)
    per axis: s_k = clip(0.3 num_ka / (den_ka + max_k den_ka), +-0.1), d += s - mean s

Writes, for CYCLES cycles updating object and probe:
    <dir>/ml_pos_obj_f64.txt      `iy ix re im`, the object change
    <dir>/ml_pos_probe_f64.txt    `m iy ix re im`, the probe change
    <dir>/ml_pos_offsets_f64.txt  `k dy dx dy' dx'`, every frame
Printed, at the start state: num_ka against minus half the central difference of
sum |x - W V|^2 over d_ka at fixed x, and gamma's num and den against minus the first and
the second central difference of the LLK along D = -g.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.fft import fft2, fftshift, ifft2

AX = (-2, -1)
POS_STEP = 0.3
POS_LIMIT = 0.1
OFFSET = 0.4
CYCLES = 4


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


class State:
    """Moved probe, windows and far field of (obj, probe) under the offsets d."""

    def __init__(self, m, obj, probe, d):
        self.pw = lp.move(np.broadcast_to(probe * m.q, (len(d), *probe.shape)), d)
        self.v = ls.views(obj, m.corners, m.n)
        self.far = ls.far(self.pw, self.v)
        self.icalc = (np.abs(self.far) ** 2).sum(1)


def back(far):
    return fftshift(ifft2(far, axes=AX, norm="ortho"), axes=AX)


def llk(m, s):
    ic = s.icalc
    return np.where(m.valid, ic - m.iobs * np.log(np.where(m.valid, ic, 1)), 0).sum()


def dmoved(m, probe, d):
    """(dW/dd_y, dW/dd_x) [nf, nm, ny, nx] of W = T_d(P q)."""
    ny, nx = probe.shape[-2:]
    spec = fft2(probe * m.q, axes=AX)[None]
    fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
    ph = np.exp(-2j * np.pi * (fy[None, :, None] * d[:, 0, None, None]
                               + fx[None, None, :] * d[:, 1, None, None]))[:, None]
    return [ifft2(spec * ph * (-2j * np.pi * f), axes=AX)
            for f in (fy[:, None] + 0 * fx[None], 0 * fy[:, None] + fx[None])]


def position_sums(m, s, probe, d):
    """[nf, 2] num and den from the magnitude projection of s's exit waves."""
    r = np.where(m.valid, np.sqrt(np.maximum(m.iobs, 0)) / np.maximum(np.sqrt(s.icalc), 1e-12),
                 1)
    x = back(s.far * r[:, None])
    c = x - s.pw * s.v[:, None]
    num, den = [], []
    for dw in dmoved(m, probe, d):
        odw = s.v[:, None] * dw
        num.append(np.real(np.conj(c) * odw).sum((1, 2, 3)))
        den.append((np.abs(odw) ** 2).sum((1, 2, 3)))
    return np.stack(num, 1), np.stack(den, 1), x


def position_step(d, num, den):
    mx = den.max(0)
    tot = den + mx
    step = np.where(tot > 0, np.clip(POS_STEP * num / np.where(tot > 0, tot, 1), -POS_LIMIT,
                                     POS_LIMIT), 0)
    return d + step - step.mean(0)


def grads(m, s, d, obj, probe):
    ic = np.maximum(s.icalc, 1e-12)
    b = back(np.where(m.valid, 2 * (1 - m.iobs / ic), 0)[:, None] * s.far)
    g_o = ls.scatter((np.conj(s.pw) * b).sum(1), m.corners, obj.shape)
    g_p = np.conj(m.q) * lp.move(np.conj(s.v)[:, None] * b, -d).sum(0)
    return g_o, g_p


def gamma(m, s, d, d_obj, d_probe):
    dv = ls.views(d_obj, m.corners, m.n)
    dpw = lp.move(np.broadcast_to(d_probe * m.q, (len(d), *d_probe.shape)), d)
    a = ls.far(s.pw, dv) + ls.far(dpw, s.v)
    b = ls.far(dpw, dv)
    po = s.far
    ic = np.maximum(s.icalc, 1e-12)
    r = np.real(np.conj(po) * a).sum(1)
    q = (np.abs(a) ** 2 + 2 * np.real(np.conj(po) * b)).sum(1)
    f = 1 - m.iobs / ic
    num = np.where(m.valid, -2 * r * f, 0).sum()
    den = np.where(m.valid, 2 * q * f + m.iobs * (2 * r / ic) ** 2, 0).sum()
    return num, den


def scale_obj_probe(m, obj, probe, d):
    s = State(m, obj, probe, d)
    k = np.sqrt(np.where(m.valid, m.iobs, 0).sum() / np.where(m.valid, s.icalc, 0).sum())
    os_, ps = np.abs(obj).sum(), np.abs(probe).sum()
    return obj * np.sqrt(ps / os_ * k), probe * np.sqrt(os_ / ps * k)


def checks(m, obj, probe, d):
    s = State(m, obj, probe, d)
    num, _, x = position_sums(m, s, probe, d)
    h = 1e-5
    fd = np.zeros_like(num)
    for a in range(2):
        e = np.zeros_like(d)
        e[:, a] = h
        cost = [(np.abs(x - State(m, obj, probe, d + sgn * e).pw * s.v[:, None]) ** 2)
                .sum((1, 2, 3)) for sgn in (1, -1)]
        fd[:, a] = -(cost[0] - cost[1]) / (4 * h)
    print("  position num vs -1/2 d|x - W V|^2/dd (max rel): %.3e" % mg.rel(num, fd)[0])
    g_o, g_p = grads(m, s, d, obj, probe)
    gn, gd = gamma(m, s, d, -g_o, -g_p)
    for h in (1e-2, 1e-3):
        h *= gn / gd
        f = [llk(m, State(m, obj - u * g_o, probe - u * g_p, d)) for u in (-h, 0, h)]
        d1, d2 = (f[2] - f[0]) / (2 * h), (f[2] - 2 * f[1] + f[0]) / h**2
        print("  gamma num vs -dL/du, den vs d2L/du2 along -g, h = %.1e gamma (rel): "
              "%.3e %.3e" % (h * gd / gn, abs(gn + d1) / abs(gn), abs(gd - d2) / abs(gd)))


def model(m, obj, probe, d, ncycle):
    obj, probe = scale_obj_probe(m, obj, probe, d)
    last, dirs = None, [np.zeros_like(obj), np.zeros_like(probe)]
    for _ in range(ncycle):
        s = State(m, obj, probe, d)
        num, den, _ = position_sums(m, s, probe, d)
        g = grads(m, s, d, obj, probe)
        beta = 0.0
        if last is not None:
            bn = sum(np.real(np.conj(a) * (a - b)).sum() for a, b in zip(g, last))
            bd = sum((np.abs(b) ** 2).sum() for b in last)
            beta = bn / bd
            beta = beta if np.isfinite(beta) and beta >= 0 else 0.0
        dirs = [beta * x - y for x, y in zip(dirs, g)]
        gn, gd = gamma(m, s, d, *dirs)
        t = gn / gd
        if np.isfinite(t):
            obj, probe = obj + t * dirs[0], probe + t * dirs[1]
        else:
            dirs = [-y for y in g]
        last = g
        print(f"  beta {beta:.6e} gamma {t:.6e} LLK {llk(m, s):.10e}")
        d = position_step(d, num, den)
    return obj, probe, d


def main():
    out = sys.argv[1]
    args = mg.load(out)
    iobs, _, _, _, _, obj0, probe = args
    m = mg.Model(*args)
    obj = obj0[0]
    nf = len(iobs)
    d0 = OFFSET * np.random.default_rng(1).uniform(-1, 1, (nf, 2))
    print("--- checks at the start state")
    checks(m, *scale_obj_probe(m, obj, probe, d0), d0)
    print(f"--- {CYCLES} cycles from offsets within {np.abs(d0).max():.3f}")
    o1, p1, d1 = model(m, obj, probe, d0, CYCLES)
    print(f"  max |offset change| {np.abs(d1 - d0).max():.4f}")
    rows = {
        "obj": lp.cplx_rows(obj.shape, o1 - obj),
        "probe": lp.cplx_rows(probe.shape, p1 - probe),
        "offsets": (np.column_stack([np.arange(nf), d0, d1]), ["%d"] + ["%.17e"] * 4),
    }
    for name, (a, fmt) in rows.items():
        path = f"{out}/ml_pos_{name}_f64.txt"
        np.savetxt(path, a, fmt=fmt)
        print(f"  wrote {path} ({len(a)} rows)")


if __name__ == "__main__":
    main()
