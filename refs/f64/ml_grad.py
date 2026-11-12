"""Float64 reference of the maximum-likelihood (Poisson) object gradient at a fixed state.

    <engine>/target/release/examples/llk_trace <dir> dm 1 2 32 40 4
    micromamba run -n ptyrs-ref python -I refs/f64/ml_grad.py <dir> [--cuda]

The state is <dir>/scan.cxi and init.h5: common::dm_scenario(32, 40, 2, 2e5, 4.0) after
scale_obj_probe, i.e. object [1, 76, 80], probe [2, 32, 32], 40 frames, no masked pixel.
Writes <dir>/ml_grad_f64.txt, one row per object pixel in row-major order, `iy ix g_re g_im`:
the [ny, nx, 2] array over (Re O, Im O) of

    g = dL/dRe(O) + i dL/dIm(O) = 2 dL/dO*,  L = sum_{frames, iobs >= 0} (icalc - iobs ln icalc),

which is PyNX's Poisson LLK times nb_obs up to a constant (get_llk divides by nb_obs, so the
normalised LLK's gradient is g / nb_obs). The model is written here, not taken from PyNX:

    psi_m = FFT_ortho(fftshift(P_m O_view q)),  icalc = sum_m |psi_m|^2,
    q = exp(i pi/(lambda z) px^2 ((y - ny/2)^2 + (x - nx/2)^2)),  corners from get_view_coord,
    g[view] += 2 conj(P_m q) fftshift(IFFT_ortho((1 - iobs/icalc) psi_m)).

Printed checks: central differences of L over every object pixel (Re and Im); PyNX
cpu_operator's ML gradient promoted to f64; with --cuda, PyNX's CUDA gradient (float32, the
visible GPU); pty-chi's forward model under torch autograd in f64.
"""
import sys
import types

import h5py
import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.special import xlogy

from pynx.ptycho.ptycho import Ptycho, PtychoData
from pynx.ptycho.shape import get_view_coord
import pynx.ptycho.cpu_operator as cpu32

AX = (-2, -1)
STEPS = (1e-3, 1e-4, 1e-5)


def load(d):
    """As ptyrs-engine/scripts/compare_pynx.py: iobs DC-centred as in the CXI file."""
    with h5py.File(f"{d}/scan.cxi", "r") as f:
        raw = f["entry_1/data_1/data"][()]
        iobs = raw.astype(np.float64)
        if raw.dtype == np.uint16:
            iobs[raw == np.iinfo(np.uint16).max] = -1
        if "entry_1/instrument_1/detector_1/mask" in f:
            iobs[:, f["entry_1/instrument_1/detector_1/mask"][()] != 0] = -1
        t = f["entry_1/sample_1/geometry_1/translation"][()].astype(np.float64)
        energy = float(f["entry_1/instrument_1/source_1/energy"][()])
        distance = float(f["entry_1/instrument_1/detector_1/distance"][()])
        pixel = float(f["entry_1/instrument_1/detector_1/x_pixel_size"][()])
    wavelength = 12.3984e-10 / (energy / 1.60218e-16)
    with h5py.File(f"{d}/init.h5", "r") as f:
        obj, probe = f["object"][()], f["probe"][()]
    return iobs, (t[0], t[1]), distance, pixel, wavelength, obj, probe


def build(iobs, pos, distance, pixel, wavelength, obj, probe, f64):
    """As compare_pynx.build."""
    data = PtychoData(iobs=iobs, positions=pos, detector_distance=distance,
                      pixel_size_detector=pixel, wavelength=wavelength)
    p = Ptycho(probe=probe, obj=obj, data=data)
    if f64:
        p._obj = np.array(obj, dtype=np.complex128, order="C", copy=True)
        p._probe = np.array(probe, dtype=np.complex128, order="C", copy=True)
        p.data.iobs = np.ascontiguousarray(fftshift(iobs, axes=AX), dtype=np.float64)
        p.pixel_size_object = np.float64(data.pixel_size_object()[0])
    return p


class Model:
    def __init__(self, iobs, pos, distance, pixel, wavelength, obj, probe):
        nf, ny, nx = iobs.shape
        assert obj.shape[0] == 1, "one object mode"
        data = PtychoData(iobs=iobs, positions=pos, detector_distance=distance,
                          pixel_size_detector=pixel, wavelength=wavelength)
        px, py = data.pixel_size_object()
        self.corners = np.array([get_view_coord(obj.shape[-2:], (ny, nx), dx, dy)[::-1]
                                 for dx, dy in zip(data.posx / px, data.posy / py)])
        y, x = np.arange(ny) - ny // 2, np.arange(nx) - nx // 2
        self.q = np.exp(1j * (np.pi / (wavelength * distance) * px**2)
                        * (y[:, None] ** 2 + x[None, :] ** 2))
        self.probe = probe
        self.pq = probe * self.q
        self.iobs = fftshift(iobs, axes=AX)  # DC at [0, 0], as PyNX holds it
        self.valid = self.iobs >= 0
        self.n = ny, nx

    def views(self, obj):
        ny, nx = self.n
        return np.stack([obj[cy:cy + ny, cx:cx + nx] for cy, cx in self.corners])

    def psi(self, v):
        return fft2(fftshift(self.pq * v[..., None, :, :], axes=AX), norm="ortho")

    def icalc(self, v):
        return (np.abs(self.psi(v)) ** 2).sum(-3)

    def llk(self, obj):
        ic = self.icalc(self.views(obj))
        return np.where(self.valid, ic - xlogy(np.maximum(self.iobs, 0), ic), 0).sum()

    def scatter(self, per_view, shape):
        ny, nx = self.n
        out = np.zeros(shape, complex)
        for (cy, cx), a in zip(self.corners, per_view):
            out[cy:cy + ny, cx:cx + nx] += a
        return out

    def residual_back(self, obj, window=1.0):
        """IFFT_ortho(window * (1 - iobs/icalc) psi_m), PyNX's corner order, [nf, nm, ny, nx]."""
        psi = self.psi(self.views(obj))
        ic = (np.abs(psi) ** 2).sum(-3)
        r = np.where(self.valid, 1 - self.iobs / ic, 0) * window
        return ifft2(r[:, None] * psi, norm="ortho")

    def grad(self, obj):
        back = fftshift(self.residual_back(obj), axes=AX)
        return self.scatter(2 * (np.conj(self.pq) * back).sum(-3), obj.shape)

    def central(self, obj, h):
        """(L(O + h e) - L(O - h e)) / 2h for every pixel e, Re as real part, Im as imaginary.
        Only the frames covering e change, so each frame's terms are differenced per pixel."""
        ny, nx = self.n
        k = np.arange(ny * nx)
        num = np.zeros(obj.shape, complex)
        for f, (cy, cx) in enumerate(self.corners):
            v0 = obj[cy:cy + ny, cx:cx + nx]
            t, ok = self.iobs[f], self.valid[f]
            for unit in (1.0, 1j):
                dv = np.zeros((ny * nx, ny * nx), complex)
                dv[k, k] = h * unit
                dv = dv.reshape(-1, ny, nx)
                ip, im = self.icalc(v0 + dv), self.icalc(v0 - dv)
                di = ip - im
                d = np.where(ok, di - np.where(t > 0, t * np.log1p(di / im), 0), 0)
                num[cy:cy + ny, cx:cx + nx] += unit * (d.sum(AX) / (2 * h)).reshape(ny, nx)
        return num


def rel(a, b):
    """max |a - b| / max |b| and |a - b| / |b|."""
    return np.abs(a - b).max() / np.abs(b).max(), np.linalg.norm(a - b) / np.linalg.norm(b)


def fit(c, m):
    """complex k minimising |c - k m|, and |c - k m| / |c|."""
    k = np.vdot(m, c) / np.vdot(m, m)
    return k, np.linalg.norm(c - k * m) / np.linalg.norm(c)


def cpu_operator_f64():
    """PyNX cpu_operator with every float promoted to f64, ML without its leading ScaleObjProbe
    (so it starts from the state given) and with its debug copy of obj_grad switched on."""
    with open(cpu32.__file__) as f:
        src = f.read()
    src = src.replace("np.complex64", "np.complex128").replace("np.float32", "np.float64")
    for old, new in (
            ("if True:\n            p = ScaleObjProbe(absolute=True) * p",
             "if True:\n            pass"),
            ("if False:\n                # TODO: debug only\n"
             "                if self.update_object:\n",
             "if True:\n                if self.update_object:\n")):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    mod = types.ModuleType("pynx.ptycho.cpu_operator_f64_mlgrad")
    mod.__package__ = "pynx.ptycho"
    mod.__file__ = cpu32.__file__
    sys.modules[mod.__name__] = mod
    exec(compile(src, cpu32.__file__, "exec"), mod.__dict__)
    return mod


def pynx_cuda_grad(args):
    """PyNX CUDA Grad object gradient (float32) at the given state."""
    import pycuda.gpuarray as cua
    import pynx.ptycho.cu_operator as cuop

    class ObjGrad(cuop.CUOperatorPtycho):
        def op(self, p):
            p._cu_obj_grad = cua.empty_like(p._cu_obj)
            p._cu_background_grad = cua.zeros_like(p._cu_background)
            p = cuop.Grad(update_object=True, reg_fac_obj=None, reg_fac_probe=None) * p
            p._obj_grad = p._cu_obj_grad.get()
            return p

    p = build(*args, f64=False)
    p = ObjGrad() * p
    return p._obj_grad[0].astype(np.complex128)


def hann(ny, nx):
    """PyNX GradPoissonFourier window, corner order."""
    def axis(n):
        i = np.arange(n)
        return np.cos((i - n * (i >= n // 2)) * np.pi / (n - 1))
    return (axis(ny)[:, None] * axis(nx)[None, :]) ** 2


def ptychi_grads(m, iobs, wavelength, obj, probe):
    """pty-chi object .grad ([ny, nx, 2] real) of the sum L and of LSQML's mean Poisson NLL, on
    pty-chi's own forward model (probe P q / sqrt(ny nx), positions = the PyNX corners), and
    the value of the loss its Autodiff reconstructor builds for LossFunctions.POISSON."""
    import torch
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    nf, ny, nx = iobs.shape
    opts = api.LSQMLOptions()
    opts.data_options.wavelength_m = wavelength
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = True
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    opts.probe_options.optimizable = False
    opts.probe_position_options.optimizable = False
    r = opts.reconstructor_options
    r.batch_size = nf
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = api.Devices.CPU
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj.copy(),
        probe_data=(probe * m.q / np.sqrt(ny * nx))[None].copy(),
        probe_position_y_px=m.corners[:, 0].astype(np.float64),
        probe_position_x_px=m.corners[:, 1].astype(np.float64),
    )
    rec = task.reconstructor
    param = rec.parameter_group.object.tensor.data
    t = task.dataset.patterns.to(param.device)
    out = {}
    torch.set_grad_enabled(True)
    for name, loss in (
            ("sum", lambda y: (y - torch.xlogy(t, y))[t >= 0].sum()),
            ("lsqml_nll_mean", torch.nn.PoissonNLLLoss(log_input=False).__call__)):
        param.grad = None
        y = rec.forward_model.forward(torch.arange(nf))
        (loss(y, t) if name != "sum" else loss(y)).backward()
        gr = param.grad[0].detach().cpu().numpy()
        out[name] = gr[..., 0] + 1j * gr[..., 1]
    with torch.no_grad():
        y = rec.forward_model.forward(torch.arange(nf))
        out["ad_poisson_loss"] = float(torch.nn.PoissonNLLLoss()(y, t))
        out["log_input_false_loss"] = float(torch.nn.PoissonNLLLoss(log_input=False)(y, t))
    return out


def main():
    d = sys.argv[1]
    args = load(d)
    iobs, _, _, _, wavelength, obj0, probe = args
    m = Model(*args)
    obj = obj0[0]
    nf, ny, nx = iobs.shape
    nb_obs = int(m.valid.sum())
    g = m.grad(obj)
    L = m.llk(obj)
    const = np.where(m.iobs > 0, xlogy(m.iobs, m.iobs) - m.iobs, 0).sum()
    print(f"object {obj.shape}, probe {probe.shape}, {nf} frames, nb_obs {nb_obs}, "
          f"max|O| {np.abs(obj).max():.4f}, max|g| {np.abs(g).max():.6e}")
    llk = (L + const) / nb_obs
    op = cpu_operator_f64()
    p = op.LLK() * op.FT() * op.ObjProbe2Psi() * build(*args, f64=True)
    llk_pynx = p.get_llk(noise=None, norm=True)["poisson"]
    print(f"L = {L:.17e}; (L + const)/nb_obs = {llk:.17e}, PyNX cpu_operator f64 normalised "
          f"Poisson LLK {llk_pynx:.17e} (rel {abs(llk - llk_pynx) / llk_pynx:.1e})")

    print("central differences (max |num - g| / max |g|, rel L2):")
    for h in STEPS:
        print(f"  h {h:.0e}: %.3e %.3e" % rel(m.central(obj, h), g))

    p = build(*args, f64=True)
    p = op.ML(nb_cycle=1, update_object=True, update_probe=False, calc_llk=False,
              zero_phase_ramp=False) * p
    gp = p._obj_grad[0]
    k, res = fit(gp, g)
    print(f"PyNX cpu_operator f64 ML obj_grad: fit to g k = {k:.12f}, residual {res:.3e}; "
          "vs g/2 (max, L2) %.3e %.3e" % rel(gp, g / 2))
    # PyNX divides psi' while still in corner order by the centred q: fftshift(psi' / q).
    back = m.residual_back(obj)
    as_pynx = m.scatter((fftshift(back / m.q, axes=AX) * np.conj(m.probe)).sum(-3), obj.shape)
    print("  PyNX obj_grad vs fftshift(psi'/q) conj(P) computed here: %.3e %.3e"
          % rel(gp, as_pynx))

    if "--cuda" in sys.argv[2:]:
        gc = pynx_cuda_grad(args)
        wback = fftshift(m.residual_back(obj, hann(ny, nx)), axes=AX)
        cands = {
            "exact g": g,
            "Hann, conj(P) conj(q)": m.scatter((np.conj(m.pq) * wback).sum(-3), obj.shape),
            "Hann, conj(P) q": m.scatter((np.conj(m.probe) * m.q * wback).sum(-3), obj.shape),
        }
        print("PyNX CUDA Grad obj_grad (float32): complex k and residual of the fit c ~ k m:")
        for name, cand in cands.items():
            k, res = fit(gc, cand)
            print(f"  {name:24s} k = {k:.6f} residual {res:.3e}")

    pc = ptychi_grads(m, iobs, wavelength, obj0, probe)
    k, res = fit(pc["sum"], g)
    print(f"pty-chi autograd .grad of L: k = {k:.12f}, residual {res:.3e}; vs g %.3e %.3e"
          % rel(pc["sum"], g))
    k, res = fit(pc["lsqml_nll_mean"], g)
    print(f"pty-chi PoissonNLLLoss(log_input=False) mean: k = {k:.6e} "
          f"(1/(nf ny nx) = {1 / (nf * ny * nx):.6e}), residual {res:.3e}")
    print(f"pty-chi Autodiff LossFunctions.POISSON loss (PoissonNLLLoss(), log_input=True): "
          f"{pc['ad_poisson_loss']}; log_input=False: {pc['log_input_false_loss']}")

    rows = np.column_stack([np.repeat(np.arange(obj.shape[0]), obj.shape[1]),
                            np.tile(np.arange(obj.shape[1]), obj.shape[0]),
                            g.real.ravel(), g.imag.ravel()])
    np.savetxt(f"{d}/ml_grad_f64.txt", rows, fmt=["%d", "%d", "%.17e", "%.17e"])
    print(f"wrote {d}/ml_grad_f64.txt ({len(rows)} rows)")


if __name__ == "__main__":
    main()
