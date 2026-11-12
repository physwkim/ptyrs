"""Loading, PyNX<->pty-chi conversion and PyNX-definition LLK for the M3 pty-chi reference runs.

Datasets come from `cargo run --release -p ptyrs --example gen_m3 -- refs/ptychi/data`:
`<name>.cxi` (u16 counts, DC at [ny/2, nx/2], positions in metres), `<name>.json` (truth
metadata, `llk_truth`), `<name>_truth.h5` and `<name>_init.h5` (object [1,nyo,nxo] and probe
[nmode,ny,nx] in PyNX normalisation).

Conventions (checked by check_llk.py):
- PyNX: icalc = sum_modes |FFT_ortho(fftshift(P * O_view * q))|^2, q = exp(i*pi/(lambda*z) *
  px^2 * ((x-nx/2)^2 + (y-ny/2)^2)), view corner = (No-N)//2 + round(pos/px).
- pty-chi: icalc = sum_modes |fft2(P' * O_view)|^2 (unnormalised, unshifted), view corner =
  pos + No/2 - N/2 for even No (SUPPORT origin), data ifftshifted on load (fft_shift=True).
- Hence the object is the same array and P' = P * q / sqrt(ny*nx); positions are PyNX's
  preprocessed positions in object pixels.

PyNX LLK (cpu_operator.LLK, normalised by nb_obs = #(iobs >= 0) as in Ptycho.get_llk):
  poisson   = sum(icalc - iobs + iobs*ln(iobs/icalc)) [iobs>0] + sum(icalc) [iobs=0]
  gaussian  = sum((iobs - icalc)^2 / (iobs + 1))
  euclidian = 4 * sum((sqrt(iobs) - sqrt(icalc))^2)
No conversion of pty-chi's logged loss is used: evaluate() recomputes icalc for all frames at
one state with the reconstructor's own forward model and applies the formulas above in float64.
The logged loss relates only approximately. pty-chi's displayed loss (MSE_SQRT, LSQML and BH) is
L = mean((sqrt(icalc+1e-7) - sqrt(iobs+1e-7))^2), so euclidian_pynx ~= 4*L over all frames at one
state, biased low by pixels where icalc and iobs are both near 1e-7 (8.3e-4 relative at the truth
state, check_llk.json); and the value logged per epoch is the mean of per-batch L taken before
each batch update, i.e. not one state. LSQML's Poisson nll is L_p = mean(icalc -
iobs*ln(icalc+1e-8)), so poisson_pynx = L_p + mean(iobs*ln(iobs) - iobs) up to the 1e-8.

The Poisson LLK is unreliable in float32 where icalc falls to rounding level while iobs > 0.
The float32 relative error of icalc reaches 1% at 1e-10 of the frame-mean icalc and 100% at
1e-14 (fermat_clean). The smooth init probe's far field falls to 1e-25, so at the init state 74%
of pty-chi's Poisson LLK comes from pixels below 1e-10 of the mean (fermat_clean and
raster_clean, check_llk.json), and PyNX CPU, pty-chi and float64 give inf, 18.3 and 21.7.
evaluate() reports that share as poisson_floor_share; overlay Poisson only where it is ~0 (it is 0
after a few epochs). Gaussian and Euclidian agree to 1e-6
at every state checked.
"""

import json
import os
import time

import h5py
import numpy as np
import torch

DET = "entry_1/instrument_1/detector_1/data"
TRANSLATION = "entry_1/sample_1/geometry_1/translation"
MSE_SQRT_EPS = 1e-7
POISSON_NLL_EPS = 1e-8
# icalc below this fraction of its frame mean has > 1% float32 rounding error (docstring).
FLOOR = 1e-10

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_modes(path):
    with h5py.File(path, "r") as f:
        return f["object"][()], f["probe"][()]


def pynx_positions(x_m, y_m, frame, geometry):
    """PyNX PtychoData position preprocessing, in object pixels as [n, 2] (y, x)."""
    from pynx.ptycho.ptycho import PtychoData

    ny, nx = frame
    d = PtychoData(
        iobs=np.zeros((1, ny, nx), np.float32),
        positions=(np.asarray(x_m, np.float64), np.asarray(y_m, np.float64)),
        detector_distance=geometry["distance_m"],
        pixel_size_detector=geometry["detector_pixel_m"],
        wavelength=geometry["wavelength_m"],
    )
    px, py = d.pixel_size_object()
    return np.stack([np.asarray(d.posy) / py, np.asarray(d.posx) / px], axis=1)


class Dataset:
    def __init__(self, name, data_dir=DATA):
        self.name = name
        self.dir = data_dir
        with open(os.path.join(data_dir, name + ".json")) as f:
            self.meta = json.load(f)
        g = self.meta["geometry"]
        self.geometry = g
        self.frame = tuple(g["frame"])
        with h5py.File(os.path.join(data_dir, self.meta["files"]["cxi"]), "r") as f:
            self.counts = f[DET][()]
            t = f[TRANSLATION][()].astype(np.float64)
        if self.counts.max() >= 65535:
            raise ValueError(f"{name}: masked pixels; the LLK here assumes none")
        self.x_m, self.y_m = t[0], t[1]
        self.pos_px = pynx_positions(self.x_m, self.y_m, self.frame, g)
        want = np.asarray(self.meta["positions"]["nominal_pp_px"])
        err = np.abs(self.pos_px - want).max()
        if err > 1e-3:
            raise ValueError(f"{name}: CXI positions differ from the generator's by {err} px")
        self.pixel = g["pixel_size_object_m"][1]
        self.nframe = len(self.counts)

    def modes(self, which):
        o, p = load_modes(os.path.join(self.dir, self.meta["files"][which]))
        if o.shape[-2] % 2 or o.shape[-1] % 2:
            raise ValueError("odd object shape: pty-chi and PyNX corners differ by one pixel")
        return o, p

    def quad_phase(self):
        """PyNX far-field quadratic phase (calc_quadratic_phase_factor, fftshifted back)."""
        ny, nx = self.frame
        f = np.pi / (self.geometry["wavelength_m"] * self.geometry["distance_m"]) * self.pixel**2
        y = np.arange(ny) - ny // 2
        x = np.arange(nx) - nx // 2
        return np.exp(1j * f * (y[:, None] ** 2 + x[None, :] ** 2))

    def probe_to_ptychi(self, p):
        return p * self.quad_phase() / np.sqrt(np.prod(self.frame))

    def probe_from_ptychi(self, p):
        return p * np.conj(self.quad_phase()) * np.sqrt(np.prod(self.frame))


def llk_sums(icalc, iobs):
    """Float64 sums of the PyNX LLK terms and pty-chi's native losses over one chunk."""
    icalc = icalc.double()
    iobs = iobs.double()
    s_obs, s_calc = iobs.sqrt(), icalc.sqrt()
    poisson = icalc - iobs + torch.xlogy(iobs, iobs) - torch.xlogy(iobs, icalc)
    floor = (icalc < FLOOR * icalc.mean(dim=(-2, -1), keepdim=True)) & (iobs > 0)
    return {
        "poisson": poisson.sum(),
        "poisson_floor": poisson[floor].sum(),
        "gaussian": ((iobs - icalc) ** 2 / (iobs + 1)).sum(),
        "euclidian": 4 * ((s_obs - s_calc) ** 2).sum(),
        "mse_sqrt": (((icalc + MSE_SQRT_EPS).sqrt() - (iobs + MSE_SQRT_EPS).sqrt()) ** 2).sum(),
        "poisson_nll": (icalc - iobs * torch.log(icalc + POISSON_NLL_EPS)).sum(),
        "nb_obs": (iobs >= 0).sum(),
        "npix": torch.tensor(iobs.numel()),
    }


@torch.no_grad()
def evaluate(task, chunk=64):
    """PyNX LLK of the current state, all frames, with the reconstructor's own forward model."""
    rec = task.reconstructor
    fm = rec.forward_model
    saved = fm.intermediate_variables
    patterns = task.dataset.patterns
    n = patterns.shape[0]
    tot = None
    for s in range(0, n, chunk):
        idx = torch.arange(s, min(n, s + chunk))
        y = fm.forward(idx)
        part = llk_sums(y, patterns[idx.to(patterns.device)].to(y.device))
        tot = part if tot is None else {k: tot[k] + part[k] for k in tot}
    fm.intermediate_variables = saved
    tot = {k: v.item() for k, v in tot.items()}
    nb, npix = tot.pop("nb_obs"), tot.pop("npix")
    out = {k: tot[k] / nb for k in ("poisson", "gaussian", "euclidian")}
    out["poisson_floor_share"] = tot["poisson_floor"] / tot["poisson"]
    out["native_mse_sqrt"] = tot["mse_sqrt"] / npix
    out["native_poisson_nll"] = tot["poisson_nll"] / npix
    out["nb_obs"] = nb
    return out


def run_epochs(task, epochs, log=None, extra=None):
    """Run one epoch at a time; wall time excludes evaluate() and `extra` and is synchronised.
    `extra(task)` returns more columns for each row."""
    more = (lambda t: {}) if extra is None else extra
    curve = [dict(epoch=0, wall_s=0.0, logged_loss=None, **evaluate(task), **more(task))]
    wall = 0.0
    for ep in range(1, epochs + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        task.run(n_epochs=1)
        torch.cuda.synchronize()
        wall += time.perf_counter() - t0
        logged = float(task.reconstructor.loss_tracker.table["loss"].iloc[-1])
        row = dict(epoch=ep, wall_s=wall, logged_loss=logged, **evaluate(task), **more(task))
        curve.append(row)
        if log is not None:
            log(row)
        # Gaussian, not Poisson: Poisson is inf wherever float32 icalc underflows to 0 (docstring).
        if not np.isfinite(row["gaussian"]):
            break
    return curve


def finite(x):
    """Non-finite floats to None, so the JSON stays strict (no Infinity/NaN tokens)."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(finite(obj), f, indent=1, allow_nan=False)


TARGET_FACTORS = (1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0)


def time_to_target(curve, truth):
    """First epoch and wall time at which each PyNX LLK is <= factor * its truth value; None if
    never. Poisson rows with poisson_floor_share > 1e-3 are skipped (rounding-dominated)."""
    out = {}
    for key in ("poisson", "gaussian", "euclidian"):
        rows = [r for r in curve if key != "poisson" or r["poisson_floor_share"] <= 1e-3]
        out[key] = {}
        for f in TARGET_FACTORS:
            hit = next((r for r in rows if r[key] <= f * truth[key]), None)
            out[key][f"{f:g}"] = None if hit is None else {"epoch": hit["epoch"],
                                                          "wall_s": hit["wall_s"]}
    return out
