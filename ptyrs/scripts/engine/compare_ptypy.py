"""PtyPy DM or RAAR on the scan, start and algorithm written by `cargo run --example ptypy_compare --
<dir> prepare <dm|raar>`.

Usage: PYTHONPATH=<ptypy checkout> python -P compare_ptypy.py <dir> <niter> [--ptyrs-guard]

Reads <dir>/scan.cxi, init.h5, views.txt and algo.txt, and runs PtyPy's CPU DM or RAAR (Full
model, double precision) from the same object and probe on the same integer views, writing the
state after each iteration i = 0..niter to <dir>/ptypy/ptypy_<i>.ptyr for `ptypy_compare <dir>
compare`.

PtyPy is set to ptyrs's run: an exact Fourier projection (fourier_relax_factor 0), probe updated
from the first iteration, one object-then-probe overlap pass, no probe support, and DM alpha 1
(ptyrs's plain DM, alpha 0) or ptyrs's RAAR beta. Each view is given at its ptyrs centre
(corner + n/2) times the object pixel, so ptyrs object pixel (0, 0) sits at the origin and a
storage's `_origin` maps PtyPy pixels onto ptyrs pixels.

PtyPy's RAAR is not ptyrs's (PyNX's) for any beta. PtyPy takes the next exit waves as
beta exit + (1 - 2 beta) PO + beta P_F(2 PO - exit) and fits object and probe to them; ptyrs
sends psi' = beta/2 (2 PO - psi) + (1 - beta/2) P_F(psi) to the Fourier projection, psi being
the previous cycle's (PO in cycle 0), and fits object and probe to P_F(psi'). The two agree
only at the start, so their curves are compared side by side.

PtyPy's inertia is a constant times the mean diffraction power (object) or the frame count
(probe), where ptyrs (PyNX) scales it by the largest normalisation of the update; no constant
matches that, so PtyrsInertia sets PtyPy's inertia before every update to the value that gives
ptyrs's regulariser.

PtyPy guards the magnitude ratio as sqrt(Iobs) / (|f| + 1e-10), ptyrs as sqrt(Iobs) / max(|f|,
1e-12); the 1e-10 moves the LLK by about 1e-10 in the first iteration, which DM amplifies.
--ptyrs-guard swaps in ptyrs's guard to compare the rest at rounding level.
"""
import os
import sys

import h5py
import numpy as np

from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.core.data import PtyScan
from ptypy.core.geometry import Geo
from ptypy.engines import projectional
from ptypy.engines import register as register_engine
from ptypy.engines.projectional import DM, RAAR
from ptypy.experiment import register as register_scan

# ptyrs DmParams::default() and RaarParams::default() obj_inertia and probe_inertia (PyNX).
OBJ_INERTIA, PROBE_INERTIA = 0.01, 0.001

SCAN = {}


@register_scan()
class PtyrsScan(PtyScan):
    """
    Frames and positions handed over in SCAN.

    Defaults:

    [name]
    default = PtyrsScan
    type = str
    help =
    """

    def load_positions(self):
        return SCAN["positions"]

    def load(self, indices):
        iobs = SCAN["iobs"]
        return ({k: np.maximum(iobs[k], 0) for k in indices}, {},
                {k: (iobs[k] >= 0).astype(float) for k in indices})


class PtyrsInertia:
    """
    Object and probe regularisers of ptyrs: inertia times the largest value of the update's
    normalisation (sum of |P|^2 over views and modes for the object, of |O|^2 over views for
    the probe).
    """

    def _max_norm(self, nrm, view, weight):
        nrm.fill(0.0)
        for pod in self.pods.values():
            nrm[getattr(pod, view)] += u.abs2(getattr(pod, weight))
        return max(s.data.max() for s in nrm.storages.values())

    def object_update(self):
        # PtyPy cfact = object_inertia * mean_power.
        self.p.object_inertia = OBJ_INERTIA * self._max_norm(self.ob_nrm, "ob_view", "probe") / self.mean_power
        super().object_update()

    def probe_update(self):
        # PtyPy cfact = probe_inertia * len(s.views) / nmodes.
        (s,) = self.pr.storages.values()
        m = self._max_norm(self.pr_nrm, "pr_view", "object")
        self.p.probe_inertia = PROBE_INERTIA * m * s.data.shape[0] / len(s.views)
        return super().probe_update()


@register_engine()
class DMPtyrsInertia(PtyrsInertia, DM):
    """
    DM with ptyrs's regularisers.

    Defaults:

    [name]
    default = DMPtyrsInertia
    type = str
    help =
    """


@register_engine()
class RAARPtyrsInertia(PtyrsInertia, RAAR):
    """
    RAAR with ptyrs's regularisers.

    Defaults:

    [name]
    default = RAARPtyrsInertia
    type = str
    help =
    """


def ptyrs_guard_projection(diff_view, a, b, c, pbound):
    """PtyPy projection_update_generalized for pbound 0 (fourier_relax_factor 0), with ptyrs's guard."""
    assert pbound == 0.0, pbound
    f = {}
    af2 = np.zeros_like(diff_view.data)
    fmask = diff_view.pod.mask.astype(diff_view.data.dtype)
    for name, pod in diff_view.pods.items():
        if pod.active:
            f[name] = pod.fw((1 - c) * pod.exit + c * pod.probe * pod.object)
            af2 += pod.downsample(u.abs2(f[name]))
    fm = (1 - fmask) + fmask * np.sqrt(np.abs(diff_view.data)) / np.maximum(np.sqrt(af2), 1e-12)
    for name, pod in diff_view.pods.items():
        if pod.active:
            pod.exit += b * pod.bw(pod.upsample(fm) * f[name]) + a * pod.probe * pod.object - (a + b) * pod.exit
    return 0.0, 0.0


def load(d):
    with h5py.File(f"{d}/scan.cxi", "r") as f:
        raw = f["entry_1/data_1/data"][()]
        iobs = raw.astype(np.float64)
        if raw.dtype == np.uint16:
            iobs[raw == np.iinfo(np.uint16).max] = -1
        if "entry_1/instrument_1/detector_1/mask" in f:
            iobs[:, f["entry_1/instrument_1/detector_1/mask"][()] != 0] = -1
        energy = float(f["entry_1/instrument_1/source_1/energy"][()])
        distance = float(f["entry_1/instrument_1/detector_1/distance"][()])
        pixel = float(f["entry_1/instrument_1/detector_1/x_pixel_size"][()])
    # ptyrs Geometry::wavelength_from_energy_joule.
    wavelength = 12.3984e-10 / (energy / 1.60218e-16)
    with h5py.File(f"{d}/init.h5", "r") as f:
        obj, probe = f["object"][()], f["probe"][()]
    corners = np.loadtxt(f"{d}/views.txt", dtype=int, ndmin=2)
    algo, param = open(f"{d}/algo.txt").read().split()
    return iobs, distance, pixel, wavelength, obj, probe, corners, algo, float(param)


def main():
    d, niter = sys.argv[1], int(sys.argv[2])
    if sys.argv[3:] == ["--ptyrs-guard"]:
        projectional.projection_update_generalized = ptyrs_guard_projection
    elif sys.argv[3:]:
        sys.exit(__doc__)
    iobs, distance, pixel, wavelength, obj, probe, corners, algo, param = load(d)
    out = f"{d}/ptypy"
    os.makedirs(out, exist_ok=True)
    n = iobs.shape[-1]
    psize = wavelength * distance / (n * pixel)
    SCAN.update(iobs=iobs, positions=(corners + n // 2) * psize)

    p = u.Param()
    p.verbose_level = "warning"
    p.data_type = "double"
    p.io = u.Param(home=out, rfile=None)
    p.io.autosave = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.io.interaction = u.Param(active=False)
    p.scans = u.Param()
    p.scans.S = u.Param(name="Full", propagation="farfield")
    p.scans.S.coherence = u.Param(num_probe_modes=probe.shape[0], num_object_modes=1)
    p.scans.S.data = u.Param(name="PtyrsScan", shape=n, psize=pixel, distance=distance,
                             energy=Geo._keV2m / wavelength, center="fftshift", auto_center=False,
                             orientation=None, save=None)
    p.engines = u.Param()
    p.engines.e0 = u.Param(numiter=niter, numiter_contiguous=1, probe_update_start=0,
                           overlap_max_iterations=1, update_object_first=True,
                           fourier_relax_factor=0.0, probe_support=None,
                           compute_log_likelihood=False)
    if algo == "dm":
        p.engines.e0.update(name="DMPtyrsInertia", alpha=param)
    elif algo == "raar":
        p.engines.e0.update(name="RAARPtyrsInertia", beta=param)
    else:
        sys.exit(f"{d}/algo.txt names {algo}, expected dm or raar")
    P = Ptycho(p, level=4)

    (ob,) = P.obj.storages.values()
    (pr,) = P.probe.storages.values()
    shift = -ob.origin / ob.psize
    off = np.round(shift).astype(int)
    assert np.abs(shift - off).max() < 1e-6, shift
    for pod in P.pods.values():
        k = pod.di_view.layer
        assert np.array_equal(pod.di_view.data, np.maximum(iobs[k], 0)), k
        assert np.array_equal(pod.ob_view.dlow, corners[k] + off), (k, pod.ob_view.dlow, corners[k])
    assert pr.data.shape == probe.shape, (pr.data.shape, probe.shape)

    # Every view lies inside both arrays, so the overlap of the two frames carries every value
    # the engine reads; PtyPy pixels outside it stay at PtyPy's initial object and are never read.
    lo = np.maximum(off, 0)
    hi = np.minimum(np.array(obj.shape[1:]) + off, ob.data.shape[1:])
    ob.data[0, lo[0]:hi[0], lo[1]:hi[1]] = obj[0, lo[0] - off[0]:hi[0] - off[0], lo[1] - off[1]:hi[1] - off[1]]
    pr.data[:] = probe
    for pod in P.pods.values():
        pod.exit = pod.probe * pod.object

    calc = sum(u.abs2(pod.fw(pod.probe * pod.object)).sum() for pod in P.pods.values())
    print(f"offset {off.tolist()}, sum calc / sum obs at the start {calc / iobs[iobs >= 0].sum():.15g}")

    eng = P.engines["e0"]
    P.runtime.iter_info = []
    eng.initialize()
    P.new_data = [(s.label, s) for s in P.diff.S.values()]
    eng.prepare()
    P.save_run(alt_file=f"{out}/ptypy_{0:04d}.ptyr", kind="minimal")
    for i in range(1, niter + 1):
        eng.iterate(1)
        P.save_run(alt_file=f"{out}/ptypy_{i:04d}.ptyr", kind="minimal")
        print(f"iteration {i}", flush=True)


if __name__ == "__main__":
    main()
