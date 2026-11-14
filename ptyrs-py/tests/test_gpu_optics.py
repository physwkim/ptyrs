"""GpuPtycho under near-field Optics against the GPU engine's own runs from the same scan and
start, `examples/engine_reference.rs`' `gpu_optics.h5`: an angular-spectrum propagator and a
Fresnel transform forward and back. Every algorithm the engine ran leaves its object, probe
and LLK records, exactly, as one device runs both; one it refused raises its error text. A
Fresnel transform forward has no plane stage and runs every algorithm; under the other two
ML alone is refused."""

import os

import h5py
import numpy as np
import pytest

import ptyrs
from conftest import DETECTOR_PIXEL, DEVICE, DISTANCE, MODULE_DIR, WAVELENGTH

NEPOCH = 3
RUNS = {
    "ap": lambda g: g.ap(NEPOCH, calc_llk=1),
    "dm": lambda g: g.dm(NEPOCH, calc_llk=1),
    "raar": lambda g: g.raar(NEPOCH, calc_llk=1),
    "ml": lambda g: g.ml(NEPOCH, calc_llk=1),
    "rpie": lambda g: g.rpie(NEPOCH, calc_llk=1),
    "lsqml": lambda g: g.lsqml(NEPOCH, 1, 5, "poisson", 7),
    "bh": lambda g: g.bh(NEPOCH, 1, 5, 0.01, 7),
}
CASES = ("angular_spectrum", "fresnel", "fresnel_back")


@pytest.fixture(scope="module", params=CASES)
def case(request):
    with h5py.File(os.path.join(MODULE_DIR, "gpu_optics.h5"), "r") as f:
        yield request.param, f[request.param]


def gpu_of(g, **options):
    """A fresh GPU state of `g`'s scan and start under its optics."""
    a, counts = g.attrs, g["counts"][()]
    w = ptyrs.Wavefield(counts.shape[1:], a["wavelength"], g["pixel"][()], a["distance"])
    # of the scan's geometry the engine reads the frame alone under optics
    scan = ptyrs.Scan(counts, g["y"][()], g["x"][()], a["wavelength"], 1.0, g["pixel"][1])
    return ptyrs.GpuPtycho(scan, g["object"][()], g["probe"][()], DEVICE,
                           optics=ptyrs.Optics.free_space(w), **options)


@pytest.mark.parametrize("algorithm", list(RUNS))
def test_matches_engine_or_raises_its_error(case, algorithm):
    name, g = case
    assert algorithm in g or (name != "fresnel" and algorithm == "ml")
    gpu = gpu_of(g)
    if algorithm not in g:
        want = g.attrs[f"{algorithm}_error"]
        assert "runs no propagator with a plane stage" in want
        with pytest.raises(ValueError) as e:
            RUNS[algorithm](gpu)
        assert str(e.value) == want
        np.testing.assert_array_equal(gpu.object, g["object"][()].astype(np.complex64))
        return
    RUNS[algorithm](gpu)
    want = g[algorithm]
    assert len(want["poisson"]) == NEPOCH
    np.testing.assert_array_equal(gpu.history["poisson"], want["poisson"][()])
    np.testing.assert_array_equal(gpu.object, want["object"][()])
    np.testing.assert_array_equal(gpu.probe, want["probe"][()])


def test_absent_optics_is_the_far_field(scenario):
    """No `optics` and the far field of the scan's geometry give the same bits."""
    scan, obj, probe, _ = scenario
    far = ptyrs.Optics.far_field(WAVELENGTH, DISTANCE, DETECTOR_PIXEL, probe.shape[1:])
    runs = []
    for optics in ({}, dict(optics=None), dict(optics=far)):
        gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, **optics)
        gpu.dm(NEPOCH, calc_llk=1)
        gpu.lsqml(NEPOCH, 1, 5, "poisson", 7)
        runs.append((gpu.object, gpu.probe, gpu.history["poisson"]))
    for run in runs[1:]:
        for got, want in zip(run, runs[0]):
            np.testing.assert_array_equal(got, want)


def test_refused_optics_raise_at_construction(case):
    """Sliced optics with one object slice, and a plane stage without fused passes."""
    name, g = case
    a, counts = g.attrs, g["counts"][()]
    scan = ptyrs.Scan(counts, g["y"][()], g["x"][()], a["wavelength"], 1.0, g["pixel"][1])
    w = ptyrs.Wavefield(counts.shape[1:], a["wavelength"], g["pixel"][()], a["distance"])
    start = g["object"][()], g["probe"][()], DEVICE
    with pytest.raises(ValueError, match="1 object slices for optics of 2"):
        ptyrs.GpuPtycho(scan, *start, optics=ptyrs.Optics.free_space(w).with_slices([w]))
    if name != "fresnel":
        with pytest.raises(ValueError, match="fused: true"):
            gpu_of(g, fused=False)
