"""Ptycho under near-field Optics against the CPU engine's `Ptycho::with_optics` from the same
scan and start, `gpu_optics.h5`'s `cpu` groups: the start's LLK, every algorithm's object,
probe and LLK records, and LSQML epochs over the state's own views, exactly, as one CPU f64
engine runs both."""

import os

import h5py
import numpy as np
import pytest

import ptyrs
from conftest import DETECTOR_PIXEL, DISTANCE, MODULE_DIR, WAVELENGTH

NEPOCH = 3
RUNS = {
    "ap": lambda p: p.ap(NEPOCH, calc_llk=1),
    "dm": lambda p: p.dm(NEPOCH, calc_llk=1),
    "raar": lambda p: p.raar(NEPOCH, calc_llk=1),
    "ml": lambda p: p.ml(NEPOCH, calc_llk=1),
    "rpie": lambda p: p.rpie(NEPOCH, calc_llk=1),
}
LLK = ("poisson", "gaussian", "euclidian", "nb_photons_calc")


@pytest.fixture(scope="module", params=("angular_spectrum", "fresnel", "fresnel_back"))
def case(request):
    with h5py.File(os.path.join(MODULE_DIR, "gpu_optics.h5"), "r") as f:
        yield f[request.param]


def wavefield(g):
    a = g.attrs
    return ptyrs.Wavefield(g["counts"].shape[1:], a["wavelength"], g["pixel"][()], a["distance"])


def state(g, optics=None):
    """A fresh CPU state of `g`'s scan and start under `optics`, its own free space if None."""
    a = g.attrs
    # of the scan's geometry the engine reads the frame alone under optics
    scan = ptyrs.Scan(g["counts"][()], g["y"][()], g["x"][()], a["wavelength"], 1.0,
                      g["pixel"][1])
    if optics is None:
        optics = ptyrs.Optics.free_space(wavefield(g))
    return ptyrs.Ptycho(scan, g["object"][()], g["probe"][()], optics=optics)


@pytest.mark.parametrize("algorithm", list(RUNS))
def test_matches_engine(case, algorithm):
    p = state(case)
    RUNS[algorithm](p)
    want = case["cpu"][algorithm]
    assert len(want["poisson"]) == NEPOCH
    np.testing.assert_array_equal(p.history["poisson"], want["poisson"][()])
    np.testing.assert_array_equal(p.object, want["object"][()])
    np.testing.assert_array_equal(p.probe, want["probe"][()])


def test_llk_and_views_take_the_optics(case):
    """`llk` of the start and LSQML epochs over `views` run under the state's optics."""
    p, want = state(case), case["cpu"]
    o, pr = case["object"][()], case["probe"][()]
    d = p.llk(o, pr)
    np.testing.assert_array_equal([d[k] for k in LLK], want["llk"][()])
    lsqml = ptyrs.Lsqml(5, "poisson", 7)
    o = o[None]
    for _ in range(NEPOCH):
        o, pr, _ = lsqml.epoch(p.views, o, pr)
    np.testing.assert_array_equal(o, want["lsqml"]["objects"][()])
    np.testing.assert_array_equal(pr, want["lsqml"]["probe"][()])


def test_absent_optics_is_the_far_field(scenario):
    """No `optics` and the far field of the scan's geometry give the same bits."""
    scan, obj, probe, _ = scenario
    far = ptyrs.Optics.far_field(WAVELENGTH, DISTANCE, DETECTOR_PIXEL, probe.shape[1:])
    runs = []
    for optics in ({}, dict(optics=None), dict(optics=far)):
        p = ptyrs.Ptycho(scan, obj, probe, **optics)
        p.dm(NEPOCH, calc_llk=1)
        o, pr, _ = ptyrs.Lsqml(5, "poisson", 7).epoch(p.views, p.object[None], p.probe)
        runs.append((p.object, p.probe, p.history["poisson"], p.llk(o[0], pr)["poisson"]))
    for run in runs[1:]:
        for got, want in zip(run, runs[0]):
            np.testing.assert_array_equal(got, want)


def test_sliced_optics_are_refused(case):
    w = wavefield(case)
    with pytest.raises(ValueError, match="Ptycho runs one object slice, got 2"):
        state(case, ptyrs.Optics.free_space(w).with_slices([w]))
