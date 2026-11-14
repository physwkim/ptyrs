"""Lsqml and Bh over Views of near-field and multislice Optics against the engine's own epochs
from the same scan and start, `examples/engine_reference.rs`' `near.h5` (pty-chi's planar
near field) and `multislice.h5` (three slices before PyNX's far field): after every epoch the
object slices, probe and steps, exactly, as one CPU f64 engine runs both."""

import os

import h5py
import numpy as np
import pytest

import ptyrs
from conftest import MODULE_DIR


@pytest.fixture(scope="module", params=["near", "multislice"])
def ref(request):
    with h5py.File(os.path.join(MODULE_DIR, f"{request.param}.h5"), "r") as f:
        yield f


def optics_of(f):
    a, frame = f.attrs, f["counts"].shape[1:]
    if a["exit"] == "free_space":
        w = ptyrs.Wavefield(frame, a["wavelength"], f["wave_pixel"][()], a["distance"])
        optics = ptyrs.Optics.free_space(w)
    else:
        assert a["exit"] == "far_field"
        optics = ptyrs.Optics.far_field(a["wavelength"], a["distance"], a["detector_pixel"],
                                        frame)
    if "gaps" not in f:
        return optics
    gaps = [ptyrs.Wavefield(frame, a["wavelength"], f["pixel"][()], d) for d in f["gaps"][()]]
    return optics.with_slices(gaps)


def start(f):
    """`f`'s views and start object slices and probe."""
    o, p = f["object"][()], f["probe"][()]
    assert len(o) == (len(f["gaps"]) + 1 if "gaps" in f else 1)
    views = ptyrs.Views(f["counts"][()], f["positions"][()], f["pixel"][()], o.shape[2:],
                        len(p), o.shape[1], optics_of(f))
    return views, o, p


def epochs(g):
    e = sorted((k for k in g if k.startswith("epoch")), key=lambda k: int(k[5:]))
    assert len(e) == 3
    return e


def test_lsqml_matches_engine(ref):
    g = ref["lsqml"]
    lsqml = ptyrs.Lsqml(int(g.attrs["batch_size"]), g.attrs["cost"], int(g.attrs["seed"]))
    views, o, p = start(ref)
    for e in epochs(g):
        o, p, steps = lsqml.epoch(views, o, p)
        want = g[e]
        np.testing.assert_array_equal(o, want["object"][()], err_msg=e)
        np.testing.assert_array_equal(p, want["probe"][()], err_msg=e)
        assert [len(s["frames"]) for s in steps] == list(want["batch_len"][()])
        for k in ("frames", "a_far", "a_obj", "a_probe"):
            np.testing.assert_array_equal(np.concatenate([s[k] for s in steps], axis=-1),
                                          want[k][()], err_msg=f"{e} {k}")
    assert not np.array_equal(o, ref["object"][()])


def test_bh_matches_engine(ref):
    g = ref["bh"]
    bh = ptyrs.Bh(int(g.attrs["batch_size"]), g.attrs["probe_rho"], int(g.attrs["seed"]))
    views, o, p = start(ref)
    for e in epochs(g):
        o, p, steps = bh.epoch(views, o, p)
        want = g[e]
        np.testing.assert_array_equal(o, want["object"][()], err_msg=e)
        np.testing.assert_array_equal(p, want["probe"][()], err_msg=e)
        assert [len(s["frames"]) for s in steps] == list(want["batch_len"][()])
        np.testing.assert_array_equal(np.concatenate([s["frames"] for s in steps]),
                                      want["frames"][()], err_msg=e)
        for k in ("beta", "alpha", "top", "bottom"):
            np.testing.assert_array_equal([s[k] for s in steps], want[k][()],
                                          err_msg=f"{e} {k}")
    assert not np.array_equal(o, ref["object"][()])
