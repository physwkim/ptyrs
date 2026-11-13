"""Lsqml's probe variation (OPR) against the engine's own LSQML epochs from the same scan and
start, `examples/engine_reference.rs`' `opr.h5`: after every epoch the object, probe,
eigenmodes, weights, main-mode weights and steps, exactly, as one CPU f64 engine runs both;
eigenmodes and weights alone with the amplitude cost, and with main-mode weights with Poisson;
over `Ptycho.views` and over views built from the far field and preprocessed positions."""

import os

import h5py
import numpy as np
import pytest

import ptyrs
from conftest import MODULE_DIR


@pytest.fixture(scope="module")
def ref():
    with h5py.File(os.path.join(MODULE_DIR, "opr.h5"), "r") as f:
        yield f


def state(f):
    a = f.attrs
    scan = ptyrs.Scan(f["counts"][()], f["y"][()], f["x"][()], a["wavelength"], a["distance"],
                      a["detector_pixel"])
    return ptyrs.Ptycho(scan, f["object"][()], f["probe"][()])


def views_of(f, kind):
    """`f`'s views from its `Ptycho` or built from its positions as the views read them."""
    if kind == "ptycho":
        return state(f).views
    a, counts, obj, probe = f.attrs, f["counts"][()], f["object"][()], f["probe"][()]
    optics = ptyrs.Optics.far_field(a["wavelength"], a["distance"], a["detector_pixel"],
                                    counts.shape[1:])
    return ptyrs.Views(counts, f["positions"][()], f["pixel"][()], obj.shape[1:], len(probe),
                       len(obj), optics)


def variation(g):
    """A run's or an epoch's `eigenmodes`, `weights` and `intensity` (None if absent)."""
    return {k: g[k][()] if k in g else None for k in ("eigenmodes", "weights", "intensity")}


@pytest.mark.parametrize("views", ["ptycho", "built"])
@pytest.mark.parametrize("run", ["opr", "intensity"])
def test_lsqml_opr_matches_engine(ref, run, views):
    g = ref[run]
    start = variation(g)
    assert (start["intensity"] is not None) == (run == "intensity")
    lsqml = ptyrs.Lsqml(int(g.attrs["batch_size"]), g.attrs["cost"], int(g.attrs["seed"]),
                        **{k: v for k, v in start.items() if v is not None})
    over = views_of(ref, views)
    o, pr = ref["object"][()][None], ref["probe"][()]
    epochs = sorted((k for k in g if k.startswith("epoch")), key=lambda k: int(k[5:]))
    assert len(epochs) == 3
    for e in epochs:
        o, pr, steps = lsqml.epoch(over, o, pr)
        want = g[e]
        np.testing.assert_array_equal(o, want["object"][()])
        np.testing.assert_array_equal(pr, want["probe"][()])
        got = dict(eigenmodes=lsqml.eigenmodes, weights=lsqml.weights, intensity=lsqml.intensity)
        for k, v in variation(want).items():
            if v is None:
                assert got[k] is None, k
            else:
                np.testing.assert_array_equal(got[k], v, err_msg=f"{e} {k}")
        assert [len(s["frames"]) for s in steps] == list(want["batch_len"][()])
        for k in ("frames", "a_far", "a_obj", "a_probe"):
            np.testing.assert_array_equal(np.concatenate([s[k] for s in steps], axis=-1),
                                          want[k][()], err_msg=f"{e} {k}")
    assert not np.array_equal(lsqml.weights, start["weights"])


def test_lsqml_opr_refusals(ref):
    p = state(ref).views
    o, pr = ref["object"][()][None], ref["probe"][()]
    v = variation(ref["intensity"])
    e, w, c = v["eigenmodes"], v["weights"], v["intensity"]
    nframe = len(w)
    plain = ptyrs.Lsqml(4, "poisson", 7)
    assert plain.eigenmodes is None and plain.weights is None and plain.intensity is None
    with pytest.raises(ValueError, match="go together"):
        ptyrs.Lsqml(4, "poisson", 7, eigenmodes=e)
    with pytest.raises(ValueError, match="go together"):
        ptyrs.Lsqml(4, "poisson", 7, weights=w)
    with pytest.raises(ValueError, match="intensity without eigenmodes"):
        ptyrs.Lsqml(4, "poisson", 7, intensity=c)
    with pytest.raises(ValueError, match=r"\[nframe, k\] for k ≥ 1"):
        ptyrs.Lsqml(4, "poisson", 7, eigenmodes=e[:0], weights=w[:, :0])
    with pytest.raises(ValueError, match=r"\[nframe, k\] for k ≥ 1"):
        ptyrs.Lsqml(4, "poisson", 7, eigenmodes=e[:1], weights=w)
    with pytest.raises(ValueError, match=f"{nframe - 1} main-mode weights for {nframe}"):
        ptyrs.Lsqml(4, "poisson", 7, eigenmodes=e, weights=w, intensity=c[1:])
    short = ptyrs.Lsqml(4, "poisson", 7, eigenmodes=e, weights=w[1:])
    with pytest.raises(ValueError, match=f"{nframe - 1} weight rows for {nframe} frames"):
        short.epoch(p, o, pr)
    small = ptyrs.Lsqml(4, "poisson", 7, eigenmodes=e[:, 1:, 1:], weights=w)
    with pytest.raises(ValueError, match=r"of \[32, 32\] probe planes"):
        small.epoch(p, o, pr)
    np.testing.assert_array_equal(short.weights, w[1:])
