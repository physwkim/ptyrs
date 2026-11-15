"""Lsqml's probe variation (OPR) against the engine's own LSQML epochs from the same scan and
start, `examples/engine_reference.rs`' `opr.h5`: after every epoch the object, probe,
eigenmodes, weights, main-mode weights and steps, exactly, as one CPU f64 engine runs both;
eigenmodes and weights alone with the amplitude cost, and with main-mode weights with Poisson;
over `Ptycho.views` and over views built from the far field and preprocessed positions, with
`Lsqml.llk`, the engine's `Lsqml::llk`, at the start and after every epoch. Without OPR,
`Lsqml.llk` is `Ptycho.llk`. Held weights and main-mode weights kept while the eigenmodes move;
the refusals."""

import os

import h5py
import numpy as np
import pytest

import ptyrs
from conftest import MODULE_DIR, drawn_offsets, rel

LLK = ("poisson", "gaussian", "euclidian", "nb_photons_calc")


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


def llk(lsqml, views, o, pr):
    """`lsqml.llk` as the references lay it out."""
    d = lsqml.llk(views, o, pr)
    return [d[k] for k in LLK]


@pytest.mark.parametrize("views", ["ptycho", "built"])
@pytest.mark.parametrize("run", ["opr", "intensity"])
def test_lsqml_opr_matches_engine(ref, run, views):
    g = ref[run]
    start = variation(g)
    assert (start["intensity"] is not None) == (run == "intensity")
    lsqml = ptyrs.Lsqml(int(g.attrs["batch_size"]), g.attrs["cost"], int(g.attrs["seed"]),
                        opr=(start["eigenmodes"], start["weights"], start["intensity"]))
    over = views_of(ref, views)
    o, pr = ref["object"][()][None], ref["probe"][()]
    epochs = sorted((k for k in g if k.startswith("epoch")), key=lambda k: int(k[5:]))
    assert len(epochs) == 3
    np.testing.assert_array_equal(llk(lsqml, over, o, pr), g["llk"][()])
    for e in epochs:
        o, pr, steps = lsqml.epoch(over, o, pr)
        want = g[e]
        np.testing.assert_array_equal(o, want["object"][()])
        np.testing.assert_array_equal(pr, want["probe"][()])
        np.testing.assert_array_equal(llk(lsqml, over, o, pr), want["llk"][()], err_msg=e)
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
    # eigenmodes and weights always, and intensity in the third place even when None
    with pytest.raises(ValueError, match="tuple of length 3"):
        ptyrs.Lsqml(4, "poisson", 7, opr=(e, w))
    for opr in ((e, None, c), (None, w, None)):
        with pytest.raises(TypeError, match="opr"):
            ptyrs.Lsqml(4, "poisson", 7, opr=opr)
    with pytest.raises(ValueError, match=r"\[nframe, k\] for k ≥ 1"):
        ptyrs.Lsqml(4, "poisson", 7, opr=(e[:0], w[:, :0], None))
    with pytest.raises(ValueError, match=r"\[nframe, k\] for k ≥ 1"):
        ptyrs.Lsqml(4, "poisson", 7, opr=(e[:1], w, None))
    with pytest.raises(ValueError, match=f"{nframe - 1} main-mode weights for {nframe}"):
        ptyrs.Lsqml(4, "poisson", 7, opr=(e, w, c[1:]))
    short = ptyrs.Lsqml(4, "poisson", 7, opr=(e, w[1:], None))
    small = ptyrs.Lsqml(4, "poisson", 7, opr=(e[:, 1:, 1:], w, None))
    for run in ("epoch", "llk"):
        with pytest.raises(ValueError, match=f"{nframe - 1} weight rows for {nframe} frames"):
            getattr(short, run)(p, o, pr)
        with pytest.raises(ValueError, match=r"of \[32, 32\] probe planes"):
            getattr(small, run)(p, o, pr)
    np.testing.assert_array_equal(short.weights, w[1:])
    # `hold` names a given value
    for kw, hold in ((dict(), ("weights",)), (dict(), ("intensity",)),
                     (dict(opr=(e, w, None)), ("intensity",)),
                     (dict(opr=(e, w, c)), ("position",)),
                     (dict(opr=(e, w, c)), ("eigenmodes",))):
        with pytest.raises(ValueError, match="hold"):
            ptyrs.Lsqml(4, "poisson", 7, hold=hold, **kw)
    with pytest.raises(TypeError):
        ptyrs.Lsqml(4, "poisson", 7, opr=(e, w, c), hold="weights")
    assert plain.hold == () and ptyrs.Lsqml(4, "poisson", 7, opr=(e, w, c)).hold == ()


@pytest.mark.parametrize("hold", [("weights",), ("intensity",), ("weights", "intensity")])
def test_lsqml_opr_hold(ref, hold):
    """Two epochs of the main-mode run's start: the held values stay exactly, the eigenmodes
    and the refined values move."""
    g = ref["intensity"]
    start = variation(g)
    lsqml = ptyrs.Lsqml(int(g.attrs["batch_size"]), g.attrs["cost"], int(g.attrs["seed"]),
                        opr=(start["eigenmodes"], start["weights"], start["intensity"]),
                        hold=hold)
    assert lsqml.hold == hold
    o, pr = ref["object"][()][None], ref["probe"][()]
    for _ in range(2):
        o, pr, _ = lsqml.epoch(state(ref).views, o, pr)
    assert np.isfinite(o).all() and np.isfinite(pr).all()
    assert not np.array_equal(lsqml.eigenmodes, start["eigenmodes"])
    for k in ("weights", "intensity"):
        assert np.array_equal(getattr(lsqml, k), start[k]) == (k in hold), k


@pytest.mark.parametrize("offsets, sampling", [(False, None), (True, None), (True, "patch"),
                                               (True, "full_object")])
def test_lsqml_llk_without_opr_is_ptycho_llk(scenario, offsets, sampling):
    """Without `opr`, `Lsqml.llk` is `Ptycho.llk` of the same arrays, offsets and sampling,
    `Sampling::auto` on both when it is not set; with offsets the two order their sums apart,
    so they agree to 1e-12, the engine's bound, not bit for bit."""
    scan, obj, probe, npos = scenario
    d = drawn_offsets(npos) if offsets else None
    state = ptyrs.Ptycho(scan, obj, probe)
    state.set_features(position=d)
    if sampling is not None:
        state.set_sampling(sampling)
    lsqml = ptyrs.Lsqml(4, "poisson", 7, position=d, sampling=sampling)
    got, want = lsqml.llk(state.views, obj[None], probe), state.llk(obj, probe)
    for k in LLK:
        assert rel(got[k], want[k]) < 1e-12, k
