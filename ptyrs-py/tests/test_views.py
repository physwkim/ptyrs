"""Views, Optics and Wavefield refuse what the engine's epochs cannot read, on `opr.h5`'s scan;
`test_opr.py` checks the views they build against the engine's."""

import os

import h5py
import numpy as np
import pytest

import ptyrs
from conftest import MODULE_DIR


@pytest.fixture(scope="module")
def ref():
    with h5py.File(os.path.join(MODULE_DIR, "opr.h5"), "r") as f:
        yield {k: f[k][()] for k in ("counts", "positions", "pixel", "object", "probe")} | {
            k: f.attrs[k] for k in ("wavelength", "distance", "detector_pixel")}


def far_field(r, frame=None):
    return ptyrs.Optics.far_field(r["wavelength"], r["distance"], r["detector_pixel"],
                                  frame or r["counts"].shape[1:])


def views(r, **kw):
    a = dict(counts=r["counts"], positions=r["positions"], pixel=r["pixel"],
             object_shape=r["object"].shape[1:], nprobe=len(r["probe"]), nobj=len(r["object"]),
             optics=far_field(r)) | kw
    return ptyrs.Views(a["counts"], a["positions"], a["pixel"], a["object_shape"], a["nprobe"],
                       a["nobj"], a["optics"])


def test_optics_slices(ref):
    frame = ref["counts"].shape[1:]
    gap = ptyrs.Wavefield(frame, ref["wavelength"], ref["pixel"], 4e-5)
    optics = far_field(ref)
    assert optics.nslice == 1
    assert optics.with_slices([gap, gap]).nslice == 3
    assert optics.with_slices([gap]).with_slices([]).nslice == 1
    with pytest.raises(ValueError, match="slice gap plane"):
        optics.with_slices([ptyrs.Wavefield((16, 16), ref["wavelength"], ref["pixel"], 4e-5)])
    with pytest.raises(ValueError, match="even and non-zero"):
        ptyrs.Wavefield((31, 32), ref["wavelength"], ref["pixel"], 4e-5)
    with pytest.raises(ValueError, match="wavelength"):
        ptyrs.Wavefield(frame, 0.0, ref["pixel"], 4e-5)


def test_views_refusals(ref):
    n = len(ref["positions"])
    with pytest.raises(ValueError, match=r"\[32, 32\] frames on the optics' \[16, 16\] plane"):
        views(ref, optics=far_field(ref, (16, 16)))
    with pytest.raises(ValueError, match=r"positions must be \[nframe, 2\]"):
        views(ref, positions=np.zeros((n, 3)))
    with pytest.raises(ValueError, match=f"{n - 1} positions for {n} frames"):
        views(ref, positions=ref["positions"][1:])
    with pytest.raises(ValueError, match="smaller than frame"):
        views(ref, object_shape=(16, 16))
    with pytest.raises(ValueError, match="nprobe and nobj"):
        views(ref, nprobe=0)
    with pytest.raises(ValueError, match="leaves object"):
        views(ref, positions=ref["positions"] * 4)


@pytest.mark.parametrize("algorithm", [ptyrs.Lsqml(4, "poisson", 7), ptyrs.Bh(4, 0.01, 7)],
                         ids=["lsqml", "bh"])
def test_epoch_refuses_other_shapes(ref, algorithm):
    v, o, p = views(ref), ref["object"][None], ref["probe"]
    want = r"differ from the views' \[1, 1, 56, 52\] and \[2, 32, 32\]"
    for o_, p_ in ((np.concatenate([o, o]), p), (o[:, :, 1:], p), (o, p[:1])):
        with pytest.raises(ValueError, match=want):
            algorithm.epoch(v, o_, p_)
    with pytest.raises(TypeError):
        algorithm.epoch(v, o[0], p)
