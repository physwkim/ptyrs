"""Ptycho and GpuPtycho `sampling`: the round trip and the refusals of pty-chi's names, the
`Sampling::auto` default far and near; the CPU's single-pass LLK the same under both models
without fractional offsets and not with them, and the GPU's under "full_object" against the
CPU's f64 to the features' 1e-4. Single passes only: the "full_object" object step is not
yet the CPU's with fractional offsets (#29)."""

import numpy as np
import pytest

import ptyrs
from conftest import (DEVICE, WAVELENGTH, drawn_offsets, intensity, rel, rel_l2, sim_object,
                      sim_probe)

TOL = 1e-4
LLK = ("poisson", "gaussian", "euclidian", "nb_photons_calc")
FUSED = dict(stack_size=7, llk_slots=3, fused=True)
REFUSED = ("FULL_OBJECT", "auto", "fullobject", "")
NEAR_PIXEL, NEAR_FRAMES = 1e-7, 9
ALGORITHMS = {
    "lsqml": lambda nframe, **kw: ptyrs.Lsqml(nframe, "poisson", 7, **kw),
    "bh": lambda nframe, **kw: ptyrs.Bh(nframe, 0.01, 7, **kw),
}


def near_field(object_side):
    """A near-field scan on 64-pixel frames over an `object_side`-pixel square object, its
    `NEAR_FRAMES` views on a grid 2 pixels apart, of the model counts of `sim_object` and two
    `sim_probe` modes; returns the scan, its optics, the object and the probe."""
    wave = ptyrs.Wavefield((64, 64), WAVELENGTH, (NEAR_PIXEL, NEAR_PIXEL), 1e-4)
    optics = ptyrs.Optics.free_space(wave)
    y, x = NEAR_PIXEL * 2.0 * np.mgrid[:3, :3].reshape(2, -1)
    obj, probe = sim_object(object_side), sim_probe(64, 2)

    def scan(counts):
        # of the scan's geometry the engine reads the frame alone under optics
        return ptyrs.Scan(counts, y, x, WAVELENGTH, 1.0, NEAR_PIXEL)

    model = ptyrs.Ptycho(scan(np.zeros((NEAR_FRAMES, 64, 64))), obj, probe, optics=optics)
    return scan(np.round(1e5 * intensity(model, obj, probe))), optics, obj, probe


def round_trip(state, default):
    assert state.sampling == default
    for name in ("full_object", "patch", "full_object"):
        state.set_sampling(name)
        assert state.sampling == name
    for name in REFUSED:
        with pytest.raises(ValueError, match="patch.*full_object"):
            state.set_sampling(name)
        assert state.sampling == "full_object"


def test_round_trip_refusals_and_auto_default(scenario):
    scan, obj, probe, _ = scenario
    round_trip(ptyrs.Ptycho(scan, obj, probe), "patch")
    round_trip(ptyrs.GpuPtycho(scan, obj, probe, DEVICE, **FUSED), "patch")
    # Sampling::auto: "full_object" in the near field when 5·frame > 4·object on both axes
    for side, default in ((72, "full_object"), (80, "patch")):
        scan, optics, obj, probe = near_field(side)
        cpu = ptyrs.Ptycho(scan, obj, probe, optics=optics)
        gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, optics=optics, **FUSED)
        assert cpu.sampling == gpu.sampling == default, side


def cpu_llk(scenario, sampling, position):
    scan, obj, probe, _ = scenario
    cpu = ptyrs.Ptycho(scan, obj, probe)
    cpu.set_sampling(sampling)
    if position is not None:
        cpu.set_features(position=position)
    return cpu.llk(obj, probe)


def test_models_agree_without_fractional_offsets(scenario):
    npos = scenario[3]
    for position, tol in ((None, 0.0), (np.zeros((npos, 2)), 1e-10)):
        patch = cpu_llk(scenario, "patch", position)
        full = cpu_llk(scenario, "full_object", position)
        for key in LLK:
            assert rel(full[key], patch[key]) <= tol, (position is None, key)


def test_models_differ_with_fractional_offsets(scenario):
    position = drawn_offsets(scenario[3])
    patch = cpu_llk(scenario, "patch", position)
    full = cpu_llk(scenario, "full_object", position)
    # past what the GPU test allows, so that it tells the models apart
    for key in LLK:
        assert rel(full[key], patch[key]) > 4 * TOL, key


@pytest.mark.parametrize("offsets", ["zero", "drawn"])
@pytest.mark.parametrize("sampling_first", [True, False])
def test_gpu_full_object_llk_matches_cpu(scenario, offsets, sampling_first):
    scan, obj, probe, npos = scenario
    position = np.zeros((npos, 2)) if offsets == "zero" else drawn_offsets(npos)
    want = cpu_llk(scenario, "full_object", position)
    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, **FUSED)
    if sampling_first:
        gpu.set_sampling("full_object")
    gpu.set_features(position=position)
    if not sampling_first:
        gpu.set_sampling("full_object")
    got = gpu.llk()
    for key in LLK:
        err = rel(got[key], want[key])
        assert err < TOL, f"{key}: {err:.2e}"


@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_lsqml_bh_refuse_other_names(algorithm):
    for name in REFUSED:
        with pytest.raises(ValueError, match="patch.*full_object"):
            ALGORITHMS[algorithm](4, sampling=name)


@pytest.mark.parametrize("field, default", [("far", "patch"), ("near", "full_object")])
@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_lsqml_bh_take_sampling(scenario, algorithm, field, default):
    """An epoch with fractional offsets under each name: the one `Sampling::auto` takes,
    "patch" in the far field and "full_object" near a small object, is the default's bit for
    bit, and the other is not."""
    if field == "far":
        scan, obj, probe, nframe = scenario
        p = ptyrs.Ptycho(scan, obj, probe)
    else:
        scan, optics, obj, probe = near_field(72)
        p, nframe = ptyrs.Ptycho(scan, obj, probe, optics=optics), NEAR_FRAMES
    position = drawn_offsets(nframe)

    def epoch(**sampling):
        run = ALGORITHMS[algorithm](nframe, position=position, **sampling)
        o, pr, _ = run.epoch(p.views, obj[None], probe)
        return o, pr, run.position

    want = epoch()
    for got, w in zip(epoch(sampling=default), want):
        np.testing.assert_array_equal(got, w)
    o, _, d = epoch(sampling={"patch": "full_object", "full_object": "patch"}[default])
    assert rel_l2(o, want[0]) > 1e-3
    assert rel_l2(d - position, want[2] - position) > 0.1
