"""Ptycho and GpuPtycho `sampling`: the round trip and the refusals of pty-chi's names, the
`Sampling::auto` default far and near; the CPU's single-pass LLK the same under both models
without fractional offsets and not with them, and the GPU's under "full_object" against the
CPU's f64 to the features' 1e-4. Single passes only: the "full_object" object step is not
yet the CPU's with fractional offsets (#29)."""

import numpy as np
import pytest

import ptyrs
from conftest import DEVICE, drawn_offsets, rel

TOL = 1e-4
LLK = ("poisson", "gaussian", "euclidian", "nb_photons_calc")
FUSED = dict(stack_size=7, llk_slots=3, fused=True)
REFUSED = ("FULL_OBJECT", "auto", "fullobject", "")


def near_field(object_side):
    """A near-field CPU and GPU state of zero counts on 64-pixel frames over an
    `object_side`-pixel square object, three views 2 pixels apart."""
    pixel = 1e-7
    optics = ptyrs.Optics.free_space(ptyrs.Wavefield((64, 64), 1.5e-10, (pixel, pixel), 1e-3))
    y = x = pixel * np.array([0.0, 2.0, 4.0])
    scan = ptyrs.Scan(np.zeros((3, 64, 64)), y, x, 1.5e-10, 1.0, pixel)
    obj = np.ones((1, object_side, object_side), complex)
    probe = np.ones((1, 64, 64), complex)
    cpu = ptyrs.Ptycho(scan, obj, probe, optics=optics)
    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, optics=optics, **FUSED)
    return cpu, gpu


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
        cpu, gpu = near_field(side)
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
