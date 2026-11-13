"""GpuPtycho.lsqml and lsqml_steps against the CPU binding's f64 Lsqml from the same start,
as the engine's own `tests/engine_gpu_lsqml.rs` compares them, to its 1e-4: each epoch's LLK
record and the final arrays of 12 epochs, and after a fresh GPU run of each of the first 3,
the last epoch's steps and the arrays."""

import numpy as np
import pytest

import ptyrs
from conftest import DEVICE, rel, rel_l2

NEPOCH = 12  # epoch 11 keeps the preconditioner epoch 10 updated
NEPOCH_STEPS = 3  # the preconditioner built, then updated
TOL = 1e-4
COSTS = pytest.mark.parametrize("cost", ["amplitude", "poisson"])
BATCHES = pytest.mark.parametrize("batch", [30, 12, 4])


@COSTS
@BATCHES
def test_lsqml_matches_cpu(scenario, cost, batch):
    scan, obj, probe, _ = scenario
    state = ptyrs.Ptycho(scan, obj, probe)
    cpu = ptyrs.Lsqml(batch, cost, 7)
    o, p = obj, probe
    want = []
    for _ in range(NEPOCH):
        o, p, _ = cpu.epoch(state, o, p)
        want.append(state.llk(o, p))

    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, stack_size=7, llk_slots=3)
    gpu.lsqml(NEPOCH, 1, batch, cost, 7)
    h = gpu.history
    assert list(h["cycle"]) == list(range(NEPOCH))
    assert h["algorithm"] == ["lsqml"] * NEPOCH
    for key in ("poisson", "gaussian", "euclidian", "nb_photons_calc"):
        err = rel(h[key], [w[key] for w in want]).max()
        assert err < TOL, f"llk {key}: {err:.2e}"
    assert rel_l2(gpu.object, o) < TOL
    assert rel_l2(gpu.probe, p) < TOL
    assert gpu.cycle == NEPOCH


@COSTS
@BATCHES
def test_lsqml_steps_match_cpu(scenario, cost, batch):
    scan, obj, probe, _ = scenario
    state = ptyrs.Ptycho(scan, obj, probe)
    cpu = ptyrs.Lsqml(batch, cost, 7)
    o, p = obj, probe
    for nepoch in range(1, NEPOCH_STEPS + 1):
        o, p, want = cpu.epoch(state, o, p)
        gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, stack_size=7, llk_slots=3)
        assert gpu.lsqml_steps() == []
        gpu.lsqml(nepoch, 0, batch, cost, 7)
        got = gpu.lsqml_steps()
        assert [list(s["frames"]) for s in got] == [list(s["frames"]) for s in want]
        for key in ("a_far", "a_obj", "a_probe"):
            err = max(rel(g[key], w[key]).max() for g, w in zip(got, want))
            assert err < TOL, f"epoch {nepoch} {key}: {err:.2e}"
        assert rel_l2(gpu.object, o) < TOL
        assert rel_l2(gpu.probe, p) < TOL


def test_lsqml_rejects(scenario):
    scan, obj, probe, _ = scenario
    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE)
    with pytest.raises(ValueError, match="batch_size"):
        gpu.lsqml(1, 0, 0, "poisson", 7)
    with pytest.raises(ValueError, match="cost"):
        gpu.lsqml(1, 0, 8, "gauss", 7)
    assert gpu.cycle == 0 and len(gpu.history["cycle"]) == 0
