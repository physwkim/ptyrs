"""GpuPtycho.bh_steps against the CPU binding's f64 Bh: after a fresh GPU run of each of the
first epochs, the last epoch's steps and the arrays, as the engine's own
`tests/engine_gpu.rs` `bh_steps_match_cpu` compares them, to its 1e-4. One batch of every
frame (CG from the second epoch) and batches of 4, the last of 2; M1 and fused passes."""

import pytest

import ptyrs
from conftest import DEVICE, rel, rel_l2

NEPOCH = 3
TOL = 1e-4


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("batch", [30, 4])
def test_bh_steps_match_cpu(scenario, fused, batch):
    scan, obj, probe, _ = scenario
    state = ptyrs.Ptycho(scan, obj, probe)
    cpu = ptyrs.Bh(batch, 0.01, 7)
    o, p = obj, probe
    for nepoch in range(1, NEPOCH + 1):
        o, p, want = cpu.epoch(state, o, p)
        gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, stack_size=7, fused=fused)
        assert gpu.bh_steps() == []
        gpu.bh(nepoch, 0, batch, 0.01, 7)
        got = gpu.bh_steps()
        assert [list(s["frames"]) for s in got] == [list(s["frames"]) for s in want]
        for key in ("beta", "alpha", "top", "bottom"):
            # a CPU 0 (β in steepest descent) must be an exact GPU 0
            err = max(rel(g[key], w[key]) for g, w in zip(got, want))
            assert err < TOL, f"epoch {nepoch} {key}: {err:.2e}"
        assert rel_l2(gpu.object, o) < TOL
        assert rel_l2(gpu.probe, p) < TOL
