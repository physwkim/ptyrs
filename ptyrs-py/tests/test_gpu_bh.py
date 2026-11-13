"""GpuPtycho.bh and bh_steps against the CPU binding's f64 Bh: after a fresh GPU run of each
of the first epochs, every epoch's LLK record, the last epoch's steps and the arrays, as the
engine's own `tests/engine_gpu.rs` `bh_steps_match_cpu` and `bh_position_matches_cpu`
compare them, to its 1e-4 (offsets to 1e-4 pixels). One batch of every frame (CG from the
second epoch) and batches of 4, the last of 2; M1 and fused passes, and the fused ones
modelling `drawn_offsets` at the default and another `position_rho`."""

import numpy as np
import pytest

import ptyrs
from conftest import DEVICE, drawn_offsets, rel, rel_l2

NEPOCH = 3
TOL = 1e-4
LLK = ("poisson", "gaussian", "euclidian", "nb_photons_calc")


@pytest.mark.parametrize("fused, moved, rho", [(False, False, {}), (True, False, {}),
                                               (True, True, {}),
                                               (True, True, dict(position_rho=0.3))],
                         ids=["m1", "fused", "position", "position-rho"])
@pytest.mark.parametrize("batch", [30, 4])
def test_bh_steps_match_cpu(scenario, batch, fused, moved, rho):
    scan, obj, probe, npos = scenario
    d0 = drawn_offsets(npos) if moved else None
    state = ptyrs.Ptycho(scan, obj, probe)
    cpu = ptyrs.Bh(batch, 0.01, 7, position=d0, **rho)
    o, p = obj, probe
    llk = []
    for nepoch in range(1, NEPOCH + 1):
        o, p, want = cpu.epoch(state, o, p)
        state.set_features(position=cpu.position)
        llk.append(state.llk(o, p))
        gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, stack_size=7, llk_slots=3, fused=fused)
        if moved:
            gpu.set_features(position=d0)
        assert gpu.bh_steps() == []
        gpu.bh(nepoch, 1, batch, 0.01, 7, **rho)
        for key in LLK:
            err = rel(gpu.history[key], [w[key] for w in llk]).max()
            assert err < TOL, f"epoch {nepoch} llk {key}: {err:.2e}"
        got = gpu.bh_steps()
        assert [list(s["frames"]) for s in got] == [list(s["frames"]) for s in want]
        for key in ("beta", "alpha", "top", "bottom"):
            # a CPU 0 (β in steepest descent) must be an exact GPU 0
            err = max(rel(g[key], w[key]) for g, w in zip(got, want))
            assert err < TOL, f"epoch {nepoch} {key}: {err:.2e}"
        assert rel_l2(gpu.object, o) < TOL
        assert rel_l2(gpu.probe, p) < TOL
        if moved:
            err = np.abs(gpu.features["position"] - cpu.position).max()
            assert err < TOL, f"epoch {nepoch} position: {err:.2e} pixels"
    if moved:
        assert (cpu.position ** 2).mean() < (d0 ** 2).mean()


def test_bh_position_rho_and_refusals(scenario):
    """`position_rho` reaches the offsets' step, and Bh refuses offsets as Lsqml does."""
    scan, obj, probe, npos = scenario
    state = ptyrs.Ptycho(scan, obj, probe)
    d0 = drawn_offsets(npos)
    moved = []
    for rho in ({}, dict(position_rho=0.1), dict(position_rho=0.3)):
        bh = ptyrs.Bh(4, 0.01, 7, position=d0, **rho)
        bh.epoch(state, obj, probe)
        moved.append(bh.position)
    assert (moved[0] == moved[1]).all() and not np.allclose(moved[0], moved[2])
    assert ptyrs.Bh(4, 0.01, 7).position is None
    with pytest.raises(ValueError, match=r"\[nframe, 2\]"):
        ptyrs.Bh(4, 0.01, 7, position=np.zeros((npos, 3)))
    with pytest.raises(ValueError, match="finite"):
        ptyrs.Bh(4, 0.01, 7, position=np.full((npos, 2), np.inf))
    short = ptyrs.Bh(4, 0.01, 7, position=np.zeros((npos + 1, 2)))
    with pytest.raises(ValueError, match=f"{npos + 1} position offsets for {npos} frames"):
        short.epoch(state, obj, probe)
