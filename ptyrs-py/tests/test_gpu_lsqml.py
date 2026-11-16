"""GpuPtycho.lsqml and lsqml_steps against the CPU binding's f64 Lsqml from the same start,
as the engine's own `tests/engine_gpu_lsqml.rs` compares them, to its 1e-4: each epoch's LLK
record and the final arrays of 12 epochs, and after a fresh GPU run of each of the first 3,
the last epoch's steps and the arrays; M1 and fused passes, and the fused ones modelling
`drawn_offsets`, whose changes are compared too, or held, which both keep. Each batch's step sizes are compared as a
vector, to its largest: a frame's `a_probe` can be 1e-3 of the others' (its probe direction
nearly orthogonal to its residual), and there the GPU's `f32` rounding of their size is
1e-4 of its own. With a probe variation (`opr`), the LLK is the CPU's `Lsqml.llk` of each
frame's own probe, and the variation's changes are compared as the offsets'."""

import numpy as np
import pytest

import ptyrs
from conftest import DEVICE, drawn_offsets, offsets_rel, rel, rel_l2

NEPOCH = 12  # epoch 11 keeps the preconditioner epoch 10 updated
NEPOCH_STEPS = 3  # the preconditioner built, then updated
TOL = 1e-4
COSTS = pytest.mark.parametrize("cost", ["amplitude", "poisson"])
BATCHES = pytest.mark.parametrize("batch", [30, 12, 4])
RUNS = pytest.mark.parametrize("fused, moved", [(False, False), (True, False), (True, True)],
                               ids=["m1", "fused", "position"])


def start(scenario, moved):
    """The scenario's scan, object and probe and, if `moved`, its `drawn_offsets`."""
    scan, obj, probe, npos = scenario
    return scan, obj, probe, drawn_offsets(npos) if moved else None


def gpu_at(scan, obj, probe, fused, d0):
    """A fresh GPU state of `obj` and `probe` modelling the offsets `d0`, if any."""
    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, stack_size=7, llk_slots=3, fused=fused)
    if d0 is not None:
        gpu.set_features(position=d0)
    return gpu


def steps_rel(a, b):
    """`max |a − b| / max |b|` of one batch's step sizes."""
    return np.abs(np.asarray(a) - b).max() / np.abs(b).max()


def offsets_match(gpu, cpu, d0):
    if d0 is None:
        assert gpu.features["position"] is None and cpu.position is None
        return
    err = offsets_rel(gpu.features["position"], cpu.position, d0)
    assert err < TOL, f"offsets: {err:.2e}"


@COSTS
@BATCHES
@RUNS
def test_lsqml_matches_cpu(scenario, cost, batch, fused, moved):
    scan, obj, probe, d0 = start(scenario, moved)
    state = ptyrs.Ptycho(scan, obj, probe)
    cpu = ptyrs.Lsqml(batch, cost, 7, position=d0)
    o, p = obj[None], probe
    want = []
    for _ in range(NEPOCH):
        o, p, _ = cpu.epoch(state.views, o, p)
        state.set_features(position=cpu.position)
        want.append(state.llk(o[0], p))

    gpu = gpu_at(scan, obj, probe, fused, d0)
    gpu.lsqml(NEPOCH, 1, batch, cost, 7)
    h = gpu.history
    assert list(h["cycle"]) == list(range(NEPOCH))
    assert h["algorithm"] == ["lsqml"] * NEPOCH
    for key in ("poisson", "gaussian", "euclidian", "nb_photons_calc"):
        err = rel(h[key], [w[key] for w in want]).max()
        assert err < TOL, f"llk {key}: {err:.2e}"
    assert rel_l2(gpu.object, o[0]) < TOL
    assert rel_l2(gpu.probe, p) < TOL
    offsets_match(gpu, cpu, d0)
    assert gpu.cycle == NEPOCH


@COSTS
def test_lsqml_held_position_matches_cpu(scenario, cost):
    """`drawn_offsets` held on both: as `test_lsqml_matches_cpu`, and the offsets kept (the
    device's f32 of them)."""
    scan, obj, probe, d0 = start(scenario, True)
    state = ptyrs.Ptycho(scan, obj, probe)
    cpu = ptyrs.Lsqml(12, cost, 7, position=d0, hold=("position",))
    assert cpu.hold == ("position",)
    state.set_features(position=d0)
    o, p = obj[None], probe
    want = []
    for _ in range(NEPOCH):
        o, p, _ = cpu.epoch(state.views, o, p)
        want.append(state.llk(o[0], p))
    np.testing.assert_array_equal(cpu.position, d0)

    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, stack_size=7, llk_slots=3, fused=True)
    gpu.set_features(position=d0, hold=("position",))
    gpu.lsqml(NEPOCH, 1, 12, cost, 7)
    h = gpu.history
    for key in ("poisson", "gaussian", "euclidian", "nb_photons_calc"):
        err = rel(h[key], [w[key] for w in want]).max()
        assert err < TOL, f"llk {key}: {err:.2e}"
    assert rel_l2(gpu.object, o[0]) < TOL
    assert rel_l2(gpu.probe, p) < TOL
    assert gpu.features["hold"] == ("position",)
    np.testing.assert_array_equal(gpu.features["position"], d0.astype(np.float32))


def opr_start(probe, npos, k, intensity):
    """The engine tests' `variation`: `k` eigenmodes of probe mode 0, it times `x`, `i y` and
    `x² − y²` in turn (the plane's centre at 0, its edges at ±2), weights near ±0.05 at each
    of `npos` frames and, with `intensity`, main-mode weights near 1."""
    ny, nx = probe.shape[1:]
    y, x = np.meshgrid(4 * np.arange(ny) / ny - 2, 4 * np.arange(nx) / nx - 2, indexing="ij")
    shapes = (x + 0j, 1j * y, x * x - y * y + 0j)
    e = np.stack([probe[0] * shapes[i % 3] for i in range(k)])
    f, i = np.meshgrid(np.arange(npos), np.arange(k), indexing="ij")
    w = 0.05 * np.sin(1.3 * f + 0.7 * i + 0.4)
    c = 1 + 0.05 * np.cos(0.9 * np.arange(npos)) if intensity else None
    return e, w, c


@COSTS
@pytest.mark.parametrize("moved, intensity, hold",
                         [(False, False, ()), (True, True, ()), (True, True, ("weights",)),
                          (False, True, ("intensity",))],
                         ids=["weights", "position", "held-weights", "held-intensity"])
def test_lsqml_opr_matches_cpu(scenario, cost, moved, intensity, hold):
    """`opr` and `hold` as `Lsqml` takes them, in batches of 12: each epoch's LLK, the final
    arrays and offsets, and the eigenmodes and refined weights; held weights kept (the
    device's `f32` of them)."""
    scan, obj, probe, d0 = start(scenario, moved)
    e, w, c = opr_start(probe, scenario[3], 3, intensity)
    views = ptyrs.Ptycho(scan, obj, probe).views
    cpu = ptyrs.Lsqml(12, cost, 7, position=d0, opr=(e, w, c), hold=hold)
    o, p = obj[None], probe
    want = []
    for _ in range(NEPOCH):
        o, p, _ = cpu.epoch(views, o, p)
        want.append(cpu.llk(views, o, p))

    gpu = gpu_at(scan, obj, probe, True, d0)
    got = gpu.lsqml(NEPOCH, 1, 12, cost, 7, opr=(e, w, c), hold=hold)
    h = gpu.history
    for key in ("poisson", "gaussian", "euclidian", "nb_photons_calc"):
        err = rel(h[key], [x[key] for x in want]).max()
        assert err < TOL, f"llk {key}: {err:.2e}"
    assert rel_l2(gpu.object, o[0]) < TOL
    assert rel_l2(gpu.probe, p) < TOL
    offsets_match(gpu, cpu, d0)
    err = offsets_rel(got[0], cpu.eigenmodes, e)
    assert err < TOL, f"eigenmodes: {err:.2e}"
    for name, g, s, x in zip(("weights", "intensity"), got[1:], (w, c),
                             (cpu.weights, cpu.intensity)):
        if s is None:
            assert g is None and x is None, name
        elif name in hold:
            np.testing.assert_array_equal(x, s, err_msg=name)
            np.testing.assert_array_equal(g, s.astype(np.float32), err_msg=name)
        else:
            err = offsets_rel(g, x, s)
            assert err < TOL, f"{name}: {err:.2e}"


@COSTS
@BATCHES
@RUNS
def test_lsqml_steps_match_cpu(scenario, cost, batch, fused, moved):
    scan, obj, probe, d0 = start(scenario, moved)
    state = ptyrs.Ptycho(scan, obj, probe)
    cpu = ptyrs.Lsqml(batch, cost, 7, position=d0)
    o, p = obj[None], probe
    for nepoch in range(1, NEPOCH_STEPS + 1):
        o, p, want = cpu.epoch(state.views, o, p)
        gpu = gpu_at(scan, obj, probe, fused, d0)
        assert gpu.lsqml_steps() == []
        gpu.lsqml(nepoch, 0, batch, cost, 7)
        got = gpu.lsqml_steps()
        assert [list(s["frames"]) for s in got] == [list(s["frames"]) for s in want]
        for key in ("a_far", "a_obj", "a_probe"):
            err = max(steps_rel(g[key], w[key]) for g, w in zip(got, want))
            assert err < TOL, f"epoch {nepoch} {key}: {err:.2e}"
        assert rel_l2(gpu.object, o[0]) < TOL
        assert rel_l2(gpu.probe, p) < TOL
        offsets_match(gpu, cpu, d0)


def test_lsqml_rejects(scenario):
    scan, obj, probe, npos = scenario
    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE)
    with pytest.raises(ValueError, match="batch_size"):
        gpu.lsqml(1, 0, 0, "poisson", 7)
    with pytest.raises(ValueError, match="cost"):
        gpu.lsqml(1, 0, 8, "gauss", 7)
    assert gpu.cycle == 0 and len(gpu.history["cycle"]) == 0
    assert ptyrs.Lsqml(8, "poisson", 7).position is None
    with pytest.raises(ValueError, match="LSQML batch_size must be >= 1"):
        ptyrs.Lsqml(0, "poisson", 7)
    with pytest.raises(ValueError, match=r"\[nframe, 2\]"):
        ptyrs.Lsqml(8, "poisson", 7, position=np.zeros((npos, 3)))
    with pytest.raises(ValueError, match="finite"):
        ptyrs.Lsqml(8, "poisson", 7, position=np.full((npos, 2), np.nan))
    with pytest.raises(ValueError, match="hold"):
        ptyrs.Lsqml(8, "poisson", 7, hold=("position",))
    short = ptyrs.Lsqml(8, "poisson", 7, position=np.zeros((npos - 1, 2)))
    with pytest.raises(ValueError, match=f"{npos - 1} position offsets for {npos} frames"):
        short.epoch(ptyrs.Ptycho(scan, obj, probe).views, obj[None], probe)


def test_lsqml_opr_rejects(scenario):
    """GPU `lsqml`'s `opr` and `hold` refused as `Lsqml` refuses them, and a variation off the
    probe's planes or the scan's frames as `Lsqml.epoch` refuses it, the state kept; M1's
    passes run no variation."""
    scan, obj, probe, npos = scenario
    e, w, c = opr_start(probe, npos, 2, True)
    gpu = gpu_at(scan, obj, probe, True, None)
    o, p = gpu.object, gpu.probe

    def run(**kw):
        return gpu.lsqml(1, 1, 4, "poisson", 7, **kw)

    assert run() is None
    gpu = gpu_at(scan, obj, probe, True, None)
    with pytest.raises(ValueError, match="tuple of length 3"):
        run(opr=(e, w))
    for opr in ((e, None, c), (None, w, None)):
        with pytest.raises(TypeError):
            run(opr=opr)
    with pytest.raises(ValueError, match=r"\[nframe, k\] for k ≥ 1"):
        run(opr=(e[:0], w[:, :0], None))
    with pytest.raises(ValueError, match=f"{npos - 1} main-mode weights for {npos}"):
        run(opr=(e, w, c[1:]))
    for kw, hold in ((dict(), ("weights",)), (dict(), ("intensity",)),
                     (dict(opr=(e, w, None)), ("intensity",)),
                     (dict(opr=(e, w, c)), ("position",)),
                     (dict(opr=(e, w, c)), ("eigenmodes",))):
        with pytest.raises(ValueError, match="hold"):
            run(hold=hold, **kw)
    with pytest.raises(TypeError, match="unknown parameters"):
        run(position=None)
    with pytest.raises(ValueError,
                       match=f"{2 * (npos - 1)} OPR weights for 2 eigenmodes over {npos} frames"):
        run(opr=(e, w[1:], None))
    with pytest.raises(ValueError, match=r"planes on a probe of \[64, 64\]"):
        run(opr=(e[:, 1:, 1:], w, None))
    assert gpu.cycle == 0 and len(gpu.history["cycle"]) == 0
    np.testing.assert_array_equal(gpu.object, o)
    np.testing.assert_array_equal(gpu.probe, p)
    m1 = gpu_at(scan, obj, probe, False, None)
    with pytest.raises(RuntimeError, match="fused"):
        m1.lsqml(1, 1, 4, "poisson", 7, opr=(e, w, c))
