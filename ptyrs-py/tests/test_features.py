"""Ptycho and GpuPtycho features: the scale, background and position features refined by AP,
DM, RAAR, ML and rPIE on the GPU's fused passes against the CPU binding's f64, as the
engine's own `tests/engine_gpu.rs` `features_match_cpu` compares them, to its 1e-4 (offsets
to 1e-4 pixels); the background's detector order against the truth's, the offsets' axes
against the binding's own forward model; the LLK of given arrays under the features against
numpy's; the round trip and the refusals."""

import numpy as np
import pytest

import ptyrs
from conftest import (DETECTOR_PIXEL, DEVICE, DISTANCE, MASKED, WAVELENGTH, intensity,
                      make_truth, mask, rel, rel_l2, scaled_probe)

N, NPOS = 64, 30  # the fused passes' smallest side
NCYCLE = 20
TOL = 1e-4
LLK = ("poisson", "gaussian", "euclidian", "nb_photons_calc")
FUSED = dict(stack_size=7, llk_slots=3, fused=True)


def flux(f):
    """Frame `f`'s flux relative to the truth's."""
    return 1 + 0.4 * np.sin(2.3 * f)


def truth_background():
    """1 to 5 counts per pixel, smooth and not symmetric under a transpose."""
    y, x = np.mgrid[:N, :N]
    return 3 + 2 * np.sin(0.21 * y) * np.cos(0.13 * x + 0.5)


@pytest.fixture(scope="module")
def featured():
    """`featured(with_flux, with_background)`: Poisson counts of the truth times `flux` and
    plus `truth_background` where asked, rounded and masked, from the truth object and probe
    `scaled_probe`d to them."""
    y, x, obj, probe, icalc = make_truth(N, NPOS)
    rng = np.random.default_rng(5)
    base = rng.poisson(icalc * 2e5 / icalc.sum(axis=(1, 2)).mean()).astype(np.float64)

    def start(with_flux, with_background):
        c = base * flux(np.arange(NPOS))[:, None, None] if with_flux else base.copy()
        if with_background:
            c += truth_background()
        c = np.round(c)
        assert c.max() < MASKED
        c = mask(c.astype(np.uint16))
        scan = ptyrs.Scan(c, y, x, WAVELENGTH, DISTANCE, DETECTOR_PIXEL)
        return scan, obj, scaled_probe(scan, c, obj, probe)

    return start


def drawn_offsets():
    """Zero-mean offsets of up to 0.4 pixels per frame."""
    f = np.arange(NPOS)
    d = np.stack([0.4 * np.sin(1.7 * f + 0.3), 0.4 * np.cos(2.3 * f)], axis=1)
    return d - d.mean(axis=0)


def features_match_cpu(start, algorithm, params, ncycle, **features):
    """`ncycle` of `algorithm` from `start` modelling `features` on the GPU and the CPU: every
    LLK record, the object, the probe and the features within 1e-4. Returns the CPU state."""
    scan, obj, probe = start
    cpu = ptyrs.Ptycho(scan, obj, probe)
    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, **FUSED)
    cpu.set_features(**features)
    gpu.set_features(**features)
    getattr(cpu, algorithm)(ncycle, **params)
    getattr(gpu, algorithm)(ncycle, **params)
    assert gpu.cycle == cpu.cycle
    hg, hc = gpu.history, cpu.history
    assert list(hg["cycle"]) == list(hc["cycle"]) and hg["algorithm"] == hc["algorithm"]
    assert len(hc["cycle"]) == ncycle
    for key in LLK:
        err = rel(hg[key], hc[key]).max()
        assert err < TOL, f"llk {key}: {err:.2e}"
    assert rel_l2(gpu.object, cpu.object) < TOL
    assert rel_l2(gpu.probe, cpu.probe) < TOL
    fg, fc = gpu.features, cpu.features
    for key in ("scale", "background", "position"):
        assert (fg[key] is None) == (features.get(key) is None) == (fc[key] is None), key
    for key in ("scale", "background"):
        if fc[key] is not None:
            err = rel(fg[key], fc[key]).max()
            assert err < TOL, f"{key}: {err:.2e}"
    if fc["position"] is not None:
        err = np.abs(fg["position"] - fc["position"]).max()
        assert err < TOL, f"position: {err:.2e} pixels"
    return cpu


CYCLES = {
    "ap": dict(obj_inertia=0.01, probe_inertia=0.001, calc_llk=1, zero_phase_ramp=True),
    "dm": dict(calc_llk=1),
    "raar": dict(calc_llk=1),
    "ml": dict(update_object=True, update_probe=True, calc_llk=1, zero_phase_ramp=True),
}


def off_truth_background(b):
    """`b`'s RMS error relative to `truth_background` over the unmasked pixels; asserts the
    masked ones kept the flat start."""
    keep = np.ones((N, N), bool)
    keep[3, :] = keep[:, 20] = False
    assert (b[~keep] == 1).all()
    t = truth_background()
    return np.sqrt(((b - t)[keep] ** 2).sum() / (t[keep] ** 2).sum())


@pytest.mark.parametrize("algorithm", CYCLES)
def test_scale_and_background_match_cpu(featured, algorithm):
    """Both stepped features from a scale of 1 and a flat background of 1 on a scan with a
    flux and a background; on the CPU the background nears the truth's in detector order."""
    flat = np.ones((N, N))
    cpu = features_match_cpu(featured(True, True), algorithm, CYCLES[algorithm], NCYCLE,
                             scale=np.ones(NPOS), background=flat)
    assert off_truth_background(cpu.features["background"]) < 0.7 * off_truth_background(flat)


@pytest.mark.parametrize("algorithm", [*CYCLES, "rpie"])
def test_position_matches_cpu(featured, algorithm):
    """Offsets refined from `drawn_offsets` on a scan drawn at the view corners; rPIE in
    pty-chi's batches of 4."""
    params = CYCLES.get(algorithm, dict(obj_alpha=0.5, probe_alpha=0.5, batch_size=4,
                                        calc_llk=1))
    d0 = drawn_offsets()
    cpu = features_match_cpu(featured(False, False), algorithm, params, NCYCLE, position=d0)
    assert (cpu.features["position"] ** 2).mean() < (d0 ** 2).mean()


def object_plane_phase(p, obj, probe):
    """`q`, the far field's object-plane phase, of `p`'s `forward` (`FT(fftshift(P·O·q))`)."""
    psi = p.forward(np.ones_like(obj), np.ones_like(probe))[0, 0]
    return np.fft.ifftshift(np.fft.ifft2(psi))


def moved_probe(p, obj, probe, d):
    """`probe` whose `P·q` is rolled by the whole-pixel offset `d` `[dy, dx]`."""
    q = object_plane_phase(p, obj, probe)
    return np.roll(probe * q, d, axis=(1, 2)) / q


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("step", [1, -2])
def test_position_axes(featured, axis, step):
    """An offset of `step` whole pixels on `axis` (0 `dy`, 1 `dx`) of every frame models
    `P·q` rolled by it: the LLK record of an AP cycle's start is the featureless LLK of that
    probe."""
    scan, obj, probe = featured(False, False)
    plain = ptyrs.Ptycho(scan, obj, probe)
    d = np.zeros(2)
    d[axis] = step
    want = plain.llk(obj, moved_probe(plain, obj, probe, d.astype(int)))
    p = ptyrs.Ptycho(scan, obj, probe)
    p.set_features(position=np.tile(d, (NPOS, 1)))
    p.ap(1, calc_llk=1, zero_phase_ramp=False)
    for key in LLK:
        assert rel(p.history[key][0], want[key]) < 1e-10, key


def test_llk_models_the_features():
    """`Ptycho.llk` of given arrays models the state's features: PyNX's normalised LLK over
    the unmasked pixels of `scale · I + background`, `I` the detector-order intensity of
    `P·q` moved by a whole-pixel offset, against numpy's; set no features, the plain LLK."""
    y, x, obj, probe, icalc = make_truth(N, NPOS)
    rng = np.random.default_rng(9)
    counts = mask(rng.poisson(icalc * 2e5 / icalc.sum(axis=(1, 2)).mean()).astype(np.uint16))
    scan = ptyrs.Scan(counts, y, x, WAVELENGTH, DISTANCE, DETECTOR_PIXEL)
    probe = scaled_probe(scan, counts, obj, probe)
    plain = ptyrs.Ptycho(scan, obj, probe)
    scale, background = flux(np.arange(NPOS)), truth_background()
    calc = (scale[:, None, None] * intensity(plain, obj, moved_probe(plain, obj, probe, (1, -2)))
            + background)
    keep, iobs = counts != MASKED, counts.astype(np.float64)
    assert (iobs[keep] == 0).any()
    terms = dict(
        poisson=calc - iobs + np.where(iobs > 0, iobs * np.log(np.maximum(iobs, 1) / calc), 0),
        gaussian=(iobs - calc) ** 2 / (iobs + 1),
        euclidian=4 * (np.sqrt(iobs) - np.sqrt(calc)) ** 2,
    )
    want = {k: v[keep].sum() / keep.sum() for k, v in terms.items()}
    want["nb_photons_calc"] = calc[keep].sum()
    p = ptyrs.Ptycho(scan, obj, probe)
    p.set_features(scale=scale, background=background, position=np.tile([1.0, -2.0], (NPOS, 1)))
    got = p.llk(obj, probe)
    for key in LLK:
        assert rel(got[key], want[key]) < 1e-10, key
    p.set_features()
    assert p.llk(obj, probe) == plain.llk(obj, probe)


def test_features_round_trip_and_refusals(featured):
    """Set features read back as set on both backends; the binding refuses a background not
    `[ny, nx]` and offsets not `[nframe, 2]`, the engine bad counts and values, M1 any feature,
    and a refused set leaves the features modelled."""
    scan, obj, probe = featured(True, True)
    cpu = ptyrs.Ptycho(scan, obj, probe)
    gpu = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, **FUSED)
    m1 = ptyrs.GpuPtycho(scan, obj, probe, DEVICE, stack_size=7, fused=False)
    for p in (cpu, gpu, m1):
        assert p.features == dict(scale=None, background=None, position=None)
    with pytest.raises(ValueError, match="fused: true"):
        m1.set_features(scale=np.ones(NPOS))
    m1.set_features()

    half = dict(scale=np.full(NPOS, 0.5), background=2 + truth_background(),
                position=drawn_offsets())
    cpu.set_features(**half)
    gpu.set_features(**half)
    for key, v in half.items():
        assert np.array_equal(cpu.features[key], v), key
        # the device keeps f32
        assert np.array_equal(gpu.features[key], v.astype(np.float32)), key

    def one_bad(v, bad):
        v = v.copy()
        v.flat[v.size // 2] = bad
        return v

    ones, bg = np.ones(NPOS), half["background"]
    bad = [dict(scale=ones[1:]), dict(background=bg[:, 1:]), dict(background=bg.T[:, :-1]),
           dict(background=bg.ravel()[None]), dict(position=np.zeros((NPOS, 3))),
           dict(position=np.zeros((NPOS - 1, 2)))]
    bad += [dict(scale=one_bad(ones, v)) for v in (0, -1, np.nan, np.inf)]
    bad += [dict(background=one_bad(bg, v)) for v in (-1, np.nan, np.inf)]
    bad += [dict(position=one_bad(np.zeros((NPOS, 2)), v)) for v in (np.nan, np.inf)]
    for b in bad:
        for p in (cpu, gpu):
            with pytest.raises(ValueError):
                p.set_features(**{**half, **b})
            for key, v in half.items():
                assert np.allclose(p.features[key], v), key
    with pytest.raises(ValueError):
        cpu.rpie(1, batch_size=4)
    with pytest.raises(ValueError):
        gpu.rpie(1, batch_size=4)
