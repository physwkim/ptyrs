"""The `ptyrs` module's tests: each new binding against the Rust result it wraps, by the CPU
binding or an engine call.

    ptyrs-py/scripts/build.sh &&
        PYTHONDONTWRITEBYTECODE=1 $PYO3_PYTHON -m pytest -p no:cacheprovider ptyrs-py/tests

The module is imported from PTYRS_PY (default ~/work/ptyrs-scratch/py, build.sh's OUT); GPU
tests run on CUDA device PTYRS_DEVICE (default 0).
"""

import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("PTYRS_PY", os.path.expanduser("~/work/ptyrs-scratch/py")))
import ptyrs  # noqa: E402

DEVICE = int(os.environ.get("PTYRS_DEVICE", "0"))
WAVELENGTH, DISTANCE, DETECTOR_PIXEL = 1.5e-10, 1.0, 75e-6


def rel_l2(a, b):
    """`|a − b| / |b|` over the whole array."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def rel(a, b):
    """Elementwise `|a − b| / |b|`, or `|a|` where `b` is 0 (a clamped step)."""
    a, b = np.asarray(a), np.asarray(b)
    return np.where(b == 0, np.abs(a), np.abs(a - b) / np.where(b == 0, 1, np.abs(b)))


MASKED = 65535


def intensity(ptycho, obj, probe):
    """`ptycho`'s model intensity of `obj` and `probe`, `[nframe, ny, nx]` in the counts'
    detector order (zero frequency at the centre; `forward` has it at `[0, 0]`)."""
    icalc = (np.abs(ptycho.forward(obj, probe)) ** 2).sum(axis=1)
    return np.fft.fftshift(icalc, axes=(-2, -1))


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def sim_object(side):
    """`base::sim::object`: a Siemens star and three Gaussian features in phase and amplitude,
    `[1, side, side]`."""
    v, u = (np.mgrid[:side, :side] - side / 2) / (side / 2)
    r = np.hypot(u, v)

    def blob(cu, cv, w):
        return np.exp(-((u - cu) ** 2 + (v - cv) ** 2) / (2 * w * w))

    star = (0.5 * (1 + np.tanh(3 * np.cos(12 * np.arctan2(v, u)))) * smoothstep(0.08, 0.12, r)
            * (1 - smoothstep(0.55, 0.6, r)))
    phase = star + 0.6 * blob(0.55, -0.45, 0.12) - 0.4 * blob(-0.5, 0.5, 0.1)
    amp = 0.95 - 0.3 * blob(-0.45, -0.55, 0.15) - 0.15 * star
    return (amp * np.exp(1j * phase))[None]


def sim_probe(n, nmode):
    """`base::sim::probe`: `nmode` Hermite-Gauss modes on one Gaussian envelope (σ = n/12)
    with a defocus curvature, mode k carrying power 4^-k of a total of 1."""
    hermite = (lambda t: np.ones_like(t), lambda t: 2 * t, lambda t: 4 * t * t - 2)
    orders = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)][:nmode]
    v, u = (np.mgrid[:n, :n] - n / 2) / (n / 12)
    r2 = u * u + v * v
    w = 0.25 ** np.arange(nmode)
    modes = []
    for k, (a, b) in enumerate(orders):
        m = hermite[a](u) * hermite[b](v) * np.exp(-r2 / 2) * np.exp(1j * (1.5 * r2 + 0.3 * k))
        modes.append(m * np.sqrt(w[k] / w.sum()) / np.linalg.norm(m))
    return np.stack(modes)


def make_truth(n=64, npos=30, nprobe=2):
    """The engine's parity scenario (`tests/common/engine/scenario.rs` `truth`) on `n`-pixel
    frames: `npos` Fermat-spiral positions about `n / 8` pixels apart, `sim_object` and
    `nprobe` modes of `sim_probe`, and their `intensity`. Returns (y, x, object, probe,
    intensity)."""
    pixel = WAVELENGTH * DISTANCE / (n * DETECTOR_PIXEL)
    k = np.arange(npos)
    r = n / 8 * pixel / np.sqrt(np.pi) * np.sqrt(k)
    th = k * np.pi * (3 - np.sqrt(5))
    y, x = r * np.sin(th), r * np.cos(th)
    # base::required_obj_shape of the centred positions, with room to spare
    ext = int(np.ceil(np.abs(np.r_[y - y.mean(), x - x.mean()] / pixel).max())) + 1
    side = (2 * ext + n + 8 + 3) // 4 * 4
    obj, probe = sim_object(side), sim_probe(n, nprobe)
    zeros = np.zeros((npos, n, n))
    model = ptyrs.Ptycho(ptyrs.Scan(zeros, y, x, WAVELENGTH, DISTANCE, DETECTOR_PIXEL), obj, probe)
    return y, x, obj, probe, intensity(model, obj, probe)


def mask(counts):
    """`counts` with row 3 and column 20 of every frame masked."""
    counts[:, 3, :] = MASKED
    counts[:, :, 20] = MASKED
    return counts


def scaled_probe(scan, counts, obj, probe):
    """`probe` scaled so that the model photons of `obj` and it on the unmasked pixels of
    `scan`'s `counts` are the counts'."""
    keep = counts != MASKED
    calc = intensity(ptyrs.Ptycho(scan, obj, probe), obj, probe)
    return probe * np.sqrt(counts[keep].sum(dtype=np.float64) / calc[keep].sum())


def make_scenario(n=64, npos=30, nprobe=2, photons=2e5, seed=3):
    """A far-field scan of `make_truth`, Poisson counts of `photons` per frame on average,
    `mask`ed, and a start of a random-phase object and the probe with a quadratic phase
    error, `scaled_probe`. Returns (scan, object, probe, frame count)."""
    rng = np.random.default_rng(seed)
    y, x, obj, probe, icalc = make_truth(n, npos, nprobe)
    k = photons / icalc.sum(axis=(1, 2)).mean()
    counts = mask(rng.poisson(icalc * k).clip(0, MASKED - 1).astype(np.uint16))
    scan = ptyrs.Scan(counts, y, x, WAVELENGTH, DISTANCE, DETECTOR_PIXEL)
    start_obj = np.exp(1j * (rng.uniform(size=obj.shape) - 0.5))
    v, u = (np.mgrid[:n, :n] - n / 2) / (n / 8)
    start_probe = scaled_probe(scan, counts, start_obj, probe * np.exp(-0.5j * (u ** 2 + v ** 2)))
    return scan, start_obj, start_probe, npos


@pytest.fixture(scope="session")
def scenario():
    return make_scenario()
