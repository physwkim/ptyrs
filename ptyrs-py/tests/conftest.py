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


def intensity(ptycho, obj, probe):
    """`ptycho`'s model intensity of `obj` and `probe`, `[nframe, ny, nx]` in the counts'
    detector order (zero frequency at the centre; `forward` has it at `[0, 0]`)."""
    icalc = (np.abs(ptycho.forward(obj, probe)) ** 2).sum(axis=1)
    return np.fft.fftshift(icalc, axes=(-2, -1))


def make_scenario(n=64, npos=30, nprobe=2, photons=2e5, seed=3):
    """A far-field scan of `npos` Fermat-spiral frames of `n` pixels drawn from a smooth object
    and `nprobe` Gaussian probe modes, Poisson counts with row 3 and column 20 masked, and a
    start of a random-phase object and the probe with a quadratic phase error, scaled to the
    counts. Returns (scan, object, probe, frame count)."""
    rng = np.random.default_rng(seed)
    pixel = WAVELENGTH * DISTANCE / (n * DETECTOR_PIXEL)
    i = np.arange(npos)
    r = n / 8 * np.sqrt(i + 0.5)
    th = i * np.pi * (3 - np.sqrt(5))
    y, x = r * np.sin(th) * pixel, r * np.cos(th) * pixel
    # base::required_obj_shape of the centred positions, with room to spare
    ext = int(np.ceil(np.abs(np.r_[y - y.mean(), x - x.mean()] / pixel).max())) + 1
    side = (2 * ext + n + 8 + 3) // 4 * 4
    yy, xx = np.mgrid[:side, :side] / side
    obj = (0.8 + 0.2 * np.cos(9 * yy) * np.sin(7 * xx)) * np.exp(1j * np.sin(5 * xx + 3 * yy))
    obj = obj[None].astype(np.complex128)
    v, u = (np.mgrid[:n, :n] - n / 2) / (n / 8)
    probe = np.stack([np.exp(-(u - m) ** 2 / 2 - v ** 2 / 2) / (1 + 2 * m) for m in range(nprobe)])
    probe = probe.astype(np.complex128)

    zeros = np.zeros((npos, n, n))
    model = ptyrs.Ptycho(ptyrs.Scan(zeros, y, x, WAVELENGTH, DISTANCE, DETECTOR_PIXEL), obj, probe)
    icalc = intensity(model, obj, probe)
    k = photons / icalc.sum(axis=(1, 2)).mean()
    counts = rng.poisson(icalc * k).clip(0, 65534).astype(np.uint16)
    counts[:, 3, :] = 65535
    counts[:, :, 20] = 65535
    scan = ptyrs.Scan(counts, y, x, WAVELENGTH, DISTANCE, DETECTOR_PIXEL)

    start_obj = np.exp(1j * (rng.uniform(size=obj.shape) - 0.5))
    start_probe = probe * np.exp(-0.5j * (u ** 2 + v ** 2))
    # the model's photons on the unmasked pixels to the counts'
    keep = counts != 65535
    calc = intensity(ptyrs.Ptycho(scan, start_obj, start_probe), start_obj, start_probe)
    start_probe *= np.sqrt(counts[keep].sum(dtype=np.float64) / calc[keep].sum())
    return scan, start_obj, start_probe, npos


@pytest.fixture(scope="session")
def scenario():
    return make_scenario()
