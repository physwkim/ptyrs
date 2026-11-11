"""Writes pynx_small.cxi with PyNX's own save_ptycho_data_cxi (libhdf5 via h5py), the
fixture tests/pynx_fixture.rs reads. Values are closed-form so the test can recompute them.

    micromamba run -n ptyrs-ref python -I make_pynx_small.py <out.cxi>
"""
import sys

import numpy as np
from pynx.ptycho.ptycho import save_ptycho_data_cxi

nz, ny, nx = 4, 4, 6
iobs = (np.arange(nz * ny * nx) * 37 % 500).astype(np.float32).reshape(nz, ny, nx)
iobs[2, 3, 1] = -1  # per-frame masked pixel (PyNX iobs < 0)
mask = np.zeros((ny, nx), dtype=bool)
mask[1, 2] = True
x = np.arange(nz) * 1.1e-7 + 1e-9 / 3
y = -np.arange(nz) * 0.7e-7
save_ptycho_data_cxi(sys.argv[1], iobs, pixel_size=75e-6, wavelength=1.5e-10,
                     detector_distance=1.2, x=x, y=y, mask=mask, instrument="fixture",
                     overwrite=True)
