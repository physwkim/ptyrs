"""Writes ptypy_small_single.ptyr and ptypy_small_double.ptyr with PtyPy's own Ptycho.save_run
(h5rw via h5py), the fixtures tests/io_ptypy_fixture.rs reads: complex64 and complex128 storages
(PtyPy data_type 'single' and 'double'). Storage values, pixel sizes and origins are closed-form
so the test can recompute them.

    PYTHONPATH=<ptypy checkout> micromamba run -n ptyrs-ref python -P make_ptypy_small.py <out_dir>
"""
import sys

import numpy as np
from ptypy import utils as u
from ptypy.core import Ptycho

for data_type in ("single", "double"):
    p = u.Param()
    p.verbose_level = "critical"
    p.data_type = data_type
    p.io = u.Param(home=sys.argv[1], rfile=None)
    p.io.autosave = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.io.interaction = u.Param(active=False)
    p.scans = u.Param()
    p.scans.S = u.Param(name="Full")
    p.scans.S.coherence = u.Param(num_probe_modes=2)
    p.scans.S.data = u.Param(name="MoonFlowerScan", shape=8, num_frames=4, density=0.2,
                             add_poisson_noise=False, save=None)
    P = Ptycho(p, level=2)

    (ob,) = P.obj.storages.values()
    (pr,) = P.probe.storages.values()
    ob.psize = (1.5e-8, 2.5e-8)
    ob.origin = (-3e-7, 4e-7)
    _, y, x = np.indices(ob.data.shape)
    ob.data[:] = (y - 0.25 * x) + 1j * (0.5 * y + x)
    m, y, x = np.indices(pr.data.shape)
    pr.data[:] = (m + 0.5 * y) + 1j * (0.25 * m - x)
    P.save_run(alt_file=f"{sys.argv[1]}/ptypy_small_{data_type}.ptyr", kind="minimal")
