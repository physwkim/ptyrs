"""Compare ptyrs-fft GPU output (examples/dump) with numpy.fft in float64.

usage: cargo run -p ptyrs-fft --release --example dump -- DIR && python check_numpy.py DIR
"""
import sys
import numpy as np

d = sys.argv[1]
n, batch = 256, 64
load = lambda f: np.fromfile(f"{d}/{f}.f32", dtype=np.float32).reshape(batch, n, n, 2)
cplx = lambda a: a[..., 0].astype(np.float64) + 1j * a[..., 1].astype(np.float64)
x, fwd, inv = (cplx(load(f)) for f in ("in", "fwd", "inv"))
refs = {"forward": (fwd, np.fft.fft2(x, axes=(1, 2))),
        "inverse": (inv, np.fft.ifft2(x, axes=(1, 2)) * n * n)}  # cuFFT inverse is unnormalized
ok = True
for name, (g, r) in refs.items():
    max_rel = np.abs(g - r).max() / np.abs(r).max()
    rms_rel = np.sqrt((np.abs(g - r) ** 2).sum() / (np.abs(r) ** 2).sum())
    print(f"{name}: max-rel {max_rel:.3e}  rms-rel {rms_rel:.3e}")
    ok &= max_rel <= 1e-5
sys.exit(0 if ok else 1)
