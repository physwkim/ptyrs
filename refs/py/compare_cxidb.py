"""ptyrs GPU against PyNX CUDA DM and ML and against pty-chi LSQML on measured data, CXIDB entry
65 (ALS 5.3.2, gold balls, 1 keV, 800 frames of 621x621 pixels, CC0), every side from one start,
through the `ptyrs` module.

    python -I compare_cxidb.py prepare RAW OUT [--device 0] [--module DIR]
    python -I compare_cxidb.py dm OUT [--cycles 200] [--device 0] [--module DIR]
    python -I compare_cxidb.py ml OUT [--cycles 100] [--fix-gamma] [--device 0] [--module DIR]
    python -I compare_cxidb.py lsqml OUT [--epochs 20] [--batch 32] [--seed 7] [--device 0]
        [--module DIR]

RAW is the entry's AuBalls_700ms_30nmStep_3_full.cxi, read with h5py alone. ptyrs::io::read_cxi
refuses it (its 621-pixel frames are odd, its translation is [N, 3] where read_cxi takes [3, N]),
so `prepare` writes a scratch conversion, OUT/scan.cxi, in the layout read_cxi reads: each frame
less the mean of data_dark, rounded and clipped at 0 (the counts stay ADU, no gain is given);
the FRAME x FRAME crop centred on the centroid of the mean frame; a pixel at 65535 in any frame
masked in every frame (detector_1/mask: pty-chi's mask is one plane); frames whose valid sum is
under MONITOR times the median dropped (PyNX's monitor rule); energy, distance and pixel as
given. The start, OUT/init.h5, is a flat object of PyNX's get_required_obj_shape(margin=16) and
one probe mode, the inverse FFT of the square root of the mean frame (masked pixels at the
median of its bright field), scaled so the engine's far field of the flat object carries the
mean frame's valid counts. The translation's axes and signs are those of the ORIENTATIONS
candidate with the lowest Poisson LLK after ORIENT_CYCLES GPU DM cycles; a candidate and its
180-degree turn are one reconstruction (the twin), so they pair up.

Every side reads OUT/scan.cxi and OUT/init.h5 with one wavelength (read_cxi's), the same
positions, mask and start. PyNX runs in a child process (`pynx`, its output in
OUT/<algo>_pynx.log), so pycuda's context and the engine's stay apart. Each run writes its
final object and probe as OUT/<algo>_<side>.h5 and prints the FRC of ptyrs against each
reference (examples/frc_pair.rs, base::frc) over the bounding box of the window centres.

dm: GpuPtycho.dm(calc_llk=1) against PyNX CUDA DM(calc_llk=1) with compare_pynx.py's figures:
the largest relative difference of each normalised LLK over the cycles (both record the estimate
a cycle starts from) and the final object's and probe's relative L2 up to a complex scale.
ml: WARMUP GPU AP cycles from init.h5 (OUT/init_ml.h5), then GpuPtycho.ml against PyNX CUDA ML
stock and with its Hann filter off, compare_pynx_ml.py's figures; --fix-gamma as there. On this
data the stock ML stops moving within a few cycles, so its figures are well above the Hann-off
ones.
lsqml: GpuPtycho.lsqml("amplitude") against pty-chi CPU f64 LSQML Gaussian
(compare_lsqml.ptychi_task, the mask as its valid_pixel_mask), both over the engine's
random_batches order, with compare_gpu_multislice.py's figures; every state's LLK is the CPU f64
Ptycho.llk. There is no truth: the figures are side against side.

Exits 1 if the starts differ (the LLK record of the start; pty-chi's start far field and its LLK
of the start), 2 if a side diverges (m3ref.exit_unless_converging): ML's on the Poisson LLK,
DM's and LSQML's on the Gaussian.
"""
import argparse
import importlib.util
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories

FRAME = 256
SATURATED = 65535
MONITOR = 0.1
# ORIENTATIONS[k]: the scan's (x, y) from the file's translation columns x and y
ORIENTATIONS = {
    "x, y": lambda x, y: (x, y), "-x, -y": lambda x, y: (-x, -y),
    "-x, y": lambda x, y: (-x, y), "x, -y": lambda x, y: (x, -y),
    "y, x": lambda x, y: (y, x), "-y, -x": lambda x, y: (-y, -x),
    "-y, x": lambda x, y: (-y, x), "y, -x": lambda x, y: (y, -x),
}
ORIENT_CYCLES = 50
WARMUP = 20
KEYS = ("poisson", "gaussian", "euclidian", "nb_photons_calc")
# the start's LLK record, ptyrs against PyNX: 5x the worst measured (2.1e-6, DM's Poisson)
START_RTOL = 1.1e-5
# pty-chi's start far field and its own Gaussian LLK of the start against ptyrs CPU f64: 5x the
# worst measured (1.5e-15, 6.1e-13)
FAR_RTOL, LLK_RTOL = 1e-14, 4e-12
CXI = {
    "data": "entry_1/instrument_1/detector_1/data",
    "dark": "entry_1/instrument_1/detector_1/data_dark",
    "mask": "entry_1/instrument_1/detector_1/mask",
    "distance": "entry_1/instrument_1/detector_1/distance",
    "x_pixel": "entry_1/instrument_1/detector_1/x_pixel_size",
    "y_pixel": "entry_1/instrument_1/detector_1/y_pixel_size",
    "energy": "entry_1/instrument_1/source_1/energy",
    "translation": "entry_1/sample_1/geometry_1/translation",
}
# ptyrs::base::Geometry::wavelength_from_energy_joule
KEV_ANGSTROM, JOULE_PER_KEV = 12.3984, 1.60218e-16


def sibling(path, name):
    """Import <ROOT>/<path>/<name>.py."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_avg():
    return "load {:.0f} {:.0f} {:.0f}".format(*os.getloadavg())


class Data:
    """OUT/scan.cxi and OUT/init.h5 as every side takes them."""

    def __init__(self, out):
        import h5py
        with h5py.File(f"{out}/scan.cxi", "r") as f:
            self.counts = f[CXI["data"]][()]
            self.mask = f[CXI["mask"]][()] != 0
            t = f[CXI["translation"]][()]
            energy = float(f[CXI["energy"]][()])
            self.distance = float(f[CXI["distance"]][()])
            self.pixel = float(f[CXI["x_pixel"]][()])
        self.x, self.y = t[0], t[1]
        self.wavelength = KEV_ANGSTROM * 1e-10 / (energy / JOULE_PER_KEV)
        self.geometry = dict(wavelength_m=self.wavelength, distance_m=self.distance,
                             detector_pixel_m=self.pixel)
        self.obj, self.probe = read_modes(f"{out}/init.h5")

    def iobs(self):
        """PyNX's and the engine's float counts, masked pixels -1."""
        iobs = self.counts.astype(np.float64)
        iobs[:, self.mask] = -1
        return iobs

    def scan(self, ptyrs):
        return ptyrs.Scan(self.iobs(), self.y, self.x, self.wavelength, self.distance,
                          self.pixel)


def read_modes(path):
    import h5py
    with h5py.File(path, "r") as f:
        return f["object"][()], f["probe"][()]


def write_modes(path, obj, probe, **more):
    """ptyrs::io::write_modes's layout (h5py's complex128), and any arrays in `more`."""
    import h5py
    with h5py.File(path, "w") as f:
        f["object"] = np.asarray(obj, np.complex128)
        f["probe"] = np.asarray(probe, np.complex128)
        for k, v in more.items():
            f[k] = v


def prepare(args):
    import h5py
    from pynx.ptycho.ptycho import PtychoData

    with h5py.File(args.raw, "r") as f:
        dark = f[CXI["dark"]][()].astype(np.float64).mean(0)
        raw = f[CXI["data"]][()]
        t = f[CXI["translation"]][()].astype(np.float64)
        energy = float(f[CXI["energy"]][()])
        distance = float(f[CXI["distance"]][()])
        pixel = float(f[CXI["x_pixel"]][()])
        assert float(f[CXI["y_pixel"]][()]) == pixel
    nframe, ny, nx = raw.shape
    assert t.shape == (nframe, 3), t.shape
    saturated = (raw == SATURATED).any(0)
    mean = np.zeros((ny, nx))
    for s in range(0, nframe, 100):
        mean += np.clip(raw[s:s + 100] - dark, 0, None).sum(0)
    yy, xx = np.indices((ny, nx))
    centre = [int(round((mean * c).sum() / mean.sum())) for c in (yy, xx)]
    crop = tuple(slice(c - FRAME // 2, c + FRAME // 2) for c in centre)
    assert all(0 <= s.start and s.stop <= n for s, n in zip(crop, (ny, nx))), crop
    counts = np.clip(np.rint(raw[(slice(None),) + crop] - dark[crop]), 0, None)
    del raw
    mask = saturated[crop]
    counts[:, mask] = 0
    counts = counts.astype(np.uint16)
    sums = counts.sum((1, 2), dtype=np.int64)
    keep = sums >= MONITOR * np.median(sums)
    counts, t = counts[keep], t[keep]
    print(f"{args.raw}: {nframe} frames of {ny}x{nx}, dark mean {dark.mean():.2f} ADU; "
          f"centroid of the mean frame (y, x) {centre}, crop rows {crop[0].start}..{crop[0].stop}"
          f" columns {crop[1].start}..{crop[1].stop}; {int(mask.sum())} pixels saturated in some "
          f"frame, masked; frames dropped by the monitor rule {np.flatnonzero(~keep).tolist()}")

    wavelength = KEV_ANGSTROM * 1e-10 / (energy / JOULE_PER_KEV)

    def flat(x, y):
        """The flat object PyNX sizes for windows at (x, y), and its pixel."""
        d = PtychoData(iobs=np.zeros((1, FRAME, FRAME), np.float32), positions=(x, y),
                       detector_distance=distance, pixel_size_detector=pixel,
                       wavelength=wavelength)
        shape = d.get_required_obj_shape(margin=16)
        return np.ones((1, *shape), np.complex128), d.pixel_size_object()[0]

    obj, opix = flat(t[:, 0], t[:, 1])

    m = counts[:, ~mask].mean(0)
    amp = np.zeros((FRAME, FRAME))
    amp[~mask] = m
    bright = m >= 0.5 * m.max()
    amp[mask] = np.median(m[bright])
    probe = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.sqrt(amp))))[None]

    sys.path.insert(0, args.module)
    import ptyrs
    one = ptyrs.Scan(counts[:1], t[:1, 1].copy(), t[:1, 0].copy(), wavelength, distance, pixel)
    icalc = np.fft.fftshift((np.abs(ptyrs.Ptycho(one, obj, probe).forward(obj, probe)) ** 2)
                            .sum(1)[0])
    probe *= np.sqrt(m.sum() / icalc[~mask].sum())
    print(f"object pixel {opix:.4e} m; probe from the mean frame ({int(bright.sum())} "
          f"bright-field pixels, masked ones at their median {np.median(m[bright]):.1f}), scaled "
          f"to {m.sum():.4e} counts per frame", flush=True)

    raw16 = counts.copy()
    raw16[:, mask] = SATURATED
    llk = {}
    for name, turn in ORIENTATIONS.items():
        x, y = turn(t[:, 0], t[:, 1])
        obj, _ = flat(x, y)
        scan = ptyrs.Scan(raw16, y.copy(), x.copy(), wavelength, distance, pixel)
        gpu = ptyrs.GpuPtycho(scan, obj, probe, args.device)
        gpu.dm(ORIENT_CYCLES, calc_llk=0)
        llk[name] = gpu.llk()["poisson"]
        print(f"  ({name}) object {obj.shape[1:]}: Poisson LLK {llk[name]:.6e} after "
              f"{ORIENT_CYCLES} GPU DM cycles ({load_avg()})", flush=True)
    best = min(llk, key=llk.get)
    print(f"scan (x, y) from the file's translation (x, y): taking ({best})")
    x, y = ORIENTATIONS[best](t[:, 0], t[:, 1])
    obj, _ = flat(x, y)

    os.makedirs(args.out, exist_ok=True)
    with h5py.File(f"{args.out}/scan.cxi", "w") as f:
        f["cxi_version"] = 140
        f[CXI["data"]] = counts
        f[CXI["mask"]] = mask.astype(np.uint8)
        f[CXI["distance"]] = distance
        f[CXI["x_pixel"]] = pixel
        f[CXI["y_pixel"]] = pixel
        f[CXI["energy"]] = energy
        f[CXI["translation"]] = np.stack([x, y, t[:, 2]])
        f["entry_1/data_1/data"] = h5py.SoftLink("/" + CXI["data"])
    write_modes(f"{args.out}/init.h5", obj, probe)
    print(f"wrote {args.out}/scan.cxi ({len(counts)} frames) and {args.out}/init.h5")


def history(gpu, n):
    h = gpu.history
    assert (np.asarray(h["cycle"]) == np.arange(n)).all(), h["cycle"]
    return np.column_stack([np.asarray(h[k]) for k in KEYS])


def rel_l2(a, b):
    """min over complex k of |a - k b| / |a|, and k."""
    k = np.vdot(b, a) / np.vdot(b, b)
    return np.linalg.norm(a - k * b) / np.linalg.norm(a), k


def pynx(args):
    """The PyNX CUDA side of `dm` or `ml` in this process; writes OUT/<algo>_pynx*.h5."""
    pm = sibling("ptyrs/scripts/engine", "compare_pynx_ml")
    cuop = pm.cuop
    tag = ""
    if args.fix_gamma:
        pm.fix_gamma()
        tag = "_fixgamma"
    data = Data(args.out)
    start = data.obj, data.probe
    if args.algo == "ml":
        start = read_modes(f"{args.out}/init_ml.h5")
    build = (data.iobs(), (data.x, data.y), data.distance, data.pixel, data.wavelength, *start)
    ml = lambda: cuop.ML(update_object=True, update_probe=True, calc_llk=1)
    runs = {"dm": [("pynx", lambda: cuop.DM(calc_llk=1), None)],
            "ml": [(f"pynx_stock{tag}", ml, True), (f"pynx_nohann{tag}", ml, False)]}[args.algo]
    if args.algo == "ml":
        # set_hann needs the processing unit, which a throwaway run sets up (compare_pynx_ml)
        cuop.ScaleObjProbe() * pm.build(*build)
    for name, op, hann in runs:
        if hann is not None:
            pm.set_hann(hann)
        p = op() ** args.cycles * pm.build(*build)
        h = p.history
        llk = np.array([[float(h[k][c]) for k in pm.KEYS] for c in range(args.cycles)])
        write_modes(f"{args.out}/{args.algo}_{name}.h5", p.get_obj(), p.get_probe(), llk=llk)
        print(f"PyNX {name}: {args.cycles} cycles ({load_avg()})", flush=True)


def run_pynx(args, algo):
    """The `pynx` child, its output (PyNX prints every LLK) in OUT/<algo>_pynx.log."""
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "pynx", args.out, algo,
           "--cycles", str(args.cycles)] + (["--fix-gamma"] if args.fix_gamma else [])
    log = f"{args.out}/{algo}_pynx.log"
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    with open(log) as f:
        lines = f.read().splitlines()
    print("\n".join(x for x in lines if x.startswith("PyNX ")), flush=True)
    if rc:
        print("\n".join(lines[-20:]))
        raise subprocess.CalledProcessError(rc, cmd)


def centres_region(data):
    """Rows and columns spanned by the window centres (ptypy_compare's region)."""
    m3ref = sibling("refs/ptychi", "m3ref")
    pos = np.round(m3ref.pynx_positions(data.x, data.y, (FRAME, FRAME), data.geometry))
    corner = (np.array(data.obj.shape[-2:]) - FRAME) // 2 + pos + FRAME // 2
    return [(int(corner[:, d].min()), int(corner[:, d].max()) + 1) for d in (0, 1)]


def frc(out, region, a, b):
    """examples/frc_pair.rs on OUT/<a>.h5 and OUT/<b>.h5."""
    (y0, y1), (x0, x1) = region
    cmd = ["cargo", "run", "--release", "-q", "--manifest-path", f"{ROOT}/Cargo.toml", "-p",
           "ptyrs", "--example", "frc_pair", "--", f"{out}/{a}.h5", f"{out}/{b}.h5",
           str(y0), str(y1), str(x0), str(x1)]
    line = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    print(f"  {a} vs {b}: {line}")


def compare(name, ours, theirs):
    """compare_pynx.py's figures of `theirs` against `ours`, both [cycle, KEYS]."""
    r = np.abs(theirs - ours) / np.abs(ours)
    print(f"--- {name}: max |rel diff| vs ptyrs over {len(r)} cycles")
    for k, v in zip(KEYS, r.T):
        print(f"  {k:16s} {v.max():.3e} (cycle {int(v.argmax())})")
    for c in sorted({0, 1, 2, 5, 10, 20, len(r) // 2, len(r) - 1}):
        if c < len(r):
            print(f"  cycle {c:4d} poisson ptyrs {ours[c, 0]:.10e} PyNX {theirs[c, 0]:.10e} "
                  f"rel {r[c, 0]:.2e}")
    return r[0].max()


def run_cuda(args, algo):
    """`dm` or `ml`: GpuPtycho against PyNX CUDA from one start."""
    m3ref = sibling("refs/ptychi", "m3ref")
    sys.path.insert(0, args.module)
    import ptyrs
    data = Data(args.out)
    scan = data.scan(ptyrs)
    print(f"{algo}: {len(data.counts)} frames, {args.cycles} cycles ({load_avg()})", flush=True)
    if algo == "ml":
        gpu = ptyrs.GpuPtycho(scan, data.obj, data.probe, args.device)
        gpu.ap(WARMUP, obj_inertia=0.01, probe_inertia=0.001, calc_llk=1, zero_phase_ramp=True)
        write_modes(f"{args.out}/init_ml.h5", gpu.object, gpu.probe)
        start = read_modes(f"{args.out}/init_ml.h5")
        gpu = ptyrs.GpuPtycho(scan, *start, args.device)
        gpu.ml(args.cycles, update_object=True, update_probe=True, calc_llk=1,
               zero_phase_ramp=True)
    else:
        gpu = ptyrs.GpuPtycho(scan, data.obj, data.probe, args.device)
        gpu.dm(args.cycles, calc_llk=1)
    ours = history(gpu, args.cycles)
    obj, probe = gpu.object, gpu.probe
    write_modes(f"{args.out}/{algo}_ptyrs.h5", obj, probe, llk=ours)
    del gpu
    run_pynx(args, algo)

    sides = ["pynx"] if algo == "dm" else [f"pynx_{s}" + ("_fixgamma" if args.fix_gamma else "")
                                           for s in ("stock", "nohann")]
    # DM minimises no LLK: on these ADU counts its Poisson LLK rises off the bright field while
    # its Gaussian falls, so DM's divergence check reads the Gaussian
    col = KEYS.index("gaussian" if algo == "dm" else "poisson")
    curves, starts = {"ptyrs": ours[:, col]}, []
    for side in sides:
        import h5py
        with h5py.File(f"{args.out}/{algo}_{side}.h5", "r") as f:
            theirs, o, p = f["llk"][()], f["object"][()], f["probe"][()]
        starts.append(compare(side, ours, theirs))
        (eo, ko), (ep, kp) = rel_l2(obj, o), rel_l2(probe, p)
        print(f"  final object rel L2 {eo:.3e} (scale {ko:.6f})  probe rel L2 {ep:.3e} "
              f"(scale {kp:.6f})")
        curves[side] = theirs[:, col]
    print(f"FRC of the final objects, ptyrs against each ({load_avg()}):")
    region = centres_region(data)
    for side in sides:
        frc(args.out, region, f"{algo}_ptyrs", f"{algo}_{side}")
    m3ref.exit_unless_converging(curves)
    return max(starts) <= START_RTOL


def ptychi_task(data, obj, probe, batch, epochs):
    """compare_lsqml.ptychi_task (Gaussian) on `data`, its mask as pty-chi's valid_pixel_mask;
    the task and m3ref.Dataset stand-in it was built from."""
    lsqml = sibling("refs/py", "compare_lsqml")
    m3ref = lsqml.m3ref
    import ptychi.api.task as at

    ds = object.__new__(m3ref.Dataset)
    ds.frame, ds.geometry = (FRAME, FRAME), data.geometry
    ds.pixel = data.wavelength * data.distance / (FRAME * data.pixel)
    ds.pos_px = m3ref.pynx_positions(data.x, data.y, ds.frame, data.geometry)
    ds.counts = data.counts
    real = at.PtychographyTask

    def masked(opts, **kw):
        kw["valid_pixel_mask"] = ~data.mask
        return real(opts, **kw)

    at.PtychographyTask = masked
    try:
        return lsqml.ptychi_task(ds, obj, probe, "gaussian", batch, epochs), ds, lsqml
    finally:
        at.PtychographyTask = real


def run_lsqml(args):
    import time

    import torch
    import ptychi.utils as utils
    from torch.utils.data import DataLoader

    sys.path.insert(0, args.module)
    import ptyrs
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)
    rb = sibling("refs/f64", "random_batches")
    data = Data(args.out)
    scan = data.scan(ptyrs)
    obj, probe = data.obj, data.probe
    cpu = ptyrs.Ptycho(scan, obj, probe)
    task, ds, lsqml = ptychi_task(data, obj, probe, args.batch, args.epochs)
    m3ref = lsqml.m3ref
    sampler = rb.Sampler(len(data.counts), args.batch, args.seed)
    rec = task.reconstructor
    rec.dataloader = DataLoader(rec.dataset, batch_sampler=sampler)

    ours = (np.abs(cpu.forward(obj, probe)) ** 2).sum(axis=1)
    theirs = lsqml.ptychi_icalc(task)
    far = np.abs(ours - theirs).max() / theirs.max()
    valid = rec.dataset.valid_pixel_mask.numpy()
    sums = m3ref.llk_sums(torch.from_numpy(theirs[:, valid]),
                          rec.dataset.patterns[:, torch.from_numpy(valid)])
    start = cpu.llk(obj, probe)["gaussian"]
    llk = abs(float(sums["gaussian"] / sums["nb_obs"]) / start - 1)
    print(f"start: far-field intensity max |ptyrs - pty-chi| / max pty-chi {far:.2e}; Gaussian "
          f"LLK ptyrs CPU f64 {start:.10e}, pty-chi's own over its valid pixels {llk:.1e} from "
          "it", flush=True)
    del ours, theirs

    curves = {"ptyrs": [start], "pty-chi": [start]}
    randn, zeroed = torch.randn, []

    def zeros(*shape, **kw):
        zeroed.append(shape)
        return torch.zeros(*shape, **kw)

    wall = time.perf_counter()
    for e in range(1, args.epochs + 1):
        torch.randn = zeros
        try:
            task.run(1)
        finally:
            torch.randn = randn
        assert sampler.epoch == e, (sampler.epoch, e)
        o = task.get_data_to_cpu("object", as_numpy=True)
        p = ds.probe_from_ptychi(task.get_data_to_cpu("probe", as_numpy=True)[0])
        curves["pty-chi"].append(cpu.llk(o, p)["gaussian"])
        print(f"pty-chi epoch {e}: {(time.perf_counter() - wall) / e:.1f} s/epoch ({load_avg()})",
              flush=True)
    write_modes(f"{args.out}/lsqml_ptychi.h5", o, p)

    own = []
    for e in range(1, args.epochs + 1):
        gpu = ptyrs.GpuPtycho(scan, obj, probe, args.device)
        gpu.lsqml(e, 1, args.batch, "amplitude", args.seed)
        h = np.asarray(gpu.history["gaussian"])
        assert len(h) == e, (len(h), e)
        own.append(h[-1])
        go, gp = gpu.object, gpu.probe
        curves["ptyrs"].append(cpu.llk(go, gp)["gaussian"])
    assert (np.array(own) == h).all(), "a shorter GPU run is not the longest run's prefix"
    write_modes(f"{args.out}/lsqml_ptyrs.h5", go, gp)

    print(f"LSQML Gaussian batch {args.batch} seed {args.seed}: epoch, Gaussian LLK GPU, "
          f"pty-chi, relative difference ({load_avg()})")
    a, b = (np.array(curves[s]) for s in ("ptyrs", "pty-chi"))
    for e, (x, y) in enumerate(zip(a, b)):
        print(f"{e:4d} {x:.10e} {y:.10e} {abs(x - y) / y:.1e}")
    (eo, ko), (ep, kp) = rel_l2(go, o), rel_l2(gp, p)
    print(f"LLK max relative difference {np.abs(a / b - 1).max():.1e}; final object rel L2 "
          f"{eo:.3e} (scale {ko:.6f}), probe rel L2 {ep:.3e} (scale {kp:.6f}); the GPU's LLK "
          f"records from the f64 llk of its states {np.abs(np.array(own) / a[1:] - 1).max():.1e}"
          f"; {len(zeroed)} torch.randn calls zeroed")
    print(f"FRC of the final objects ({load_avg()}):")
    frc(args.out, centres_region(data), "lsqml_ptyrs", "lsqml_ptychi")
    m3ref.exit_unless_converging(curves)
    return far <= FAR_RTOL and llk <= LLK_RTOL


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = a.add_subparsers(dest="cmd", required=True)
    module = os.path.expanduser("~/work/ptyrs-scratch/py")
    p = sub.add_parser("prepare")
    p.add_argument("raw")
    p.add_argument("out")
    for name, n in (("dm", 200), ("ml", 100)):
        q = sub.add_parser(name)
        q.add_argument("out")
        q.add_argument("--cycles", type=int, default=n)
        q.add_argument("--fix-gamma", action="store_true")
    q = sub.add_parser("lsqml")
    q.add_argument("out")
    q.add_argument("--epochs", type=int, default=20)
    q.add_argument("--batch", type=int, default=32)
    q.add_argument("--seed", type=int, default=7)
    q = sub.add_parser("pynx")
    q.add_argument("out")
    q.add_argument("algo", choices=("dm", "ml"))
    q.add_argument("--cycles", type=int, required=True)
    q.add_argument("--fix-gamma", action="store_true")
    for q in sub.choices.values():
        q.add_argument("--device", type=int, default=0)
        q.add_argument("--module", default=module)
    args = a.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "pynx":
        pynx(args)
    elif not (run_lsqml(args) if args.cmd == "lsqml" else run_cuda(args, args.cmd)):
        sys.exit(1)


if __name__ == "__main__":
    main()
