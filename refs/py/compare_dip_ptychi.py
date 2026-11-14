"""pty-chi's deep image prior against ptyrs::nn::dip with the same network, start, loss and
optimizer on one M3 dataset: pty-chi's DIPPlanarObject with its default UNet, and the example
nn_dip_ptychi, which rebuilds that UNet in burn from the weights `init` exports and trains it
through Dip.

    python -I compare_dip_ptychi.py init DATA_DIR OUT_DIR [--name fermat_clean] [--seed 0]
    python -I compare_dip_ptychi.py run DATA_DIR OUT_DIR [--name fermat_clean] [--dtype f64|f32]
        [--epochs 100] [--lr 1e-3]
    python -I compare_dip_ptychi.py plot DATA_DIR OUT_DIR [--name fermat_clean] [--dtype f64]
        [--rerun DIR]

init draws the network and its input code under --seed in float64 and writes OUT_DIR/dip_init.h5:
the UNet's state_dict by name, the code `nn_input` [1, 32, h, w] and the start object `object0`
[h, w, 2] (real and imaginary parts) it generates. run loads that file at --dtype, exits 1 if the
generated start differs from object0 by more than START_TOL of its largest modulus, runs
--epochs epochs and writes OUT_DIR/ptychi_<dtype>.tsv (epoch, PyNX Poisson LLK after it with
epoch 0 the start, the epoch's synchronised wall seconds) and the last object to
OUT_DIR/ptychi_<dtype>_object.h5. plot draws OUT_DIR/ptychi_<dtype>.tsv against
OUT_DIR/ptyrs_<dtype>.tsv (the LLK, its relative difference and the seconds per iteration) and
the phase of the two last objects with their difference over the box of probe centres into
OUT_DIR/dip_<dtype>.png, and prints the numbers. Each phase is taken from the mean over the box,
as the intensities do not see a constant phase. --rerun names the OUT_DIR of a second pty-chi
run from the same dip_init.h5, drawn as pty-chi's own run-to-run difference: torch's CUDA
kernels are not bitwise reproducible (upsample_bilinear2d backward adds in no fixed order).

Matched config: one Adam step per epoch on the network weights (batch = all frames, torch's
epsilon 1e-8), the truth probe and the positions fixed, PATCH shift at the PyNX integer view
corners. The loss is torch.nn.PoissonNLLLoss(log_input=False), the Poisson NLL of the intensity,
swapped in for pty-chi's own: maps.py:78 maps LossFunctions.POISSON to torch.nn.PoissonNLLLoss and
reconstructors/ad_general.py:43 builds it without arguments, so log_input=True takes the predicted
intensity for its logarithm. The swapped loss is the PyNX Poisson LLK ptyrs differentiates, less
a constant and over npix = nb_obs, with icalc + 1e-8 in its logarithm.
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ in the refs directories
sys.path[:0] = [os.path.join(HERE, "..", "ptychi")]
import m3ref  # noqa: E402

INIT = "dip_init.h5"
# Generated start against object0, relative to its largest modulus. float32 convolutions run in
# TF32 by default (torch.backends.cudnn.allow_tf32), so only a broken load fails there.
START_TOL = {"f64": 1e-10, "f32": 1e-2}
CHANNELS = 32


def build(ds, dtype, lr):
    import ptychi.api as api
    from ptychi.api.task import PtychographyTask

    ny, nx = ds.frame
    real, cplx = {"f64": (np.float64, np.complex128), "f32": (np.float32, np.complex64)}[dtype]
    opts = api.AutodiffPtychographyOptions()
    r = opts.reconstructor_options
    r.loss_function = api.LossFunctions.POISSON  # replaced after the build, see the docstring
    r.batch_size = ds.nframe
    r.num_epochs = 1
    r.default_dtype = {"f64": api.Dtypes.FLOAT64, "f32": api.Dtypes.FLOAT32}[dtype]
    r.default_device = api.Devices.GPU
    # upsample_bilinear2d's backward has no deterministic CUDA kernel.
    r.allow_nondeterministic_algorithms = True
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH
    d = opts.data_options
    d.wavelength_m = ds.geometry["wavelength_m"]
    d.fft_shift = True
    d.save_data_on_device = True  # as GpuPtycho keeps the counts
    o = opts.object_options
    o.optimizable = True
    o.optimizer = api.Optimizers.ADAM
    o.step_size = lr
    o.pixel_size_m = ds.pixel
    # Corners as given: the view of corner c starts at c + origin - (n - 1) / 2.
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    dip = o.experimental.deep_image_prior_options
    dip.enabled = True
    dip.model = api.DIPModels.UNET
    dip.model_params = {"num_in_channels": CHANNELS, "num_out_channels": 2}
    dip.net_input_channels = CHANNELS
    opts.probe_options.optimizable = False
    opts.probe_position_options.optimizable = False
    obj, _ = ds.modes("init")  # its shape only: the network generates the object
    _, probe = ds.modes("truth")
    corners = (np.array(obj.shape[-2:]) - ds.frame) // 2 + np.round(ds.pos_px)
    task = PtychographyTask(
        opts,
        diffraction_data=ds.counts.astype(real),
        object_data=obj.astype(cplx),
        probe_data=ds.probe_to_ptychi(probe)[None].astype(cplx),
        probe_position_y_px=corners[:, 0],
        probe_position_x_px=corners[:, 1],
    )
    task.reconstructor.loss_function = torch.nn.PoissonNLLLoss(log_input=False)
    return task


@torch.no_grad()
def start_object(task):
    """The generated object as [h, w, 2] float64."""
    return torch.view_as_real(task.object.generate()[0]).double().cpu().numpy()


@torch.no_grad()
def poisson(task):
    """m3ref.evaluate's PyNX Poisson LLK, the network run once rather than per frame chunk."""
    fm = task.reconstructor.forward_model
    task.object.generate()
    fm.dip_generate = lambda: None
    try:
        return m3ref.evaluate(task)["poisson"]
    finally:
        del fm.dip_generate


def init(args):
    ds = m3ref.Dataset(args.name, args.data_dir)
    torch.manual_seed(args.seed)
    task = build(ds, "f64", 1e-3)
    model = task.object.model
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, INIT)
    with h5py.File(path, "w") as f:
        for k, v in model.state_dict().items():
            f[k] = v.cpu().numpy()
        f["nn_input"] = task.object.nn_input.cpu().numpy()
        f["object0"] = start_object(task)
        f.attrs["seed"] = args.seed
        f.attrs["name"] = args.name
    print(f"{path}: {sum(p.numel() for p in model.parameters())} weights, code "
          f"{tuple(task.object.nn_input.shape)}")


def load(task, path):
    """dip_init.h5 into the task's network and code; returns object0."""
    model, obj = task.object.model, task.object
    with h5py.File(path, "r") as f:
        state = {k: torch.from_numpy(np.asarray(f[k][()])) for k in model.state_dict()}
        code, object0 = f["nn_input"][()], f["object0"][()]
    model.load_state_dict(state)
    with torch.no_grad():
        obj.nn_input.copy_(torch.from_numpy(code))
    return object0


def run(args):
    ds = m3ref.Dataset(args.name, args.data_dir)
    task = build(ds, args.dtype, args.lr)
    object0 = load(task, os.path.join(args.out_dir, INIT))
    d = np.abs(start_object(task) - object0).max() / np.abs(object0).max()
    print(f"start object against {INIT}: max difference / max modulus {d:.2e}", flush=True)
    if not d <= START_TOL[args.dtype]:
        sys.exit(1)
    print(f"load {os.getloadavg()}", flush=True)

    rows = [(0, poisson(task), 0.0)]
    print(f"pty-chi DIP {args.dtype} lr {args.lr}: epoch, Poisson LLK, s", flush=True)
    print(f"{0:4d} {rows[0][1]:.10e}", flush=True)
    for e in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        task.run(n_epochs=1)
        torch.cuda.synchronize()
        s = time.perf_counter() - t
        rows.append((e, poisson(task), s))
        print(f"{e:4d} {rows[-1][1]:.10e} {s:.3f}", flush=True)
        if not np.isfinite(rows[-1][1]):
            break
    print(f"load {os.getloadavg()}; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    tag = f"ptychi_{args.dtype}"
    with open(os.path.join(args.out_dir, tag + ".tsv"), "w") as f:
        f.write("iter\tllk\tseconds\n")
        for e, llk, s in rows:
            f.write(f"{e}\t{llk!r}\t{s!r}\n")
    with torch.no_grad():
        o = task.object.generate().cpu().numpy().astype(np.complex128)
    _, probe = ds.modes("truth")
    with h5py.File(os.path.join(args.out_dir, tag + "_object.h5"), "w") as f:
        f["object"] = o
        f["probe"] = probe


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = a.add_subparsers(dest="cmd", required=True)
    for cmd in ("init", "run", "plot"):
        p = sub.add_parser(cmd)
        p.add_argument("data_dir")
        p.add_argument("out_dir")
        p.add_argument("--name", default="fermat_clean")
        if cmd == "init":
            p.add_argument("--seed", type=int, default=0)
        else:
            p.add_argument("--dtype", choices=("f64", "f32"), default="f64")
        if cmd == "run":
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--lr", type=float, default=1e-3)
        if cmd == "plot":
            p.add_argument("--rerun")
    args = a.parse_args()
    {"init": init, "run": run, "plot": plot}[args.cmd](args)


# Light chart surface, ink and series slots of the reference palette (dataviz skill).
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
PTYCHI, PTYRS, RERUN = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3b0ac", "#e66767", "#b8302f"]
AT = (0, 1, 2, 3, 10, 20, 50, 100)


def table(path):
    with open(path) as f:
        head = f.readline().split()
        rows = np.array([[float(v) for v in line.split()] for line in f])
    return {k: rows[:, i] for i, k in enumerate(head)}


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    out, tag = args.out_dir, args.dtype
    sides = [("pty-chi", PTYCHI, "-", os.path.join(out, f"ptychi_{tag}"))]
    if args.rerun:
        sides.append(("pty-chi rerun", RERUN, ":", os.path.join(args.rerun, f"ptychi_{tag}")))
    sides.append(("ptyrs", PTYRS, "--", os.path.join(out, f"ptyrs_{tag}")))
    tabs = {name: table(base + ".tsv") for name, _, _, base in sides}
    objs = {name: load_object(base + "_object.h5") for name, _, _, base in sides}

    # The box of probe centres, every pixel of it under the probe's core.
    ds = m3ref.Dataset(args.name, args.data_dir)
    shape = np.array(objs["pty-chi"].shape)
    centres = (shape - ds.frame) // 2 + np.round(ds.pos_px) + np.array(ds.frame) // 2
    (y0, x0), (y1, x1) = centres.min(0).astype(int), centres.max(0).astype(int) + 1
    crop = {k: o[y0:y1, x0:x1] for k, o in objs.items()}
    ref = tabs["pty-chi"]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "text.color": INK,
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.facecolor": SURFACE,
        "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    fig = plt.figure(figsize=(13, 7.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    ax_llk, ax_rel, ax_s = top.subplots(1, 3)
    for name, color, ls, _ in sides:
        t = tabs[name]
        ax_llk.plot(t["iter"], t["llk"], color=color, ls=ls, lw=1.5, label=name)
        if name != "pty-chi":
            n = min(len(t["llk"]), len(ref["llk"]))
            rel = np.abs(t["llk"][:n] - ref["llk"][:n]) / ref["llk"][:n]
            ax_rel.plot(t["iter"][1:n], rel[1:], color=color, ls=ls, lw=1.5,
                        label=f"{name} vs pty-chi")
        ax_s.plot(t["iter"][1:], t["seconds"][1:], color=color, ls=ls, lw=1.5, label=name)
        if "loss_seconds" in t:
            ax_s.plot(t["iter"][1:], t["loss_seconds"][1:], color=color, ls=":", lw=1,
                      label=f"{name}, its loss")
    ax_llk.set(yscale="log", xlabel="iteration", ylabel="Poisson LLK (PyNX, normalised)",
               title="LLK after each iteration")
    ax_rel.set(yscale="log", xlabel="iteration", ylabel="|LLK − LLK(pty-chi)| / LLK(pty-chi)",
               title="Relative LLK difference from pty-chi")
    ax_s.set(yscale="log", xlabel="iteration", ylabel="seconds",
             title="Seconds per iteration (synchronised)")
    for ax in (ax_llk, ax_rel, ax_s):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.title.set_size(10)

    images = bottom.subplots(1, 2 + len(sides) - 1)
    blues = LinearSegmentedColormap.from_list("blues", BLUES)
    diverging = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    phases = [rephase(crop["pty-chi"]), rephase(crop["ptyrs"])]
    # Colour scales over the central 99 % of pixels; the printed numbers keep the extremes.
    lo, hi = np.percentile(np.concatenate([p.ravel() for p in phases]), [0.5, 99.5])
    extent = (x0, x1, y1, y0)
    for ax, ph, name in zip(images, phases, ("pty-chi", "ptyrs")):
        im = ax.imshow(ph, cmap=blues, vmin=lo, vmax=hi, extent=extent)
        ax.set_title(f"{name}: phase after {int(tabs[name]['iter'][-1])}", fontsize=10)
    bottom.colorbar(im, ax=images[:2], shrink=0.8, label="rad")
    deltas = [(f"{name} − pty-chi", rephase(crop[name] * np.conj(crop["pty-chi"])))
              for name, _, _, _ in sides[1:]]
    deltas.sort(key=lambda d: d[0] != "ptyrs − pty-chi")
    big = max(np.percentile(np.abs(d), 99.5) for _, d in deltas)
    for ax, (title, d) in zip(images[2:], deltas):
        im = ax.imshow(d, cmap=diverging, vmin=-big, vmax=big, extent=extent)
        ax.set_title(f"{title}: phase", fontsize=10)
    bottom.colorbar(im, ax=images[2:], shrink=0.8, label="rad")
    for ax in images:
        ax.grid(False)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Deep image prior, pty-chi's UNet from one start, {tag}, "
                 f"box of probe centres [{y0}:{y1}, {x0}:{x1}]", fontsize=11)
    png = os.path.join(out, f"dip_{tag}.png")
    fig.savefig(png, dpi=110)
    print(png)

    print("iteration " + " ".join(f"{name:>16}" for name, _, _, _ in sides))
    for i in AT:
        if all(i < len(tabs[name]["llk"]) for name, _, _, _ in sides):
            print(f"{i:9d} " + " ".join(f"{tabs[name]['llk'][i]:16.10e}"
                                        for name, _, _, _ in sides))
    for name, _, _, _ in sides:
        s = tabs[name]["seconds"]
        line = f"{name}: first iteration {s[1]:.3f} s, median of the rest {np.median(s[2:]):.3f} s"
        if "loss_seconds" in tabs[name]:
            line += f" (its loss {np.median(tabs[name]['loss_seconds'][2:]):.3f} s)"
        print(line)
    for title, d in deltas:
        print(f"{title}: phase over the box, rms {np.sqrt(np.mean(d**2)):.3e}, "
              f"max {np.abs(d).max():.3e} rad")


def rephase(o):
    """The phase of o from that of its mean."""
    m = o.mean()
    return np.angle(o * np.conj(m / abs(m)))


def load_object(path):
    with h5py.File(path, "r") as f:
        o = f["object"][()]
    return o.reshape(o.shape[-2:])


if __name__ == "__main__":
    main()
