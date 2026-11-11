"""pty-chi rPIE on the scan and starting point written by `cargo run --example llk_trace -- <dir> rpie`.

Usage: python -I compare_ptychi_rpie.py <dir> [--gpu]

Maps the Rust start onto pty-chi's model: the same object; the probe times the quadratic
phase and 1/sqrt(ny*nx), since pty-chi has no quadratic phase and an unnormalised FFT; and
positions whose integer windows start at the Rust window corners (views.txt). Runs
RPIEReconstructor in f64 with SGD and the parameters in rpie.txt, on consecutive unshuffled
batches, with every constraint pty-chi enables by default switched off. After each epoch the
state is mapped back and its normalised LLK computed by PyNX cpu_operator promoted to f64,
the LLK the Rust history records; prints per-epoch relative differences against
llk_rust.txt, writes llk_ptychi.txt, and compares the final arrays with final_rust.h5.
"""
import importlib.util
import pathlib
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import ptychi.api as api
from ptychi.api.task import PtychographyTask


def sibling(name):
    """Import a script next to this one (python -I keeps its directory off sys.path)."""
    sys.dont_write_bytecode = True  # no __pycache__ in the source tree
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pynx = sibling("compare_pynx")


def main():
    d = sys.argv[1]
    device = api.Devices.GPU if "--gpu" in sys.argv[2:] else api.Devices.CPU
    rust = np.loadtxt(f"{d}/llk_rust.txt", ndmin=2)
    nepoch = len(rust)
    assert (rust[:, 0] == np.arange(nepoch)).all(), "llk_rust.txt must hold every epoch"
    rust = rust[:, 1:]
    params = dict(line.split() for line in open(f"{d}/rpie.txt"))
    views = np.loadtxt(f"{d}/views.txt", ndmin=2)
    re_im = np.loadtxt(f"{d}/quad.txt")
    iobs, pos, distance, pixel, wavelength, obj0, probe0 = pynx.load(d)
    nmode, ny, nx = probe0.shape
    quad = (re_im[:, 0] + 1j * re_im[:, 1]).reshape(ny, nx)
    root_n = np.sqrt(ny * nx)
    assert obj0.shape[0] == 1 and (iobs >= 0).all(), "one object mode, no masked pixels"

    op = pynx.cpu_operator_f64()

    def llk(obj, probe):
        p = pynx.build(iobs, pos, distance, pixel, wavelength, obj, probe, f64=True)
        p = op.LLK() * op.FT() * op.ObjProbe2Psi() * p
        h = p.get_llk(noise=None, norm=True)
        return [h["poisson"], h["gaussian"], h["euclidian"], h["nb_photons_calc"]]

    opts = api.RPIEOptions()
    opts.data_options.wavelength_m = wavelength
    opts.data_options.fft_shift = True
    o = opts.object_options
    o.optimizable = float(params["obj_step"]) > 0
    o.optimizer = api.Optimizers.SGD
    o.step_size = float(params["obj_step"])
    o.alpha = float(params["obj_alpha"])
    o.pixel_size_m = float(pixel * distance * wavelength)  # unused by the Fourier propagator
    # window start = round(position + origin - (n - 1) / 2) = the Rust corner
    o.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
    o.position_origin_coords = [(ny - 1) / 2, (nx - 1) / 2]
    o.remove_object_probe_ambiguity.enabled = False
    pr = opts.probe_options
    pr.optimizable = float(params["probe_step"]) > 0
    pr.optimizer = api.Optimizers.SGD
    pr.step_size = float(params["probe_step"])
    pr.alpha = float(params["probe_alpha"])
    pr.orthogonalize_incoherent_modes.enabled = False
    pr.orthogonalize_opr_modes.enabled = False
    opts.probe_position_options.optimizable = False
    r = opts.reconstructor_options
    batch = int(params["batch_size"])
    r.batch_size = batch
    r.num_epochs = nepoch
    r.default_dtype = api.Dtypes.FLOAT64
    r.default_device = device
    r.allow_nondeterministic_algorithms = False
    r.forward_model_options.shift_operator = api.ShiftOperators.PATCH

    task = PtychographyTask(
        opts,
        diffraction_data=iobs,
        object_data=obj0.copy(),
        probe_data=(probe0 * quad / root_n)[None].copy(),
        probe_position_y_px=views[:, 0].astype(np.float64),
        probe_position_x_px=views[:, 1].astype(np.float64),
    )
    rec = task.reconstructor
    # pty-chi shuffles; the Rust batches are consecutive frames in scan order
    rec.dataloader = DataLoader(rec.dataset, batch_size=batch, shuffle=False)

    def state():
        obj = task.get_data_to_cpu("object", as_numpy=True).astype(np.complex128)
        probe = task.get_data_to_cpu("probe", as_numpy=True)[0] * np.conj(quad) * root_n
        return obj, probe

    h = []
    for _ in range(nepoch):
        task.run(1)
        h.append(llk(*state()))
    h = np.array(h)
    np.savetxt(f"{d}/llk_ptychi.txt", np.column_stack([np.arange(nepoch), h]),
               fmt=["%d"] + ["%.17e"] * 4)
    rel = np.abs(h - rust) / np.abs(rust)
    print(f"--- pty-chi rPIE ({device}): max |rel diff| vs Rust over {nepoch} epochs")
    for k, v in zip(pynx.KEYS, rel.T):
        i = int(v.argmax())
        print(f"  {k:16s} {v.max():.3e} (epoch {i})")
    obj, probe = state()
    with h5py.File(f"{d}/final_rust.h5", "r") as f:
        obj_r, probe_r = f["object"][()], f["probe"][()]
    for name, a, b in (("object", obj_r, obj), ("probe", probe_r, probe)):
        print(f"  final {name} rel L2 {np.linalg.norm(a - b) / np.linalg.norm(a):.3e}")
    for c in sorted({0, 1, 2, 5, 10, nepoch // 2, nepoch - 1}):
        if c < nepoch:
            print(f"  epoch {c:4d} poisson rust {rust[c, 0]:.10e} pty-chi {h[c, 0]:.10e}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
