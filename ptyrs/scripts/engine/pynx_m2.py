"""M2 (DESIGN §0 fold condition 2): PyNX CUDA AP and DM per-cycle time on the scan.cxi and
init.h5 `cargo run --example dist_m2_time` times ptyrs on, with compare_pynx.py's loader.

Usage: python -I pynx_m2.py time <dir> <ap|dm> <ncycle> <repeats>
       python -I pynx_m2.py prof <dir> <ap|dm> <k>

Both run 5 untimed cycles at PyNX's default stack (16) without LLK or phase-ramp removal.
`time` then times `repeats` runs of `ncycle` cycles and prints their median; `prof` runs `k`
cycles between cuProfilerStart and cuProfilerStop, for `ncu --profile-from-start off`.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pynx import build, load  # noqa: E402
import pycuda.driver as drv  # noqa: E402
import pynx.ptycho.cu_operator as cuop  # noqa: E402


def op(algo):
    if algo == "dm":
        return cuop.DM(calc_llk=0, zero_phase_ramp=False)
    return cuop.AP(update_object=True, update_probe=True, calc_llk=0, zero_phase_ramp=False)


def main():
    mode, d, algo = sys.argv[1], sys.argv[2], sys.argv[3]
    p = build(*load(d), f64=False)
    p = op(algo) ** 5 * p
    cuop.default_processing_unit.finish()
    nframe = p.data.iobs.shape[0]
    if mode == "time":
        ncycle, repeats = int(sys.argv[4]), int(sys.argv[5])
        ms = []
        for _ in range(repeats):
            t = time.perf_counter()
            p = op(algo) ** ncycle * p
            cuop.default_processing_unit.finish()
            ms.append(1e3 * (time.perf_counter() - t) / ncycle)
        runs = " ".join(f"{m:.3f}" for m in ms)
        print(f"pynx {algo}: {nframe} frames, median {sorted(ms)[len(ms) // 2]:.3f} ms/cycle"
              f" (runs {runs}) over {ncycle} cycles")
    else:
        k = int(sys.argv[4])
        drv.start_profiler()
        p = op(algo) ** k * p
        cuop.default_processing_unit.finish()
        drv.stop_profiler()
        print(f"pynx {algo}: profiled {k} cycles")


if __name__ == "__main__":
    main()
