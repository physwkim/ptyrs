"""Check ptyrs::live's result PVs (pv_out) with a p4p client: the
live_pv_out_demo example publishes a LiveObject snapshot preview (rows
row0..row1 of two planes hold a known pattern, the rest the flat fill), a
two-mode probe and an LLK history; every update received is compared with
the same patterns computed here, together with its dimensions, binning and
Planes attribute.

  cargo build --release -p ptyrs --examples
  micromamba run -n epicsrs-pva python ptyrs/scripts/live/pv_out_check.py
"""

import argparse
import os
import subprocess
import sys
import threading

import numpy as np
from p4p.client.thread import Context

FILL = np.complex64(1.0 + 0.5j)
PVS = ("OBJ:AMP", "OBJ:PHASE", "PROBE:AMP", "PROBE:PHASE", "LLK")


def polar(a, p):
    return (a * np.cos(p)).astype(np.float32) + 1j * (a * np.sin(p)).astype(np.float32)


def obj_pattern(n, row0, row1, u):
    z, y, x = np.meshgrid(np.arange(2.0), np.arange(row0, row1, dtype=float), np.arange(float(n)),
                          indexing="ij")
    a = 0.6 + 0.3 * np.sin(0.011 * x + 0.017 * y + z)
    o = np.full((2, n, n), FILL, np.complex64)
    o[:, row0:row1] = polar(a, 0.05 * x - 0.03 * y + 0.9 * z + 0.4 * u)
    return o


def probe_pattern(np_, u):
    m, y, x = np.meshgrid(np.arange(2.0), np.arange(float(np_)), np.arange(float(np_)), indexing="ij")
    c = float(np_ // 2)
    r2 = (x - c) ** 2 + (y - c) ** 2
    w = 10.0 + 5.0 * m
    return polar(np.exp(-r2 / (2.0 * w * w)), 0.002 * r2 + m + 0.1 * u)


def binned(planes, b):
    z, ny, nx = planes.shape
    return planes.astype(np.complex128).reshape(z, ny // b, b, nx // b, b).mean(axis=(2, 4))


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=f"{root}/target/release")
    ap.add_argument("--port", type=int, default=15175)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--row0", type=int, default=300)
    ap.add_argument("--row1", type=int, default=460)
    ap.add_argument("--bin", type=int, default=4)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--posts", type=int, default=5)
    a = ap.parse_args()
    assert a.n % a.bin == 0
    prefix = "PTYOUT:"
    demo = subprocess.Popen(
        [f"{a.target}/examples/live_pv_out_demo", "--port", str(a.port), "--gpu", str(a.gpu),
         "--prefix", prefix, "--n", str(a.n), "--row0", str(a.row0), "--row1", str(a.row1),
         "--bin", str(a.bin), "--probe", str(a.probe), "--posts", str(a.posts)],
        stdout=subprocess.PIPE, text=True,
    )
    for line in demo.stdout:
        print(line, end="")
        if line.startswith("pvout listening"):
            break
    conf = {
        "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{a.port}",
        "EPICS_PVA_AUTO_ADDR_LIST": "NO",
        "EPICS_PVA_ADDR_LIST": "",
    }
    got = {pv: {} for pv in PVS}
    lock, done = threading.Lock(), threading.Event()

    def on_update(pv):
        def cb(v):
            with lock:
                got[pv][v["uniqueId"]] = v
                if all(a.posts in got[p] for p in PVS):
                    done.set()
        return cb

    ctx = Context("pva", conf=conf, useenv=False, nt=False)
    subs = [ctx.monitor(prefix + pv, on_update(pv), request="record[queueSize=16]field()")
            for pv in PVS]
    done.wait(60)
    for s in subs:
        s.close()
    ctx.close()
    for line in demo.stdout:
        print(line, end="")
    rc = demo.wait(30)

    bad, ok = [], {}

    def check(what, good, detail=""):
        ok[what] = ok.get(what, 0) + bool(good)
        if not good:
            bad.append(f"{what} {detail}")

    check("demo-exit-0", rc == 0, f"exit {rc}")
    for pv in PVS:
        missing = set(range(1, a.posts + 1)) - set(got[pv])
        check("uids-1..posts-received", not missing, f"{pv} missing {sorted(missing)}")

    def phase_close(got_p, want_p, tol):
        return np.abs(np.angle(np.exp(1j * (got_p.astype(np.float64) - want_p)))).max() < tol

    def image(v, nx, ny, binning, planes):
        dims = [(d["size"], d["binning"]) for d in v["dimension"]]
        attrs = {at["name"]: at["value"] for at in v["attribute"]}
        return dims == [(nx, binning), (ny, binning)] and attrs.get("Planes") == planes, dims, attrs

    ob = a.n // a.bin
    for u in sorted(got["OBJ:AMP"]):
        if u < 0:
            continue
        want = binned(obj_pattern(a.n, a.row0, a.row1, u), a.bin)
        amp, ph = got["OBJ:AMP"][u], got["OBJ:PHASE"].get(u)
        if ph is None:
            continue
        for v, name in ((amp, "OBJ:AMP"), (ph, "OBJ:PHASE")):
            good, dims, attrs = image(v, ob, 2 * ob, a.bin, 2)
            check("obj-dims-binning-planes", good, f"{name} uid {u} {dims} {attrs}")
        ga = np.asarray(amp["value"]).reshape(2, ob, ob)
        gp = np.asarray(ph["value"]).reshape(2, ob, ob)
        check("obj-float32", ga.dtype == np.float32 and gp.dtype == np.float32, f"{ga.dtype}")
        err = np.abs(ga - np.abs(want)).max()
        check("obj-amp", err < 2e-6, f"uid {u} max err {err:.3g}")
        check("obj-phase", phase_close(gp, np.angle(want), 1e-5), f"uid {u}")
        # Rows outside row0..row1 (backed or not) carry the fill.
        out_rows = np.r_[0:a.row0 // a.bin, -(-a.row1 // a.bin):ob]
        check("obj-fill-rows", np.allclose(ga[:, out_rows], abs(FILL), atol=1e-7)
              and np.allclose(gp[:, out_rows], np.angle(FILL), atol=1e-7), f"uid {u}")
    for u in sorted(got["PROBE:AMP"]):
        if u < 0 or u not in got["PROBE:PHASE"]:
            continue
        want = probe_pattern(a.probe, u).astype(np.complex128)
        amp, ph = got["PROBE:AMP"][u], got["PROBE:PHASE"][u]
        for v, name in ((amp, "PROBE:AMP"), (ph, "PROBE:PHASE")):
            good, dims, attrs = image(v, a.probe, 2 * a.probe, 1, 2)
            check("probe-dims-binning-planes", good, f"{name} uid {u} {dims} {attrs}")
        ga = np.asarray(amp["value"]).reshape(2, a.probe, a.probe)
        gp = np.asarray(ph["value"]).reshape(2, a.probe, a.probe)
        err = np.abs(ga - np.abs(want)).max()
        check("probe-amp", err < 2e-6, f"uid {u} max err {err:.3g}")
        # Phase is only defined where the amplitude is not lost in f32.
        lit = np.abs(want) > 1e-3
        check("probe-phase", phase_close(gp[lit], np.angle(want)[lit], 1e-4), f"uid {u}")
    for u, v in sorted(got["LLK"].items()):
        if u < 0:
            continue
        g = np.asarray(v["value"])
        want = 1e6 * 0.8 ** np.arange(u + 1)
        dims = [(d["size"], d["binning"]) for d in v["dimension"]]
        check("llk-dims", dims == [(u + 1, 1)], f"uid {u} {dims}")
        check("llk-values", g.dtype == np.float64 and np.allclose(g, want, rtol=1e-12, atol=0),
              f"uid {u} {g}")
    for what, n in sorted(ok.items()):
        print(f"pv_out_check.py {what} ok={n}")
    for b in bad:
        print(f"pv_out_check.py FAIL {b}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
