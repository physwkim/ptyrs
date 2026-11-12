"""Cross-check ptyrs::live's lz4/bslz4 codecs against C liblz4 (python-lz4)
and C bitshuffle, bit for bit, in both directions:

  Rust payloads (examples/live_codec_cross --dump) -> C decode
  pty-sim payloads received over PVA (p4p)    -> C decode
  C payloads (lz4.block, bitshuffle)          -> Rust decode (--check)

Also asserts that bitshuffle.compress_lz4 emits bare blocks, without the
12-byte [u64 BE size][u32 BE block] header the Eiger stream carries and
ADEiger strips before filling the NDArray.

  cargo build --release -p ptyrs --bins --examples
  micromamba run -n epicsrs-pva python ptyrs/scripts/live/codec_cross.py
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time

import bitshuffle
import lz4.block
import numpy as np
from p4p.client.thread import Context

BLOCK = 4096  # bshuf default block for 2-byte elements


def bare_blocks(payload, n):
    """True if payload is bshuf_compress_lz4's bare block sequence for n
    u16 elements: [u32 BE len][lz4] per block, then n % 8 raw elements."""
    blocked = n - n % 8
    pos, start = 0, 0
    while start < blocked:
        if pos + 4 > len(payload):
            return False
        (clen,) = struct.unpack_from(">I", payload, pos)
        pos += 4 + clen
        start += min(BLOCK, blocked - start)
    return pos == len(payload) - 2 * (n % 8)


def c_decode(codec, payload, n):
    if codec == "lz4":
        return np.frombuffer(lz4.block.decompress(payload, uncompressed_size=2 * n), "<u2")
    arr = np.frombuffer(payload, np.uint8)
    return bitshuffle.decompress_lz4(arr, (n,), np.dtype("<u2"))


class Tally:
    def __init__(self):
        self.ok, self.bad = {}, []

    def __call__(self, what, good, detail=""):
        self.ok[what] = self.ok.get(what, 0) + bool(good)
        if not good:
            self.bad.append(f"{what} {detail}")


def capture_sim(target, codec, raw, port, frames, t):
    """Run pty-sim and decode what arrives over PVA with the C library."""
    conf = {
        "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{port}",
        "EPICS_PVA_AUTO_ADDR_LIST": "NO",
        "EPICS_PVA_ADDR_LIST": "",
    }
    sim = subprocess.Popen(
        [f"{target}/pty-sim", "--codec", codec, "--rate", "300", "--frames", str(frames),
         "--pool", str(len(raw)), "--port", str(port), "--linger-ms", "1000"],
        stdout=subprocess.PIPE, text=True,
    )
    for line in sim.stdout:
        if line.startswith("sim listening"):
            break
    got, done = [], threading.Event()

    def on_frame(v):
        uid = v["uniqueId"]
        if uid >= 0:
            got.append((uid, v["codec.name"], bytes(v["value"])))
            if len(got) >= frames:
                done.set()

    ctx = Context("pva", conf=conf, useenv=False, nt=False)
    ctx_pos = Context("pva", conf=conf, useenv=False, nt=False)
    req = "record[queueSize=1024,pipeline=true]field()"
    subs = [ctx.monitor("PTY:FRAME", on_frame, request=req),
            ctx_pos.monitor("PTY:POS", lambda v: None)]
    done.wait(60)
    for s in subs:
        s.close()
    ctx.close()
    ctx_pos.close()
    sim.wait(30)
    t(f"sim-{codec}-received-all", len(got) == frames, f"{len(got)}/{frames}")
    for uid, name, payload in got:
        ref = raw[uid % len(raw)]
        t(f"sim-{codec}->C", name == codec and np.array_equal(c_decode(codec, payload, ref.size), ref),
          f"uid {uid}")
        if codec == "bslz4":
            t("sim-bslz4-bare-blocks", bare_blocks(payload, ref.size), f"uid {uid}")


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--target", default=f"{root}/target/release")
    ap.add_argument("--port", type=int, default=15077)
    ap.add_argument("--sim-frames", type=int, default=300)
    a = ap.parse_args()
    t = Tally()
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([f"{a.target}/examples/live_codec_cross", "--dump", d], check=True)
        eiger = []
        identical = {"lz4": 0, "bslz4": 0}
        for line in open(f"{d}/cases.txt"):
            name, w, h, count = line.split()
            for i in range(int(count)):
                f = lambda ext: f"{d}/{name}_{i}.{ext}"
                raw = np.fromfile(f("raw"), "<u2")
                assert raw.size == int(w) * int(h)
                if name == "eiger":
                    eiger.append(raw)
                for codec in ("lz4", "bslz4"):
                    rs = open(f(f"rs.{codec}"), "rb").read()
                    t(f"rust-{codec}->C", np.array_equal(c_decode(codec, rs, raw.size), raw), f"{name}_{i}")
                c_lz4 = lz4.block.compress(raw.tobytes(), store_size=False)
                c_bs = bitshuffle.compress_lz4(raw).tobytes()
                head = struct.unpack(">Q", c_bs[:8])[0]
                t("C-bslz4-no-12-byte-header", head != raw.nbytes and bare_blocks(c_bs, raw.size),
                  f"{name}_{i} first u64 {head}")
                identical["lz4"] += c_lz4 == open(f("rs.lz4"), "rb").read()
                identical["bslz4"] += c_bs == open(f("rs.bslz4"), "rb").read()
                open(f("c.lz4"), "wb").write(c_lz4)
                open(f("c.bslz4"), "wb").write(c_bs)
        for k, codec in enumerate(("lz4", "bslz4")):
            capture_sim(a.target, codec, eiger, a.port + k, a.sim_frames, t)
        rc = subprocess.run([f"{a.target}/examples/live_codec_cross", "--check", d]).returncode
        t("C->rust (codec_cross --check)", rc == 0, f"exit {rc}")
    for what, n in sorted(t.ok.items()):
        print(f"codec_cross.py {what} ok={n}")
    print(f"codec_cross.py payload bytes identical to Rust's: lz4={identical['lz4']} bslz4={identical['bslz4']}")
    for b in t.bad:
        print(f"codec_cross.py FAIL {b}")
    sys.exit(1 if t.bad else 0)


if __name__ == "__main__":
    main()
