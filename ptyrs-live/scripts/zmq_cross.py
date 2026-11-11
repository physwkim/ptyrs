"""Cross-check ptyrs-live's Eiger stream v1 path against the system C libzmq
(libzmq.so.5 through ctypes), C liblz4 and C bitshuffle, in both directions:

  pty-sim --out zmq (vendored libzmq PUSH) -> C libzmq PULL, read part by
      part the way ADEiger's StreamAPI reads it, decoded with the C codecs
  C libzmq PUSH (C-compressed frames) + p4p POS server -> pty-recv
      --source zmq, which compares every frame with its own synthesis

Reference frames come from `codec_cross --dump` (pty-sim's frames at
--pool 16).

  cargo build --release -p ptyrs-live --bins --examples
  micromamba run -n epicsrs-pva python ptyrs-live/scripts/zmq_cross.py
"""

import argparse
import ctypes
import json
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
from p4p import Type, Value
from p4p.client.thread import Context
from p4p.server import Server
from p4p.server.thread import SharedPV

PULL, PUSH = 7, 8
SNDMORE = 2
RCVMORE, LINGER, RCVTIMEO = 13, 17, 27
W, H, POOL = 1028, 1062, 16
ENCODING = {"none": "<", "lz4": "lz4<", "bslz4": "bs16-lz4<"}


class Zmq:
    """The few libzmq calls ADEiger's StreamAPI makes, plus PUSH."""

    def __init__(self, path):
        L = ctypes.CDLL(path)
        vp, c_int, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        L.zmq_ctx_new.restype = vp
        L.zmq_socket.restype, L.zmq_socket.argtypes = vp, (vp, c_int)
        L.zmq_connect.argtypes = L.zmq_bind.argtypes = (vp, ctypes.c_char_p)
        L.zmq_setsockopt.argtypes = (vp, c_int, vp, sz)
        L.zmq_getsockopt.argtypes = (vp, c_int, vp, ctypes.POINTER(sz))
        L.zmq_msg_init.argtypes = L.zmq_msg_close.argtypes = (vp,)
        L.zmq_msg_recv.argtypes = (vp, vp, c_int)
        L.zmq_msg_data.restype, L.zmq_msg_data.argtypes = vp, (vp,)
        L.zmq_msg_size.restype, L.zmq_msg_size.argtypes = sz, (vp,)
        L.zmq_send.argtypes = (vp, ctypes.c_char_p, sz, c_int)
        L.zmq_close.argtypes = L.zmq_ctx_term.argtypes = (vp,)
        L.zmq_strerror.restype = ctypes.c_char_p
        self.L = L
        self.ctx = L.zmq_ctx_new()
        v = [ctypes.c_int() for _ in range(3)]
        L.zmq_version(*[ctypes.byref(x) for x in v])
        self.version = ".".join(str(x.value) for x in v)

    def err(self, what):
        e = self.L.zmq_errno()
        raise OSError(e, f"{what}: {self.L.zmq_strerror(e).decode()}")

    def socket(self, kind, timeout_ms=10000):
        s = self.L.zmq_socket(self.ctx, kind)
        for opt, val in ((LINGER, 5000), (RCVTIMEO, timeout_ms)):
            x = ctypes.c_int(val)
            self.L.zmq_setsockopt(s, opt, ctypes.byref(x), 4)
        return s

    def recv(self, s):
        """One part and whether more follow, as zmq_msg_recv gives it."""
        m = (ctypes.c_uint64 * 8)()
        self.L.zmq_msg_init(m)
        if self.L.zmq_msg_recv(m, s, 0) < 0:
            self.err("zmq_msg_recv")
        data = ctypes.string_at(self.L.zmq_msg_data(m), self.L.zmq_msg_size(m))
        self.L.zmq_msg_close(m)
        more, n = ctypes.c_int(), ctypes.c_size_t(4)
        self.L.zmq_getsockopt(s, RCVMORE, ctypes.byref(more), ctypes.byref(n))
        return data, bool(more.value)

    def send_parts(self, s, parts):
        for i, p in enumerate(parts):
            flags = SNDMORE if i + 1 < len(parts) else 0
            if self.L.zmq_send(s, p, len(p), flags) < 0:
                self.err("zmq_send")

    def close(self, *socks):
        for s in socks:
            self.L.zmq_close(s)
        self.L.zmq_ctx_term(self.ctx)


class Tally:
    def __init__(self):
        self.ok, self.bad = {}, []

    def __call__(self, what, good, detail=""):
        self.ok[what] = self.ok.get(what, 0) + bool(good)
        if not good:
            self.bad.append(f"{what} {detail}")


def c_decode(encoding, blob, n):
    """ADEiger's uncompress(): lz4 of the whole blob, or bitshuffle/lz4 from
    byte 12 with the default block size."""
    if encoding == "<":
        return np.frombuffer(blob, "<u2")
    if encoding == "lz4<":
        return np.frombuffer(lz4.block.decompress(blob, uncompressed_size=2 * n), "<u2")
    body = np.frombuffer(blob[12:], np.uint8)
    return bitshuffle.decompress_lz4(body, (n,), np.dtype("<u2"), 0)


def position(uid):
    return (uid % 128) * 50e-9, ((uid // 128) % 128) * 50e-9


def rust_push_to_c_pull(z, target, codec, raw, port, zport, frames, t):
    """pty-sim's series read by the system libzmq the way ADEiger reads it."""
    conf = {
        "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{port}",
        "EPICS_PVA_AUTO_ADDR_LIST": "NO",
        "EPICS_PVA_ADDR_LIST": "",
    }
    sim = subprocess.Popen(
        [f"{target}/pty-sim", "--codec", codec, "--rate", "300", "--frames", str(frames),
         "--pool", str(POOL), "--port", str(port), "--out", "zmq", "--zmq-port", str(zport),
         "--linger-ms", "500"],
        stdout=subprocess.PIPE, text=True,
    )
    for line in sim.stdout:
        if line.startswith("sim listening"):
            break
    # The simulator binds PUSH once POS has a subscriber.
    ctx = Context("pva", conf=conf, useenv=False, nt=False)
    sub = ctx.monitor("PTY:POS", lambda v: None)
    s = z.socket(PULL)
    z.L.zmq_connect(s, f"tcp://127.0.0.1:{zport}".encode())
    tag = f"rust-push-{codec}"
    try:
        part, more = z.recv(s)
        h = json.loads(part)
        t(f"{tag} dheader", h["htype"] == "dheader-1.0" and h["header_detail"] == "basic"
          and more, f"{h} more={more}")
        part, more = z.recv(s)
        cfg = json.loads(part)
        t(f"{tag} config", not more and (cfg["x_pixels_in_detector"], cfg["y_pixels_in_detector"],
                                         cfg["nimages"]) == (W, H, frames), f"{cfg} more={more}")
        series, got = h["series"], 0
        while True:
            part, more = z.recv(s)
            head = json.loads(part)
            if head["htype"].startswith("dseries_end"):
                t(f"{tag} dseries_end", head == {"htype": "dseries_end-1.0", "series": series}
                  and not more, f"{head} more={more}")
                break
            frame = head["frame"]
            parts = [(part, more)] + [z.recv(s) for _ in range(3)]
            mores = [m for _, m in parts]
            d = json.loads(parts[1][0])
            blob = parts[2][0]
            tm = json.loads(parts[3][0])
            t(f"{tag} image parts", mores == [True, True, True, False] and head["htype"] == "dimage-1.0"
              and head["series"] == series and frame == got, f"frame {frame} {head} {mores}")
            t(f"{tag} dimage_d", d == {"htype": "dimage_d-1.0", "shape": [W, H], "type": "uint16",
                                       "encoding": ENCODING[codec], "size": len(blob)}, f"{frame} {d}")
            t(f"{tag} dconfig", tm["htype"] == "dconfig-1.0"
              and tm["stop_time"] - tm["start_time"] == tm["real_time"] > 0, f"{frame} {tm}")
            if codec == "bslz4":
                t(f"{tag} bshuf header", struct.unpack(">QI", blob[:12]) == (2 * W * H, 8192),
                  f"{frame} {blob[:12].hex()}")
            ref = raw[frame % POOL]
            t(f"{tag} C decode", np.array_equal(c_decode(d["encoding"], blob, ref.size), ref),
              f"frame {frame}")
            got += 1
        t(f"{tag} frames", got == frames, f"{got}/{frames}")
    finally:
        z.L.zmq_close(s)
        sub.close()
        ctx.close()
        sim.wait(30)


def c_push_to_rust_pull(z, target, codec, raw, port, zport, frames, t):
    """C-compressed frames over the system libzmq into pty-recv, positions
    from a p4p server."""
    T = Type([("uniqueId", "i"), ("x", "d"), ("y", "d")], id="ptyrs:position:1.0")
    pv = SharedPV(initial=Value(T, {"uniqueId": -1, "x": 0.0, "y": 0.0}))
    srv = Server(providers=[{"PTY:POS": pv}], useenv=False, conf={
        "EPICS_PVAS_INTF_ADDR_LIST": "127.0.0.1",
        "EPICS_PVAS_SERVER_PORT": str(port),
        "EPICS_PVAS_BROADCAST_PORT": str(port + 1),
    })
    blobs = []
    for r in raw:
        if codec == "lz4":
            blobs.append(lz4.block.compress(r.tobytes(), store_size=False))
        else:
            blobs.append(struct.pack(">QI", r.nbytes, 8192) + bitshuffle.compress_lz4(r).tobytes())
    s = z.socket(PUSH)
    if z.L.zmq_bind(s, f"tcp://127.0.0.1:{zport}".encode()) < 0:
        z.err("zmq_bind")
    recv = subprocess.Popen(
        [f"{target}/pty-recv", "--addr", f"127.0.0.1:{port}", "--source", "zmq",
         "--zmq", f"tcp://127.0.0.1:{zport}", "--frames", str(frames), "--pool", str(POOL),
         "--verify-every", "1", "--workers", "4", "--util-min", "0", "--connect-timeout-s", "30"],
        stdout=subprocess.PIPE, text=True,
    )
    out = []
    reader = threading.Thread(target=lambda: out.extend(recv.stdout))
    reader.start()
    series, t0 = 3, time.time_ns()
    cfg = {"x_pixels_in_detector": W, "y_pixels_in_detector": H, "nimages": frames,
           "ntrigger": 1, "frame_time": 1 / 300, "count_time": 1 / 300}
    hdr = {"htype": "dheader-1.0", "series": series, "header_detail": "basic"}
    # Blocks until pty-recv connects; it subscribes POS first.
    z.send_parts(s, [json.dumps(hdr).encode(), json.dumps(cfg).encode()])
    for i in range(frames):
        x, y = position(i)
        pv.post(Value(T, {"uniqueId": i, "x": x, "y": y}))
        blob = blobs[i % POOL]
        start = time.time_ns() - t0
        z.send_parts(s, [
            json.dumps({"htype": "dimage-1.0", "series": series, "frame": i, "hash": ""}).encode(),
            json.dumps({"htype": "dimage_d-1.0", "shape": [W, H], "type": "uint16",
                        "encoding": ENCODING[codec], "size": len(blob)}).encode(),
            blob,
            json.dumps({"htype": "dconfig-1.0", "start_time": start, "stop_time": start + 3333,
                        "real_time": 3333}).encode(),
        ])
        time.sleep(1 / 300)
    z.send_parts(s, [json.dumps({"htype": "dseries_end-1.0", "series": series}).encode()])
    rc = recv.wait(120)
    reader.join()
    z.L.zmq_close(s)
    srv.stop()
    text = "".join(out)
    line = lambda k: next((l for l in out if l.startswith(k)), "")
    tag = f"c-push-{codec}"
    t(f"{tag} pty-recv exit 0", rc == 0, f"exit {rc}\n{text}")
    t(f"{tag} lost=0", f"expected={frames} received_unique={frames} gpu_frames={frames} lost=0"
      in line("recv expected"), line("recv expected"))
    t(f"{tag} rejected=0", " rejected=0 " in line("recv expected"), line("recv expected"))
    v = line("recv verify")
    t(f"{tag} every frame verified", f"sampled={frames} sampled_bad=0" in v
      and "h2d_mismatch_slots=0 e2e_mismatch_slots=0" in v, v)
    t(f"{tag} series end", line("recv zmq").startswith(
        f"recv zmq messages={frames + 2} headers=1 series_end=1 rejected=0"), line("recv zmq"))


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--target", default=f"{root}/target/release")
    ap.add_argument("--libzmq", default="/lib/x86_64-linux-gnu/libzmq.so.5")
    ap.add_argument("--port", type=int, default=15087)
    ap.add_argument("--zmq-port", type=int, default=19997)
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    z = Zmq(a.libzmq)
    print(f"zmq_cross.py system libzmq {z.version} ({a.libzmq})")
    t = Tally()
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([f"{a.target}/examples/codec_cross", "--dump", d], check=True)
        raw = [np.fromfile(f"{d}/eiger_{i}.raw", "<u2") for i in range(POOL)]
    assert all(r.size == W * H for r in raw)
    for k, codec in enumerate(("none", "lz4", "bslz4")):
        rust_push_to_c_pull(z, a.target, codec, raw, a.port + 2 * k, a.zmq_port, a.frames, t)
    for k, codec in enumerate(("lz4", "bslz4")):
        c_push_to_rust_pull(z, a.target, codec, raw, a.port + 10 + 2 * k, a.zmq_port + 1,
                            a.frames, t)
    z.close()
    for what, n in sorted(t.ok.items()):
        print(f"zmq_cross.py {what} ok={n}")
    for b in t.bad:
        print(f"zmq_cross.py FAIL {b}")
    sys.exit(1 if t.bad else 0)


if __name__ == "__main__":
    main()
