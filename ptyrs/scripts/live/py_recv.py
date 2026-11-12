"""Python baseline for the pty-recv input path (DESIGN.md §0 fold condition 1),
written the way a beamline scientist would: p4p monitors FRAME and POS on
separate connections -> decode (python-lz4 / bitshuffle, full frame) ->
match position by uniqueId -> crop 256x256 -> u16 into a preallocated
numpy ring. Stops at the host ring: no GPU upload (cupy is not installed).

  --mode threads  one process, decode thread pool fed by a queue.Queue
  --mode procs    multiprocessing workers; the p4p callback copies each
                  payload into a shared-memory input ring, workers decode
                  into a shared-memory output ring

Run through py_input_bench.sh for the same CPU pinning as m0_input_bench.sh.
Exit 2 if frames were lost, 3 if the ring contents fail verification.
"""

import argparse
import ctypes
import itertools
import multiprocessing as mp
import os
import queue
import socket
import sys
import threading
import time

import bitshuffle
import lz4.block
import numpy as np
from p4p.client.thread import Context

CODECS = ("lz4", "bslz4")
POS_TABLE = 1 << 16


def decode(cid, payload, n):
    if cid == 0:
        return np.frombuffer(lz4.block.decompress(payload, uncompressed_size=2 * n), "<u2")
    return bitshuffle.decompress_lz4(payload, (n,), np.dtype("<u2"))


def position(uid):
    """pty-sim's raster (synth::position)."""
    return (uid % 128) * 50e-9, (uid // 128 % 128) * 50e-9


def cpu_ns(pids):
    """utime + stime of every thread, per pid."""
    tick = 1e9 / os.sysconf("SC_CLK_TCK")
    out = np.zeros(len(pids))
    for i, p in enumerate(pids):
        with open(f"/proc/{p}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        out[i] = (int(fields[11]) + int(fields[12])) * tick
    return out


def summary(ns):
    if len(ns) == 0:
        return "n=0"
    us = np.asarray(ns, dtype=np.float64) / 1e3
    p = np.percentile(us, [50, 99, 99.9])
    return f"n={len(us)} p50_us={p[0]:.0f} p99_us={p[1]:.0f} p999_us={p[2]:.0f} max_us={us.max():.0f}"


class WorkerStats:
    def __init__(self):
        self.done = self.committed = self.decode_err = self.pos_missing = self.pos_first = 0
        self.t_last = 0
        self.rx_to_ring, self.pub_to_ring, self.decode = [], [], []

    def commit(self, t_rx, t_pub, t_dec):
        now = time.monotonic_ns()
        self.rx_to_ring.append(now - t_rx)
        self.pub_to_ring.append(time.time_ns() - t_pub)
        self.decode.append(t_dec)
        self.committed += 1
        self.t_last = now


class Geometry:
    def __init__(self, a):
        self.w, self.h, self.c = a.width, a.height, a.crop
        self.n = a.width * a.height
        self.x0, self.y0 = a.width // 2 - a.crop // 2, a.height // 2 - a.crop // 2

    def crop(self, frame):
        return frame.reshape(self.h, self.w)[self.y0:self.y0 + self.c, self.x0:self.x0 + self.c]


def wait_pos(lookup, uid, deadline_ns, st):
    """Poll for the position of uid; None after the deadline."""
    xy = lookup(uid)
    if xy is not None:
        st.pos_first += 1
        return xy
    while time.monotonic_ns() < deadline_ns:
        time.sleep(20e-6)
        xy = lookup(uid)
        if xy is not None:
            return xy
    return None


# ---- variant (a): one process, thread pool ----------------------------------

def thread_worker(q, g, ring, ring_uid, ring_xy, slots, counter, pos, wait_ns, st):
    while True:
        item = q.get()
        if item is None:
            return
        uid, cid, payload, t_rx, t_pub = item
        t0 = time.monotonic_ns()
        try:
            frame = decode(cid, payload, g.n)
        except Exception:
            st.decode_err += 1
            st.done += 1
            continue
        t_dec = time.monotonic_ns() - t0
        xy = wait_pos(pos.get, uid, t_rx + wait_ns, st)
        if xy is None:
            st.pos_missing += 1
        else:
            s = next(counter) % slots
            ring[s] = g.crop(frame)
            ring_uid[s] = uid
            ring_xy[s] = xy
            st.commit(t_rx, t_pub, t_dec)
        st.done += 1


class Threads:
    def __init__(self, a, g):
        self.slots = a.slots
        self.ring = np.empty((a.slots, g.c, g.c), np.uint16)
        self.ring_uid = np.full(a.slots, -1, np.int64)
        self.ring_xy = np.zeros((a.slots, 2))
        self.pos = {}
        self.q = queue.Queue(maxsize=a.queue)
        self.stats = [WorkerStats() for _ in range(a.workers)]
        counter = itertools.count()
        self.threads = [
            threading.Thread(
                target=thread_worker,
                args=(self.q, g, self.ring, self.ring_uid, self.ring_xy, a.slots, counter,
                      self.pos, a.pos_wait_ms * 1_000_000, st),
                daemon=True)
            for st in self.stats
        ]
        for t in self.threads:
            t.start()
        self.pids = [os.getpid()]

    def on_pos(self, uid, x, y):
        self.pos[uid] = (x, y)

    def dispatch(self, uid, cid, payload, t_rx, t_pub):
        try:
            self.q.put_nowait((uid, cid, payload, t_rx, t_pub))
            return True
        except queue.Full:
            return False

    def done(self):
        return sum(st.done for st in self.stats)

    def finish(self):
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()
        return self.stats, self.ring, self.ring_uid, self.ring_xy


# ---- variant (b): multiprocessing, shared-memory rings ----------------------

def shared(ctype, shape, dtype):
    raw = mp.RawArray(ctype, int(np.prod(shape)))
    return raw, np.frombuffer(raw, dtype).reshape(shape)


def proc_worker(q, res, g, wait_ns, shm):
    inbuf = np.frombuffer(shm["inbuf"], np.uint8).reshape(shm["nin"], -1)
    busy = np.frombuffer(shm["busy"], np.uint8)
    ring = np.frombuffer(shm["ring"], np.uint16).reshape(shm["slots"], g.c, g.c)
    ring_uid = np.frombuffer(shm["ring_uid"], np.int64)
    ring_xy = np.frombuffer(shm["ring_xy"], np.float64).reshape(-1, 2)
    pos_id = np.frombuffer(shm["pos_id"], np.int64)
    pos_xy = np.frombuffer(shm["pos_xy"], np.float64).reshape(-1, 2)
    counter, done = shm["counter"], shm["done"]
    slots = shm["slots"]

    def lookup(uid):
        k = uid % POS_TABLE
        if pos_id[k] != uid:
            return None
        xy = (pos_xy[k, 0], pos_xy[k, 1])
        return xy if pos_id[k] == uid else None

    st = WorkerStats()
    while True:
        item = q.get()
        if item is None:
            break
        s, nbytes, uid, cid, t_rx, t_pub = item
        t0 = time.monotonic_ns()
        try:
            frame = decode(cid, inbuf[s, :nbytes], g.n)
            ok = True
        except Exception:
            ok = False
        busy[s] = 0
        if not ok:
            st.decode_err += 1
        else:
            t_dec = time.monotonic_ns() - t0
            xy = wait_pos(lookup, uid, t_rx + wait_ns, st)
            if xy is None:
                st.pos_missing += 1
            else:
                with counter.get_lock():
                    o = counter.value
                    counter.value += 1
                o %= slots
                ring[o] = g.crop(frame)
                ring_uid[o] = uid
                ring_xy[o] = xy
                st.commit(t_rx, t_pub, t_dec)
        with done.get_lock():
            done.value += 1
    st.rx_to_ring = np.array(st.rx_to_ring, np.int64)
    st.pub_to_ring = np.array(st.pub_to_ring, np.int64)
    st.decode = np.array(st.decode, np.int64)
    res.put(st)


class Procs:
    def __init__(self, a, g):
        # fork before p4p starts any thread; children inherit the RawArrays.
        ctx = mp.get_context("fork")
        nbytes = 2 * g.n
        slot_bytes = nbytes + nbytes // 255 + 65536  # LZ4 / bitshuffle-LZ4 bound
        self.nin, self.slot_bytes, self.slots = a.queue, slot_bytes, a.slots
        shm = {"nin": a.queue, "slots": a.slots}
        shm["inbuf"], self.inbuf = shared(ctypes.c_uint8, (a.queue, slot_bytes), np.uint8)
        shm["busy"], self.busy = shared(ctypes.c_uint8, (a.queue,), np.uint8)
        shm["ring"], self.ring = shared(ctypes.c_uint16, (a.slots, g.c, g.c), np.uint16)
        shm["ring_uid"], self.ring_uid = shared(ctypes.c_int64, (a.slots,), np.int64)
        shm["ring_xy"], self.ring_xy = shared(ctypes.c_double, (a.slots, 2), np.float64)
        shm["pos_id"], self.pos_id = shared(ctypes.c_int64, (POS_TABLE,), np.int64)
        shm["pos_xy"], self.pos_xy = shared(ctypes.c_double, (POS_TABLE, 2), np.float64)
        self.ring_uid[:] = -1
        self.pos_id[:] = -1
        shm["counter"] = ctx.Value(ctypes.c_int64, 0)
        shm["done"] = self.done_ctr = ctx.Value(ctypes.c_int64, 0)
        self.q, self.res = ctx.Queue(), ctx.Queue()
        self.procs = [
            ctx.Process(target=proc_worker, args=(self.q, self.res, g, a.pos_wait_ms * 1_000_000, shm),
                        daemon=True)
            for _ in range(a.workers)
        ]
        for p in self.procs:
            p.start()
        self.pids = [os.getpid()] + [p.pid for p in self.procs]
        self.cursor = 0

    def on_pos(self, uid, x, y):
        k = uid % POS_TABLE
        self.pos_id[k] = -1
        self.pos_xy[k] = (x, y)
        self.pos_id[k] = uid

    def dispatch(self, uid, cid, payload, t_rx, t_pub):
        s = self.cursor
        if self.busy[s]:
            free = np.flatnonzero(self.busy == 0)
            if len(free) == 0:
                return False
            s = int(free[0])
        n = len(payload)
        if n > self.slot_bytes:
            raise ValueError(f"payload {n} B exceeds input slot {self.slot_bytes} B")
        self.inbuf[s, :n] = payload
        self.busy[s] = 1
        self.cursor = (s + 1) % self.nin
        self.q.put((s, n, uid, cid, t_rx, t_pub))
        return True

    def done(self):
        return self.done_ctr.value

    def finish(self):
        for _ in self.procs:
            self.q.put(None)
        stats = [self.res.get() for _ in self.procs]
        for p in self.procs:
            p.join()
        return stats, self.ring, self.ring_uid, self.ring_xy


# ---- receive loop -------------------------------------------------------------

class Rx:
    def __init__(self, frames):
        self.seen = np.zeros(frames, bool)
        self.unique = self.dup = self.beyond = self.placeholders = self.rejected = 0
        self.gap_events = self.gap_frames = self.out_of_order = self.queue_full = 0
        self.dispatched = self.pos_updates = 0
        self.last_uid = -1
        self.first_rx = self.last_rx = None
        self.cpu_first = None
        self.error = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("threads", "procs"), required=True)
    ap.add_argument("--addr", default="127.0.0.1:15075")
    ap.add_argument("--frames", type=int, default=30_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--queue", type=int, default=256, help="dispatch queue / input ring slots")
    ap.add_argument("--slots", type=int, default=4096, help="output ring slots")
    ap.add_argument("--width", type=int, default=1028)
    ap.add_argument("--height", type=int, default=1062)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--pool", type=int, default=64, help="pty-sim --pool, for verification")
    ap.add_argument("--pipeline", type=int, default=1024, help="FRAME queueSize, pipeline=true")
    ap.add_argument("--pos-wait-ms", type=int, default=1000)
    ap.add_argument("--idle-s", type=float, default=3.0)
    ap.add_argument("--connect-s", type=float, default=60.0)
    a = ap.parse_args()
    g = Geometry(a)
    host, port = a.addr.rsplit(":", 1)
    cpus = sorted(os.sched_getaffinity(0))
    with open("/proc/self/sched") as f:
        util_min = next((l.split(":")[1].strip() for l in f if l.startswith("uclamp.min")), "n/a")
    print(f"pyrecv mode={a.mode} workers={a.workers} queue={a.queue} slots={a.slots} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} cpus={cpus[0]}-{cpus[-1]} "
          f"util_min={util_min} python={sys.version.split()[0]}", flush=True)

    sink = Threads(a, g) if a.mode == "threads" else Procs(a, g)
    rx = Rx(a.frames)
    image_bytes = 2 * g.n

    def on_frame(v):
        t_rx = time.monotonic_ns()
        try:
            uid = v["uniqueId"]
            if uid < 0:
                rx.placeholders += 1
                return
            name = v["codec.name"]
            if name not in CODECS or v["uncompressedSize"] != image_bytes:
                rx.rejected += 1
                return
            rx.last_rx = t_rx
            if rx.first_rx is None:
                # first_rx last: the main loop reads the other two once it is set.
                rx.cpu_first = cpu_ns(sink.pids)
                rx.first_rx = t_rx
            if uid >= a.frames:
                rx.beyond += 1
                return
            if rx.seen[uid]:
                rx.dup += 1
                return
            rx.seen[uid] = True
            rx.unique += 1
            if uid > rx.last_uid + 1:
                rx.gap_events += 1
                rx.gap_frames += uid - rx.last_uid - 1
            elif uid < rx.last_uid:
                rx.out_of_order += 1
            rx.last_uid = max(rx.last_uid, uid)
            t_pub = v["timeStamp.secondsPastEpoch"] * 1_000_000_000 + v["timeStamp.nanoseconds"]
            if sink.dispatch(uid, CODECS.index(name), v["value"], t_rx, t_pub):
                rx.dispatched += 1
            else:
                rx.queue_full += 1
        except Exception as e:
            if rx.error is None:
                rx.error = repr(e)

    def on_pos(v):
        uid = v["uniqueId"]
        if uid >= 0:
            sink.on_pos(uid, v["x"], v["y"])
            rx.pos_updates += 1

    deadline = time.monotonic() + a.connect_s
    while True:
        try:
            socket.create_connection((host, int(port)), timeout=1).close()
            break
        except OSError:
            if time.monotonic() > deadline:
                sys.exit(f"pyrecv cannot reach {a.addr}")
            time.sleep(0.05)
    conf = {"EPICS_PVA_NAME_SERVERS": a.addr, "EPICS_PVA_AUTO_ADDR_LIST": "NO", "EPICS_PVA_ADDR_LIST": ""}
    ctx_frame = Context("pva", conf=conf, useenv=False, nt=False)
    ctx_pos = Context("pva", conf=conf, useenv=False, nt=False)
    subs = [
        ctx_pos.monitor("PTY:POS", on_pos, request="record[queueSize=4096,pipeline=true]field()"),
        ctx_frame.monitor("PTY:FRAME", on_frame,
                          request=f"record[queueSize={a.pipeline},pipeline=true]field()"),
    ]

    sample = None
    last_unique = 0
    while True:
        time.sleep(0.1)
        now = time.monotonic_ns()
        if rx.first_rx is None:
            if now / 1e9 > deadline:
                print("pyrecv no frame within connect timeout", flush=True)
                break
            continue
        if rx.unique != last_unique:
            last_unique = rx.unique
            sample = (now, cpu_ns(sink.pids))
        if rx.unique >= a.frames or now - rx.last_rx > a.idle_s * 1e9:
            break
    for s in subs:
        s.close()
    ctx_frame.close()
    ctx_pos.close()
    drain_deadline = time.monotonic() + 30
    while sink.done() < rx.dispatched and time.monotonic() < drain_deadline:
        time.sleep(0.01)
    t_drain, cpu_drain = time.monotonic_ns(), cpu_ns(sink.pids) if rx.first_rx else 0
    stats, ring, ring_uid, ring_xy = sink.finish()

    committed = sum(st.committed for st in stats)
    t_last = max((st.t_last for st in stats), default=0)
    first = rx.first_rx or t_drain
    lost = a.frames - min(committed, a.frames)
    cat = lambda f: np.concatenate([np.asarray(getattr(st, f), np.int64) for st in stats])
    print(f"pyrecv expected={a.frames} received_unique={rx.unique} ring_frames={committed} lost={lost} "
          f"dup={rx.dup} beyond={rx.beyond} gap_events={rx.gap_events} gap_frames={rx.gap_frames} "
          f"out_of_order={rx.out_of_order} queue_full={rx.queue_full} rejected={rx.rejected} "
          f"placeholders={rx.placeholders} decode_err={sum(st.decode_err for st in stats)} "
          f"pos_missing={sum(st.pos_missing for st in stats)} pos_updates={rx.pos_updates}")
    if rx.error:
        print(f"pyrecv first_error={rx.error}")
    span = max((rx.last_rx or first) - first, 1) / 1e9
    nan = np.full(len(sink.pids), np.nan)
    steady = (sample[1] - rx.cpu_first) / (sample[0] - first) if sample and sample[0] > first else nan
    total = (cpu_drain - rx.cpu_first) / max(t_drain - first, 1) if rx.cpu_first is not None else nan
    print(f"pyrecv rx_rate_hz={rx.unique / span:.1f} cpu_cores_steady={steady.sum():.2f} "
          f"cpu_cores_total={total.sum():.2f} main_process_cores_steady={steady[0]:.2f} "
          f"processes={len(sink.pids)} drain_after_last_rx_ms={(t_last - (rx.last_rx or first)) / 1e6:.1f}")
    print(f"pyrecv rx_to_ring {summary(cat('rx_to_ring'))}")
    print(f"pyrecv pub_to_ring {summary(cat('pub_to_ring'))}")
    print(f"pyrecv decode {summary(cat('decode'))}")
    print(f"pyrecv pos_before_frame={sum(st.pos_first for st in stats)}")

    # Every filled slot holds the crop of pool entry uid % pool and the
    # position pty-sim posted for uid.
    filled = np.flatnonzero(ring_uid >= 0)
    ref, crop_bad, pos_bad = {}, 0, 0
    for s in filled:
        uid = int(ring_uid[s])
        crop_bad += not np.array_equal(ref.setdefault(uid % a.pool, ring[s]), ring[s])
        pos_bad += tuple(ring_xy[s]) != position(uid)
    print(f"pyrecv verify slots_checked={len(filled)} pool_entries={len(ref)} crop_mismatch={crop_bad} "
          f"pos_mismatch={pos_bad}")
    sys.exit(2 if lost else 3 if crop_bad or pos_bad else 0)


if __name__ == "__main__":
    main()
