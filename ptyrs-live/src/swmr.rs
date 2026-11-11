//! HDF5 SWMR live input in the layout areaDetector's NDFileHDF5 plugin
//! writes by default: frames at [`DATA`], one per chunk, hard-linked from
//! `/entry/data/data`, and one per-frame NDAttribute series each for the
//! uniqueId, the EPICS timestamp and the scan position. [`SwmrOut`] writes
//! it (pty-sim `--out h5`); [`run_swmr`] turns it into the same [`Work`]
//! events as the PVA and ZMQ sources.
//!
//! rust-hdf5 0.7.10's `SwmrFileReader` hands out decoded slices only, so
//! the reader threads here decode and the frames reach the workers as
//! [`Codec::None`].

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use crossbeam_channel::{Receiver, Sender};
use rust_hdf5::FilterPipeline;
use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};

use crate::codec::{Codec, bslz4_compress_u16, bytemuck_u16, lz4_compress};
use crate::eiger::{Progress, blob_prefix};
use crate::pipeline::{Placement, SourceStats, Work};
use crate::pos::{Pos, PosTable};
use crate::stats::{Histogram, mono_ns, wall_ns};

pub const DATA: &str = "entry/instrument/detector/data";
pub const UID: &str = "entry/instrument/NDAttributes/NDArrayUniqueId";
pub const TS_SEC: &str = "entry/instrument/NDAttributes/NDArrayEpicsTSSec";
pub const TS_NSEC: &str = "entry/instrument/NDAttributes/NDArrayEpicsTSnSec";
pub const POS_X: &str = "entry/instrument/NDAttributes/PosX";
pub const POS_Y: &str = "entry/instrument/NDAttributes/PosY";
/// Frames per chunk of an attribute series. A flush rewrites the chunk
/// still filling in place, so a reader sees every frame the extent counts.
const ATTR_CHUNK: u64 = 1024;
/// 1990-01-01 less 1970-01-01, s.
const EPICS_EPOCH_S: i64 = 631_152_000;
/// Frames the index thread takes per refresh, so the positions it posts
/// ahead of their frames stay far below the [`PosTable`] capacity.
const MAX_BATCH: u64 = 1024;
/// Consecutive failed refreshes or reads (a header caught mid-rewrite)
/// before a reader gives up.
const MAX_RETRIES: u32 = 1000;

/// One frame as the stored chunk of [`DATA`] under `codec`'s HDF5 filter.
pub fn filtered_chunk(codec: Codec, img: &[u16], w: usize, h: usize) -> Vec<u8> {
    let raw = bytemuck_u16(img);
    match codec {
        Codec::None => raw.to_vec(),
        // Filter 32008: [u64 BE total bytes][u32 BE block bytes], the
        // Eiger blob prefix as well.
        Codec::Bslz4 => {
            let mut c = blob_prefix(codec, w, h);
            c.extend_from_slice(&bslz4_compress_u16(img));
            c
        }
        // Filter 32004 with one block: [u64 BE total][u32 BE block =
        // total][u32 BE stored]; a stored size equal to the block's marks
        // raw bytes, so a block that does not shrink goes in raw.
        Codec::Lz4 => {
            let z = lz4_compress(raw);
            let body: &[u8] = if z.len() < raw.len() { &z } else { raw };
            let mut c = Vec::with_capacity(16 + body.len());
            c.extend_from_slice(&(raw.len() as u64).to_be_bytes());
            c.extend_from_slice(&(raw.len() as u32).to_be_bytes());
            c.extend_from_slice(&(body.len() as u32).to_be_bytes());
            c.extend_from_slice(body);
            c
        }
    }
}

/// The writing side, as NDFileHDF5 in SWMR mode lays the file out.
pub struct SwmrOut {
    f: SwmrFileWriter,
    filtered: bool,
    data: usize,
    uid: usize,
    sec: usize,
    nsec: usize,
    x: usize,
    y: usize,
}

impl SwmrOut {
    /// Creates `path` with its groups, datasets and link, then starts SWMR,
    /// after which readers may attach.
    pub fn create(path: &Path, codec: Codec, (w, h): (usize, usize)) -> rust_hdf5::Result<Self> {
        let mut f = SwmrFileWriter::create(path)?;
        for (parent, name) in [
            ("/", "entry"),
            ("/entry", "instrument"),
            ("/entry/instrument", "detector"),
            ("/entry/instrument", "NDAttributes"),
            ("/entry", "data"),
        ] {
            f.create_group(parent, name)?;
        }
        let dims = [h as u64, w as u64];
        let data = match codec {
            Codec::None => f.create_streaming_dataset::<u16>(DATA, &dims)?,
            Codec::Lz4 => {
                f.create_streaming_dataset_compressed::<u16>(DATA, &dims, FilterPipeline::lz4())?
            }
            Codec::Bslz4 => f.create_streaming_dataset_compressed::<u16>(
                DATA,
                &dims,
                FilterPipeline::bshuf_lz4(2),
            )?,
        };
        let c = [ATTR_CHUNK];
        let uid = f.create_streaming_dataset_chunked::<i32>(UID, &[], &c)?;
        let sec = f.create_streaming_dataset_chunked::<u32>(TS_SEC, &[], &c)?;
        let nsec = f.create_streaming_dataset_chunked::<u32>(TS_NSEC, &[], &c)?;
        let x = f.create_streaming_dataset_chunked::<f64>(POS_X, &[], &c)?;
        let y = f.create_streaming_dataset_chunked::<f64>(POS_Y, &[], &c)?;
        f.create_hard_link("/entry/data", "data", &format!("/{DATA}"))?;
        f.start_swmr()?;
        Ok(Self {
            f,
            filtered: codec != Codec::None,
            data,
            uid,
            sec,
            nsec,
            x,
            y,
        })
    }

    /// Frame `uid` as its stored chunk ([`filtered_chunk`]) with its
    /// attributes; readers see it after the next [`Self::flush`].
    pub fn append(
        &mut self,
        chunk: &[u8],
        uid: i32,
        (x, y): (f64, f64),
        wall_ns: i64,
    ) -> rust_hdf5::Result<()> {
        if self.filtered {
            self.f.append_frame_raw(self.data, chunk, 0)?;
        } else {
            self.f.append_frame(self.data, chunk)?;
        }
        let sec = (wall_ns.div_euclid(1_000_000_000) - EPICS_EPOCH_S) as u32;
        let nsec = wall_ns.rem_euclid(1_000_000_000) as u32;
        self.f.append_frame(self.sec, &sec.to_le_bytes())?;
        self.f.append_frame(self.nsec, &nsec.to_le_bytes())?;
        self.f.append_frame(self.x, &x.to_le_bytes())?;
        self.f.append_frame(self.y, &y.to_le_bytes())?;
        self.f.append_frame(self.uid, &uid.to_le_bytes())
    }

    pub fn flush(&mut self) -> rust_hdf5::Result<()> {
        self.f.flush()
    }

    pub fn close(self) -> rust_hdf5::Result<()> {
        self.f.close()
    }
}

pub struct SwmrCfg {
    pub path: PathBuf,
    pub readers: usize,
    pub width: usize,
    pub height: usize,
    /// Sleep between refreshes that found nothing new.
    pub poll: Duration,
    /// Frames visible but not yet taken by a reader.
    pub queue: usize,
}

pub struct SwmrReport {
    /// From the first attempt to the file opening, ms.
    pub open_ms: i64,
    pub refreshes: u64,
    /// Refreshes and reads that failed and were retried.
    pub retries: u64,
    pub frames: u64,
    /// Frames visible to the index thread but not yet read, at most.
    pub max_backlog: u64,
    /// Read and decode of one frame.
    pub read: Histogram,
}

struct Job {
    index: u64,
    uid: i32,
    t_pub: i64,
}

/// Frames every dataset of the layout holds. A refresh may land between two
/// header rewrites of one flush, so no single dataset's extent is trusted.
fn visible(r: &mut SwmrFileReader) -> rust_hdf5::Result<u64> {
    let mut k = u64::MAX;
    for name in [DATA, UID, TS_SEC, TS_NSEC, POS_X, POS_Y] {
        k = k.min(r.dataset_shape(name)?[0]);
    }
    Ok(k)
}

/// Retries `op` through transient failures, sleeping `poll` in between.
fn retry<T>(
    poll: Duration,
    retries: &mut u64,
    what: &str,
    mut op: impl FnMut() -> rust_hdf5::Result<T>,
) -> Result<T, String> {
    let mut n = 0;
    loop {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) if n == MAX_RETRIES => return Err(format!("{what}: {e}")),
            Err(_) => {
                n += 1;
                *retries += 1;
                std::thread::sleep(poll);
            }
        }
    }
}

struct Attrs {
    uid: Vec<i32>,
    sec: Vec<u32>,
    nsec: Vec<u32>,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn read_attrs(r: &mut SwmrFileReader, k0: u64, n: u64) -> rust_hdf5::Result<Attrs> {
    Ok(Attrs {
        uid: r.read_slice(UID, &[k0], &[n])?,
        sec: r.read_slice(TS_SEC, &[k0], &[n])?,
        nsec: r.read_slice(TS_NSEC, &[k0], &[n])?,
        x: r.read_slice(POS_X, &[k0], &[n])?,
        y: r.read_slice(POS_Y, &[k0], &[n])?,
    })
}

/// Follows the file at `cfg.path` until `stop`: one index thread (the
/// caller) refreshes, posts each new frame's position to `pos` and queues
/// the frame; `cfg.readers` threads, each with its own handle on the file,
/// read and decode the frames into `st`. Waits for the file to appear.
pub fn run_swmr(
    cfg: &SwmrCfg,
    st: &Mutex<SourceStats>,
    pos: &PosTable,
    tx: Sender<Work>,
    place: &Placement,
    stop: &AtomicBool,
    progress: &Progress,
) -> Result<SwmrReport, String> {
    let mut rep = SwmrReport {
        open_ms: 0,
        refreshes: 0,
        retries: 0,
        frames: 0,
        max_backlog: 0,
        read: Histogram::new(100_000),
    };
    let t0 = mono_ns();
    // Until the writer has started SWMR it holds the file exclusively.
    let mut idx = loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(rep);
        }
        if let Ok(mut r) = SwmrFileReader::open(&cfg.path)
            && r.dataset_shape(DATA).is_ok()
        {
            break r;
        }
        std::thread::sleep(cfg.poll);
    };
    rep.open_ms = (mono_ns() - t0) / 1_000_000;
    let shape = idx.dataset_shape(DATA).map_err(|e| e.to_string())?;
    let esize = idx.dataset_element_size(DATA).map_err(|e| e.to_string())?;
    if shape[1..] != [cfg.height as u64, cfg.width as u64] || esize != 2 {
        return Err(format!(
            "{DATA}: {shape:?} of {esize}-byte elements, expected [n, {}, {}] u16",
            cfg.height, cfg.width
        ));
    }
    let (jtx, jrx) = crossbeam_channel::bounded::<Job>(cfg.queue);
    std::thread::scope(|sc| {
        let readers: Vec<_> = (0..cfg.readers)
            .map(|i| {
                let (jrx, tx) = (jrx.clone(), tx.clone());
                std::thread::Builder::new()
                    .name(format!("h5-read{i}"))
                    .spawn_scoped(sc, move || {
                        place.enter(place.readers[i].as_ref().map(std::slice::from_ref));
                        read_frames(cfg, jrx, st, tx, progress)
                    })
                    .unwrap()
            })
            .collect();
        drop((jrx, tx));
        let indexed = index(&mut idx, cfg, pos, jtx, stop, &mut rep);
        for h in readers {
            let (read, retries) = h.join().unwrap()?;
            rep.read.merge(&read);
            rep.retries += retries;
        }
        indexed
    })?;
    Ok(rep)
}

fn index(
    r: &mut SwmrFileReader,
    cfg: &SwmrCfg,
    pos: &PosTable,
    jobs: Sender<Job>,
    stop: &AtomicBool,
    rep: &mut SwmrReport,
) -> Result<(), String> {
    let mut next = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let k = retry(cfg.poll, &mut rep.retries, "refresh", || {
            r.refresh()?;
            visible(r)
        })?;
        rep.refreshes += 1;
        if k <= next {
            std::thread::sleep(cfg.poll);
            continue;
        }
        let t = wall_ns();
        rep.max_backlog = rep.max_backlog.max(k - next + jobs.len() as u64);
        let n = (k - next).min(MAX_BATCH);
        let a = retry(cfg.poll, &mut rep.retries, "attributes", || {
            read_attrs(r, next, n)
        })?;
        for j in 0..n as usize {
            let uid = a.uid[j];
            pos.insert(
                uid,
                Pos {
                    x: a.x[j],
                    y: a.y[j],
                    t_rx: t,
                },
            );
            let t_pub = (a.sec[j] as i64 + EPICS_EPOCH_S) * 1_000_000_000 + a.nsec[j] as i64;
            let job = Job {
                index: next + j as u64,
                uid,
                t_pub,
            };
            // Every reader gone means one failed; its error is reported.
            if jobs.send(job).is_err() {
                return Ok(());
            }
        }
        next += n;
        rep.frames = next;
    }
    Ok(())
}

fn read_frames(
    cfg: &SwmrCfg,
    jobs: Receiver<Job>,
    st: &Mutex<SourceStats>,
    tx: Sender<Work>,
    progress: &Progress,
) -> Result<(Histogram, u64), String> {
    let mut retries = 0;
    let mut r = retry(cfg.poll, &mut retries, "open", || {
        SwmrFileReader::open(&cfg.path)
    })?;
    let mut hist = Histogram::new(100_000);
    let (mut seen, h, w) = (0u64, cfg.height as u64, cfg.width as u64);
    for job in jobs {
        let t = mono_ns();
        // The index thread saw the frame, so one refresh shows it here.
        let b = retry(cfg.poll, &mut retries, "frame", || {
            if job.index >= seen {
                r.refresh()?;
                seen = visible(&mut r)?;
            }
            r.read_slice_raw(DATA, &[job.index, 0, 0], &[1, h, w])
        })?;
        hist.record(mono_ns() - t);
        let mut s = st.lock().unwrap();
        // Received once read, not once visible: a finished file's whole
        // backlog is visible at once. Stamped under the lock, so receive
        // times keep the queue's order.
        let work = Work {
            payload: Bytes::from(b),
            codec: Codec::None,
            uid: job.uid,
            t_rx: wall_ns(),
            t_pub: job.t_pub,
        };
        s.accept(work, &tx);
        progress.unique.store(s.unique, Ordering::Relaxed);
        progress.last_rx.store(s.last_rx, Ordering::Relaxed);
    }
    Ok((hist, retries))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    /// Removed when the test ends, pass or fail.
    struct Tmp(PathBuf);

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn tmp(name: &str) -> Tmp {
        let p = std::env::temp_dir().join(format!("ptyrs-swmr-{}-{name}.h5", std::process::id()));
        let _ = std::fs::remove_file(&p);
        Tmp(p)
    }

    const W: usize = 24;
    const H: usize = 16;

    /// Sparse counts, so both codecs compress.
    fn frame(uid: i32) -> Vec<u16> {
        (0..W * H)
            .map(|i| {
                let v = ((i as u64 * 2_654_435_761 + uid as u64 * 40_503) >> 7) & 0xff;
                if v < 40 { v as u16 } else { 0 }
            })
            .collect()
    }

    /// Not short decimals, so a lossy round trip would show.
    fn pos_of(uid: i32) -> (f64, f64) {
        (uid as f64 * 1.25e-7 + 1e-9 / 3.0, -(uid as f64) * 0.7e-7)
    }

    fn wall_of(uid: i32) -> i64 {
        1_700_000_000_123_456_789 + uid as i64 * 333_333
    }

    fn write(out: &mut SwmrOut, codec: Codec, uids: std::ops::Range<i32>) {
        for uid in uids {
            let c = filtered_chunk(codec, &frame(uid), W, H);
            out.append(&c, uid, pos_of(uid), wall_of(uid)).unwrap();
        }
    }

    fn cfg(path: &Path, readers: usize) -> SwmrCfg {
        SwmrCfg {
            path: path.to_path_buf(),
            readers,
            width: W,
            height: H,
            poll: Duration::from_micros(50),
            queue: 64,
        }
    }

    /// Runs the source next to `writer` until `frames` frames arrive.
    fn follow(
        cfg: &SwmrCfg,
        frames: u64,
        writer: impl FnOnce() + Send,
    ) -> (Vec<Work>, PosTable, Result<SwmrReport, String>) {
        let st = Mutex::new(SourceStats::new(frames));
        let pos = PosTable::new(1 << 12);
        let (tx, rx) = crossbeam_channel::bounded(frames as usize);
        let place = Placement::new("", 1, 0, cfg.readers, 0);
        let (stop, progress) = (AtomicBool::new(false), Progress::default());
        let mut got = Vec::new();
        let rep = std::thread::scope(|sc| {
            sc.spawn(writer);
            let src = sc.spawn(|| run_swmr(cfg, &st, &pos, tx, &place, &stop, &progress));
            while (got.len() as u64) < frames {
                match rx.recv_timeout(Duration::from_secs(20)) {
                    Ok(w) => got.push(w),
                    Err(_) => break,
                }
            }
            stop.store(true, Ordering::Relaxed);
            src.join().unwrap()
        });
        (got, pos, rep)
    }

    fn check(got: &[Work], pos: &PosTable, frames: i32) {
        let mut uids: Vec<i32> = got.iter().map(|w| w.uid).collect();
        uids.sort_unstable();
        assert_eq!(uids, (0..frames).collect::<Vec<_>>());
        for w in got {
            assert_eq!(w.codec, Codec::None);
            assert_eq!(&w.payload[..], bytemuck_u16(&frame(w.uid)), "uid {}", w.uid);
            assert_eq!(w.t_pub, wall_of(w.uid));
            let p = pos.get(w.uid).expect("position posted");
            assert_eq!((p.x, p.y), pos_of(w.uid));
            assert!(p.t_rx <= w.t_rx, "uid {}: frame before position", w.uid);
        }
        // One stamp per read, in queue order: a closed file's frames are
        // all visible at one instant but are not received at one.
        assert!(got.windows(2).all(|p| p[0].t_rx <= p[1].t_rx));
        assert!(got[0].t_rx < got[got.len() - 1].t_rx);
    }

    #[test]
    fn closed_file_round_trips_every_codec() {
        for codec in [Codec::None, Codec::Lz4, Codec::Bslz4] {
            let t = tmp(codec.name());
            let mut out = SwmrOut::create(&t.0, codec, (W, H)).unwrap();
            write(&mut out, codec, 0..40);
            out.close().unwrap();
            let (got, pos, rep) = follow(&cfg(&t.0, 2), 40, || {});
            let rep = rep.unwrap();
            check(&got, &pos, 40);
            assert_eq!(rep.frames, 40, "{codec:?}");
            // The NDFileHDF5 link to the frames.
            let mut r = SwmrFileReader::open(&t.0).unwrap();
            let b = r.read_slice_raw("entry/data/data", &[7, 0, 0], &[1, H as u64, W as u64]);
            assert_eq!(b.unwrap(), bytemuck_u16(&frame(7)), "{codec:?}");
        }
    }

    /// Frames and attributes become visible flush by flush while the
    /// readers follow; the source starts before the file exists.
    #[test]
    fn follows_a_live_writer_across_flushes() {
        let t = tmp("live");
        let codec = Codec::Bslz4;
        let n = 300;
        let path = t.0.clone();
        let (got, pos, rep) = follow(&cfg(&t.0, 3), n as u64, move || {
            std::thread::sleep(Duration::from_millis(20));
            // The writer is not Send; it lives on this thread.
            let mut out = SwmrOut::create(&path, codec, (W, H)).unwrap();
            std::thread::sleep(Duration::from_millis(5));
            for k in (0..n).step_by(7) {
                write(&mut out, codec, k..(k + 7).min(n));
                out.flush().unwrap();
                std::thread::sleep(Duration::from_micros(300));
            }
            out.close().unwrap();
        });
        let rep = rep.unwrap();
        check(&got, &pos, n);
        assert_eq!(rep.frames, n as u64);
        assert!(rep.refreshes > n as u64 / 7, "{} refreshes", rep.refreshes);
    }

    #[test]
    fn lz4_stores_an_incompressible_frame_raw() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let img: Vec<u16> = (0..W * H)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u16
            })
            .collect();
        let c = filtered_chunk(Codec::Lz4, &img, W, H);
        assert_eq!(c.len(), 16 + W * H * 2);
        let t = tmp("raw-lz4");
        let mut out = SwmrOut::create(&t.0, Codec::Lz4, (W, H)).unwrap();
        out.append(&c, 0, (0.0, 0.0), 0).unwrap();
        out.close().unwrap();
        let mut r = SwmrFileReader::open(&t.0).unwrap();
        let b = r.read_slice_raw(DATA, &[0, 0, 0], &[1, H as u64, W as u64]);
        assert_eq!(b.unwrap(), bytemuck_u16(&img));
    }

    #[test]
    fn refuses_a_frame_size_other_than_the_geometry() {
        let t = tmp("shape");
        let mut out = SwmrOut::create(&t.0, Codec::None, (W, H)).unwrap();
        write(&mut out, Codec::None, 0..2);
        out.close().unwrap();
        let mut c = cfg(&t.0, 1);
        c.height = W;
        let (got, _, rep) = follow(&c, 2, || {});
        assert!(got.is_empty());
        let e = rep.err().expect("shape refused");
        assert!(e.contains("expected [n, 24, 24] u16"), "{e}");
    }
}
