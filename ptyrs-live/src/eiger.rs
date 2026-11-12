//! Eiger stream API v1 over ZMQ PUSH/PULL, as ADEiger reads it
//! (`eigerApp/src/streamApi.cpp`): a `dheader-1.0` message per series, a
//! 4-part image message per frame, a `dseries_end-1.0` message at the end.
//!
//! Image parts: `dimage-1.0` {series, frame, hash}, `dimage_d-1.0`
//! {shape [x, y], type, encoding, size}, the data blob, `dconfig-1.0`
//! {start_time, stop_time, real_time} in ns since the series start. A
//! `bs16-lz4<` blob starts with the bitshuffle HDF5-filter header
//! (u64 BE uncompressed bytes, u32 BE block bytes) that ADEiger skips.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::codec::{Codec, bshuf_block_elems};
use crate::handoff::WorkTx;
use crate::pipeline::{Geometry, SourceStats, Work};
use crate::stats::wall_ns;

/// ADEiger's `ZMQ_PORT`.
pub const PORT: u16 = 9999;

/// Header part of a multipart message; the fields present depend on `htype`.
#[derive(Serialize, Deserialize, Default)]
struct Head {
    htype: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    series: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    header_detail: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct ImageDesc {
    htype: String,
    shape: [u64; 2],
    #[serde(rename = "type")]
    dtype: String,
    encoding: String,
    size: u64,
}

#[derive(Serialize, Deserialize)]
struct ImageTimes {
    htype: String,
    start_time: i64,
    stop_time: i64,
    real_time: i64,
}

/// The `header_detail: basic` config part, reduced to what the receiver
/// reads. `ptyrs_series_start_ns` is a simulator extension: the wall clock
/// at the series start, which turns `dconfig.start_time` into a publish
/// time. A real Eiger does not send it.
#[derive(Serialize, Deserialize, Default)]
pub struct SeriesConfig {
    pub x_pixels_in_detector: u64,
    pub y_pixels_in_detector: u64,
    pub nimages: u64,
    pub ntrigger: u64,
    pub frame_time: f64,
    pub count_time: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ptyrs_series_start_ns: Option<i64>,
}

fn json<T: Serialize>(v: &T) -> Vec<u8> {
    serde_json::to_vec(v).expect("plain structs serialise")
}

pub fn encoding(c: Codec) -> &'static str {
    match c {
        Codec::None => "<",
        Codec::Lz4 => "lz4<",
        Codec::Bslz4 => "bs16-lz4<",
    }
}

const BSHUF_HEADER: usize = 12;

/// Global header, `header_detail: basic`: the header part and the config part.
pub fn header_message(series: u64, config: &SeriesConfig) -> [Vec<u8>; 2] {
    let head = Head {
        htype: "dheader-1.0".into(),
        series: Some(series),
        header_detail: Some("basic".into()),
        ..Default::default()
    };
    [json(&head), json(config)]
}

/// What the JSON parts of one image message say about its blob.
pub struct ImageMeta {
    pub series: u64,
    pub frame: u64,
    pub width: usize,
    pub height: usize,
    pub codec: Codec,
    /// [`blob_prefix`] plus the codec payload.
    pub blob_len: usize,
    /// Exposure start, ns since the series start.
    pub start_ns: i64,
    pub real_ns: i64,
}

/// The three JSON parts of an image message; the blob goes between the
/// second and the third.
pub fn image_parts(m: &ImageMeta) -> [Vec<u8>; 3] {
    let head = Head {
        htype: "dimage-1.0".into(),
        series: Some(m.series),
        frame: Some(m.frame),
        // The detector sends the blob's md5; nothing here checks it.
        hash: Some(String::new()),
        ..Default::default()
    };
    let desc = ImageDesc {
        htype: "dimage_d-1.0".into(),
        shape: [m.width as u64, m.height as u64],
        dtype: "uint16".into(),
        encoding: encoding(m.codec).into(),
        size: m.blob_len as u64,
    };
    let times = ImageTimes {
        htype: "dconfig-1.0".into(),
        start_time: m.start_ns,
        stop_time: m.start_ns + m.real_ns,
        real_time: m.real_ns,
    };
    [json(&head), json(&desc), json(&times)]
}

/// Bytes that precede the codec payload in the blob: the bitshuffle header
/// for `bs16-lz4<`, nothing otherwise.
pub fn blob_prefix(codec: Codec, w: usize, h: usize) -> Vec<u8> {
    match codec {
        Codec::Bslz4 => {
            let mut p = Vec::with_capacity(BSHUF_HEADER);
            p.extend_from_slice(&((w * h * 2) as u64).to_be_bytes());
            p.extend_from_slice(&((bshuf_block_elems(2) * 2) as u32).to_be_bytes());
            p
        }
        _ => Vec::new(),
    }
}

/// The codec payload of `blob`, `None` unless it starts with
/// [`blob_prefix`]: the workers' bslz4 decode takes the default block only.
pub fn blob_payload(codec: Codec, blob: &Bytes, w: usize, h: usize) -> Option<Bytes> {
    let p = blob_prefix(codec, w, h);
    blob.starts_with(&p).then(|| blob.slice(p.len()..))
}

pub fn end_message(series: u64) -> Vec<u8> {
    json(&Head {
        htype: "dseries_end-1.0".into(),
        series: Some(series),
        ..Default::default()
    })
}

/// What one multipart message turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Header,
    Image,
    End,
    Rejected,
}

enum Msg {
    Header,
    Image(Work),
    End,
}

/// Receiver side: validates each message against the series header and the
/// frame geometry and hands images to [`SourceStats::accept`] with the codec
/// payload as a zero-copy slice of the blob part.
pub struct EigerFrames {
    geom: Geometry,
    series: Option<u64>,
    t0: Option<i64>,
    pub headers: u64,
    pub ends: u64,
}

impl EigerFrames {
    pub fn new(geom: Geometry) -> Self {
        Self {
            geom,
            series: None,
            t0: None,
            headers: 0,
            ends: 0,
        }
    }

    pub fn on_message(&mut self, parts: &[Bytes], st: &mut SourceStats, tx: &WorkTx) -> Kind {
        let t_rx = wall_ns();
        match self.classify(parts, t_rx) {
            Ok(Msg::Image(w)) => {
                st.accept(w, tx);
                Kind::Image
            }
            Ok(Msg::Header) => Kind::Header,
            Ok(Msg::End) => Kind::End,
            Err(e) => {
                st.reject(e);
                Kind::Rejected
            }
        }
    }

    fn classify(&mut self, parts: &[Bytes], t_rx: i64) -> Result<Msg, String> {
        let head: Head = parse(&parts[0], "header part")?;
        match head.htype.split('-').next().unwrap_or("") {
            "dheader" => {
                let detail = head.header_detail.as_deref().unwrap_or("");
                // ADEiger: basic adds one part, all adds seven.
                let want = match detail {
                    "basic" => 2,
                    "all" => 8,
                    "none" => 1,
                    d => return Err(format!("dheader: unknown header_detail {d:?}")),
                };
                if parts.len() != want {
                    return Err(format!("dheader {detail}: {} parts", parts.len()));
                }
                self.t0 = None;
                if want > 1 {
                    let c: SeriesConfig = parse(&parts[1], "detector config")?;
                    self.t0 = c.ptyrs_series_start_ns;
                }
                self.series = Some(head.series.ok_or("dheader without series")?);
                self.headers += 1;
                Ok(Msg::Header)
            }
            "dseries_end" => {
                if head.series.is_none() || head.series != self.series {
                    return Err(format!(
                        "dseries_end {:?} in {:?}",
                        head.series, self.series
                    ));
                }
                self.ends += 1;
                Ok(Msg::End)
            }
            "dimage" => self.image(&head, parts, t_rx).map(Msg::Image),
            t => Err(format!("unknown htype {t:?}")),
        }
    }

    fn image(&self, head: &Head, parts: &[Bytes], t_rx: i64) -> Result<Work, String> {
        let [_, desc, blob, times] = parts else {
            return Err(format!("dimage: {} parts, expected 4", parts.len()));
        };
        let Some(series) = self.series else {
            return Err("dimage before any dheader".into());
        };
        if head.series != Some(series) {
            return Err(format!(
                "dimage series {:?} in series {series}",
                head.series
            ));
        }
        let frame = head.frame.ok_or("dimage without frame")?;
        let uid = i32::try_from(frame).map_err(|_| format!("frame {frame} past i32"))?;
        let d: ImageDesc = parse(desc, "dimage_d")?;
        let g = self.geom;
        if d.shape != [g.width as u64, g.height as u64] || d.dtype != "uint16" {
            return Err(format!(
                "image {:?} {} is not a {}x{} uint16 frame",
                d.shape, d.dtype, g.width, g.height
            ));
        }
        if d.size != blob.len() as u64 {
            return Err(format!("size {} but blob has {} bytes", d.size, blob.len()));
        }
        let codec = match d.encoding.as_str() {
            "<" => Codec::None,
            "lz4<" => Codec::Lz4,
            "bs16-lz4<" => Codec::Bslz4,
            e => return Err(format!("unsupported encoding {e:?}")),
        };
        let payload = blob_payload(codec, blob, g.width, g.height)
            .ok_or("bs16-lz4< blob header does not match the frame")?;
        let t: ImageTimes = parse(times, "dconfig")?;
        Ok(Work {
            payload,
            codec,
            uid,
            t_rx,
            // Without the simulator's series clock, publish time is arrival.
            t_pub: self.t0.map_or(t_rx, |t0| t0 + t.start_time),
        })
    }
}

fn parse<'a, T: Deserialize<'a>>(b: &'a [u8], what: &str) -> Result<T, String> {
    serde_json::from_slice(b).map_err(|e| format!("{what}: {e}"))
}

/// A received ZMQ part kept alive as the owner of a [`Bytes`].
struct Part(zmq::Message);

impl AsRef<[u8]> for Part {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Default, Debug)]
pub struct PullReport {
    pub messages: u64,
    pub headers: u64,
    pub ends: u64,
    pub rejected: u64,
}

/// Progress the polling thread watches while the PULL loop runs.
#[derive(Default)]
pub struct Progress {
    pub unique: AtomicU64,
    pub last_rx: AtomicI64,
}

impl Progress {
    /// What the polling thread reads of `s`; `last_rx` stays 0 until a
    /// frame arrives.
    pub fn publish(&self, s: &SourceStats) {
        self.unique.store(s.unique, Ordering::Relaxed);
        self.last_rx
            .store(s.rx.map_or(0, |r| r.last), Ordering::Relaxed);
    }
}

/// PULL from `endpoint` until the series ends or `stop` is set. Every part
/// is moved into a [`Bytes`] without copying, so the workers decode
/// straight out of the ZMQ receive buffers.
pub fn run_pull(
    endpoint: &str,
    rcvhwm: i32,
    mut frames: EigerFrames,
    st: Arc<Mutex<SourceStats>>,
    tx: WorkTx,
    stop: &AtomicBool,
    progress: &Progress,
) -> zmq::Result<PullReport> {
    let ctx = zmq::Context::new();
    let sock = ctx.socket(zmq::PULL)?;
    sock.set_rcvhwm(rcvhwm)?;
    sock.set_rcvtimeo(20)?;
    sock.connect(endpoint)?;
    let mut rep = PullReport::default();
    let mut parts: Vec<Bytes> = Vec::with_capacity(8);
    while !stop.load(Ordering::Relaxed) {
        let first = match sock.recv_msg(0) {
            Ok(m) => m,
            Err(zmq::Error::EAGAIN) => continue,
            Err(e) => return Err(e),
        };
        parts.clear();
        parts.push(Bytes::from_owner(Part(first)));
        // The parts of one message arrive together, so these never wait.
        while sock.get_rcvmore()? {
            parts.push(Bytes::from_owner(Part(sock.recv_msg(0)?)));
        }
        rep.messages += 1;
        let mut s = st.lock().unwrap();
        let kind = frames.on_message(&parts, &mut s, &tx);
        progress.publish(&s);
        drop(s);
        match kind {
            Kind::Rejected => rep.rejected += 1,
            Kind::End => break,
            Kind::Header | Kind::Image => {}
        }
    }
    rep.headers = frames.headers;
    rep.ends = frames.ends;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{bslz4_compress_u16, bytemuck_u16, lz4_compress};
    use crate::handoff::{WorkRx, work_queue};

    fn queue() -> (WorkTx, WorkRx) {
        work_queue(std::num::NonZeroUsize::new(16).unwrap())
    }

    fn geom(w: usize, h: usize) -> Geometry {
        Geometry::centred(w, h, 16, w / 2, h / 2)
    }

    fn bytes(parts: Vec<Vec<u8>>) -> Vec<Bytes> {
        parts.into_iter().map(Bytes::from).collect()
    }

    fn image(codec: Codec, frame: u64, img: &[u16], w: usize, h: usize) -> Vec<Bytes> {
        let bare = match codec {
            Codec::None => bytemuck_u16(img).to_vec(),
            Codec::Lz4 => lz4_compress(bytemuck_u16(img)),
            Codec::Bslz4 => bslz4_compress_u16(img),
        };
        let mut blob = blob_prefix(codec, w, h);
        blob.extend_from_slice(&bare);
        let [a, b, c] = image_parts(&ImageMeta {
            series: 7,
            frame,
            width: w,
            height: h,
            codec,
            blob_len: blob.len(),
            start_ns: 1000,
            real_ns: 50,
        });
        bytes(vec![a, b, blob, c])
    }

    #[test]
    fn series_round_trips_every_encoding() {
        let (w, h) = (40, 30);
        let img: Vec<u16> = (0..w * h).map(|i| (i % 97) as u16).collect();
        let (tx, rx) = queue();
        let mut st = SourceStats::new(3);
        let mut f = EigerFrames::new(geom(w, h));
        let cfg = SeriesConfig {
            ptyrs_series_start_ns: Some(5_000),
            ..Default::default()
        };
        let msg =
            |p: Vec<Bytes>, f: &mut EigerFrames, st: &mut SourceStats| f.on_message(&p, st, &tx);
        assert_eq!(
            msg(bytes(header_message(7, &cfg).into()), &mut f, &mut st),
            Kind::Header
        );
        for (uid, c) in [Codec::None, Codec::Lz4, Codec::Bslz4]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                msg(image(c, uid as u64, &img, w, h), &mut f, &mut st),
                Kind::Image
            );
        }
        assert_eq!(msg(bytes(vec![end_message(7)]), &mut f, &mut st), Kind::End);
        let got: Vec<Work> = rx.try_iter().collect();
        assert_eq!(got.len(), 3);
        let mut out = vec![0u16; w * h];
        let mut bs = crate::codec::Bslz4Scratch::default();
        for (i, wk) in got.iter().enumerate() {
            assert_eq!((wk.uid, wk.t_pub), (i as i32, 6_000));
            match wk.codec {
                Codec::None => assert_eq!(&wk.payload[..], bytemuck_u16(&img)),
                Codec::Lz4 => {
                    crate::codec::lz4_decompress_into(
                        &wk.payload,
                        crate::codec::bytemuck_u16_mut(&mut out),
                    )
                    .unwrap();
                    assert_eq!(out, img);
                }
                Codec::Bslz4 => {
                    crate::codec::bslz4_decompress_into_u16(&wk.payload, &mut out, &mut bs)
                        .unwrap();
                    assert_eq!(out, img);
                }
            }
        }
        assert_eq!((st.unique, st.rejected), (3, 0));
    }

    #[test]
    fn rejects_what_adeiger_would_misread() {
        let (w, h) = (40, 30);
        let img = vec![1u16; w * h];
        let (tx, _rx) = queue();
        let mut st = SourceStats::new(10);
        let mut f = EigerFrames::new(geom(w, h));
        // Image before any header.
        assert_eq!(
            f.on_message(&image(Codec::Lz4, 0, &img, w, h), &mut st, &tx),
            Kind::Rejected
        );
        let hdr = bytes(header_message(7, &SeriesConfig::default()).into());
        assert_eq!(f.on_message(&hdr, &mut st, &tx), Kind::Header);
        // Wrong shape.
        assert_eq!(
            f.on_message(&image(Codec::Lz4, 1, &img, h, w), &mut st, &tx),
            Kind::Rejected
        );
        // bshuf header with another block size.
        let mut p = image(Codec::Bslz4, 2, &img, w, h);
        let mut blob = p[2].to_vec();
        blob[11] ^= 1;
        p[2] = Bytes::from(blob);
        assert_eq!(f.on_message(&p, &mut st, &tx), Kind::Rejected);
        // size field disagrees with the blob.
        let mut p = image(Codec::None, 3, &img, w, h);
        p[2] = p[2].slice(2..);
        assert_eq!(f.on_message(&p, &mut st, &tx), Kind::Rejected);
        // Without the simulator clock, publish time falls back to arrival.
        let (tx2, rx2) = queue();
        assert_eq!(
            f.on_message(&image(Codec::None, 4, &img, w, h), &mut st, &tx2),
            Kind::Image
        );
        let wk = rx2.try_recv().unwrap();
        assert_eq!(wk.t_pub, wk.t_rx);
        assert_eq!((st.rejected, st.unique), (4, 1));
    }
}
