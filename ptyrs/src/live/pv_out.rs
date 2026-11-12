//! Live results as NTNDArray PVs (DESIGN.md §3 B6): `<prefix>OBJ:AMP`,
//! `<prefix>OBJ:PHASE`, `<prefix>PROBE:AMP`, `<prefix>PROBE:PHASE` and
//! `<prefix>LLK`, which areaDetector viewers, Phoebus and pvaPy clients
//! subscribe to unchanged. The `n` planes of an image (object slices, probe
//! modes) of `ny x nx` pixels are stacked along y, their memory order, as
//! dimensions `[nx, n * ny]` with an Int32 attribute `Planes = n`; a binned
//! preview carries its bin in the dimensions' `binning`. `LLK` is the
//! history, one Float64 per epoch, oldest first.

use std::num::NonZeroUsize;

use crossbeam_channel::{Receiver, never, select};
use epics_pva_rs::error::PvaResult;
use epics_pva_rs::nt::nd_array::{
    NdArrayBuffer, NdAttribute, NdCodec, NdDimension, NdTimeStamp, NtNdArray, nt_nd_array_desc,
    nt_nd_array_value,
};
use epics_pva_rs::pvdata::{PvField, ScalarValue};
use epics_pva_rs::server_native::{SharedPV, SharedSource};
use serde::Deserialize;

use crate::live::follow::View;
use crate::live::stats::wall_ns;

/// `[recon.pv]`: the live results as these PVs, on a PVA server set up from
/// the `EPICS_PVAS_*` environment.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct PvCfg {
    pub prefix: String,
    /// The object preview's bin.
    #[serde(default = "PvCfg::bin")]
    pub bin: NonZeroUsize,
    /// Least time between object posts; each costs the live run one more
    /// download of its object.
    #[serde(default = "PvCfg::period_ms")]
    pub period_ms: u64,
    /// Least time between LLK posts; each makes the live run wait for its
    /// last epoch.
    #[serde(default = "PvCfg::period_ms")]
    pub llk_period_ms: u64,
}

impl PvCfg {
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            bin: Self::bin(),
            period_ms: Self::period_ms(),
            llk_period_ms: Self::period_ms(),
        }
    }

    fn bin() -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }

    fn period_ms() -> u64 {
        500
    }
}

/// Modulus and argument of complex planes, binned.
#[derive(Debug, Clone, PartialEq)]
pub struct AmpPhase {
    pub planes: usize,
    /// `[ny, nx]` of one binned plane.
    pub shape: [usize; 2],
    pub bin: usize,
    pub amp: Vec<f32>,
    pub phase: Vec<f32>,
}

impl AmpPhase {
    /// `planes` planes of `shape` interleaved complex pixels. Each
    /// `bin x bin` block (fewer pixels at the far edges) is averaged as
    /// complex numbers before the modulus and argument are taken, so a block
    /// straddling the phase wrap keeps its phase.
    pub fn new(px: &[[f32; 2]], planes: usize, shape: [usize; 2], bin: usize) -> Self {
        let [ny, nx] = shape;
        assert_eq!(px.len(), planes * ny * nx, "{planes} planes of {ny}x{nx}");
        let [by, bx] = [ny.div_ceil(bin), nx.div_ceil(bin)];
        let mut sum = vec![[0f64; 2]; planes * by * bx];
        for (r, row) in px.chunks_exact(nx).enumerate() {
            let out = &mut sum[((r / ny) * by + r % ny / bin) * bx..][..bx];
            for (x, p) in row.iter().enumerate() {
                out[x / bin][0] += f64::from(p[0]);
                out[x / bin][1] += f64::from(p[1]);
            }
        }
        let count = |i: usize, n: usize| (n - i * bin).min(bin) as f64;
        let (amp, phase) = sum
            .iter()
            .enumerate()
            .map(|(i, &[re, im])| {
                let k = count(i / bx % by, ny) * count(i % bx, nx);
                ((re / k).hypot(im / k) as f32, im.atan2(re) as f32)
            })
            .unzip();
        Self {
            planes,
            shape: [by, bx],
            bin,
            amp,
            phase,
        }
    }
}

pub struct PvOut {
    obj: [SharedPV; 2],
    probe: [SharedPV; 2],
    llk: SharedPV,
}

impl PvOut {
    /// Opens the five PVs, empty, and adds them to `source`.
    pub fn new(source: &SharedSource, prefix: &str) -> PvaResult<Self> {
        let empty = array_value(NdArrayBuffer::Float(Vec::new().into()), vec![], vec![], -1);
        let open = |name: &str| -> PvaResult<SharedPV> {
            let pv = SharedPV::build_mailbox();
            pv.open(nt_nd_array_desc(), empty.clone())?;
            source.add(format!("{prefix}{name}"), pv.clone());
            Ok(pv)
        };
        Ok(Self {
            obj: [open("OBJ:AMP")?, open("OBJ:PHASE")?],
            probe: [open("PROBE:AMP")?, open("PROBE:PHASE")?],
            llk: open("LLK")?,
        })
    }

    /// Returns the fewest subscribers either PV reached, as do the other
    /// posts.
    pub fn post_object(&self, img: AmpPhase, uid: i32) -> PvaResult<usize> {
        post_image(&self.obj, img, uid)
    }

    pub fn post_probe(&self, img: AmpPhase, uid: i32) -> PvaResult<usize> {
        post_image(&self.probe, img, uid)
    }

    pub fn post_llk(&self, llk: &[f64], uid: i32) -> PvaResult<usize> {
        let dims = vec![dim(llk.len(), 1)];
        let v = array_value(
            NdArrayBuffer::Double(llk.to_vec().into()),
            dims,
            vec![],
            uid,
        );
        self.llk.try_post_checked(v)
    }
}

/// What [`publish`] posted.
#[derive(Debug, Default, PartialEq)]
pub struct Posts {
    pub objects: u64,
    /// The LLK history as last posted.
    pub llk: Vec<f64>,
}

/// Posts `probe` once, then each object copy `views` delivers, binned by
/// `bin` with the epochs it follows as its uid, and the LLK history each
/// time `llk` extends it, with its length as its uid, until both senders
/// drop.
pub fn publish(
    out: &PvOut,
    probe: AmpPhase,
    views: Receiver<View>,
    llk: Receiver<Vec<f64>>,
    bin: NonZeroUsize,
) -> PvaResult<Posts> {
    out.post_probe(probe, 0)?;
    let mut posts = Posts::default();
    let uid = |n: u64| i32::try_from(n).unwrap_or(i32::MAX);
    let (mut views, mut llk) = (Some(views), Some(llk));
    let (no_view, no_llk) = (never(), never());
    while views.is_some() || llk.is_some() {
        select! {
            recv(views.as_ref().unwrap_or(&no_view)) -> v => match v {
                Ok(v) => {
                    let img = AmpPhase::new(&v.pixels, 1, v.shape, bin.get());
                    out.post_object(img, uid(v.epochs))?;
                    posts.objects += 1;
                }
                Err(_) => views = None,
            },
            recv(llk.as_ref().unwrap_or(&no_llk)) -> l => match l {
                Ok(l) => {
                    posts.llk.extend(l);
                    out.post_llk(&posts.llk, uid(posts.llk.len() as u64))?;
                }
                Err(_) => llk = None,
            },
        }
    }
    Ok(posts)
}

fn post_image(pvs: &[SharedPV; 2], img: AmpPhase, uid: i32) -> PvaResult<usize> {
    let [ny, nx] = img.shape;
    let dims = vec![dim(nx, img.bin), dim(img.planes * ny, img.bin)];
    let attrs = vec![NdAttribute::scalar(
        "Planes",
        ScalarValue::Int(img.planes as i32),
    )];
    let mut fewest = usize::MAX;
    for (pv, data) in pvs.iter().zip([img.amp, img.phase]) {
        let v = array_value(
            NdArrayBuffer::Float(data.into()),
            dims.clone(),
            attrs.clone(),
            uid,
        );
        fewest = fewest.min(pv.try_post_checked(v)?);
    }
    Ok(fewest)
}

fn dim(size: usize, binning: usize) -> NdDimension {
    NdDimension {
        size: size as i32,
        offset: 0,
        full_size: size as i32,
        binning: binning as i32,
        reverse: false,
    }
}

fn array_value(
    value: NdArrayBuffer,
    dimension: Vec<NdDimension>,
    attribute: Vec<NdAttribute>,
    uid: i32,
) -> PvField {
    let bytes = match &value {
        NdArrayBuffer::Float(a) => a.len() * 4,
        NdArrayBuffer::Double(a) => a.len() * 8,
        _ => unreachable!("results are Float32 or Float64"),
    } as i64;
    let wall = wall_ns();
    let ts = NdTimeStamp {
        seconds_past_epoch: wall.div_euclid(1_000_000_000),
        nanoseconds: wall.rem_euclid(1_000_000_000) as i32,
        user_tag: 0,
    };
    nt_nd_array_value(&NtNdArray {
        value,
        codec: NdCodec::default(),
        compressed_size: bytes,
        uncompressed_size: bytes,
        unique_id: uid,
        data_time_stamp: ts.clone(),
        alarm: Default::default(),
        time_stamp: ts,
        dimension,
        attribute,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polar(a: f64, p: f64) -> [f32; 2] {
        [(a * p.cos()) as f32, (a * p.sin()) as f32]
    }

    #[test]
    fn bin_one_is_per_pixel_modulus_and_argument() {
        let px: Vec<[f32; 2]> = (0..2 * 3 * 5)
            .map(|i| polar(0.1 + i as f64 * 0.03, i as f64 * 0.7 - 10.0))
            .collect();
        let img = AmpPhase::new(&px, 2, [3, 5], 1);
        assert_eq!((img.planes, img.shape, img.bin), (2, [3, 5], 1));
        for (i, p) in px.iter().enumerate() {
            let (re, im) = (f64::from(p[0]), f64::from(p[1]));
            assert_eq!(img.amp[i], re.hypot(im) as f32);
            assert_eq!(img.phase[i], im.atan2(re) as f32);
        }
    }

    /// Two planes of 3 x 5 in 2 x 2 blocks: 2 x 3 blocks per plane, the last
    /// row and column of blocks partial; values from the plane, row and
    /// column so that blocks never mix across planes.
    #[test]
    fn blocks_average_as_complex_numbers_within_one_plane() {
        let (ny, nx, bin) = (3usize, 5usize, 2usize);
        let v = |z: usize, y: usize, x: usize| {
            polar(1.0 + z as f64, 0.1 * y as f64 + 0.3 * x as f64 + z as f64)
        };
        let px: Vec<[f32; 2]> = (0..2)
            .flat_map(|z| (0..ny).flat_map(move |y| (0..nx).map(move |x| v(z, y, x))))
            .collect();
        let img = AmpPhase::new(&px, 2, [ny, nx], bin);
        assert_eq!(img.shape, [2, 3]);
        for z in 0..2 {
            for (by, bx) in (0..2).flat_map(|by| (0..3).map(move |bx| (by, bx))) {
                let (mut re, mut im, mut k) = (0f64, 0f64, 0f64);
                for y in by * bin..((by + 1) * bin).min(ny) {
                    for x in bx * bin..((bx + 1) * bin).min(nx) {
                        let p = v(z, y, x);
                        re += f64::from(p[0]);
                        im += f64::from(p[1]);
                        k += 1.0;
                    }
                }
                let i = (z * 2 + by) * 3 + bx;
                assert_eq!(img.amp[i], (re / k).hypot(im / k) as f32, "z {z} block {i}");
                assert_eq!(img.phase[i], im.atan2(re) as f32, "z {z} block {i}");
            }
        }
    }

    /// Phases just either side of the wrap average to the wrap, where a
    /// mean of the phases themselves would give 0.
    #[test]
    fn a_block_across_the_phase_wrap_keeps_its_phase() {
        use std::f64::consts::PI;
        let e = 0.05;
        let px = [polar(1.0, PI - e), polar(1.0, -PI + e)];
        let img = AmpPhase::new(&px, 1, [1, 2], 2);
        assert!(
            (f64::from(img.phase[0].abs()) - PI).abs() < 1e-6,
            "phase {}",
            img.phase[0]
        );
        assert!(
            (img.amp[0] - e.cos() as f32).abs() < 1e-6,
            "amp {}",
            img.amp[0]
        );
    }

    /// `uniqueId`, the dimension sizes and binnings, and the first pixel.
    fn header(v: &PvField) -> (i32, Vec<(i32, i32)>, Option<ScalarValue>) {
        let PvField::Structure(s) = v else {
            panic!("{v:?}")
        };
        let int = |f: Option<&PvField>| match f {
            Some(PvField::Scalar(ScalarValue::Int(i))) => *i,
            f => panic!("{f:?}"),
        };
        let Some(PvField::StructureArray(ds)) = s.get_field("dimension") else {
            panic!("no dimension")
        };
        let dims = ds
            .iter()
            .map(|d| {
                let d = d.as_ref().unwrap();
                (int(d.get_field("size")), int(d.get_field("binning")))
            })
            .collect();
        let first = match s.get_field("value") {
            Some(PvField::Union { value, .. }) => match &**value {
                PvField::ScalarArrayTyped(a) => a.get(0),
                PvField::ScalarArray(a) => a.first().cloned(),
                _ => None,
            },
            _ => None,
        };
        (int(s.get_field("uniqueId")), dims, first)
    }

    /// A client subscribed to `OBJ:AMP` and `LLK` before the first post
    /// sees each object copy [`publish`] posts, binned, with its epochs as
    /// the uid, and the LLK history each batch extends, with its length as
    /// the uid; `PROBE:PHASE` holds the probe's two modes stacked along y.
    #[test]
    fn a_subscriber_sees_each_object_copy_the_llk_and_the_probe() {
        use std::sync::Arc;
        use std::time::Duration;

        use epics_pva_rs::client_native::PvaClient;
        use epics_pva_rs::server_native::PvaServer;

        use crate::live::follow::View;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let source = SharedSource::new();
        let out = PvOut::new(&source, "T:").unwrap();
        let _in = rt.enter();
        let server = PvaServer::isolated(Arc::new(source)).unwrap();
        let client = PvaClient::builder().server_addr(server.tcp_addr()).build();
        let monitor = |pv: &str| {
            let (mtx, mrx) = crossbeam_channel::unbounded();
            let sub = rt
                .block_on(client.pvmonitor_handle(
                    pv,
                    move |_, v| {
                        let _ = mtx.send(header(v));
                    },
                    |_| {},
                ))
                .unwrap();
            let wait = move || mrx.recv_timeout(Duration::from_secs(10)).unwrap();
            assert_eq!(wait().0, -1, "{pv}: the empty array before any post");
            (sub, wait)
        };
        let (_obj, wait) = monitor("T:OBJ:AMP");
        let (_llk, wait_llk) = monitor("T:LLK");

        let probe_px: Vec<[f32; 2]> = (0..2 * 4 * 4).map(|i| polar(1.0, i as f64 * 0.1)).collect();
        let probe = AmpPhase::new(&probe_px, 2, [4, 4], 1);
        let (vtx, vrx) = crossbeam_channel::bounded(0);
        let (ltx, lrx) = crossbeam_channel::unbounded();
        let bin = NonZeroUsize::new(2).unwrap();
        let publisher = std::thread::spawn(move || publish(&out, probe, vrx, lrx, bin));
        for (batch, len) in [(vec![2.5, 1.5], 2), (vec![0.75], 3)] {
            ltx.send(batch).unwrap();
            let (uid, dims, first) = wait_llk();
            assert_eq!((uid, dims), (len, vec![(len, 1)]));
            assert_eq!(first, Some(ScalarValue::Double(2.5)));
        }
        drop(ltx);
        for epochs in [3u32, 7] {
            let pixels: Vec<[f32; 2]> = (0..5 * 6)
                .map(|i| polar(f64::from(epochs) + i as f64, 0.3))
                .collect();
            let want = AmpPhase::new(&pixels, 1, [5, 6], 2);
            vtx.send(View {
                epochs: epochs.into(),
                shape: [5, 6],
                pixels,
            })
            .unwrap();
            let (uid, dims, first) = wait();
            assert_eq!(uid, epochs as i32);
            assert_eq!(dims, [(3, 2), (3, 2)]);
            assert_eq!(first, Some(ScalarValue::Float(want.amp[0])));
        }
        drop(vtx);
        let posts = publisher.join().unwrap().unwrap();
        assert_eq!(
            posts,
            Posts {
                objects: 2,
                llk: vec![2.5, 1.5, 0.75]
            }
        );
        let (uid, dims, _) = header(&rt.block_on(client.pvget("T:PROBE:PHASE")).unwrap());
        assert_eq!((uid, dims), (0, vec![(4, 1), (8, 1)]));
    }
}
