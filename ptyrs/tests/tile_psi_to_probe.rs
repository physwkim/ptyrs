//! `ptyrs::tile::TileKernels::psi_to_probe` against the SIMT
//! `ptyrs::gpu::Kernels::psi_to_probe` and an f64 reference: whole and split stack tables, a
//! device-written entry (including `nf == 0` and entries the kernels clamp). No launch
//! compiles: the set compiled every kernel when it was built.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use cutile::tile_kernel::jit_compile_count;
use ptyrs::gpu::{C32, Kernels, StackCtl, StackTable, UpdateSums, ViewAddr, ViewDims, Views};
use ptyrs::tile::TileKernels;

type Z = (f64, f64);

/// Deterministic values in `[-scale, scale)`.
fn noise(len: usize, seed: u64, scale: f32) -> Vec<C32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
    };
    (0..len)
        .map(|_| C32 {
            re: next(),
            im: next(),
        })
        .collect()
}

/// Pixels per frame.
fn n(d: &ViewDims) -> usize {
    d.frame[0] as usize * d.frame[1] as usize
}

fn up<T: DeviceCopy>(stream: &CudaStream, host: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(stream, host).unwrap()
}

/// 20x40 frames (partial 8x32 tiles in both directions) on two 50x60 object modes and
/// three probe modes, six views per frame. `fy`, `fx` cover zero and non-zero, frames 1
/// and 3 reach the object's last row and frames 3 and 4 its last column.
fn dims() -> (ViewDims, Vec<ViewAddr>) {
    let d = ViewDims {
        nframe: 5,
        nmode: 6,
        nprobe: 3,
        nobj: 2,
        frame: [20, 40],
        obj: [50, 60],
        quad_phase: 0.003,
    };
    let offsets = [
        (0, 0, 0.0, 0.0),
        (29, 20, 0.25, 0.0),
        (13, 7, 0.0, 0.7),
        (29, 19, 0.6, 0.9),
        (5, 19, 0.5, 0.5),
    ];
    let views = offsets
        .iter()
        .flat_map(|&(oy, ox, fy, fx)| {
            (0..d.nmode).map(move |m| ViewAddr {
                oy,
                ox,
                fy,
                fx,
                probe: m % d.nprobe,
                obj: m / d.nprobe,
            })
        })
        .collect();
    (d, views)
}

/// Adds the probe sums of `views` (from view 0 of `psi`) in f64: each probe pixel takes its
/// bilinear object sample over the corners of non-zero weight.
fn probe_sums_f64(
    d: &ViewDims,
    views: &[ViewAddr],
    psi: &[C32],
    obj: &[C32],
    num: &mut [Z],
    den: &mut [f64],
) {
    let [ny, nx] = [d.frame[0] as usize, d.frame[1] as usize];
    let [nyo, nxo] = [d.obj[0] as usize, d.obj[1] as usize];
    let n = ny * nx;
    for (kl, v) in views.iter().enumerate() {
        for y in 0..ny {
            for x in 0..nx {
                let (fy, fx) = (v.fy as f64, v.fx as f64);
                let (y0, x0) = (v.oy as usize + y, v.ox as usize + x);
                let at = |yy: usize, xx: usize| (v.obj as usize * nyo + yy) * nxo + xx;
                let o = [
                    (y0, x0, (1.0 - fy) * (1.0 - fx)),
                    (y0, x0 + 1, (1.0 - fy) * fx),
                    (y0 + 1, x0, fy * (1.0 - fx)),
                    (y0 + 1, x0 + 1, fy * fx),
                ]
                .iter()
                .filter(|c| c.2 != 0.0)
                .fold((0.0, 0.0), |s, &(yy, xx, w)| {
                    let o = obj[at(yy, xx)];
                    (s.0 + w * o.re as f64, s.1 + w * o.im as f64)
                });
                let (dy, dx) = (y as f64 - (ny / 2) as f64, x as f64 - (nx / 2) as f64);
                let a = d.quad_phase as f64 * (dy * dy + dx * dx);
                let oq = (o.0 * a.cos() - o.1 * a.sin(), o.0 * a.sin() + o.1 * a.cos());
                let s = psi[kl * n + ((y + ny / 2) % ny) * nx + (x + nx / 2) % nx];
                let j = v.probe as usize * n + y * nx + x;
                num[j].0 += oq.0 * s.re as f64 + oq.1 * s.im as f64;
                num[j].1 += oq.0 * s.im as f64 - oq.1 * s.re as f64;
                den[j] += o.0 * o.0 + o.1 * o.1;
            }
        }
    }
}

/// Psi of the frames `f` of `all` (`fsize` values each), padded with NaN to `nf_max`
/// frames: a kernel reading past its stack spreads NaN.
fn stack_psi(all: &[C32], fsize: usize, f: Range<u32>, nf_max: u32) -> Vec<C32> {
    let mut psi = all[f.start as usize * fsize..f.end as usize * fsize].to_vec();
    let nan = C32 {
        re: f32::NAN,
        im: f32::NAN,
    };
    psi.resize(nf_max as usize * fsize, nan);
    psi
}

/// Writes `{ f0, nf }` to the one entry of `table`.
fn set_entry(k: &Kernels, stream: &CudaStream, table: &StackTable, f0: u32, nf: u32) {
    stream.synchronize().unwrap();
    let ctl = [StackCtl { f0, nf }];
    k.context().bind_to_thread().unwrap();
    // SAFETY: `table` holds one entry; no work on it is in flight.
    unsafe { cuda_core::memory::memcpy_htod_sync(table.ptr(), ctl.as_ptr(), 8) }.unwrap();
}

/// The sums as host `f32`s, real and imaginary parts of `num` interleaved.
fn host(stream: &CudaStream, s: &UpdateSums) -> (Vec<f32>, Vec<f32>) {
    let num = s.num.to_host_vec(stream).unwrap();
    let num = num.iter().flat_map(|c| [c.re, c.im]).collect();
    (num, s.den.to_host_vec(stream).unwrap())
}

/// `sqrt(sum (got - want)^2 / sum want^2)`.
fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (e, r) = got.iter().zip(want).fold((0.0, 0.0), |(e, r), (&g, &w)| {
        (e + (g as f64 - w).powi(2), r + w * w)
    });
    (e / r).sqrt()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Fixture {
    k: Kernels,
    stream: Arc<CudaStream>,
    ex: ExecutionContext,
    d: ViewDims,
    view_list: Vec<ViewAddr>,
    views: Views,
    obj_host: Vec<C32>,
    obj: DeviceBuffer<C32>,
    /// Psi of every frame of the scan.
    psi_host: Vec<C32>,
    /// Non-zero starting sums: the kernels add to what is there.
    num0: Vec<C32>,
    den0: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
        let stream: Arc<CudaStream> = ctx.default_stream();
        let ex = ExecutionContext::new(stream.clone());
        let (d, view_list) = dims();
        let views = Views::new(&stream, d, &view_list).unwrap();
        let obj_host = noise(d.nobj as usize * (d.obj[0] * d.obj[1]) as usize, 2, 1.0);
        let obj = up(&stream, &obj_host);
        let psi_host = noise(d.nframe as usize * Self::fsize(&d), 3, 1.0);
        let len = d.nprobe as usize * n(&d);
        let num0 = noise(len, 20, 1.0);
        let den0 = noise(len, 21, 1.0).iter().map(|c| c.re.abs()).collect();
        Self {
            k,
            stream,
            ex,
            d,
            view_list,
            views,
            obj_host,
            obj,
            psi_host,
            num0,
            den0,
        }
    }

    /// Psi values per frame.
    fn fsize(d: &ViewDims) -> usize {
        d.nmode as usize * n(d)
    }

    fn start(&self) -> UpdateSums {
        let s = &self.stream;
        UpdateSums::new(up(s, &self.num0), up(s, &self.den0), s).unwrap()
    }

    fn frame_views(&self, f: Range<u32>) -> &[ViewAddr] {
        let m = self.d.nmode;
        &self.view_list[(f.start * m) as usize..(f.end * m) as usize]
    }

    /// The starting sums plus the f64 sums of frames `f`.
    fn want(&self, f: Range<u32>) -> (Vec<f64>, Vec<f64>) {
        let mut num: Vec<Z> = self
            .num0
            .iter()
            .map(|c| (c.re as f64, c.im as f64))
            .collect();
        let mut den: Vec<f64> = self.den0.iter().map(|&v| v as f64).collect();
        let psi = &self.psi_host[f.start as usize * Self::fsize(&self.d)..];
        let views = self.frame_views(f);
        probe_sums_f64(&self.d, views, psi, &self.obj_host, &mut num, &mut den);
        (num.iter().flat_map(|&(a, b)| [a, b]).collect(), den)
    }

    /// Asserts Tile against SIMT and f64, and that both kept the bits of every probe mode
    /// no view of frames `f` uses; returns those modes.
    fn check(
        &self,
        what: &str,
        f: Range<u32>,
        tile: &(Vec<f32>, Vec<f32>),
        simt: &(Vec<f32>, Vec<f32>),
    ) -> Vec<u32> {
        let want = self.want(f.clone());
        let as_f64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        for (part, t, s, w) in [
            ("num", &tile.0, &simt.0, &want.0),
            ("den", &tile.1, &simt.1, &want.1),
        ] {
            let (vs_simt, vs_f64, simt_f64) = (rel_l2(t, &as_f64(s)), rel_l2(t, w), rel_l2(s, w));
            let same = t.iter().zip(s).filter(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "{what} {part}: rel L2 Tile-SIMT {vs_simt:.2e}, Tile-f64 {vs_f64:.2e}, \
                 SIMT-f64 {simt_f64:.2e}; {} of {} values bit-equal to SIMT",
                same.count(),
                t.len()
            );
            assert!(
                vs_simt < 1e-6,
                "{what} {part}: Tile vs SIMT rel L2 {vs_simt:.2e}"
            );
            assert!(
                vs_f64 < 1e-5,
                "{what} {part}: Tile vs f64 rel L2 {vs_f64:.2e}"
            );
        }
        let views = self.frame_views(f);
        let idle: Vec<u32> = (0..self.d.nprobe)
            .filter(|&p| views.iter().all(|v| v.probe != p))
            .collect();
        let n = n(&self.d);
        let num0: Vec<f32> = self.num0.iter().flat_map(|c| [c.re, c.im]).collect();
        for &p in &idle {
            let px = p as usize * n..(p as usize + 1) * n;
            let re = 2 * px.start..2 * px.end;
            for (who, got) in [("SIMT", simt), ("Tile", tile)] {
                assert_eq!(
                    bits(&got.0[re.clone()]),
                    bits(&num0[re.clone()]),
                    "{what} {who}: num of probe mode {p}, without views, changed"
                );
                assert_eq!(
                    bits(&got.1[px.clone()]),
                    bits(&self.den0[px.clone()]),
                    "{what} {who}: den of probe mode {p}, without views, changed"
                );
            }
        }
        idle
    }
}

#[test]
fn psi_to_probe_matches_simt_and_ignores_stack_split() {
    let fx = Fixture::new();
    let (stream, ex, views) = (&fx.stream, &fx.ex, &fx.views);
    let fsize = Fixture::fsize(&fx.d);
    let whole = StackTable::new(stream, &fx.d, std::slice::from_ref(&(0..fx.d.nframe))).unwrap();
    let psi = up(stream, &fx.psi_host);
    let stacks = [0..2, 2..5];
    let split = StackTable::new(stream, &fx.d, &stacks).unwrap();
    let tk_whole = TileKernels::new(stream, views, &whole).unwrap();
    let tk_split = TileKernels::new(stream, views, &split).unwrap();
    let jit = jit_compile_count();
    let psi_parts = stacks.map(|f| up(stream, &stack_psi(&fx.psi_host, fsize, f, split.nf_max())));

    let mut simt = fx.start();
    fx.k.psi_to_probe(ex, views, whole.at(0), &psi, &fx.obj, &mut simt)
        .unwrap();
    let simt = host(stream, &simt);
    let tile_whole = || {
        let mut s = fx.start();
        tk_whole
            .psi_to_probe(ex, whole.at(0), &psi, &fx.obj, &mut s)
            .unwrap();
        host(stream, &s)
    };
    let tile = tile_whole();
    fx.check("psi_to_probe 0..5", 0..5, &tile, &simt);

    let mut parts = fx.start();
    for (s, psi) in psi_parts.iter().enumerate() {
        tk_split
            .psi_to_probe(ex, split.at(s), psi, &fx.obj, &mut parts)
            .unwrap();
    }
    let parts = host(stream, &parts);
    assert_eq!(
        bits(&parts.0),
        bits(&tile.0),
        "num depends on the stack split"
    );
    assert_eq!(
        bits(&parts.1),
        bits(&tile.1),
        "den depends on the stack split"
    );
    let again = tile_whole();
    assert_eq!(bits(&again.0), bits(&tile.0), "num differs run to run");
    assert_eq!(bits(&again.1), bits(&tile.1), "den differs run to run");
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

/// One launch per entry value, written to the device between them: the kernels read the
/// entry when they run and clamp it to `nf_max` frames inside the scan. Every mode of an
/// entry with no frames after the clamp keeps its sums bit for bit.
#[test]
fn psi_to_probe_follows_the_device_entry() {
    let fx = Fixture::new();
    let (stream, ex, views) = (&fx.stream, &fx.ex, &fx.views);
    let fsize = Fixture::fsize(&fx.d);
    let dev = StackTable::device(stream, &fx.d, 3).unwrap();
    let tk = TileKernels::new(stream, views, &dev).unwrap();
    let jit = jit_compile_count();
    let nframe = fx.d.nframe;
    // With the modes each leaves without views.
    let entries: [((u32, u32), &[u32]); 11] = [
        ((0, 0), &[0, 1, 2]),
        ((2, 0), &[0, 1, 2]),
        ((5, 0), &[0, 1, 2]),
        ((0, 3), &[]),
        ((1, 3), &[]),
        ((3, 2), &[]),
        ((0, 4), &[]),
        ((4, 3), &[]),
        ((9, 2), &[0, 1, 2]),
        ((u32::MAX, u32::MAX), &[0, 1, 2]),
        ((0, u32::MAX), &[]),
    ];
    for ((f0, nf), idle) in entries {
        let c0 = f0.min(nframe);
        let f = c0..c0 + nf.min(dev.nf_max()).min(nframe - c0);
        let psi = up(
            stream,
            &stack_psi(&fx.psi_host, fsize, f.clone(), dev.nf_max()),
        );
        set_entry(&fx.k, stream, &dev, f0, nf);
        let mut simt = fx.start();
        fx.k.psi_to_probe(ex, views, dev.at(0), &psi, &fx.obj, &mut simt)
            .unwrap();
        let mut tile = fx.start();
        tk.psi_to_probe(ex, dev.at(0), &psi, &fx.obj, &mut tile)
            .unwrap();
        let what = format!("psi_to_probe entry {{ f0: {f0}, nf: {nf} }} = frames {f:?}");
        let (simt, tile) = (host(stream, &simt), host(stream, &tile));
        assert_eq!(fx.check(&what, f, &tile, &simt), idle, "{what}: idle modes");
    }
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

#[test]
fn psi_to_probe_checks_buffers() {
    let fx = Fixture::new();
    let (stream, ex, views) = (&fx.stream, &fx.ex, &fx.views);
    let fsize = Fixture::fsize(&fx.d);
    let table = StackTable::new(stream, &fx.d, &[0..2, 2..5]).unwrap();
    let tk = TileKernels::new(stream, views, &table).unwrap();
    let jit = jit_compile_count();
    let psi = up(stream, &fx.psi_host[..3 * fsize]);
    let shape = |r: Result<(), ptyrs::gpu::Error>, what: &str| {
        assert!(
            matches!(r, Err(ptyrs::gpu::Error::Shape(_))),
            "{what}: {r:?}"
        );
    };
    let mut sums = UpdateSums::zeroed(stream, fx.num0.len()).unwrap();
    // Two frames of psi do not hold `nf_max` = 3.
    let short_psi = up(stream, &fx.psi_host[..2 * fsize]);
    let r = tk.psi_to_probe(ex, table.at(1), &short_psi, &fx.obj, &mut sums);
    shape(r, "psi");
    // One object mode does not hold two.
    let short_obj = up(stream, &fx.obj_host[..fx.obj_host.len() / 2]);
    let r = tk.psi_to_probe(ex, table.at(1), &psi, &short_obj, &mut sums);
    shape(r, "obj");
    // Two probe modes of sums do not hold three.
    let mut short = UpdateSums::zeroed(stream, 2 * n(&fx.d)).unwrap();
    let r = tk.psi_to_probe(ex, table.at(1), &psi, &fx.obj, &mut short);
    shape(r, "sums");
    // Psi 8 bytes into its allocation, aligned for `C32` but not to 16 bytes.
    let wide = up(stream, &fx.psi_host[..3 * fsize + 1]);
    let bytes = wide
        .storage()
        .slice_bytes(8, 3 * fsize * size_of::<C32>())
        .unwrap();
    // SAFETY: the bytes hold initialized `C32` values.
    let shifted = unsafe { DeviceBuffer::<C32>::from_storage(bytes, 3 * fsize) }.unwrap();
    let r = tk.psi_to_probe(ex, table.at(1), &shifted, &fx.obj, &mut sums);
    shape(r, "misaligned psi");
    tk.psi_to_probe(ex, table.at(1), &psi, &fx.obj, &mut sums)
        .unwrap();
    // A stack of another `nf_max`, or of a table for another scan length, is refused; one
    // of another table of the same geometry launches.
    let other = StackTable::new(stream, &fx.d, &[0..1, 1..5]).unwrap();
    let r = tk.psi_to_probe(ex, other.at(0), &psi, &fx.obj, &mut sums);
    shape(r, "stack of nf_max 4");
    let short_scan = ViewDims { nframe: 4, ..fx.d };
    let other = StackTable::new(stream, &short_scan, std::slice::from_ref(&(0..3))).unwrap();
    let r = tk.psi_to_probe(ex, other.at(0), &psi, &fx.obj, &mut sums);
    shape(r, "stack of a 4-frame scan");
    let same = StackTable::new(stream, &fx.d, &[0..3, 3..5]).unwrap();
    tk.psi_to_probe(ex, same.at(1), &psi, &fx.obj, &mut sums)
        .unwrap();
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}
