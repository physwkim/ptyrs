//! `ptyrs::tile::TileKernels::psi_to_obj` against the SIMT `ptyrs::gpu::Kernels::psi_to_obj`
//! and an f64 reference: whole and split stack tables, a device-written entry (including
//! `nf == 0` and entries the kernels clamp), and object tiles no view of the stack touches.
//! No launch compiles: the set compiled every kernel when it was built.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use cutile::tile_kernel::jit_compile_count;
use ptyrs::gpu::{
    C32, Kernels, OBJ_TILE, StackCtl, StackTable, UpdateSums, ViewAddr, ViewDims, Views, view_tiles,
};
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

/// 24x32 frames on two 82x100 object modes, whose last tile row and column are partial,
/// and two probe modes (four views per frame). `fy`, `fx` cover zero and non-zero, frame 3
/// reaches the object's far corner, and object tiles row 3 x column 0, rows 4-5 x columns
/// 0-3 and rows 0-2 x columns 4-6 hold no view.
fn dims() -> (ViewDims, Vec<ViewAddr>) {
    let d = ViewDims {
        nframe: 5,
        nmode: 4,
        nprobe: 2,
        nobj: 2,
        frame: [24, 32],
        obj: [82, 100],
        quad_phase: 0.003,
    };
    let offsets = [
        (0, 0, 0.0, 0.0),
        (26, 28, 0.25, 0.0),
        (13, 7, 0.0, 0.7),
        (57, 67, 0.6, 0.9),
        (5, 20, 0.5, 0.5),
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

/// Adds the object sums of `views` (from view 0 of `psi`), scattering each probe pixel to
/// its bilinear corners of non-zero weight in f64.
fn obj_sums_f64(
    d: &ViewDims,
    views: &[ViewAddr],
    psi: &[C32],
    probe: &[C32],
    num: &mut [Z],
    den: &mut [f64],
) {
    let [ny, nx] = [d.frame[0] as usize, d.frame[1] as usize];
    let [nyo, nxo] = [d.obj[0] as usize, d.obj[1] as usize];
    let n = ny * nx;
    for (kl, v) in views.iter().enumerate() {
        for y in 0..ny {
            for x in 0..nx {
                let p = probe[v.probe as usize * n + y * nx + x];
                let p = (p.re as f64, p.im as f64);
                let (dy, dx) = (y as f64 - (ny / 2) as f64, x as f64 - (nx / 2) as f64);
                let a = d.quad_phase as f64 * (dy * dy + dx * dx);
                let pq = (p.0 * a.cos() - p.1 * a.sin(), p.0 * a.sin() + p.1 * a.cos());
                let s = psi[kl * n + ((y + ny / 2) % ny) * nx + (x + nx / 2) % nx];
                let u = (
                    pq.0 * s.re as f64 + pq.1 * s.im as f64,
                    pq.0 * s.im as f64 - pq.1 * s.re as f64,
                );
                let (fy, fx) = (v.fy as f64, v.fx as f64);
                let (y0, x0) = (v.oy as usize + y, v.ox as usize + x);
                let at = |yy: usize, xx: usize| (v.obj as usize * nyo + yy) * nxo + xx;
                for (yy, xx, w) in [
                    (y0, x0, (1.0 - fy) * (1.0 - fx)),
                    (y0, x0 + 1, (1.0 - fy) * fx),
                    (y0 + 1, x0, fy * (1.0 - fx)),
                    (y0 + 1, x0 + 1, fy * fx),
                ] {
                    if w == 0.0 {
                        continue;
                    }
                    let j = at(yy, xx);
                    num[j] = (num[j].0 + w * u.0, num[j].1 + w * u.1);
                    den[j] += w * (p.0 * p.0 + p.1 * p.1);
                }
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

/// Per object pixel of every mode, whether its tile is one that `views` touch.
fn lit(d: &ViewDims, views: &[ViewAddr]) -> Vec<bool> {
    let [nyo, nxo] = [d.obj[0] as usize, d.obj[1] as usize];
    let [nty, ntx] = d.tiles().map(|v| v as usize);
    let mut tiles = vec![false; nty * ntx];
    for v in views {
        for [ty, tx] in view_tiles(d, v) {
            tiles[ty as usize * ntx + tx as usize] = true;
        }
    }
    let t = OBJ_TILE as usize;
    (0..d.nobj as usize * nyo * nxo)
        .map(|i| {
            let p = i % (nyo * nxo);
            tiles[(p / nxo / t) * ntx + (p % nxo) / t]
        })
        .collect()
}

/// The sums as host `f32`s, real and imaginary parts of `num` interleaved.
fn host(stream: &CudaStream, s: &UpdateSums) -> (Vec<f32>, Vec<f32>) {
    let num = s.num.to_host_vec(stream).unwrap();
    let num = num.iter().flat_map(|c| [c.re, c.im]).collect();
    (num, s.den.to_host_vec(stream).unwrap())
}

/// Asserts every value of `got` outside the `lit` pixels has the bits of `start`, and
/// returns how many pixels that covered.
fn assert_unlit_kept(
    what: &str,
    got: &(Vec<f32>, Vec<f32>),
    start: &(Vec<f32>, Vec<f32>),
    lit: &[bool],
) -> usize {
    let mut dark = 0;
    for (i, &l) in lit.iter().enumerate() {
        if l {
            continue;
        }
        dark += 1;
        for (part, g, s) in [
            ("num.re", got.0[2 * i], start.0[2 * i]),
            ("num.im", got.0[2 * i + 1], start.0[2 * i + 1]),
            ("den", got.1[i], start.1[i]),
        ] {
            assert_eq!(
                g.to_bits(),
                s.to_bits(),
                "{what}: {part} of pixel {i}, in a tile without views, changed"
            );
        }
    }
    dark
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
    probe_host: Vec<C32>,
    probe: DeviceBuffer<C32>,
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
        let probe_host = noise(d.nprobe as usize * n(&d), 2, 1.0);
        let probe = up(&stream, &probe_host);
        let psi_host = noise(d.nframe as usize * Self::fsize(&d), 3, 1.0);
        let len = d.nobj as usize * (d.obj[0] * d.obj[1]) as usize;
        let num0 = noise(len, 10, 1.0);
        let den0 = noise(len, 11, 1.0).iter().map(|c| c.re.abs()).collect();
        Self {
            k,
            stream,
            ex,
            d,
            view_list,
            views,
            probe_host,
            probe,
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

    fn start_host(&self) -> (Vec<f32>, Vec<f32>) {
        let num = self.num0.iter().flat_map(|c| [c.re, c.im]).collect();
        (num, self.den0.clone())
    }

    /// The starting sums plus the f64 sums of frames `f`.
    fn want(&self, f: Range<u32>) -> (Vec<f64>, Vec<f64>) {
        let mut num: Vec<Z> = self
            .num0
            .iter()
            .map(|c| (c.re as f64, c.im as f64))
            .collect();
        let mut den: Vec<f64> = self.den0.iter().map(|&v| v as f64).collect();
        let m = self.d.nmode;
        let views = &self.view_list[(f.start * m) as usize..(f.end * m) as usize];
        let psi = &self.psi_host[f.start as usize * Self::fsize(&self.d)..];
        obj_sums_f64(&self.d, views, psi, &self.probe_host, &mut num, &mut den);
        (num.iter().flat_map(|&(a, b)| [a, b]).collect(), den)
    }

    /// Asserts Tile against SIMT and f64, and that it kept the tiles without views of
    /// frames `f`; returns how many pixels those were.
    fn check(
        &self,
        what: &str,
        f: Range<u32>,
        tile: &(Vec<f32>, Vec<f32>),
        simt: &(Vec<f32>, Vec<f32>),
    ) -> usize {
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
        let m = self.d.nmode;
        let lit = lit(
            &self.d,
            &self.view_list[(f.start * m) as usize..(f.end * m) as usize],
        );
        let start = self.start_host();
        assert_unlit_kept(&format!("{what} SIMT"), simt, &start, &lit);
        assert_unlit_kept(&format!("{what} Tile"), tile, &start, &lit)
    }
}

#[test]
fn psi_to_obj_matches_simt_and_ignores_stack_split() {
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
    fx.k.psi_to_obj(ex, views, whole.at(0), &psi, &fx.probe, &mut simt)
        .unwrap();
    let simt = host(stream, &simt);
    let tile_whole = || {
        let mut s = fx.start();
        tk_whole
            .psi_to_obj(ex, whole.at(0), &psi, &fx.probe, &mut s)
            .unwrap();
        host(stream, &s)
    };
    let tile = tile_whole();
    let dark = fx.check("psi_to_obj 0..5", 0..5, &tile, &simt);
    // Row 3 x column 0, rows 4-5 x columns 0-3, rows 0-2 x columns 4-6, both modes.
    assert_eq!(
        dark,
        2 * (16 * 16 + 18 * 64 + 48 * 36),
        "pixels in tiles without views"
    );

    let mut parts = fx.start();
    for (s, psi) in psi_parts.iter().enumerate() {
        tk_split
            .psi_to_obj(ex, split.at(s), psi, &fx.probe, &mut parts)
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
/// entry when they run and clamp it to `nf_max` frames inside the scan. An entry with no
/// frames after the clamp leaves the sums as they were, bit for bit.
#[test]
fn psi_to_obj_follows_the_device_entry() {
    let fx = Fixture::new();
    let (stream, ex, views) = (&fx.stream, &fx.ex, &fx.views);
    let fsize = Fixture::fsize(&fx.d);
    let dev = StackTable::device(stream, &fx.d, 3).unwrap();
    let tk = TileKernels::new(stream, views, &dev).unwrap();
    let jit = jit_compile_count();
    let nframe = fx.d.nframe;
    let entries = [
        (0, 0),
        (2, 0),
        (5, 0),
        (1, 3),
        (3, 2),
        (0, 4),
        (4, 3),
        (9, 2),
        (u32::MAX, u32::MAX),
        (0, u32::MAX),
    ];
    for (f0, nf) in entries {
        let c0 = f0.min(nframe);
        let f = c0..c0 + nf.min(dev.nf_max()).min(nframe - c0);
        let psi = up(
            stream,
            &stack_psi(&fx.psi_host, fsize, f.clone(), dev.nf_max()),
        );
        set_entry(&fx.k, stream, &dev, f0, nf);
        let mut simt = fx.start();
        fx.k.psi_to_obj(ex, views, dev.at(0), &psi, &fx.probe, &mut simt)
            .unwrap();
        let mut tile = fx.start();
        tk.psi_to_obj(ex, dev.at(0), &psi, &fx.probe, &mut tile)
            .unwrap();
        let what = format!("psi_to_obj entry {{ f0: {f0}, nf: {nf} }} = frames {f:?}");
        let (simt, tile) = (host(stream, &simt), host(stream, &tile));
        let dark = fx.check(&what, f.clone(), &tile, &simt);
        if f.is_empty() {
            assert_eq!(dark, fx.num0.len(), "{what}: lit pixels without frames");
        }
    }
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

#[test]
fn psi_to_obj_checks_buffers() {
    let fx = Fixture::new();
    let (stream, ex, views) = (&fx.stream, &fx.ex, &fx.views);
    let fsize = Fixture::fsize(&fx.d);
    let table = StackTable::new(stream, &fx.d, &[0..2, 2..5]).unwrap();
    let tk = TileKernels::new(stream, views, &table).unwrap();
    let jit = jit_compile_count();
    let psi = up(stream, &fx.psi_host[..3 * fsize]);
    let len = fx.num0.len();
    let shape = |r: Result<(), ptyrs::gpu::Error>, what: &str| {
        assert!(
            matches!(r, Err(ptyrs::gpu::Error::Shape(_))),
            "{what}: {r:?}"
        );
    };
    let mut sums = UpdateSums::zeroed(stream, len).unwrap();
    // Two frames of psi do not hold `nf_max` = 3.
    let short_psi = up(stream, &fx.psi_host[..2 * fsize]);
    let r = tk.psi_to_obj(ex, table.at(1), &short_psi, &fx.probe, &mut sums);
    shape(r, "psi");
    // One probe mode does not hold two.
    let short_probe = up(stream, &fx.probe_host[..n(&fx.d)]);
    let r = tk.psi_to_obj(ex, table.at(1), &psi, &short_probe, &mut sums);
    shape(r, "probe");
    // One object mode of sums does not hold two.
    let mut short = UpdateSums::zeroed(stream, len / 2).unwrap();
    let r = tk.psi_to_obj(ex, table.at(1), &psi, &fx.probe, &mut short);
    shape(r, "sums");
    // Psi 8 bytes into its allocation, aligned for `C32` but not to 16 bytes.
    let wide = up(stream, &fx.psi_host[..3 * fsize + 1]);
    let bytes = wide
        .storage()
        .slice_bytes(8, 3 * fsize * size_of::<C32>())
        .unwrap();
    // SAFETY: the bytes hold initialized `C32` values.
    let shifted = unsafe { DeviceBuffer::<C32>::from_storage(bytes, 3 * fsize) }.unwrap();
    let r = tk.psi_to_obj(ex, table.at(1), &shifted, &fx.probe, &mut sums);
    shape(r, "misaligned psi");
    tk.psi_to_obj(ex, table.at(1), &psi, &fx.probe, &mut sums)
        .unwrap();
    // A stack of another `nf_max`, or of a table for another scan length, is refused; one
    // of another table of the same geometry launches.
    let other = StackTable::new(stream, &fx.d, &[0..1, 1..5]).unwrap();
    let r = tk.psi_to_obj(ex, other.at(0), &psi, &fx.probe, &mut sums);
    shape(r, "stack of nf_max 4");
    let short_scan = ViewDims { nframe: 4, ..fx.d };
    let other = StackTable::new(stream, &short_scan, std::slice::from_ref(&(0..3))).unwrap();
    let r = tk.psi_to_obj(ex, other.at(0), &psi, &fx.probe, &mut sums);
    shape(r, "stack of a 4-frame scan");
    let same = StackTable::new(stream, &fx.d, &[0..3, 3..5]).unwrap();
    tk.psi_to_obj(ex, same.at(1), &psi, &fx.probe, &mut sums)
        .unwrap();
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}
