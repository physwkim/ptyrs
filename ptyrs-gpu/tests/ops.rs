//! The M1 pixel kernels and the M2 fused passes against f64 CPU references of the contract
//! in the `ptyrs_gpu` crate docs (PyNX `cpu_operator.py` semantics, orthonormal FFT).

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use ptyrs_fft::{FftError, Plan2d};
use ptyrs_gpu::{
    C32, Error, Exit, FUSED_SIDES, FusedBufs, FusedInputs, FusedSums, Kernels, Lines, LlkPartial,
    LlkScratch, LlkSums, MASKED, OBJ_TILE, StackCtl, StackRef, StackTable, UpdateSums, ViewAddr,
    ViewDims, Views, record, view_tiles,
};

type Z = (f64, f64);

/// A stream, an eager execution context on it (its leases end when the test drops it) and
/// the kernels.
fn setup() -> (Arc<CudaStream>, ExecutionContext, Kernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream = ctx.default_stream();
    (stream.clone(), ExecutionContext::new(stream), kernels)
}

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

fn up<T: DeviceCopy>(stream: &CudaStream, host: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(stream, host).unwrap()
}

/// A table of the one stack `frames`.
fn one_stack(stream: &CudaStream, d: &ViewDims, frames: Range<u32>) -> StackTable {
    StackTable::new(stream, d, std::slice::from_ref(&frames)).unwrap()
}

/// `host` followed by zeros up to `len` values.
fn up_padded(stream: &CudaStream, host: &[C32], len: usize) -> DeviceBuffer<C32> {
    let mut v = host.to_vec();
    v.resize(len, C32::default());
    up(stream, &v)
}

fn z(c: C32) -> Z {
    (c.re as f64, c.im as f64)
}

fn mul(a: Z, b: Z) -> Z {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `conj(a) * b`.
fn cmul_conj(a: Z, b: Z) -> Z {
    (a.0 * b.0 + a.1 * b.1, a.0 * b.1 - a.1 * b.0)
}

fn norm2(a: Z) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

/// `sqrt(sum |got - want|^2 / sum |want|^2)`.
fn rel_l2(got: &[C32], want: &[Z]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (e, r) = got.iter().zip(want).fold((0.0, 0.0), |(e, r), (g, w)| {
        (
            e + norm2((g.re as f64 - w.0, g.im as f64 - w.1)),
            r + norm2(*w),
        )
    });
    (e / r).sqrt()
}

fn rel_l2_re(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (e, r) = got.iter().zip(want).fold((0.0, 0.0), |(e, r), (&g, &w)| {
        (e + (g as f64 - w).powi(2), r + w * w)
    });
    (e / r).sqrt()
}

/// A scan: frame `f` sits at `offsets[f]` for all its views; mode `m` is probe
/// `m % nprobe` times object `m / nprobe`.
struct Scan {
    d: ViewDims,
    views: Vec<ViewAddr>,
    obj: Vec<C32>,
    probe: Vec<C32>,
}

impl Scan {
    fn new(d: ViewDims, offsets: &[(u32, u32, f32, f32)], probe_scale: f32) -> Self {
        assert_eq!(offsets.len(), d.nframe as usize);
        assert_eq!(d.nmode, d.nprobe * d.nobj);
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
        let nobj = d.nobj as usize * d.obj[0] as usize * d.obj[1] as usize;
        let nprobe = d.nprobe as usize * d.frame[0] as usize * d.frame[1] as usize;
        Self {
            d,
            views,
            obj: noise(nobj, 1, 1.0),
            probe: noise(nprobe, 2, probe_scale),
        }
    }

    fn n(&self) -> usize {
        self.d.frame[0] as usize * self.d.frame[1] as usize
    }

    fn nk(&self, frames: &Range<u32>) -> usize {
        (frames.end - frames.start) as usize * self.d.nmode as usize
    }

    fn views_of(&self, frames: &Range<u32>) -> &[ViewAddr] {
        let m = self.d.nmode as usize;
        &self.views[frames.start as usize * m..frames.end as usize * m]
    }

    /// `(y, x, y * nx + x)` over a frame.
    fn pixels(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        let [ny, nx] = [self.d.frame[0] as usize, self.d.frame[1] as usize];
        (0..ny).flat_map(move |y| (0..nx).map(move |x| (y, x, y * nx + x)))
    }

    /// Origin-at-`[0, 0]` offset of natural pixel `(y, x)`; also the FFT-order frequency of
    /// detector pixel `(y, x)`.
    fn shifted(&self, y: usize, x: usize) -> usize {
        let [ny, nx] = [self.d.frame[0] as usize, self.d.frame[1] as usize];
        ((y + ny / 2) % ny) * nx + (x + nx / 2) % nx
    }

    fn quad(&self, y: usize, x: usize) -> Z {
        let dy = y as f64 - (self.d.frame[0] / 2) as f64;
        let dx = x as f64 - (self.d.frame[1] / 2) as f64;
        let a = self.d.quad_phase as f64 * (dy * dy + dx * dx);
        (a.cos(), a.sin())
    }

    /// Object indices and bilinear weights of the corners view `v`'s sample at probe pixel
    /// `(y, x)` reads (non-zero weight).
    fn corners(&self, v: &ViewAddr, y: usize, x: usize) -> impl Iterator<Item = (usize, f64)> {
        let [nyo, nxo] = [self.d.obj[0] as usize, self.d.obj[1] as usize];
        let (fy, fx) = (v.fy as f64, v.fx as f64);
        let (y0, x0) = (v.oy as usize + y, v.ox as usize + x);
        let at = move |yy: usize, xx: usize| {
            assert!(
                yy < nyo && xx < nxo,
                "{v:?} reads ({yy}, {xx}) outside the object"
            );
            (v.obj as usize * nyo + yy) * nxo + xx
        };
        [
            ((y0, x0), (1.0 - fy) * (1.0 - fx)),
            ((y0, x0 + 1), (1.0 - fy) * fx),
            ((y0 + 1, x0), fy * (1.0 - fx)),
            ((y0 + 1, x0 + 1), fy * fx),
        ]
        .into_iter()
        .filter(|&(_, w)| w != 0.0)
        .map(move |((yy, xx), w)| (at(yy, xx), w))
    }

    fn sample(&self, obj: &[C32], v: &ViewAddr, y: usize, x: usize) -> Z {
        self.corners(v, y, x).fold((0.0, 0.0), |s, (i, w)| {
            (s.0 + w * obj[i].re as f64, s.1 + w * obj[i].im as f64)
        })
    }

    /// Exit waves of `frames` for object `obj` and the scan's probe.
    fn psi(&self, frames: Range<u32>, obj: &[C32]) -> Vec<Z> {
        let n = self.n();
        let mut out = vec![(0.0, 0.0); self.nk(&frames) * n];
        for (kl, v) in self.views_of(&frames).iter().enumerate() {
            for (y, x, i) in self.pixels() {
                let o = self.sample(obj, v, y, x);
                let p = z(self.probe[v.probe as usize * n + i]);
                out[kl * n + self.shifted(y, x)] = mul(mul(o, p), self.quad(y, x));
            }
        }
        out
    }

    /// Adds the stack's object sums, scattered from each probe pixel to its four corners.
    fn obj_sums(&self, frames: Range<u32>, psi: &[C32], num: &mut [Z], den: &mut [f64]) {
        let n = self.n();
        for (kl, v) in self.views_of(&frames).iter().enumerate() {
            for (y, x, i) in self.pixels() {
                let p = z(self.probe[v.probe as usize * n + i]);
                let u = cmul_conj(mul(p, self.quad(y, x)), z(psi[kl * n + self.shifted(y, x)]));
                for (j, w) in self.corners(v, y, x) {
                    num[j] = (num[j].0 + w * u.0, num[j].1 + w * u.1);
                    den[j] += w * norm2(p);
                }
            }
        }
    }

    /// Adds the stack's probe sums.
    fn probe_sums(&self, frames: Range<u32>, psi: &[C32], num: &mut [Z], den: &mut [f64]) {
        let n = self.n();
        for (kl, v) in self.views_of(&frames).iter().enumerate() {
            for (y, x, i) in self.pixels() {
                let o = self.sample(&self.obj, v, y, x);
                let u = cmul_conj(mul(o, self.quad(y, x)), z(psi[kl * n + self.shifted(y, x)]));
                let j = v.probe as usize * n + i;
                num[j] = (num[j].0 + u.0, num[j].1 + u.1);
                den[j] += norm2(o);
            }
        }
    }
}

/// 24x32 frames on two 50x60 object modes and two probe modes (four views per frame).
/// Frames 1 and 3 reach the object's last row and column, each once through its last
/// probe row or column (`f == 0`) and once through the `+ 1` neighbour (`f > 0`); the
/// subpixel offsets cover `fy` and `fx` zero and non-zero.
fn scan() -> Scan {
    scan_in([50, 60])
}

/// [`scan`] on an object of `obj` `[nyo, nxo]`.
fn scan_in(obj: [u32; 2]) -> Scan {
    let d = ViewDims {
        nframe: 5,
        nmode: 4,
        nprobe: 2,
        nobj: 2,
        frame: [24, 32],
        obj,
        quad_phase: 0.003,
    };
    let offsets = [
        (0, 0, 0.0, 0.0),
        (25, 28, 0.25, 0.0),
        (13, 7, 0.0, 0.7),
        (26, 27, 0.0, 0.9),
        (5, 20, 0.5, 0.5),
    ];
    Scan::new(d, &offsets, 1.0)
}

/// PyNX `LLKAll` terms of one unmasked pixel: Poisson, Gaussian, Euclidean, calc.
fn llk_terms(obs: f64, calc: f64) -> [f64; 4] {
    let poisson = if obs == 0.0 {
        calc
    } else {
        calc - obs + obs * (obs / calc).ln()
    };
    [
        poisson,
        (obs - calc).powi(2) / (obs + 1.0),
        4.0 * (obs.sqrt() - calc.sqrt()).powi(2),
        calc,
    ]
}

fn assert_llk(what: &str, got: &[LlkPartial], want: &[[f64; 4]]) {
    assert_eq!(got.len(), want.len());
    for (f, (g, w)) in got.iter().zip(want).enumerate() {
        let g = [g.poisson, g.gaussian, g.euclidean, g.icalc];
        for (j, (&g, &w)) in g.iter().zip(w).enumerate() {
            let rel = (g as f64 - w).abs() / w.abs().max(1.0);
            assert!(
                rel < 1e-5,
                "{what} frame {f} term {j}: {g} vs {w} (rel {rel:.2e})"
            );
        }
    }
    let sums = LlkSums::from_partials(got);
    let total = |j: usize| want.iter().map(|w| w[j]).sum::<f64>();
    let pairs = [
        (sums.poisson, total(0)),
        (sums.gaussian, total(1)),
        (sums.euclidean, total(2)),
        (sums.icalc, total(3)),
    ];
    for (g, w) in pairs {
        assert!(
            (g - w).abs() / w.abs().max(1.0) < 1e-5,
            "{what} total {g} vs {w}"
        );
    }
}

#[test]
fn obj_probe_to_psi_matches_f64() {
    let (stream, ex, k) = setup();
    let s = scan();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let (obj, probe) = (up(&stream, &s.obj), up(&stream, &s.probe));
    let stacks = [0..5, 2..5];
    let table = StackTable::new(&stream, &s.d, &stacks).unwrap();
    for (i, frames) in stacks.into_iter().enumerate() {
        let mut psi = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..5)) * s.n()).unwrap();
        k.obj_probe_to_psi(&ex, &views, table.at(i), &obj, &probe, &mut psi)
            .unwrap();
        let got = psi.to_host_vec(&stream).unwrap();
        let (stack, rest) = got.split_at(s.nk(&frames) * s.n());
        let err = rel_l2(stack, &s.psi(frames.clone(), &s.obj));
        println!("obj_probe_to_psi {frames:?}: rel L2 {err:.2e}");
        assert!(err < 2e-6, "frames {frames:?}: rel L2 {err:.2e}");
        assert!(
            rest.iter().all(|c| *c == C32::default()),
            "frames {frames:?}: planes past the stack written"
        );
    }
}

/// Uploads `num`, `den` as sums.
fn sums(stream: &CudaStream, start: &(Vec<C32>, Vec<f32>)) -> UpdateSums {
    UpdateSums::new(up(stream, &start.0), up(stream, &start.1), stream).unwrap()
}

/// `whole` matches the f64 sums, and `parts` (the same frames in two stacks) matches
/// `whole` bit for bit.
fn check_sums(
    what: &str,
    stream: &CudaStream,
    whole: &UpdateSums,
    parts: &UpdateSums,
    want: (&[Z], &[f64]),
) {
    let num = whole.num.to_host_vec(stream).unwrap();
    let den = whole.den.to_host_vec(stream).unwrap();
    let (en, ed) = (rel_l2(&num, want.0), rel_l2_re(&den, want.1));
    println!("{what} sums: num rel L2 {en:.2e}, den {ed:.2e}");
    assert!(en < 1e-5 && ed < 1e-5, "{what}: num {en:.2e}, den {ed:.2e}");
    let bits = |v: &[C32]| -> Vec<u32> {
        v.iter()
            .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
            .collect()
    };
    let fbits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(
        bits(&num),
        bits(&parts.num.to_host_vec(stream).unwrap()),
        "{what} num depends on the stack split"
    );
    assert_eq!(
        fbits(&den),
        fbits(&parts.den.to_host_vec(stream).unwrap()),
        "{what} den depends on the stack split"
    );
}

#[test]
fn update_sums_match_f64_and_ignore_stack_split() {
    let (stream, ex, k) = setup();
    let s = scan();
    let n = s.n();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let (obj, probe) = (up(&stream, &s.obj), up(&stream, &s.probe));
    let psi_host = noise(s.nk(&(0..5)) * n, 3, 1.0);
    let psi = up(&stream, &psi_host);
    let whole_stack = one_stack(&stream, &s.d, 0..5);
    let part_stacks = StackTable::new(&stream, &s.d, &[0..2, 2..5]).unwrap();
    let (split, part_len) = (s.nk(&(0..2)) * n, s.nk(&(2..5)) * n);
    let psi_parts = [
        up_padded(&stream, &psi_host[..split], part_len),
        up(&stream, &psi_host[split..]),
    ];
    // Non-zero starting sums: the kernels add to what is there.
    let start = |len: usize, seed: u64| {
        let den = noise(len, seed + 1, 1.0)
            .iter()
            .map(|c| c.re.abs())
            .collect();
        (noise(len, seed, 1.0), den)
    };
    let f64_start = |st: &(Vec<C32>, Vec<f32>)| -> (Vec<Z>, Vec<f64>) {
        (
            st.0.iter().map(|&c| z(c)).collect(),
            st.1.iter().map(|&v| v as f64).collect(),
        )
    };

    let st = start(s.obj.len(), 10);
    let (mut whole, mut parts) = (sums(&stream, &st), sums(&stream, &st));
    k.psi_to_obj(&ex, &views, whole_stack.at(0), &psi, &probe, &mut whole)
        .unwrap();
    for (i, part) in psi_parts.iter().enumerate() {
        k.psi_to_obj(&ex, &views, part_stacks.at(i), part, &probe, &mut parts)
            .unwrap();
    }
    let (mut num, mut den) = f64_start(&st);
    s.obj_sums(0..5, &psi_host, &mut num, &mut den);
    check_sums("object", &stream, &whole, &parts, (&num, &den));

    let st = start(s.probe.len(), 20);
    let (mut whole, mut parts) = (sums(&stream, &st), sums(&stream, &st));
    k.psi_to_probe(&ex, &views, whole_stack.at(0), &psi, &obj, &mut whole)
        .unwrap();
    for (i, part) in psi_parts.iter().enumerate() {
        k.psi_to_probe(&ex, &views, part_stacks.at(i), part, &obj, &mut parts)
            .unwrap();
    }
    let (mut num, mut den) = f64_start(&st);
    s.probe_sums(0..5, &psi_host, &mut num, &mut den);
    check_sums("probe", &stream, &whole, &parts, (&num, &den));
}

/// A tile's owner block looks for an earlier frame of the stack covering it with a block's
/// worth of threads at a time: frames `256..NF` share a tile no frame before 256 touches,
/// so only the second stride finds their earlier frames. More blocks would wrongly own that
/// tile than the GPU holds at once, so late ones would add to sums already written.
#[test]
fn psi_to_obj_owns_each_tile_once_past_a_block_of_frames() {
    const NF: u32 = 2000;
    let (stream, ex, k) = setup();
    let d = ViewDims {
        nframe: NF,
        nmode: 1,
        nprobe: 1,
        nobj: 1,
        frame: [2, 2],
        obj: [64, 64],
        quad_phase: 0.003,
    };
    let offsets: Vec<_> = (0..NF)
        .map(|f| {
            let at = if f < 256 { 0 } else { 32 };
            (at + f % 7, at + f / 7 % 7, 0.0, 0.0)
        })
        .collect();
    let s = Scan::new(d, &offsets, 1.0);
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let probe = up(&stream, &s.probe);
    let psi_host = noise(s.nk(&(0..NF)) * s.n(), 3, 1.0);
    let st = (
        noise(s.obj.len(), 10, 1.0),
        noise(s.obj.len(), 11, 1.0)
            .iter()
            .map(|c| c.re.abs())
            .collect::<Vec<_>>(),
    );
    let (mut whole, mut parts) = (sums(&stream, &st), sums(&stream, &st));
    k.psi_to_obj(
        &ex,
        &views,
        one_stack(&stream, &s.d, 0..NF).at(0),
        &up(&stream, &psi_host),
        &probe,
        &mut whole,
    )
    .unwrap();
    let split = StackTable::new(&stream, &s.d, &[0..NF / 2, NF / 2..NF]).unwrap();
    let half = s.nk(&(0..NF / 2)) * s.n();
    for (i, part) in psi_host.chunks_exact(half).enumerate() {
        k.psi_to_obj(
            &ex,
            &views,
            split.at(i),
            &up(&stream, part),
            &probe,
            &mut parts,
        )
        .unwrap();
    }
    let mut num: Vec<Z> = st.0.iter().map(|&c| z(c)).collect();
    let mut den: Vec<f64> = st.1.iter().map(|&v| v as f64).collect();
    s.obj_sums(0..NF, &psi_host, &mut num, &mut den);
    check_sums("object", &stream, &whole, &parts, (&num, &den));
}

#[test]
fn psi_to_obj_is_the_adjoint_of_obj_probe_to_psi() {
    let (stream, ex, k) = setup();
    let s = scan();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let all = one_stack(&stream, &s.d, 0..5);
    let probe = up(&stream, &s.probe);
    let o = noise(s.obj.len(), 7, 1.0);
    // <A o, A o> = <o, A^H A o>: no cancellation, so f32 rounding stays small.
    let mut a_o = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..5)) * s.n()).unwrap();
    k.obj_probe_to_psi(&ex, &views, all.at(0), &up(&stream, &o), &probe, &mut a_o)
        .unwrap();
    let mut ah = UpdateSums::zeroed(&stream, o.len()).unwrap();
    k.psi_to_obj(&ex, &views, all.at(0), &a_o, &probe, &mut ah)
        .unwrap();
    let dot = |a: &[C32], b: &[C32]| {
        a.iter().zip(b).fold((0.0, 0.0), |s, (&a, &b)| {
            let t = cmul_conj(z(a), z(b));
            (s.0 + t.0, s.1 + t.1)
        })
    };
    let a_o = a_o.to_host_vec(&stream).unwrap();
    let lhs = dot(&a_o, &a_o);
    let rhs = dot(&o, &ah.num.to_host_vec(&stream).unwrap());
    let rel = norm2((lhs.0 - rhs.0, lhs.1 - rhs.1)).sqrt() / norm2(lhs).sqrt();
    println!("adjoint: <Ao,Ao> {lhs:?}, <o,A^H Ao> {rhs:?}, rel {rel:.2e}");
    assert!(rel < 1e-5, "rel {rel:.2e}");
}

fn c32_bits(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// The tiled object update leaves the tiles without views bit-exact and reads none of their
/// sums; every other pixel gets the bits of the flat update.
#[test]
fn object_update_skips_tiles_without_views() {
    let (stream, ex, k) = setup();
    // Views reach row 50 and column 60: tile row 4 and tile columns 4, 5 have none.
    let s = scan_in([80, 96]);
    let len = s.obj.len();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let all = one_stack(&stream, &s.d, 0..5);
    let probe = up(&stream, &s.probe);
    let psi = up(&stream, &noise(s.nk(&(0..5)) * s.n(), 3, 1.0));
    let [nyo, nxo] = s.d.obj.map(|v| v as usize);
    let (t, ntx) = (OBJ_TILE as usize, nxo.div_ceil(OBJ_TILE as usize));
    let tile_len = views.tile_len().to_host_vec(&stream).unwrap();
    let lit = |i: usize| tile_len[(i / nxo % nyo) / t * ntx + i % nxo / t] != 0;
    let nlit = (0..len).filter(|&i| lit(i)).count();
    assert!(
        nlit > 0 && nlit < len,
        "{nlit} of {len} pixels on tiles with views"
    );

    let mut flat_obj = up(&stream, &s.obj);
    let mut flat = UpdateSums::zeroed(&stream, len).unwrap();
    k.psi_to_obj(&ex, &views, all.at(0), &psi, &probe, &mut flat)
        .unwrap();
    k.update_from_sums(&ex, &mut flat_obj, &mut flat, len as u32, 0.01)
        .unwrap();

    // Read, the poisoned sums would raise max(den) or put NaN into the object.
    let nan = C32 {
        re: f32::NAN,
        im: f32::NAN,
    };
    let num0 = (0..len)
        .map(|i| if lit(i) { C32::default() } else { nan })
        .collect();
    let den0: Vec<f32> = (0..len).map(|i| if lit(i) { 0.0 } else { 1e30 }).collect();
    let mut obj = up(&stream, &s.obj);
    let mut tiled = sums(&stream, &(num0, den0.clone()));
    k.psi_to_obj(&ex, &views, all.at(0), &psi, &probe, &mut tiled)
        .unwrap();
    k.update_obj_from_sums(&ex, &views, &mut obj, &mut tiled, 0.01)
        .unwrap();

    let got = c32_bits(&obj.to_host_vec(&stream).unwrap());
    let (flat, start) = (
        c32_bits(&flat_obj.to_host_vec(&stream).unwrap()),
        c32_bits(&s.obj),
    );
    let (num, den) = (
        c32_bits(&tiled.num.to_host_vec(&stream).unwrap()),
        tiled.den.to_host_vec(&stream).unwrap(),
    );
    for i in 0..len {
        let want = if lit(i) { flat[i] } else { start[i] };
        assert_eq!(got[i], want, "pixel {i}, tile with views: {}", lit(i));
        // The update consumes the sums of the tiles with views and touches no other.
        let (num_want, den_want) = if lit(i) {
            ((0, 0), 0)
        } else {
            (c32_bits(&[nan])[0], den0[i].to_bits())
        };
        assert_eq!(num[i], num_want, "num {i}, tile with views: {}", lit(i));
        assert_eq!(
            den[i].to_bits(),
            den_want,
            "den {i}, tile with views: {}",
            lit(i)
        );
    }
    let moved = (0..len).filter(|&i| got[i] != start[i]).count();
    assert!(moved > nlit / 2, "only {moved} of {nlit} pixels updated");
}

/// `obj_den_max` writes the max of `den` over the tiles with views; `normalize_obj` with that
/// word gives the bits of `update_obj_from_sums`, and with a larger word from elsewhere (the
/// max another device holds) the plain f32 formula at that max, on the tiles with views only.
#[test]
fn object_update_split_takes_the_max_it_is_given() {
    let (stream, ex, k) = setup();
    let s = scan_in([80, 96]);
    let len = s.obj.len();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let all = one_stack(&stream, &s.d, 0..5);
    let probe = up(&stream, &s.probe);
    let psi = up(&stream, &noise(s.nk(&(0..5)) * s.n(), 6, 1.0));
    let [nyo, nxo] = s.d.obj.map(|v| v as usize);
    let (t, ntx) = (OBJ_TILE as usize, nxo.div_ceil(OBJ_TILE as usize));
    let tile_len = views.tile_len().to_host_vec(&stream).unwrap();
    let lit = |i: usize| tile_len[(i / nxo % nyo) / t * ntx + i % nxo / t] != 0;
    // Unlit pixels carry a den above every lit one: read, it would become the max.
    let den0: Vec<f32> = (0..len).map(|i| if lit(i) { 0.0 } else { 1e30 }).collect();
    let start = (vec![C32::default(); len], den0);
    let accumulated = || {
        let mut sums = sums(&stream, &start);
        k.psi_to_obj(&ex, &views, all.at(0), &psi, &probe, &mut sums)
            .unwrap();
        sums
    };
    let host_sums = |sums: &UpdateSums| {
        (
            sums.num.to_host_vec(&stream).unwrap(),
            sums.den.to_host_vec(&stream).unwrap(),
        )
    };

    let mut joined = up(&stream, &s.obj);
    let mut sums = accumulated();
    let (num, den) = host_sums(&sums);
    k.update_obj_from_sums(&ex, &views, &mut joined, &mut sums, 0.01)
        .unwrap();
    let joined = c32_bits(&joined.to_host_vec(&stream).unwrap());
    let joined_sums = host_sums(&sums);

    let local = (0..len)
        .filter(|&i| lit(i))
        .fold(0.0f32, |m, i| m.max(den[i]));
    let mut sums = accumulated();
    k.obj_den_max(&ex, &views, &mut sums).unwrap();
    let word = sums.max_bits.to_host_vec(&stream).unwrap()[0];
    assert_eq!(word, local.to_bits(), "obj_den_max");
    let mut split = up(&stream, &s.obj);
    let max = up(&stream, &[word]);
    k.normalize_obj(&ex, &views, &mut split, &mut sums, &max, 0.01)
        .unwrap();
    assert!(
        c32_bits(&split.to_host_vec(&stream).unwrap()) == joined,
        "split object differs from update_obj_from_sums"
    );
    let split_sums = host_sums(&sums);
    assert!(
        c32_bits(&split_sums.0) == c32_bits(&joined_sums.0)
            && split_sums
                .1
                .iter()
                .map(|v| v.to_bits())
                .eq(joined_sums.1.iter().map(|v| v.to_bits())),
        "split sums differ from update_obj_from_sums"
    );

    let global = 4.0 * local;
    let mut sums = accumulated();
    let mut obj = up(&stream, &s.obj);
    let max = up(&stream, &[global.to_bits()]);
    k.normalize_obj(&ex, &views, &mut obj, &mut sums, &max, 0.01)
        .unwrap();
    let got = c32_bits(&obj.to_host_vec(&stream).unwrap());
    let reg = 0.01f32 * global;
    for i in 0..len {
        let want = if lit(i) {
            let d = (den[i] + reg).max(1e-12);
            let part = |a: f32, b: f32| (a + reg * b) / d;
            let (a, b) = (num[i], s.obj[i]);
            (part(a.re, b.re).to_bits(), part(a.im, b.im).to_bits())
        } else {
            c32_bits(&s.obj[i..=i])[0]
        };
        assert_eq!(got[i], want, "pixel {i}, tile with views: {}", lit(i));
    }
    let (num_after, den_after) = host_sums(&sums);
    for i in (0..len).filter(|&i| lit(i)) {
        assert_eq!(
            (c32_bits(&num_after[i..=i])[0], den_after[i].to_bits()),
            ((0, 0), 0),
            "sums at {i} not consumed"
        );
    }
    let short = DeviceBuffer::<u32>::zeroed(&stream, 0).unwrap();
    assert!(
        matches!(
            k.normalize_obj(&ex, &views, &mut obj, &mut sums, &short, 0.01),
            Err(Error::Shape(_))
        ),
        "max_bits without its word"
    );
}

/// [`Views::from_parts`] over slabs with padding past each run (the live layout) gives the
/// bits of the packed lists: a run ends at its length, not at the next tile's start.
#[test]
fn views_over_padded_slabs_match_packed_lists() {
    let (stream, ex, k) = setup();
    let s = scan_in([80, 96]);
    let packed = Views::new(&stream, s.d, &s.views).unwrap();
    let start = packed.tile_start().to_host_vec(&stream).unwrap();
    let len = packed.tile_len().to_host_vec(&stream).unwrap();
    let list = packed.tile_views().to_host_vec(&stream).unwrap();
    let cap = *len.iter().max().unwrap() as usize + 2;
    // Padding names view 0, which touches tile 0: read, it would add to the sums.
    let mut slab = vec![0u32; len.len() * cap];
    for (t, &l) in len.iter().enumerate() {
        let run = &list[start[t] as usize..][..l as usize];
        slab[t * cap..][..run.len()].copy_from_slice(run);
    }
    let slab_start: Vec<u32> = (0..=len.len()).map(|t| (t * cap) as u32).collect();
    // SAFETY: the runs are those of `Views::new` for these views, each within its slab.
    let slabs = unsafe {
        Views::from_parts(
            s.d,
            up(&stream, &s.views),
            up(&stream, &slab_start),
            up(&stream, &slab),
            up(&stream, &len),
        )
    }
    .unwrap();

    let table = StackTable::new(&stream, &s.d, &[0..2, 2..5]).unwrap();
    let probe = up(&stream, &s.probe);
    let psi = up(&stream, &noise(s.nk(&(0..5)) * s.n(), 4, 1.0));
    let mut out = Vec::new();
    for views in [&packed, &slabs] {
        let mut obj = up(&stream, &s.obj);
        let mut sums = UpdateSums::zeroed(&stream, s.obj.len()).unwrap();
        for i in 0..table.len() {
            k.psi_to_obj(&ex, views, table.at(i), &psi, &probe, &mut sums)
                .unwrap();
        }
        k.update_obj_from_sums(&ex, views, &mut obj, &mut sums, 0.01)
            .unwrap();
        out.push(c32_bits(&obj.to_host_vec(&stream).unwrap()));
    }
    assert!(out[0] == out[1], "slab views differ from packed views");
    assert!(out[0] != c32_bits(&s.obj), "the update moved nothing");

    let short = |n: usize| DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
    // SAFETY: rejected before any launch.
    let r = unsafe {
        Views::from_parts(
            s.d,
            up(&stream, &s.views),
            short(len.len()),
            short(1),
            short(len.len()),
        )
    };
    assert!(
        matches!(r, Err(Error::Shape(_))),
        "tile_start without its end"
    );
}

/// Scan counts: every 7th pixel masked, every 5th of the rest zero, others 1..=60.
fn counts(len: usize) -> Vec<u16> {
    (0..len)
        .map(|i| match i {
            _ if i % 7 == 3 => MASKED,
            _ if i % 5 == 1 => 0,
            _ => 1 + ((i * 37) % 60) as u16,
        })
        .collect()
}

#[test]
fn intensity_amplitude_and_llk_match_f64() {
    let (stream, ex, k) = setup();
    let s = scan();
    let (d, n, nm) = (s.d, s.n(), s.d.nmode as usize);
    let frames = 1..4;
    let nf = 3;
    // Far-field stack scaled so the calculated intensity is near the counts.
    let psi_host = noise(s.nk(&frames) * n, 5, 50.0);
    let counts_host = counts(d.nframe as usize * n);

    let mut icalc_ref = vec![0.0; nf * n];
    let mut proj_ref: Vec<Z> = psi_host.iter().map(|&c| z(c)).collect();
    let mut llk_ref = vec![[0.0; 4]; nf];
    for fl in 0..nf {
        for (y, x, i) in s.pixels() {
            let at = s.shifted(y, x);
            let plane = |m: usize| (fl * nm + m) * n + at;
            let calc = (0..nm).map(|m| norm2(z(psi_host[plane(m)]))).sum::<f64>() / n as f64;
            icalc_ref[fl * n + i] = calc;
            let obs = counts_host[(frames.start as usize + fl) * n + i];
            let scale = if obs == MASKED {
                1.0 / n as f64
            } else {
                for (l, t) in llk_ref[fl].iter_mut().zip(llk_terms(obs as f64, calc)) {
                    *l += t;
                }
                (obs as f64 / calc.max(1e-12)).sqrt() / n as f64
            };
            for m in 0..nm {
                let v = &mut proj_ref[plane(m)];
                *v = (v.0 * scale, v.1 * scale);
            }
        }
    }

    let counts = up(&stream, &counts_host);
    let table = one_stack(&stream, &d, frames);
    let stack = table.at(0);
    let mut psi = up(&stream, &psi_host);
    let mut icalc = DeviceBuffer::<f32>::zeroed(&stream, nf * n).unwrap();
    k.mode_intensity(&ex, &d, stack, &psi, &mut icalc).unwrap();
    let mut llk = DeviceBuffer::<LlkPartial>::zeroed(&stream, nf).unwrap();
    let mut scratch = LlkScratch::new(&stream, &d, table.nf_max()).unwrap();
    k.llk(&ex, stack, &counts, &psi, &mut scratch, &mut llk)
        .unwrap();
    k.apply_amplitude(&ex, &d, stack, &counts, &mut psi)
        .unwrap();

    let err = rel_l2_re(&icalc.to_host_vec(&stream).unwrap(), &icalc_ref);
    println!("mode_intensity: rel L2 {err:.2e}");
    assert!(err < 2e-6, "mode_intensity rel L2 {err:.2e}");
    assert_llk("llk", &llk.to_host_vec(&stream).unwrap(), &llk_ref);
    let err = rel_l2(&psi.to_host_vec(&stream).unwrap(), &proj_ref);
    println!("apply_amplitude: rel L2 {err:.2e}");
    assert!(err < 2e-6, "apply_amplitude rel L2 {err:.2e}");
}

/// `llk` over frames of 48 x 98 pixels, three chunks of which the last is short, on a
/// stack starting at frame 1 of a table whose `nf_max` exceeds the stack: each frame
/// against f64, the record past the stack untouched, and a scratch made for fewer frames
/// than the table refused.
#[test]
fn llk_sums_frames_over_several_chunks() {
    let (stream, ex, k) = setup();
    let d = ViewDims {
        nframe: 4,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [48, 98],
        obj: [64, 128],
        quad_phase: 0.0,
    };
    let s = Scan::new(d, &[(0, 0, 0.0, 0.0); 4], 1.0);
    let (n, nm) = (s.n(), d.nmode as usize);
    let table = StackTable::new(&stream, &d, &[0..3, 1..3]).unwrap();
    let (frames, nf_max) = (1..3, table.nf_max() as usize);
    let psi_host = noise(nf_max * nm * n, 7, 50.0);
    let counts_host = counts(d.nframe as usize * n);
    let mut want = vec![[0.0; 4]; 2];
    for (fl, w) in want.iter_mut().enumerate() {
        for (y, x, i) in s.pixels() {
            let at = s.shifted(y, x);
            let calc = (0..nm)
                .map(|m| norm2(z(psi_host[(fl * nm + m) * n + at])))
                .sum::<f64>()
                / n as f64;
            let obs = counts_host[(frames.start as usize + fl) * n + i];
            if obs != MASKED {
                for (l, t) in w.iter_mut().zip(llk_terms(obs as f64, calc)) {
                    *l += t;
                }
            }
        }
    }
    let counts = up(&stream, &counts_host);
    let psi = up(&stream, &psi_host);
    let sentinel = LlkPartial {
        poisson: -1.0,
        gaussian: -2.0,
        euclidean: -3.0,
        icalc: -4.0,
    };
    let mut llk = up(&stream, &[sentinel; 3]);
    let mut scratch = LlkScratch::new(&stream, &d, table.nf_max()).unwrap();
    k.llk(&ex, table.at(1), &counts, &psi, &mut scratch, &mut llk)
        .unwrap();
    let got = llk.to_host_vec(&stream).unwrap();
    assert_llk("llk", &got[..2], &want);
    assert_eq!(got[2], sentinel, "record past the stack written");

    let mut short = LlkScratch::new(&stream, &d, 2).unwrap();
    let r = k.llk(&ex, table.at(1), &counts, &psi, &mut short, &mut llk);
    assert!(matches!(r, Err(Error::Shape(_))), "{r:?}");
}

#[test]
fn update_from_sums_matches_f64() {
    let (stream, ex, k) = setup();
    let inertia = 0.01f32;
    // 300_001 values make each block of the max pass stride; its maximum is the last value.
    for (len, seed) in [(6000usize, 20u64), (300_001, 30)] {
        let x_host = noise(len, seed, 1.0);
        let num_host = noise(len, seed + 1, 3.0);
        let mut den_host: Vec<f32> = noise(len, seed + 2, 2.0)
            .iter()
            .map(|c| c.re.abs())
            .collect();
        den_host[len - 1] = 7.5;
        let reg = inertia as f64 * den_host.iter().fold(0.0f64, |m, &v| m.max(v as f64));
        let want: Vec<Z> = (0..len)
            .map(|i| {
                let (a, b) = (z(num_host[i]), z(x_host[i]));
                let d = (den_host[i] as f64 + reg).max(1e-12);
                ((a.0 + reg * b.0) / d, (a.1 + reg * b.1) / d)
            })
            .collect();
        // A tail past `n` the update must neither use nor consume.
        let tail = (C32 { re: 5.0, im: -5.0 }, 1e30f32);
        let (mut num_t, mut den_t) = (num_host.clone(), den_host.clone());
        num_t.push(tail.0);
        den_t.push(tail.1);
        let mut x = up(&stream, &x_host);
        let mut sums = UpdateSums::new(up(&stream, &num_t), up(&stream, &den_t), &stream).unwrap();
        k.update_from_sums(&ex, &mut x, &mut sums, len as u32, inertia)
            .unwrap();
        let got = x.to_host_vec(&stream).unwrap();
        let (num_after, den_after) = (
            sums.num.to_host_vec(&stream).unwrap(),
            sums.den.to_host_vec(&stream).unwrap(),
        );
        assert!(
            num_after[..len] == vec![C32::default(); len][..]
                && den_after[..len] == vec![0.0; len][..],
            "n={len}: consumed sums not zeroed"
        );
        assert_eq!(
            (num_after[len], den_after[len]),
            tail,
            "n={len}: tail written"
        );
        let err = rel_l2(&got, &want);
        println!("update_from_sums n={len}: rel L2 {err:.2e}");
        assert!(err < 1e-6, "n={len}: rel L2 {err:.2e}");

        // Bit for bit the plain f32 formula, each operation rounded on its own (what a Tile
        // or CPU port computes), not a mix of fused and separate roundings.
        let reg = inertia * den_host.iter().fold(0.0f32, |m, &v| m.max(v));
        let mismatches = (0..len)
            .filter(|&i| {
                let d = (den_host[i] + reg).max(1e-12);
                let part = |a: f32, b: f32| (a + reg * b) / d;
                let (a, b) = (num_host[i], x_host[i]);
                let w = (part(a.re, b.re), part(a.im, b.im));
                (got[i].re.to_bits(), got[i].im.to_bits()) != (w.0.to_bits(), w.1.to_bits())
            })
            .count();
        assert_eq!(
            mismatches, 0,
            "n={len}: {mismatches} values differ from plain f32"
        );
    }
}

#[test]
fn dm_reflect_and_add_match_f64() {
    let (stream, ex, k) = setup();
    let (len, alpha) = (5000usize, 0.02f32);
    let a = alpha as f64;
    let (psi_host, po_host) = (noise(len, 40, 1.0), noise(len, 41, 1.0));
    let psi1: Vec<Z> = psi_host
        .iter()
        .zip(&po_host)
        .map(|(&p, &o)| {
            let (p, o) = (z(p), z(o));
            (
                (2.0 - a) * o.0 - (1.0 - a) * p.0,
                (2.0 - a) * o.1 - (1.0 - a) * p.1,
            )
        })
        .collect();
    let po1: Vec<Z> = po_host
        .iter()
        .zip(&psi1)
        .map(|(&o, p1)| {
            (
                (1.0 - a) * (o.re as f64 - p1.0),
                (1.0 - a) * (o.im as f64 - p1.1),
            )
        })
        .collect();
    let sum: Vec<Z> = psi1
        .iter()
        .zip(&po1)
        .map(|(p, o)| (p.0 + o.0, p.1 + o.1))
        .collect();

    let (mut psi, mut po) = (up(&stream, &psi_host), up(&stream, &po_host));
    k.dm_reflect(&ex, &mut psi, &mut po, len as u32, alpha)
        .unwrap();
    let e_psi = rel_l2(&psi.to_host_vec(&stream).unwrap(), &psi1);
    let e_po = rel_l2(&po.to_host_vec(&stream).unwrap(), &po1);
    k.add_c32(&ex, &mut psi, &po, len as u32).unwrap();
    let e_sum = rel_l2(&psi.to_host_vec(&stream).unwrap(), &sum);
    println!("dm_reflect: psi {e_psi:.2e}, po {e_po:.2e}; add_c32 {e_sum:.2e}");
    assert!(e_psi < 1e-6 && e_po < 1e-6 && e_sum < 1e-6);

    // First DM cycle: psi = PO is a fixed point, and the add-back term vanishes.
    let (mut psi, mut po) = (up(&stream, &po_host), up(&stream, &po_host));
    k.dm_reflect(&ex, &mut psi, &mut po, len as u32, alpha)
        .unwrap();
    let want: Vec<Z> = po_host.iter().map(|&c| z(c)).collect();
    let e_fix = rel_l2(&psi.to_host_vec(&stream).unwrap(), &want);
    let rest = po
        .to_host_vec(&stream)
        .unwrap()
        .iter()
        .fold(0.0f32, |m, c| m.max(c.re.abs()).max(c.im.abs()));
    println!("dm_reflect fixed point: psi rel L2 {e_fix:.2e}, max |po| {rest:.2e}");
    assert!(e_fix < 1e-6 && rest < 1e-6);
}

/// f64 orthonormal 2D DFT of an `n x n` plane (forward kernel `exp(-2 pi i jk / n)`).
fn dft2_ortho(plane: &[Z], n: usize, inverse: bool) -> Vec<Z> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let tw: Vec<Z> = (0..n)
        .map(|j| {
            let a = sign * 2.0 * PI * j as f64 / n as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let line = |get: &dyn Fn(usize) -> Z, k: usize| {
        (0..n).fold((0.0, 0.0), |s, j| {
            let t = mul(get(j), tw[(j * k) % n]);
            (s.0 + t.0, s.1 + t.1)
        })
    };
    let mut rows = vec![(0.0, 0.0); n * n];
    for r in 0..n {
        for k in 0..n {
            rows[r * n + k] = line(&|j| plane[r * n + j], k);
        }
    }
    let mut out = vec![(0.0, 0.0); n * n];
    for c in 0..n {
        for k in 0..n {
            let v = line(&|j| rows[j * n + c], k);
            out[k * n + c] = (v.0 / n as f64, v.1 / n as f64);
        }
    }
    out
}

/// `obj_probe_to_psi`, cuFFT, `llk`, `apply_amplitude`, inverse cuFFT against PyNX's
/// orthonormal `IFT * ApplyAmplitude(calc_llk) * FT * ObjProbe2Psi` in f64.
#[test]
fn fourier_projection_matches_pynx_orthonormal_f64() {
    const N: usize = 32;
    let (stream, ex, k) = setup();
    let d = ViewDims {
        nframe: 3,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [N as u32, N as u32],
        obj: [48, 48],
        quad_phase: 0.002,
    };
    let s = Scan::new(
        d,
        &[(0, 0, 0.0, 0.0), (15, 3, 0.3, 0.0), (10, 15, 0.5, 0.25)],
        10.0,
    );
    let (n, nm, nf) = (N * N, 2, 3);

    // PyNX in f64: FT, counts near 1.2x the model, LLK, amplitude projection, IFT.
    let mut far: Vec<Z> = s
        .psi(0..3, &s.obj)
        .chunks(n)
        .flat_map(|p| dft2_ortho(p, N, false))
        .collect();
    let mut counts_host = vec![0u16; nf * n];
    let mut llk_ref = vec![[0.0; 4]; nf];
    for f in 0..nf {
        for (y, x, i) in s.pixels() {
            let at = |m: usize| (f * nm + m) * n + s.shifted(y, x);
            let calc: f64 = (0..nm).map(|m| norm2(far[at(m)])).sum();
            let obs = match i {
                _ if i % 11 == 4 => MASKED,
                _ if i % 13 == 2 => 0,
                _ => (1.2 * calc).round().min(60000.0) as u16,
            };
            counts_host[f * n + i] = obs;
            if obs == MASKED {
                continue;
            }
            for (l, t) in llk_ref[f].iter_mut().zip(llk_terms(obs as f64, calc)) {
                *l += t;
            }
            let r = (obs as f64).sqrt() / calc.sqrt().max(1e-12);
            for m in 0..nm {
                far[at(m)] = (far[at(m)].0 * r, far[at(m)].1 * r);
            }
        }
    }
    let want: Vec<Z> = far.chunks(n).flat_map(|p| dft2_ortho(p, N, true)).collect();

    let views = Views::new(&stream, d, &s.views).unwrap();
    let table = one_stack(&stream, &d, 0..3);
    let mut plan = Plan2d::new(&stream, N, nf * nm).unwrap();
    let counts = up(&stream, &counts_host);
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, nf * nm * n).unwrap();
    k.obj_probe_to_psi(
        &ex,
        &views,
        table.at(0),
        &up(&stream, &s.obj),
        &up(&stream, &s.probe),
        &mut psi,
    )
    .unwrap();
    plan.forward_inplace(&ex, &mut psi).unwrap();
    let mut llk = DeviceBuffer::<LlkPartial>::zeroed(&stream, nf).unwrap();
    let mut scratch = LlkScratch::new(&stream, &d, table.nf_max()).unwrap();
    k.llk(&ex, table.at(0), &counts, &psi, &mut scratch, &mut llk)
        .unwrap();
    k.apply_amplitude(&ex, &d, table.at(0), &counts, &mut psi)
        .unwrap();
    plan.inverse_inplace(&ex, &mut psi).unwrap();

    assert_llk("chain llk", &llk.to_host_vec(&stream).unwrap(), &llk_ref);
    let err = rel_l2(&psi.to_host_vec(&stream).unwrap(), &want);
    println!("FT, apply_amplitude, IFT vs PyNX ortho f64: rel L2 {err:.2e}");
    assert!(err < 1e-5, "rel L2 {err:.2e}");
}

#[test]
fn views_and_frame_ranges_are_checked() {
    let (stream, ex, k) = setup();
    let s = scan();
    let shape_err = |r: Result<Views, Error>| matches!(r, Err(Error::Shape(_)));
    assert!(
        shape_err(Views::new(&stream, s.d, &s.views[1..])),
        "wrong view count"
    );
    let with = |f: fn(&mut ViewAddr)| {
        let mut v = s.views.clone();
        f(&mut v[6]);
        Views::new(&stream, s.d, &v)
    };
    assert!(
        shape_err(with(|v| v.oy = 27)),
        "view past the object's last row"
    );
    assert!(
        shape_err(with(|v| v.ox = 29)),
        "view past the object's last column"
    );
    assert!(shape_err(with(|v| v.fy = 1.0)), "fy = 1");
    assert!(shape_err(with(|v| v.fx = -0.1)), "fx < 0");
    assert!(shape_err(with(|v| v.probe = 2)), "probe mode out of range");
    assert!(shape_err(with(|v| v.obj = 2)), "object mode out of range");
    // Frame 1 mode 2 must be probe 0 times object 1 at the frame's (25, 28, 0.25, 0).
    assert!(shape_err(with(|v| v.probe = 1)), "probe not m % nprobe");
    assert!(shape_err(with(|v| v.obj = 0)), "object not m / nprobe");
    assert!(shape_err(with(|v| v.oy = 24)), "mode off the frame's row");
    assert!(
        shape_err(with(|v| v.fx = 0.5)),
        "mode off the frame's subpixel"
    );
    let uneven = ViewDims { nobj: 1, ..s.d };
    assert!(
        matches!(
            StackTable::new(&stream, &uneven, std::slice::from_ref(&(0..1))),
            Err(Error::Shape(_))
        ),
        "nmode != nprobe * nobj"
    );
    let odd = ViewDims {
        frame: [24, 31],
        ..s.d
    };
    assert!(
        shape_err(Views::new(&stream, odd, &s.views)),
        "odd frame side"
    );

    let table_err = |r: Result<StackTable, Error>| matches!(r, Err(Error::Shape(_)));
    for stacks in [&[0..2, 3..3][..], &[0..1, 4..6], &[]] {
        assert!(
            table_err(StackTable::new(&stream, &s.d, stacks)),
            "stacks {stacks:?}"
        );
    }
    for nf_max in [0, 6] {
        assert!(
            table_err(StackTable::device(&stream, &s.d, nf_max)),
            "nf_max {nf_max}"
        );
    }

    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let (obj, probe) = (up(&stream, &s.obj), up(&stream, &s.probe));
    let table = StackTable::new(&stream, &s.d, &[0..2, 1..5]).unwrap();
    let mut short = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..4)) * s.n() - 1).unwrap();
    let r = k.obj_probe_to_psi(&ex, &views, table.at(0), &obj, &probe, &mut short);
    assert!(
        matches!(r, Err(Error::Shape(_))),
        "psi shorter than nf_max frames"
    );
    let other = one_stack(&stream, &ViewDims { nframe: 4, ..s.d }, 0..4);
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..5)) * s.n()).unwrap();
    let r = k.obj_probe_to_psi(&ex, &views, other.at(0), &obj, &probe, &mut psi);
    assert!(
        matches!(r, Err(Error::Shape(_))),
        "table made for another frame count"
    );
}

/// A view may end on the object's last row only with `fy == 0`, since its `+ 1` neighbour
/// would leave the object and nothing wraps; likewise for the last column and `fx`.
#[test]
fn check_view_rejects_a_neighbour_past_the_object() {
    let d = scan().d;
    let ([ny, nx], [nyo, nxo]) = (d.frame, d.obj);
    let last = ViewAddr {
        oy: nyo - ny,
        ox: nxo - nx,
        ..ViewAddr::default()
    };
    let ok = |v: ViewAddr| d.check_view(&v).is_ok();
    assert!(ok(last), "last row and column, fy = fx = 0");
    assert!(!ok(ViewAddr { fy: 0.25, ..last }), "fy > 0 on the last row");
    assert!(
        !ok(ViewAddr { fx: 0.25, ..last }),
        "fx > 0 on the last column"
    );
    let inner = ViewAddr {
        oy: last.oy - 1,
        ox: last.ox - 1,
        fy: 0.25,
        fx: 0.25,
        ..last
    };
    assert!(ok(inner), "+ 1 neighbours on the last row and column");
}

/// `view_tiles` lists, once each, every tile holding an object pixel a view's samples read,
/// and `Views::new` lists each view in exactly those tiles' runs.
#[test]
fn view_tiles_cover_the_pixels_each_view_reads() {
    let (stream, _, _) = setup();
    let s = scan();
    let [nyo, nxo] = s.d.obj.map(|n| n as usize);
    let [nty, ntx] = s.d.tiles().map(|n| n as usize);
    let side = OBJ_TILE as usize;
    let mut runs = vec![Vec::new(); nty * ntx];
    for (k, v) in s.views.iter().enumerate() {
        let tiles: Vec<_> = view_tiles(&s.d, v).collect();
        let mut once = tiles.clone();
        once.sort();
        once.dedup();
        assert_eq!(once.len(), tiles.len(), "view {k} lists a tile twice");
        for (y, x, _) in s.pixels() {
            for (i, _) in s.corners(v, y, x) {
                let (oy, ox) = (i / nxo % nyo, i % nxo);
                let tile = [(oy / side) as u32, (ox / side) as u32];
                assert!(
                    tiles.contains(&tile),
                    "view {k} reads ({oy}, {ox}) outside {tiles:?}"
                );
            }
        }
        for [ty, tx] in tiles {
            runs[ty as usize * ntx + tx as usize].push(k as u32);
        }
    }
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let start = views.tile_start().to_host_vec(&stream).unwrap();
    let list = views.tile_views().to_host_vec(&stream).unwrap();
    let len = views.tile_len().to_host_vec(&stream).unwrap();
    for (t, run) in runs.iter().enumerate() {
        let lo = start[t] as usize;
        assert_eq!(&list[lo..lo + len[t] as usize], run, "tile {t}");
    }
}

/// Every stack launch over one stack, on per-stack outputs it zeroes first; the update sums
/// accumulate across runs.
struct StackRun {
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    psi: DeviceBuffer<C32>,
    icalc: DeviceBuffer<f32>,
    llk: DeviceBuffer<LlkPartial>,
    llk_scratch: LlkScratch,
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
}

impl StackRun {
    fn new(stream: &CudaStream, s: &Scan, nf_max: u32) -> Self {
        let (nf, n) = (nf_max as usize, s.n());
        Self {
            obj: up(stream, &s.obj),
            probe: up(stream, &s.probe),
            psi: DeviceBuffer::zeroed(stream, s.nk(&(0..nf_max)) * n).unwrap(),
            icalc: DeviceBuffer::zeroed(stream, nf * n).unwrap(),
            llk: DeviceBuffer::zeroed(stream, nf).unwrap(),
            llk_scratch: LlkScratch::new(stream, &s.d, nf_max).unwrap(),
            obj_sums: UpdateSums::zeroed(stream, s.obj.len()).unwrap(),
            probe_sums: UpdateSums::zeroed(stream, s.probe.len()).unwrap(),
        }
    }

    fn run(
        &mut self,
        ctx: &ExecutionContext,
        k: &Kernels,
        views: &Views,
        stack: StackRef<'_>,
        counts: &DeviceBuffer<u16>,
    ) -> Result<(), Error> {
        self.clear(ctx)?;
        self.ops(ctx, k, views, stack, counts)
    }

    /// Zeroes the per-stack outputs.
    fn clear(&mut self, ctx: &ExecutionContext) -> Result<(), Error> {
        ptyrs_gpu::zero(ctx, &mut self.psi)?;
        ptyrs_gpu::zero(ctx, &mut self.icalc)?;
        ptyrs_gpu::zero(ctx, &mut self.llk)
    }

    /// The stack launches alone, onto whatever the buffers hold.
    fn ops(
        &mut self,
        ctx: &ExecutionContext,
        k: &Kernels,
        views: &Views,
        stack: StackRef<'_>,
        counts: &DeviceBuffer<u16>,
    ) -> Result<(), Error> {
        let (obj, probe) = (&self.obj, &self.probe);
        let d = *views.dims();
        k.obj_probe_to_psi(ctx, views, stack, obj, probe, &mut self.psi)?;
        k.mode_intensity(ctx, &d, stack, &self.psi, &mut self.icalc)?;
        k.llk(
            ctx,
            stack,
            counts,
            &self.psi,
            &mut self.llk_scratch,
            &mut self.llk,
        )?;
        k.psi_to_obj(ctx, views, stack, &self.psi, probe, &mut self.obj_sums)?;
        k.psi_to_probe(ctx, views, stack, &self.psi, obj, &mut self.probe_sums)?;
        k.apply_amplitude(ctx, &d, stack, counts, &mut self.psi)
    }

    /// Fills every buffer the stack launches write with noise.
    fn fill_noise(&mut self, stream: &CudaStream) {
        let re =
            |len: usize, seed| -> Vec<f32> { noise(len, seed, 1.0).iter().map(|c| c.re).collect() };
        let sums = |len: usize, seed| {
            let (num, den) = (
                up(stream, &noise(len, seed, 1.0)),
                up(stream, &re(len, seed + 1)),
            );
            UpdateSums::new(num, den, stream).unwrap()
        };
        let llk: Vec<_> = noise(2 * self.llk.len(), 3, 1.0)
            .chunks(2)
            .map(|c| LlkPartial {
                poisson: c[0].re,
                gaussian: c[0].im,
                euclidean: c[1].re,
                icalc: c[1].im,
            })
            .collect();
        self.psi = up(stream, &noise(self.psi.len(), 1, 1.0));
        self.icalc = up(stream, &re(self.icalc.len(), 2));
        self.llk = up(stream, &llk);
        self.obj_sums = sums(self.obj.len(), 4);
        self.probe_sums = sums(self.probe.len(), 6);
    }

    fn bits(&self, stream: &CudaStream) -> Vec<u32> {
        let c = |b: &DeviceBuffer<C32>| -> Vec<u32> {
            let v = b.to_host_vec(stream).unwrap();
            v.iter()
                .flat_map(|c| [c.re, c.im])
                .map(f32::to_bits)
                .collect()
        };
        let f = |b: &DeviceBuffer<f32>| -> Vec<u32> {
            let v = b.to_host_vec(stream).unwrap();
            v.iter().map(|x| x.to_bits()).collect()
        };
        let llk = self.llk.to_host_vec(stream).unwrap();
        let llk = llk
            .iter()
            .flat_map(|p| [p.poisson, p.gaussian, p.euclidean, p.icalc].map(f32::to_bits));
        [
            c(&self.psi),
            f(&self.icalc),
            llk.collect(),
            c(&self.obj_sums.num),
            f(&self.obj_sums.den),
            c(&self.probe_sums.num),
            f(&self.probe_sums.den),
        ]
        .concat()
    }
}

/// A graph captured once on a device-written entry runs whichever stack the entry holds when
/// it replays, bit for bit like eager launches on a host-made table, sums accumulating over
/// the replays, and an entry past `nf_max` or the scan is clamped.
#[test]
fn one_graph_runs_whichever_stack_its_entry_holds() {
    let (stream, _, k) = setup();
    let s = scan();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let counts = up(&stream, &counts(s.d.nframe as usize * s.n()));
    let host = StackTable::new(&stream, &s.d, &[0..3, 3..5, 1..4, 4..5]).unwrap();
    let mut eager = StackRun::new(&stream, &s, host.nf_max());
    let mut want: Vec<_> = (0..host.len())
        .map(|i| {
            record(|ctx| eager.run(ctx, &k, &views, host.at(i), &counts))
                .sync_on(&stream)
                .unwrap();
            eager.bits(&stream)
        })
        .collect();
    // f0 past the scan: no frames, so the outputs stay zeroed and the sums as they were.
    record(|ctx| eager.clear(ctx)).sync_on(&stream).unwrap();
    want.push(eager.bits(&stream));

    let dev = StackTable::device(&stream, &s.d, 3).unwrap();
    let mut it = StackRun::new(&stream, &s, dev.nf_max());
    let side = k.context().new_stream().unwrap();
    let graph = CudaGraph::capture(
        side.clone(),
        record(|ctx| it.run(ctx, &k, &views, dev.at(0), &counts)),
    )
    .unwrap();
    let entries = [(0, 3), (3, 2), (1, 3), (4, 3), (9, 2)];
    for ((f0, nf), want) in entries.into_iter().zip(&want) {
        let ctl = [StackCtl { f0, nf }];
        k.context().bind_to_thread().unwrap();
        // SAFETY: `dev` holds one entry; no work on it is in flight.
        unsafe { cuda_core::memory::memcpy_htod_sync(dev.ptr(), ctl.as_ptr(), 8) }.unwrap();
        graph.launch().sync_on(&side).unwrap();
        assert!(
            it.bits(&stream) == *want,
            "entry {{ f0: {f0}, nf: {nf} }} differs from eager"
        );
    }
}

/// A captured stack replayed against an entry with `nf == 0` leaves every buffer it writes
/// as it was, bit for bit, wherever the entry's `f0` points; with `nf == 1` it does not.
#[test]
fn empty_stack_replay_writes_nothing() {
    let (stream, _, k) = setup();
    let s = scan();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let counts = up(&stream, &counts(s.d.nframe as usize * s.n()));
    let dev = StackTable::device(&stream, &s.d, 3).unwrap();
    let mut it = StackRun::new(&stream, &s, dev.nf_max());
    it.fill_noise(&stream);
    let side = k.context().new_stream().unwrap();
    let graph = CudaGraph::capture(
        side.clone(),
        record(|ctx| it.ops(ctx, &k, &views, dev.at(0), &counts)),
    )
    .unwrap();
    let start = it.bits(&stream);
    for (f0, nf, empty) in [(2, 0, true), (0, 0, true), (4, 0, true), (2, 1, false)] {
        let ctl = [StackCtl { f0, nf }];
        k.context().bind_to_thread().unwrap();
        // SAFETY: `dev` holds one entry; no work on it is in flight.
        unsafe { cuda_core::memory::memcpy_htod_sync(dev.ptr(), ctl.as_ptr(), 8) }.unwrap();
        graph.launch().sync_on(&side).unwrap();
        assert_eq!(
            it.bits(&stream) == start,
            empty,
            "entry {{ f0: {f0}, nf: {nf} }}"
        );
    }
}

/// One graph of `bin_views`, replayed over a device entry holding the stacks in ascending
/// frame order, leaves in each tile slab the packed run of `Views::new` cut to the slab's
/// capacity, with the cut views counted in `overflow` and the slots past the run untouched;
/// a replay with `nf == 0` changes nothing.
/// 12 frames of 4 views: [`scan`]'s offsets, each repeat shifted up and left.
fn scan12() -> Scan {
    let base = scan();
    let d = ViewDims {
        nframe: 12,
        ..base.d
    };
    let offsets: Vec<_> = (0..12)
        .map(|f| {
            let (oy, ox, fy, fx): (u32, u32, f32, f32) = [
                (0, 0, 0.0, 0.0),
                (25, 28, 0.25, 0.0),
                (13, 7, 0.0, 0.7),
                (26, 27, 0.0, 0.9),
                (5, 20, 0.5, 0.5),
            ][f % 5];
            let back = (f / 5) as u32;
            (
                oy.saturating_sub(3 * back),
                ox.saturating_sub(4 * back),
                fy,
                fx,
            )
        })
        .collect();
    Scan::new(d, &offsets, 1.0)
}

#[test]
fn bin_views_appends_each_stack_to_the_tile_slabs() {
    let (stream, _, k) = setup();
    // The 9-frame stack has 36 views, more than one warp ballot.
    let s = scan12();
    let d = s.d;
    let packed = Views::new(&stream, d, &s.views).unwrap();
    let pstart = packed.tile_start().to_host_vec(&stream).unwrap();
    let plen = packed.tile_len().to_host_vec(&stream).unwrap();
    let plist = packed.tile_views().to_host_vec(&stream).unwrap();
    let ntiles = plen.len();
    let full = *plen.iter().max().unwrap();
    let dev = StackTable::device(&stream, &d, 9).unwrap();
    let side = k.context().new_stream().unwrap();
    for cap in [full, full - 1] {
        let slot_start: Vec<u32> = (0..=ntiles as u32).map(|t| t * cap).collect();
        let unset = vec![u32::MAX; ntiles * cap as usize];
        // SAFETY: empty runs; `bin_views` fills them in ascending frame order.
        let mut views = unsafe {
            Views::from_parts(
                d,
                up(&stream, &s.views),
                up(&stream, &slot_start),
                up(&stream, &unset),
                up(&stream, &vec![0u32; ntiles]),
            )
        }
        .unwrap();
        let mut overflow = up(&stream, &[0u32]);
        let graph = CudaGraph::capture(
            side.clone(),
            record(|ctx| k.bin_views(ctx, &mut views, dev.at(0), &mut overflow)),
        )
        .unwrap();
        let state = |views: &Views, overflow: &DeviceBuffer<u32>| {
            (
                views.tile_len().to_host_vec(&stream).unwrap(),
                views.tile_views().to_host_vec(&stream).unwrap(),
                overflow.to_host_vec(&stream).unwrap()[0],
            )
        };
        let mut before = state(&views, &overflow);
        let mut kend = 0;
        // (12, 3) clamps to no frames.
        for (f0, nf) in [(0, 0), (0, 2), (5, 0), (2, 9), (11, 0), (11, 1), (12, 3)] {
            let ctl = [StackCtl { f0, nf }];
            k.context().bind_to_thread().unwrap();
            // SAFETY: `dev` holds one entry; no work on it is in flight.
            unsafe { cuda_core::memory::memcpy_htod_sync(dev.ptr(), ctl.as_ptr(), 8) }.unwrap();
            graph.launch().sync_on(&side).unwrap();
            let got = state(&views, &overflow);
            let what = format!("cap {cap}, entry {{ f0: {f0}, nf: {nf} }}");
            if nf == 0 || f0 >= d.nframe {
                assert!(got == before, "{what} changed the slabs");
                continue;
            }
            kend = (f0 + nf) * d.nmode;
            let (len, list, cut) = &got;
            let mut want_cut = 0;
            for t in 0..ntiles {
                let run = &plist[pstart[t] as usize..][..plen[t] as usize];
                let want: Vec<u32> = run.iter().copied().filter(|&v| v < kend).collect();
                let kept = want.len().min(cap as usize);
                want_cut += want.len() - kept;
                let slab = &list[t * cap as usize..][..cap as usize];
                assert_eq!(len[t] as usize, kept, "{what}: tile {t} length");
                assert_eq!(&slab[..kept], &want[..kept], "{what}: tile {t} run");
                assert!(
                    slab[kept..].iter().all(|&v| v == u32::MAX),
                    "{what}: tile {t} wrote past its run"
                );
            }
            assert_eq!(*cut as usize, want_cut, "{what}: overflow");
            before = got;
        }
        assert_eq!(kend, d.nframe * d.nmode, "every frame binned");
        assert_eq!(before.2 > 0, cap < full, "cap {cap}: overflow");
    }
}

/// An order of [`scan12`]'s frames that moves every frame.
const ORDER12: [u32; 12] = [7, 2, 11, 0, 5, 9, 1, 10, 3, 8, 6, 4];

/// `s`'s views with frame `p` taken from frame `order[p]`.
fn reordered_views(s: &Scan, order: &[u32]) -> Vec<ViewAddr> {
    order
        .iter()
        .flat_map(|&q| s.views_of(&(q..q + 1)).iter().copied())
        .collect()
}

type HostViews = (Vec<ViewAddr>, Vec<u32>, Vec<u32>, Vec<u32>);

/// `v`'s records, tile starts, run lengths and tile slots.
fn host_views(stream: &CudaStream, v: &Views) -> HostViews {
    (
        v.addr().to_host_vec(stream).unwrap(),
        v.tile_start().to_host_vec(stream).unwrap(),
        v.tile_len().to_host_vec(stream).unwrap(),
        v.tile_views().to_host_vec(stream).unwrap(),
    )
}

/// `reorder_views` by a permutation, captured once and replayed twice, leaves the records
/// and tile lists `Views::new` makes of the frames in that order, from packed lists and from
/// padded slabs alike; the inverse permutation then gives back the source exactly.
#[test]
fn reorder_views_by_a_permutation_and_its_inverse_restores_the_views() {
    let (stream, _, k) = setup();
    let s = scan12();
    let d = s.d;
    let mut inv = [0u32; 12];
    for (p, &q) in (0..).zip(&ORDER12) {
        inv[q as usize] = p;
    }
    let src = Views::new(&stream, d, &s.views).unwrap();
    let packed = Views::new(&stream, d, &reordered_views(&s, &ORDER12)).unwrap();
    let want = host_views(&stream, &packed);
    let order = up(&stream, &ORDER12);
    let mut mid = Views::reordered(&stream, &src).unwrap();
    let side = k.context().new_stream().unwrap();
    let graph = CudaGraph::capture(
        side.clone(),
        record(|ctx| k.reorder_views(ctx, &src, &order, &mut mid)),
    )
    .unwrap();
    for replay in 0..2 {
        graph.launch().sync_on(&side).unwrap();
        assert!(host_views(&stream, &mid) == want, "replay {replay}");
    }
    let mut back = Views::reordered(&stream, &mid).unwrap();
    let inv = up(&stream, &inv);
    record(|ctx| k.reorder_views(ctx, &mid, &inv, &mut back))
        .sync_on(&stream)
        .unwrap();
    assert!(host_views(&stream, &back) == host_views(&stream, &src));

    // Slabs with spare slots, filled by `bin_views`: a reorder keeps their capacities and
    // replaces the runs an earlier reorder left.
    let (wstart, wlen, wlist) = (&want.1, &want.2, &want.3);
    let ntiles = wlen.len();
    let cap = wlen.iter().max().unwrap() + 3;
    let start: Vec<u32> = (0..=ntiles as u32).map(|t| t * cap).collect();
    // SAFETY: empty runs, which `bin_views` fills with the one stack of every frame.
    let mut slab = unsafe {
        Views::from_parts(
            d,
            up(&stream, &s.views),
            up(&stream, &start),
            up(&stream, &vec![u32::MAX; ntiles * cap as usize]),
            up(&stream, &vec![0u32; ntiles]),
        )
    }
    .unwrap();
    let all = one_stack(&stream, &d, 0..d.nframe);
    let mut overflow = up(&stream, &[0u32]);
    record(|ctx| k.bin_views(ctx, &mut slab, all.at(0), &mut overflow))
        .sync_on(&stream)
        .unwrap();
    let mut slab_mid = Views::reordered(&stream, &slab).unwrap();
    for order in [&inv, &order] {
        record(|ctx| k.reorder_views(ctx, &slab, order, &mut slab_mid))
            .sync_on(&stream)
            .unwrap();
    }
    let (addr, got_start, len, list) = host_views(&stream, &slab_mid);
    assert!(addr == want.0, "slab records");
    assert_eq!(got_start, start, "slab starts");
    assert_eq!(&len, wlen, "slab run lengths");
    for t in 0..ntiles {
        let run = |list: &[u32], at: u32| list[at as usize..][..wlen[t] as usize].to_vec();
        assert_eq!(run(&list, start[t]), run(wlist, wstart[t]), "slab tile {t}");
    }
}

/// The stack launches over reordered views match, bit for bit, the same launches over
/// `Views::new` of the frames in that order, stack after stack.
#[test]
fn stack_pass_over_reordered_views_matches_the_frames_in_that_order() {
    let (stream, _, k) = setup();
    let s = scan12();
    let (d, n) = (s.d, s.n());
    let src = Views::new(&stream, d, &s.views).unwrap();
    let order = up(&stream, &ORDER12);
    let mut dst = Views::reordered(&stream, &src).unwrap();
    record(|ctx| k.reorder_views(ctx, &src, &order, &mut dst))
        .sync_on(&stream)
        .unwrap();
    let host = Views::new(&stream, d, &reordered_views(&s, &ORDER12)).unwrap();
    let c = counts(d.nframe as usize * n);
    let c: Vec<u16> = ORDER12
        .iter()
        .flat_map(|&q| c[q as usize * n..][..n].iter().copied())
        .collect();
    let counts = up(&stream, &c);
    let table = StackTable::new(&stream, &d, &[0..5, 5..9, 9..12]).unwrap();
    let mut got = StackRun::new(&stream, &s, table.nf_max());
    let mut want = StackRun::new(&stream, &s, table.nf_max());
    for i in 0..table.len() {
        record(|ctx| got.run(ctx, &k, &dst, table.at(i), &counts))
            .sync_on(&stream)
            .unwrap();
        record(|ctx| want.run(ctx, &k, &host, table.at(i), &counts))
            .sync_on(&stream)
            .unwrap();
        assert!(got.bits(&stream) == want.bits(&stream), "stack {i}");
    }
}

/// Side of the square image the line FFTs and the `Plan2d` of [`Iteration`] round-trip.
const AUX: u32 = 128;

/// State that one call of every launch method updates in place.
struct Iteration {
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    psi: DeviceBuffer<C32>,
    po: DeviceBuffer<C32>,
    icalc: DeviceBuffer<f32>,
    llk: DeviceBuffer<LlkPartial>,
    llk_scratch: LlkScratch,
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
    aux: DeviceBuffer<C32>,
    tmp: DeviceBuffer<C32>,
    plan: Plan2d,
    stacks: StackTable,
}

impl Iteration {
    fn new(stream: &CudaStream, s: &Scan) -> Self {
        let (nf, n) = (s.d.nframe as usize, s.n());
        let nk = s.nk(&(0..s.d.nframe)) * n;
        let aux = (AUX * AUX) as usize;
        Self {
            obj: up(stream, &s.obj),
            probe: up(stream, &s.probe),
            psi: up(stream, &noise(nk, 50, 1.0)),
            po: DeviceBuffer::zeroed(stream, nk).unwrap(),
            icalc: DeviceBuffer::zeroed(stream, nf * n).unwrap(),
            llk: DeviceBuffer::zeroed(stream, nf).unwrap(),
            llk_scratch: LlkScratch::new(stream, &s.d, s.d.nframe).unwrap(),
            obj_sums: UpdateSums::zeroed(stream, s.obj.len()).unwrap(),
            probe_sums: UpdateSums::zeroed(stream, s.probe.len()).unwrap(),
            aux: up(stream, &noise(aux, 51, 1.0)),
            tmp: DeviceBuffer::zeroed(stream, aux).unwrap(),
            plan: Plan2d::new(stream, AUX as usize, 1).unwrap(),
            stacks: one_stack(stream, &s.d, 0..s.d.nframe),
        }
    }

    /// A DM iteration, then 2D FFT round trips of `aux` through the line kernels and through
    /// every `Plan2d` transform.
    fn run(
        &mut self,
        ctx: &ExecutionContext,
        k: &Kernels,
        views: &Views,
        counts: &DeviceBuffer<u16>,
    ) -> Result<(), Error> {
        let d = *views.dims();
        let stack = self.stacks.at(0);
        let nk = self.psi.len() as u32;
        k.obj_probe_to_psi(ctx, views, stack, &self.obj, &self.probe, &mut self.po)?;
        k.dm_reflect(ctx, &mut self.psi, &mut self.po, nk, 0.02)?;
        k.mode_intensity(ctx, &d, stack, &self.psi, &mut self.icalc)?;
        k.llk(
            ctx,
            stack,
            counts,
            &self.psi,
            &mut self.llk_scratch,
            &mut self.llk,
        )?;
        k.apply_amplitude(ctx, &d, stack, counts, &mut self.psi)?;
        k.add_c32(ctx, &mut self.psi, &self.po, nk)?;
        k.psi_to_obj(
            ctx,
            views,
            stack,
            &self.psi,
            &self.probe,
            &mut self.obj_sums,
        )?;
        k.psi_to_probe(
            ctx,
            views,
            stack,
            &self.psi,
            &self.obj,
            &mut self.probe_sums,
        )?;
        let np = self.probe.len() as u32;
        k.update_obj_from_sums(ctx, views, &mut self.obj, &mut self.obj_sums, 0.01)?;
        k.update_from_sums(ctx, &mut self.probe, &mut self.probe_sums, np, 0.01)?;

        let rows = Lines {
            n: AUX,
            count: AUX,
            line_stride: AUX,
            elem_stride: 1,
        };
        let cols = Lines {
            line_stride: 1,
            elem_stride: AUX,
            ..rows
        };
        k.fft_lines(ctx, &self.aux, &mut self.tmp, rows, false)?;
        k.fft_lines_inplace(ctx, &mut self.tmp, cols, false)?;
        k.fft_lines(ctx, &self.tmp, &mut self.aux, rows, true)?;
        k.fft_lines_inplace(ctx, &mut self.aux, cols, true)?;
        k.scale_c32(ctx, &mut self.aux, AUX * AUX, 1.0 / (AUX * AUX) as f32)?;

        self.plan.forward(ctx, &self.aux, &mut self.tmp)?;
        self.plan.inverse_inplace(ctx, &mut self.tmp)?;
        self.plan.forward_inplace(ctx, &mut self.tmp)?;
        self.plan.inverse(ctx, &self.tmp, &mut self.aux)?;
        let n4 = ((AUX * AUX) as f32).powi(2);
        k.scale_c32(ctx, &mut self.aux, AUX * AUX, 1.0 / n4)
    }

    /// The bits of every buffer `run` writes.
    fn bits(&self, stream: &CudaStream) -> Vec<Vec<u32>> {
        let c = |b: &DeviceBuffer<C32>| {
            b.to_host_vec(stream)
                .unwrap()
                .iter()
                .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                .collect()
        };
        let f = |b: &DeviceBuffer<f32>| {
            b.to_host_vec(stream)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        let llk = self.llk.to_host_vec(stream).unwrap();
        vec![
            c(&self.obj),
            c(&self.probe),
            c(&self.psi),
            c(&self.po),
            f(&self.icalc),
            llk.iter()
                .flat_map(|p| [p.poisson, p.gaussian, p.euclidean, p.icalc].map(f32::to_bits))
                .collect(),
            c(&self.obj_sums.num),
            f(&self.obj_sums.den),
            c(&self.probe_sums.num),
            f(&self.probe_sums.den),
            c(&self.aux),
            c(&self.tmp),
        ]
    }
}

#[test]
fn captured_iteration_replays_bitwise_like_eager() {
    let (stream, _, k) = setup();
    let s = scan();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let counts = up(&stream, &counts(s.d.nframe as usize * s.n()));

    let mut eager = Iteration::new(&stream, &s);
    let start = eager.bits(&stream);
    let want: Vec<_> = (0..4)
        .map(|_| {
            record(|ctx| eager.run(ctx, &k, &views, &counts))
                .sync_on(&stream)
                .unwrap();
            eager.bits(&stream)
        })
        .collect();
    assert!(
        want.iter()
            .flatten()
            .flatten()
            .all(|&b| f32::from_bits(b).is_finite()),
        "eager state went non-finite"
    );
    assert!(
        want[0][0] != want[1][0],
        "a second iteration must move the object"
    );

    let mut it = Iteration::new(&stream, &s);
    let side = k.context().new_stream().unwrap();
    let graph =
        CudaGraph::capture(side.clone(), record(|ctx| it.run(ctx, &k, &views, &counts))).unwrap();
    assert!(
        it.bits(&stream) == start,
        "capture must not execute the work"
    );
    for (replay, want) in want[..2].iter().enumerate() {
        graph.launch().sync_on(&side).unwrap();
        assert!(
            it.bits(&stream) == *want,
            "replay {replay} differs from eager"
        );
    }

    // A replay holds its accesses until it completes: another stream can neither write
    // `obj`, which the graph reads, nor read `icalc`, which it only writes, nor run the
    // plan, whose work area the graph writes.
    let replay = ExecutionContext::new(side.clone());
    // SAFETY: `replay` keeps the launch's leases until `side` is synchronized below, and
    // those leases refuse every other access to the buffers until then.
    unsafe { graph.launch().execute(&replay) }.unwrap();
    let other = ExecutionContext::new(stream.clone());
    let not_ready = cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_READY;
    let refused = k.scale_c32(&other, &mut it.obj, 1, 1.0);
    assert!(
        matches!(refused, Err(Error::Device(DeviceError::Driver(DriverError(e)))) if e == not_ready),
        "{refused:?}"
    );
    let refused = it.icalc.to_host_vec(&stream);
    assert!(
        matches!(refused, Err(DriverError(e)) if e == not_ready),
        "{refused:?}"
    );
    let (a, mut b) = (
        DeviceBuffer::<C32>::zeroed(&stream, it.aux.len()).unwrap(),
        DeviceBuffer::<C32>::zeroed(&stream, it.aux.len()).unwrap(),
    );
    let refused = it.plan.forward(&other, &a, &mut b);
    assert!(
        matches!(refused, Err(FftError::Device(DeviceError::Driver(DriverError(e)))) if e == not_ready),
        "{refused:?}"
    );
    side.synchronize().unwrap();
    drop(replay);
    assert!(
        it.bits(&stream) == want[2],
        "in-flight replay differs from eager"
    );

    // After the replays, the buffers are ordinary eager buffers again.
    record(|ctx| it.run(ctx, &k, &views, &counts))
        .sync_on(&stream)
        .unwrap();
    assert!(
        it.bits(&stream) == want[3],
        "eager run after replays differs"
    );
}

/// In-place unnormalized radix-2 FFT, forward kernel `exp(-2 pi i jk / n)`.
fn fft_f64(a: &mut [Z], inverse: bool) {
    let n = a.len();
    assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let tw: Vec<Z> = (0..n / 2)
        .map(|k| {
            let t = sign * 2.0 * PI * k as f64 / n as f64;
            (t.cos(), t.sin())
        })
        .collect();
    let mut len = 2;
    while len <= n {
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (u, v) = (a[start + k], mul(a[start + k + len / 2], tw[k * (n / len)]));
                a[start + k] = (u.0 + v.0, u.1 + v.1);
                a[start + k + len / 2] = (u.0 - v.0, u.1 - v.1);
            }
        }
        len <<= 1;
    }
}

/// Unnormalized 2D FFT of each `[ny, nx]` plane of `planes`, in place.
fn fft2_f64(planes: &mut [Z], [ny, nx]: [u32; 2], inverse: bool) {
    let (ny, nx) = (ny as usize, nx as usize);
    let mut col = vec![(0.0, 0.0); ny];
    for p in planes.chunks_exact_mut(ny * nx) {
        for row in p.chunks_exact_mut(nx) {
            fft_f64(row, inverse);
        }
        for x in 0..nx {
            for (y, c) in col.iter_mut().enumerate() {
                *c = p[y * nx + x];
            }
            fft_f64(&mut col, inverse);
            for (y, c) in col.iter().enumerate() {
                p[y * nx + x] = *c;
            }
        }
    }
}

/// The M1 Fourier projection in f64 of `frames`' exit waves `psi` (M1 layout) against the
/// scan's `counts`: the projected waves and each frame's LLK terms of the far field.
fn project_f64(s: &Scan, frames: Range<u32>, psi: &[Z], counts: &[u16]) -> (Vec<Z>, Vec<[f64; 4]>) {
    far_f64(s, frames, psi, counts, |obs, calc| {
        obs.map_or(1.0, |obs| (obs / calc.max(1e-12)).sqrt())
    })
}

/// Engine's `poisson_grad` in f64, back in the detector plane, and the LLK terms.
fn grad_f64(s: &Scan, frames: Range<u32>, psi: &[Z], counts: &[u16]) -> (Vec<Z>, Vec<[f64; 4]>) {
    far_f64(s, frames, psi, counts, |obs, calc| {
        obs.map_or(0.0, |obs| 2.0 * (1.0 - obs / calc.max(1e-12)))
    })
}

/// `psi`'s far field with every pixel's modes times `scale(obs, calc)` in the ortho plane
/// (`obs` `None` where masked), back in the detector plane, and each frame's LLK terms.
fn far_f64(
    s: &Scan,
    frames: Range<u32>,
    psi: &[Z],
    counts: &[u16],
    scale: impl Fn(Option<f64>, f64) -> f64,
) -> (Vec<Z>, Vec<[f64; 4]>) {
    let (n, nm) = (s.n(), s.d.nmode as usize);
    let nf = (frames.end - frames.start) as usize;
    let mut far = psi.to_vec();
    fft2_f64(&mut far, s.d.frame, false);
    let mut llk = vec![[0.0; 4]; nf];
    for (fl, llk) in llk.iter_mut().enumerate() {
        for (y, x, i) in s.pixels() {
            let at = |m: usize| (fl * nm + m) * n + s.shifted(y, x);
            let calc = (0..nm).map(|m| norm2(far[at(m)])).sum::<f64>() / n as f64;
            let obs = counts[(frames.start as usize + fl) * n + i];
            let obs = (obs != MASKED).then_some(obs as f64);
            if let Some(obs) = obs {
                for (l, t) in llk.iter_mut().zip(llk_terms(obs, calc)) {
                    *l += t;
                }
            }
            let scale = scale(obs, calc) / n as f64;
            for m in 0..nm {
                far[at(m)] = (far[at(m)].0 * scale, far[at(m)].1 * scale);
            }
        }
    }
    fft2_f64(&mut far, s.d.frame, true);
    (far, llk)
}

/// Five frames of `frame` with `[nprobe, nobj]` modes on an object 40 pixels larger; frames
/// 1 and 3 reach the object's last row and column as in [`scan_in`].
fn fused_scan(frame: [u32; 2], [nprobe, nobj]: [u32; 2]) -> Scan {
    let d = ViewDims {
        nframe: 5,
        nmode: nprobe * nobj,
        nprobe,
        nobj,
        frame,
        obj: [frame[0] + 40, frame[1] + 40],
        quad_phase: 0.0002,
    };
    let offsets = [
        (0, 0, 0.0, 0.0),
        (39, 40, 0.25, 0.0),
        (13, 7, 0.0, 0.7),
        (40, 39, 0.0, 0.9),
        (5, 20, 0.5, 0.5),
    ];
    Scan::new(d, &offsets, 10.0)
}

/// Counts near 1.2x the scan's model intensity, with masked and zero pixels.
fn fused_counts(s: &Scan) -> Vec<u16> {
    let (n, nm) = (s.n(), s.d.nmode as usize);
    let mut far = s.psi(0..s.d.nframe, &s.obj);
    fft2_f64(&mut far, s.d.frame, false);
    let mut counts = vec![0u16; s.d.nframe as usize * n];
    for (f, plane) in counts.chunks_exact_mut(n).enumerate() {
        for (y, x, i) in s.pixels() {
            let at = |m: usize| (f * nm + m) * n + s.shifted(y, x);
            let calc = (0..nm).map(|m| norm2(far[at(m)])).sum::<f64>() / n as f64;
            plane[i] = match i {
                _ if i % 13 == 4 => MASKED,
                _ if i % 11 == 2 => 0,
                _ => (1.2 * calc).round().min(60000.0) as u16,
            };
        }
    }
    counts
}

fn to_c32(v: &[Z]) -> Vec<C32> {
    v.iter()
        .map(|&(re, im)| C32 {
            re: re as f32,
            im: im as f32,
        })
        .collect()
}

/// A scan's fused inputs on the device.
struct FusedScan {
    s: Scan,
    counts_host: Vec<u16>,
    views: Views,
    counts: DeviceBuffer<u16>,
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
}

impl FusedScan {
    fn new(stream: &CudaStream, frame: [u32; 2], modes: [u32; 2]) -> Self {
        let s = fused_scan(frame, modes);
        let counts_host = fused_counts(&s);
        Self {
            views: Views::new(stream, s.d, &s.views).unwrap(),
            counts: up(stream, &counts_host),
            obj: up(stream, &s.obj),
            probe: up(stream, &s.probe),
            counts_host,
            s,
        }
    }

    fn inputs(&self) -> FusedInputs<'_> {
        FusedInputs {
            views: &self.views,
            counts: &self.counts,
            obj: &self.obj,
            probe: &self.probe,
        }
    }
}

/// Frames and `[nprobe, nobj]` modes of the f64 fused tests: every side along both axes,
/// and every side's column block at the most modes it takes.
const FUSED_CASES: [([u32; 2], [u32; 2]); 6] = [
    ([128, 256], [2, 2]),
    ([256, 128], [2, 2]),
    ([64, 512], [2, 2]),
    ([512, 64], [2, 1]),
    ([64, 128], [16, 1]),
    ([128, 64], [8, 1]),
];

/// AP: the projected exit waves, LLK partials and object and probe sums of `fused_project`,
/// and `fused_llk`, against the f64 M1 chain, on [`FUSED_CASES`].
#[test]
fn fused_ap_matches_f64() {
    let (stream, ex, k) = setup();
    for (frame, modes) in FUSED_CASES {
        let fs = FusedScan::new(&stream, frame, modes);
        let (s, n) = (&fs.s, fs.s.n());
        let stacks = [0..3, 3..5];
        let table = StackTable::new(&stream, &s.d, &stacks).unwrap();
        let mut bufs = FusedBufs::new(&stream, &s.d, table.nf_max()).unwrap();
        let mut obj_sums = UpdateSums::zeroed(&stream, s.obj.len()).unwrap();
        let mut probe_sums = UpdateSums::zeroed(&stream, s.probe.len()).unwrap();
        let mut llk = DeviceBuffer::<LlkPartial>::zeroed(&stream, 3).unwrap();
        let mut llk_only = DeviceBuffer::<LlkPartial>::zeroed(&stream, 3).unwrap();
        let (mut num, mut den) = (vec![(0.0, 0.0); s.obj.len()], vec![0.0; s.obj.len()]);
        let (mut pnum, mut pden) = (vec![(0.0, 0.0); s.probe.len()], vec![0.0; s.probe.len()]);
        for (i, frames) in stacks.into_iter().enumerate() {
            let stack = table.at(i);
            let nf = (frames.end - frames.start) as usize;
            k.fused_project(
                &ex,
                &fs.inputs(),
                stack,
                Exit::Ap {
                    llk: Some(&mut llk),
                },
                &mut bufs,
                FusedSums {
                    obj: Some(&mut obj_sums),
                    probe: Some(&mut probe_sums),
                },
            )
            .unwrap();
            let work = bufs.work().to_host_vec(&stream).unwrap();
            k.fused_llk(&ex, &fs.inputs(), stack, &mut bufs, &mut llk_only)
                .unwrap();

            let po = s.psi(frames.clone(), &s.obj);
            let (want, llk_ref) = project_f64(s, frames.clone(), &po, &fs.counts_host);
            let err = rel_l2(&work[..s.nk(&frames) * n], &want);
            println!("fused AP {frame:?} {frames:?}: rel L2 {err:.2e}");
            assert!(err < 1e-5, "{frame:?} {frames:?}: rel L2 {err:.2e}");
            let got = llk.to_host_vec(&stream).unwrap();
            assert_llk("fused AP llk", &got[..nf], &llk_ref);
            let alone = llk_only.to_host_vec(&stream).unwrap();
            let llk_bits = |v: &[LlkPartial]| -> Vec<[u32; 4]> {
                v.iter()
                    .map(|p| [p.poisson, p.gaussian, p.euclidean, p.icalc].map(f32::to_bits))
                    .collect()
            };
            assert_eq!(
                llk_bits(&alone[..nf]),
                llk_bits(&got[..nf]),
                "{frame:?} {frames:?}: fused_llk differs from the AP partials"
            );
            let want = to_c32(&want);
            s.obj_sums(frames.clone(), &want, &mut num, &mut den);
            s.probe_sums(frames, &want, &mut pnum, &mut pden);
        }
        for (what, sums, num, den) in [
            ("object", &obj_sums, &num, &den),
            ("probe", &probe_sums, &pnum, &pden),
        ] {
            let en = rel_l2(&sums.num.to_host_vec(&stream).unwrap(), num);
            let ed = rel_l2_re(&sums.den.to_host_vec(&stream).unwrap(), den);
            println!("fused AP {frame:?} {what} sums: num rel L2 {en:.2e}, den {ed:.2e}");
            assert!(
                en < 1e-5 && ed < 1e-5,
                "{frame:?} {what}: {en:.2e}, {ed:.2e}"
            );
        }
    }
}

/// ML: `Exit::Grad`'s gradient waves and its object and probe sums (engine's
/// `Ptycho::obj_grad` and `Ptycho::probe_grad`), against engine's
/// `poisson_grad` chain in f64; its LLK partials bit for bit against `Exit::Ap`'s.
#[test]
fn fused_grad_matches_f64() {
    let (stream, ex, k) = setup();
    for (frame, modes) in FUSED_CASES {
        let fs = FusedScan::new(&stream, frame, modes);
        let (s, n) = (&fs.s, fs.s.n());
        let stacks = [0..3, 3..5];
        let table = StackTable::new(&stream, &s.d, &stacks).unwrap();
        let mut bufs = FusedBufs::new(&stream, &s.d, table.nf_max()).unwrap();
        let mut obj_sums = UpdateSums::zeroed(&stream, s.obj.len()).unwrap();
        let mut probe_sums = UpdateSums::zeroed(&stream, s.probe.len()).unwrap();
        let mut llk = DeviceBuffer::<LlkPartial>::zeroed(&stream, 3).unwrap();
        let mut llk_ap = DeviceBuffer::<LlkPartial>::zeroed(&stream, 3).unwrap();
        let (mut num, mut den) = (vec![(0.0, 0.0); s.obj.len()], vec![0.0; s.obj.len()]);
        let (mut pnum, mut pden) = (vec![(0.0, 0.0); s.probe.len()], vec![0.0; s.probe.len()]);
        for (i, frames) in stacks.into_iter().enumerate() {
            let stack = table.at(i);
            let nf = (frames.end - frames.start) as usize;
            k.fused_project(
                &ex,
                &fs.inputs(),
                stack,
                Exit::Ap {
                    llk: Some(&mut llk_ap),
                },
                &mut bufs,
                FusedSums::default(),
            )
            .unwrap();
            k.fused_project(
                &ex,
                &fs.inputs(),
                stack,
                Exit::Grad {
                    llk: Some(&mut llk),
                },
                &mut bufs,
                FusedSums {
                    obj: Some(&mut obj_sums),
                    probe: Some(&mut probe_sums),
                },
            )
            .unwrap();
            let work = bufs.work().to_host_vec(&stream).unwrap();

            let po = s.psi(frames.clone(), &s.obj);
            let (want, llk_ref) = grad_f64(s, frames.clone(), &po, &fs.counts_host);
            let err = rel_l2(&work[..s.nk(&frames) * n], &want);
            println!("fused grad {frame:?} {frames:?}: rel L2 {err:.2e}");
            assert!(err < 1e-5, "{frame:?} {frames:?}: rel L2 {err:.2e}");
            let got = llk.to_host_vec(&stream).unwrap();
            assert_llk("fused grad llk", &got[..nf], &llk_ref);
            let llk_bits = |v: &[LlkPartial]| -> Vec<[u32; 4]> {
                v.iter()
                    .map(|p| [p.poisson, p.gaussian, p.euclidean, p.icalc].map(f32::to_bits))
                    .collect()
            };
            assert_eq!(
                llk_bits(&llk_ap.to_host_vec(&stream).unwrap()[..nf]),
                llk_bits(&got[..nf]),
                "{frame:?} {frames:?}: grad LLK partials differ from AP's"
            );
            let want = to_c32(&want);
            s.obj_sums(frames.clone(), &want, &mut num, &mut den);
            s.probe_sums(frames, &want, &mut pnum, &mut pden);
        }
        for (what, sums, num, den) in [
            ("object", &obj_sums, &num, &den),
            ("probe", &probe_sums, &pnum, &pden),
        ] {
            let en = rel_l2(&sums.num.to_host_vec(&stream).unwrap(), num);
            let ed = rel_l2_re(&sums.den.to_host_vec(&stream).unwrap(), den);
            println!("fused grad {frame:?} {what} sums: num rel L2 {en:.2e}, den {ed:.2e}");
            assert!(
                en < 1e-5 && ed < 1e-5,
                "{frame:?} {what}: {en:.2e}, {ed:.2e}"
            );
        }
    }
}

/// DM: the first cycle's `psi = P(PO)` and a later cycle's `P(R) + (1 - alpha)(PO - R)` with
/// its object and probe sums, against f64, the later cycle starting from the first one's
/// output.
#[test]
fn fused_dm_matches_f64() {
    let (stream, ex, k) = setup();
    let alpha = 0.02f32;
    let (a, b) = (2.0 - alpha as f64, 1.0 - alpha as f64);
    for (frame, modes) in FUSED_CASES {
        let fs = FusedScan::new(&stream, frame, modes);
        let (s, n) = (&fs.s, fs.s.n());
        let stacks = [0..3, 3..5];
        let table = StackTable::new(&stream, &s.d, &stacks).unwrap();
        let mut bufs = FusedBufs::new(&stream, &s.d, table.nf_max()).unwrap();
        let mut obj_sums = UpdateSums::zeroed(&stream, s.obj.len()).unwrap();
        let mut probe_sums = UpdateSums::zeroed(&stream, s.probe.len()).unwrap();
        let (mut num, mut den) = (vec![(0.0, 0.0); s.obj.len()], vec![0.0; s.obj.len()]);
        let (mut pnum, mut pden) = (vec![(0.0, 0.0); s.probe.len()], vec![0.0; s.probe.len()]);
        for (i, frames) in stacks.into_iter().enumerate() {
            let (stack, len) = (table.at(i), s.nk(&frames) * n);
            let mut psi = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..3)) * n).unwrap();
            let po = s.psi(frames.clone(), &s.obj);
            k.fused_project(
                &ex,
                &fs.inputs(),
                stack,
                Exit::DmFirst { psi: &mut psi },
                &mut bufs,
                FusedSums::default(),
            )
            .unwrap();
            let first = psi.to_host_vec(&stream).unwrap();
            let (want, _) = project_f64(s, frames.clone(), &po, &fs.counts_host);
            let err = rel_l2(&first[..len], &want);
            println!("fused DM first {frame:?} {frames:?}: rel L2 {err:.2e}");
            assert!(err < 1e-5, "{frame:?} {frames:?} first: rel L2 {err:.2e}");

            k.fused_project(
                &ex,
                &fs.inputs(),
                stack,
                Exit::Dm {
                    psi: &mut psi,
                    alpha,
                },
                &mut bufs,
                FusedSums {
                    obj: Some(&mut obj_sums),
                    probe: Some(&mut probe_sums),
                },
            )
            .unwrap();
            let refl: Vec<Z> = po
                .iter()
                .zip(&first)
                .map(|(o, &p)| {
                    let p = z(p);
                    (a * o.0 - b * p.0, a * o.1 - b * p.1)
                })
                .collect();
            let (proj, _) = project_f64(s, frames.clone(), &refl, &fs.counts_host);
            let want: Vec<Z> = proj
                .iter()
                .zip(po.iter().zip(&refl))
                .map(|(p, (o, r))| (p.0 + b * (o.0 - r.0), p.1 + b * (o.1 - r.1)))
                .collect();
            let got = psi.to_host_vec(&stream).unwrap();
            let err = rel_l2(&got[..len], &want);
            println!("fused DM {frame:?} {frames:?}: rel L2 {err:.2e}");
            assert!(err < 1e-5, "{frame:?} {frames:?}: rel L2 {err:.2e}");
            let want = to_c32(&want);
            s.obj_sums(frames.clone(), &want, &mut num, &mut den);
            s.probe_sums(frames, &want, &mut pnum, &mut pden);
        }
        for (what, sums, num, den) in [
            ("object", &obj_sums, &num, &den),
            ("probe", &probe_sums, &pnum, &pden),
        ] {
            let en = rel_l2(&sums.num.to_host_vec(&stream).unwrap(), num);
            let ed = rel_l2_re(&sums.den.to_host_vec(&stream).unwrap(), den);
            println!("fused DM {frame:?} {what} sums: num rel L2 {en:.2e}, den {ed:.2e}");
            assert!(
                en < 1e-5 && ed < 1e-5,
                "{frame:?} {what}: {en:.2e}, {ed:.2e}"
            );
        }
    }
}

/// Every frame's AP waves, LLK partials and DM waves, and the object and probe sums of both,
/// are the same bits whatever stacks the frames are split into.
#[test]
fn fused_results_do_not_depend_on_the_stack_split() {
    let (stream, ex, k) = setup();
    let fs = FusedScan::new(&stream, [128, 256], [2, 2]);
    let (s, n, nm) = (&fs.s, fs.s.n(), fs.s.d.nmode as usize);
    let mut bufs = FusedBufs::new(&stream, &s.d, 5).unwrap();
    let start_psi = noise(5 * nm * n, 60, 1.0);
    // Stacks of these lengths, in frame order.
    let lens: [&[u32]; 3] = [&[5], &[2, 3], &[1; 5]];
    let splits = lens.map(|lens| {
        lens.iter()
            .scan(0, |f0, &nf| {
                *f0 += nf;
                Some(*f0 - nf..*f0)
            })
            .collect::<Vec<_>>()
    });
    let mut results = vec![];
    for split in &splits {
        let table = StackTable::new(&stream, &s.d, split).unwrap();
        let mut ap_sums = UpdateSums::zeroed(&stream, s.obj.len()).unwrap();
        let mut dm_sums = UpdateSums::zeroed(&stream, s.obj.len()).unwrap();
        let mut ap_probe = UpdateSums::zeroed(&stream, s.probe.len()).unwrap();
        let mut dm_probe = UpdateSums::zeroed(&stream, s.probe.len()).unwrap();
        let mut llk = DeviceBuffer::<LlkPartial>::zeroed(&stream, 5).unwrap();
        let (mut ap, mut dm, mut partials) = (vec![], vec![], vec![]);
        for (i, frames) in split.iter().enumerate() {
            let (stack, len) = (table.at(i), s.nk(frames) * n);
            let nf = (frames.end - frames.start) as usize;
            k.fused_project(
                &ex,
                &fs.inputs(),
                stack,
                Exit::Ap {
                    llk: Some(&mut llk),
                },
                &mut bufs,
                FusedSums {
                    obj: Some(&mut ap_sums),
                    probe: Some(&mut ap_probe),
                },
            )
            .unwrap();
            ap.extend(c32_bits(&bufs.work().to_host_vec(&stream).unwrap()[..len]));
            let got = llk.to_host_vec(&stream).unwrap();
            partials.extend(
                got[..nf]
                    .iter()
                    .map(|p| [p.poisson, p.gaussian, p.euclidean, p.icalc].map(f32::to_bits)),
            );
            // The same start waves for a frame wherever its stack begins.
            let at = frames.start as usize * nm * n;
            let mut psi = up_padded(&stream, &start_psi[at..], start_psi.len());
            k.fused_project(
                &ex,
                &fs.inputs(),
                stack,
                Exit::Dm {
                    psi: &mut psi,
                    alpha: 0.02,
                },
                &mut bufs,
                FusedSums {
                    obj: Some(&mut dm_sums),
                    probe: Some(&mut dm_probe),
                },
            )
            .unwrap();
            dm.extend(c32_bits(&psi.to_host_vec(&stream).unwrap()[..len]));
        }
        let sums = |u: &UpdateSums| {
            let num = c32_bits(&u.num.to_host_vec(&stream).unwrap());
            let den: Vec<u32> = u
                .den
                .to_host_vec(&stream)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            (num, den)
        };
        results.push((
            ap,
            partials,
            dm,
            [&ap_sums, &dm_sums, &ap_probe, &dm_probe].map(sums),
        ));
    }
    for (split, r) in splits.iter().zip(&results).skip(1) {
        let w = &results[0];
        assert!(r.0 == w.0, "{split:?}: AP waves differ");
        assert!(r.1 == w.1, "{split:?}: LLK partials differ");
        assert!(r.2 == w.2, "{split:?}: DM waves differ");
        for (i, what) in ["AP object", "DM object", "AP probe", "DM probe"]
            .iter()
            .enumerate()
        {
            assert!(r.3[i] == w.3[i], "{split:?}: {what} sums differ");
        }
    }
}

/// A captured AP and DM stack replayed against an entry with `nf == 0` leaves `psi`, the
/// LLK partials and the object and probe sums of both as they were, bit for bit; with
/// `nf == 1` it does not.
#[test]
fn fused_empty_stack_replay_writes_nothing() {
    let (stream, _, k) = setup();
    let fs = FusedScan::new(&stream, [256, 128], [2, 2]);
    let (s, n) = (&fs.s, fs.s.n());
    let dev = StackTable::device(&stream, &s.d, 3).unwrap();
    let mut bufs = FusedBufs::new(&stream, &s.d, 3).unwrap();
    let mut psi = up(&stream, &noise(s.nk(&(0..3)) * n, 70, 1.0));
    let llk_start: Vec<_> = noise(6, 71, 1.0)
        .chunks(2)
        .map(|c| LlkPartial {
            poisson: c[0].re,
            gaussian: c[0].im,
            euclidean: c[1].re,
            icalc: c[1].im,
        })
        .collect();
    let mut llk = up(&stream, &llk_start);
    let noisy_sums = |len, seed| {
        let den: Vec<f32> = noise(len, seed + 1, 1.0).iter().map(|c| c.re).collect();
        UpdateSums::new(
            up(&stream, &noise(len, seed, 1.0)),
            up(&stream, &den),
            &stream,
        )
        .unwrap()
    };
    let (mut ap_sums, mut dm_sums) = (noisy_sums(s.obj.len(), 72), noisy_sums(s.obj.len(), 74));
    let mut ap_probe = noisy_sums(s.probe.len(), 76);
    let mut dm_probe = noisy_sums(s.probe.len(), 78);
    let side = k.context().new_stream().unwrap();
    let inputs = fs.inputs();
    let graph = CudaGraph::capture(
        side.clone(),
        record(|ctx| {
            let ap = Exit::Ap {
                llk: Some(&mut llk),
            };
            let sums = FusedSums {
                obj: Some(&mut ap_sums),
                probe: Some(&mut ap_probe),
            };
            k.fused_project(ctx, &inputs, dev.at(0), ap, &mut bufs, sums)?;
            let dm = Exit::Dm {
                psi: &mut psi,
                alpha: 0.02,
            };
            let sums = FusedSums {
                obj: Some(&mut dm_sums),
                probe: Some(&mut dm_probe),
            };
            k.fused_project(ctx, &inputs, dev.at(0), dm, &mut bufs, sums)
        }),
    )
    .unwrap();
    let bits = |psi: &DeviceBuffer<C32>,
                llk: &DeviceBuffer<LlkPartial>,
                sums: [&UpdateSums; 4]|
     -> Vec<u32> {
        let mut c: Vec<C32> = psi.to_host_vec(&stream).unwrap();
        for u in sums {
            c.extend(u.num.to_host_vec(&stream).unwrap());
        }
        let mut out: Vec<u32> = c
            .iter()
            .flat_map(|c| [c.re, c.im])
            .map(f32::to_bits)
            .collect();
        for u in sums {
            out.extend(
                u.den
                    .to_host_vec(&stream)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits()),
            );
        }
        out.extend(
            llk.to_host_vec(&stream)
                .unwrap()
                .iter()
                .flat_map(|p| [p.poisson, p.gaussian, p.euclidean, p.icalc].map(f32::to_bits)),
        );
        out
    };
    let start = bits(&psi, &llk, [&ap_sums, &dm_sums, &ap_probe, &dm_probe]);
    for (f0, nf, empty) in [(2, 0, true), (0, 0, true), (4, 0, true), (2, 1, false)] {
        let ctl = [StackCtl { f0, nf }];
        k.context().bind_to_thread().unwrap();
        // SAFETY: `dev` holds one entry; no work on it is in flight.
        unsafe { cuda_core::memory::memcpy_htod_sync(dev.ptr(), ctl.as_ptr(), 8) }.unwrap();
        graph.launch().sync_on(&side).unwrap();
        assert_eq!(
            bits(&psi, &llk, [&ap_sums, &dm_sums, &ap_probe, &dm_probe]) == start,
            empty,
            "entry {{ f0: {f0}, nf: {nf} }}"
        );
    }
}

/// The fused passes take frame sides of 128 or 256 and as many modes as one column block
/// holds, and buffers for at least the table's `nf_max` frames.
#[test]
fn fused_shapes_are_checked() {
    let (stream, ex, k) = setup();
    let base = fused_scan([128, 256], [2, 2]).d;
    let dims = |frame: [u32; 2], nprobe| ViewDims {
        frame,
        nprobe,
        nobj: 1,
        nmode: nprobe,
        obj: frame.map(|s| s + 40),
        ..base
    };
    for (frame, nprobe, ok) in [
        ([64, 128], 16, true),
        ([64, 128], 17, false),
        ([128, 128], 8, true),
        ([128, 128], 9, false),
        ([256, 128], 4, true),
        ([256, 128], 5, false),
        ([512, 128], 2, true),
        ([512, 128], 3, false),
        ([128, 512], 1, true),
        ([24, 32], 1, false),
        ([1024, 128], 1, false),
        ([128, 32], 1, false),
    ] {
        let r = FusedBufs::new(&stream, &dims(frame, nprobe), 2);
        assert_eq!(
            r.is_ok(),
            ok,
            "frame {frame:?}, {nprobe} modes: {:?}",
            r.err()
        );
        if let Err(e) = r {
            assert!(matches!(e, Error::Shape(_)), "{e:?}");
        }
    }
    for &n in FUSED_SIDES {
        let r = FusedBufs::new(&stream, &dims([n, n], 1), 2);
        assert!(r.is_ok(), "side {n}: {:?}", r.err());
    }
    let e = FusedBufs::new(&stream, &dims([24, 32], 1), 2)
        .err()
        .unwrap();
    assert!(format!("{e}").contains(&format!("{FUSED_SIDES:?}")), "{e}");
    assert!(
        matches!(FusedBufs::new(&stream, &base, 0), Err(Error::Shape(_))),
        "nf_max 0"
    );

    let fs = FusedScan::new(&stream, [128, 256], [2, 2]);
    let table = one_stack(&stream, &fs.s.d, 0..3);
    let mut short = FusedBufs::new(&stream, &fs.s.d, 2).unwrap();
    let mut out = DeviceBuffer::<LlkPartial>::zeroed(&stream, 3).unwrap();
    let r = k.fused_llk(&ex, &fs.inputs(), table.at(0), &mut short, &mut out);
    assert!(matches!(r, Err(Error::Shape(_))), "bufs for 2 of 3 frames");
    let mut bufs = FusedBufs::new(&stream, &fs.s.d, 3).unwrap();
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, fs.s.nk(&(0..3)) * fs.s.n() - 1).unwrap();
    let r = k.fused_project(
        &ex,
        &fs.inputs(),
        table.at(0),
        Exit::DmFirst { psi: &mut psi },
        &mut bufs,
        FusedSums::default(),
    );
    assert!(
        matches!(r, Err(Error::Shape(_))),
        "psi short of nf_max frames"
    );
    let mut probe_sums = UpdateSums::zeroed(&stream, fs.s.probe.len() - 1).unwrap();
    let r = k.fused_project(
        &ex,
        &fs.inputs(),
        table.at(0),
        Exit::Ap { llk: None },
        &mut bufs,
        FusedSums {
            obj: None,
            probe: Some(&mut probe_sums),
        },
    );
    assert!(matches!(r, Err(Error::Shape(_))), "probe sums short");
}
