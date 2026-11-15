//! What a multislice object's last slice adds to BH under the position model, against
//! references in f64: [`BhKernels::phase_grad`] and [`BhKernels::phase_tangent`] against the
//! derivatives of the plane phase `OpticsKernels::slice_phase` puts on the exit waves, its
//! rates measured from that kernel; [`BhKernels::pos_second`] against central differences of
//! [`BhKernels::pos_spec`]; the moved probes' entry against the waves that hold them; and the
//! partials' segments.

use std::f64::consts::TAU;

use cuda_core::CudaContext;

use super::*;
use crate::base::C64;
use crate::gpu::{OpticsKernels, SlicePhase, SpectrumBuf, StackTable};

const NFRAME: usize = 5;
const NMODE: usize = 4;
const NPROBE: usize = 2;
const NOBJ: usize = NMODE / NPROBE;
const NF_MAX: usize = 3;
/// The last stack is shorter than `NF_MAX`, so its last frame's planes stay as they were.
const STACKS: [Range<u32>; 2] = [0..3, 3..5];
/// The scan frame at each epoch position.
const ORDER: [u32; NFRAME] = [3, 0, 4, 2, 1];
/// Two chunks of `GAMMA_CHUNK` pixels per frame.
const SHAPE: [usize; 2] = [64, 64];
const N: usize = SHAPE[0] * SHAPE[1];
const CHUNK: usize = GAMMA_CHUNK as usize;
const NCHUNK: usize = N.div_ceil(CHUNK);
const OBJ: [usize; 2] = [96, 96];
/// Values of a stack's fields, `[nf_max * nmode, ny, nx]`.
const FIELD: usize = NF_MAX * NMODE * N;
/// The last slice's plane phase, radians per squared pixel.
const QUAD: f32 = 0.013;
const KINDS: [u32; 3] = [BH_ENTRY_PROBE, BH_ENTRY_MOVED, BH_ENTRY_WAVES];

struct Fixture {
    stream: Arc<CudaStream>,
    ctx: ExecutionContext,
    kernels: Kernels,
    bh: BhKernels,
    optics: OpticsKernels,
    dims: ViewDims,
    table: StackTable,
    order: EpochOrder,
    /// The epoch's views, as `order` holds them.
    views: Vec<ViewAddr>,
    /// The views `order` was set from.
    _src: Views,
}

fn fixture() -> Fixture {
    let cuda = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cuda).unwrap();
    let bh = BhKernels::load(&kernels).unwrap();
    let optics = OpticsKernels::load(&kernels).unwrap();
    let stream = cuda.default_stream();
    let ctx = ExecutionContext::new(stream.clone());
    let dims = ViewDims {
        nframe: NFRAME as u32,
        nmode: NMODE as u32,
        nprobe: NPROBE as u32,
        nobj: NOBJ as u32,
        frame: SHAPE.map(|n| n as u32),
        obj: OBJ.map(|n| n as u32),
        quad_phase: 0.013,
    };
    let addr: Vec<ViewAddr> = (0..NFRAME as u32)
        .flat_map(|f| {
            (0..NMODE as u32).map(move |m| ViewAddr {
                oy: 3 + 5 * f,
                ox: 27 - 4 * f,
                fy: 0.0,
                fx: 0.0,
                probe: m % NPROBE as u32,
                obj: m / NPROBE as u32,
            })
        })
        .collect();
    let src = Views::new(&stream, dims, &addr).unwrap();
    let table = StackTable::new(&stream, &dims, &STACKS).unwrap();
    let mut order = EpochOrder::new(&kernels, &stream, &src).unwrap();
    order.set(&ctx, &kernels, &src, &ORDER).unwrap();
    let views = order.views().addr().to_host_vec(&stream).unwrap();
    Fixture {
        stream,
        ctx,
        kernels,
        bh,
        optics,
        dims,
        table,
        order,
        views,
        _src: src,
    }
}

/// Deterministic values in `[-1, 1)`, exact in f32.
fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn c32s(f: &[f32]) -> Vec<C32> {
    f.chunks(2).map(|p| C32 { re: p[0], im: p[1] }).collect()
}

fn c64(c: C32) -> C64 {
    C64::new(c.re.into(), c.im.into())
}

impl Fixture {
    /// Noise of `len` values on the device, and as the host sees them.
    fn field(&self, len: usize, seed: u64) -> (DeviceBuffer<C32>, Vec<C64>) {
        let host = c32s(&noise(2 * len, seed));
        let dev = DeviceBuffer::from_host(&self.stream, &host).unwrap();
        (dev, host.into_iter().map(c64).collect())
    }

    fn host(&self, b: &DeviceBuffer<C32>) -> Vec<C64> {
        let h = b.to_host_vec(&self.stream).unwrap();
        h.into_iter().map(c64).collect()
    }

    fn floats(&self, b: &DeviceBuffer<C32>) -> Vec<f32> {
        let h = b.to_host_vec(&self.stream).unwrap();
        h.iter().flat_map(|c| [c.re, c.im]).collect()
    }

    fn offsets(&self, d: &[f32]) -> DeviceBuffer<f32> {
        DeviceBuffer::from_host(&self.stream, d).unwrap()
    }
}

/// The stored index of natural pixel `(y, x)`, as `psi_index`.
fn stored(y: usize, x: usize) -> usize {
    (y + SHAPE[0] / 2) % SHAPE[0] * SHAPE[1] + (x + SHAPE[1] / 2) % SHAPE[1]
}

/// The planes of an entry of `kind`.
fn entry_len(kind: u32) -> usize {
    match kind {
        BH_ENTRY_PROBE => NPROBE * N,
        BH_ENTRY_MOVED => NF_MAX * NPROBE * N,
        _ => FIELD,
    }
}

fn entry(kind: u32, b: &DeviceBuffer<C32>) -> BhEntry<'_> {
    match kind {
        BH_ENTRY_PROBE => BhEntry::Probe(b),
        BH_ENTRY_MOVED => BhEntry::Moved(b),
        _ => BhEntry::Waves(b),
    }
}

/// The phase's rates per pixel of offset along `y` and `x` at each stored pixel, as
/// `slice_phase` moves its angle: its phase at a unit offset over that at none.
fn phase_rates(fx: &Fixture) -> [Vec<f64>; 2] {
    let ones = vec![C32 { re: 1.0, im: 0.0 }; FIELD];
    let phase = |o: [f32; 2]| {
        let offsets = fx.offsets(&[o; NFRAME].concat());
        let mut w = DeviceBuffer::from_host(&fx.stream, &ones).unwrap();
        let p = SlicePhase {
            quad: QUAD,
            offsets: Some(&offsets),
        };
        let at = fx.table.at(0);
        fx.optics
            .slice_phase(&fx.ctx, &fx.dims, at, p, &mut w, false)
            .unwrap();
        fx.host(&w)[..N].to_vec()
    };
    let p0 = phase([0.0, 0.0]);
    [[1.0, 0.0], [0.0, 1.0]].map(|o| {
        let p = phase(o);
        (0..N).map(|k| (p[k] * p0[k].conj()).arg()).collect()
    })
}

/// `X = S w`, the slice's exit before the phase, of stack `frames`' planes at stored pixels:
/// `slice` through the epoch's views at whole pixels, `w` the views' waves of an entry of
/// `kind`.
fn exit(fx: &Fixture, frames: &Range<u32>, slice: &[C64], entry: &[C64], kind: u32) -> Vec<C64> {
    let mut x = vec![C64::default(); frames.len() * NMODE * N];
    for kl in 0..frames.len() * NMODE {
        let v = fx.views[frames.start as usize * NMODE + kl];
        assert_eq!((v.fy, v.fx), (0.0, 0.0), "views at whole pixels");
        let (oy, ox, obj, p) = (
            v.oy as usize,
            v.ox as usize,
            v.obj as usize,
            v.probe as usize,
        );
        for d in 0..N {
            let (py, px) = (d / SHAPE[1], d % SHAPE[1]);
            let at = stored(py, px);
            let o = slice[(obj * OBJ[0] + oy + py) * OBJ[1] + ox + px];
            let w = match kind {
                BH_ENTRY_PROBE => entry[p * N + d],
                BH_ENTRY_MOVED => entry[(kl / NMODE * NPROBE + p) * N + at],
                _ => entry[kl * N + at],
            };
            x[kl * N + at] = o * w;
        }
    }
    x
}

/// The first float of chunk `c`'s pair of position `pos`'s segment `seg` of `nseg`.
fn slot(pos: usize, seg: usize, nseg: usize, c: usize) -> usize {
    2 * ((pos * nseg + seg) * NCHUNK + c)
}

/// Panics unless `|got - want| <= tol` at every value, `tol` 0 asking `==`.
fn within(what: &str, got: &[f32], want: &[f64], tol: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for i in 0..got.len() {
        let (g, w) = (got[i] as f64, want[i]);
        let ok = if tol[i] == 0.0 {
            g == w
        } else {
            (g - w).abs() <= tol[i]
        };
        assert!(
            ok,
            "{what}: value {i} is {g:e}, want {w:e} within {:e}",
            tol[i]
        );
    }
}

/// Panics unless `got` is `want` to `rel` of `want`'s largest magnitude.
fn close(what: &str, got: &[C64], want: &[C64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{what}");
    let top = want.iter().map(|w| w.norm()).fold(0.0, f64::max);
    let (i, err) = (0..got.len())
        .map(|i| (i, (got[i] - want[i]).norm()))
        .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
    assert!(
        err <= rel * top,
        "{what}: value {i} is {}, want {} ({err:e} of {top:e})",
        got[i],
        want[i]
    );
}

/// `phase_grad`'s partials into segment 1 of two at `quad`, per entry kind, against `(1 /
/// 2π) Σ Re(conj(r) i s_a X)`: the derivative of `Re Σ conj(r) Φ X` along offset `a`, the
/// phase's angle moving by `s_a` per pixel of it, over `2π`. Segment 0 and the positions of
/// the stacks not yet run stay as they were.
fn phase_grad_case(quad: f32) {
    let fx = fixture();
    let rates = phase_rates(&fx);
    let (slice, slice_h) = fx.field(fx.dims.obj_len(), 1);
    for (k, kind) in KINDS.into_iter().enumerate() {
        let seed = 10 * k as u64 + 10;
        let mut bufs = BhPositions::zeroed(&fx.stream, &fx.dims, 2).unwrap();
        let was = noise(bufs.grads.len(), seed);
        bufs.grads.copy_from_host(&fx.stream, &was).unwrap();
        let mut off = BhOffsets {
            bufs: &mut bufs,
            order: &fx.order,
            scale: 1.0,
            step: 1.0,
        };
        let mut want: Vec<f64> = was.iter().map(|&w| w.into()).collect();
        let mut tol = vec![0.0; want.len()];
        for (s, frames) in STACKS.iter().enumerate() {
            let seed = seed + 3 * s as u64;
            let (e, e_h) = fx.field(entry_len(kind), seed + 1);
            let (r, r_h) = fx.field(FIELD, seed + 2);
            let last = BhLastSlice {
                slice: &slice,
                entry: entry(kind, &e),
                quad,
            };
            let at = fx.table.at(s);
            fx.bh
                .phase_grad(&fx.ctx, at, last, &r, (&mut off, 1))
                .unwrap();
            let x = exit(&fx, frames, &slice_h, &e_h, kind);
            let q = f64::from(quad) / f64::from(QUAD);
            for fl in 0..frames.len() {
                let pos = frames.start as usize + fl;
                for c in 0..NCHUNK {
                    let o = slot(pos, 1, 2, c);
                    for (a, rate) in rates.iter().enumerate() {
                        let (mut sum, mut mag) = (0.0, 0.0);
                        for d in c * CHUNK..N.min((c + 1) * CHUNK) {
                            let at = stored(d / SHAPE[1], d % SHAPE[1]);
                            for m in 0..NMODE {
                                let j = (fl * NMODE + m) * N + at;
                                let t = r_h[j].conj() * C64::i() * (q * rate[at]) * x[j];
                                sum += t.re;
                                mag += t.norm();
                            }
                        }
                        want[o + a] = sum / TAU;
                        // Exact zeros without the phase; else f32 sums over the chunk's
                        // pixels and modes.
                        tol[o + a] = 1e-5 * mag / TAU;
                    }
                }
            }
            let got = off.bufs.grads.to_host_vec(&fx.stream).unwrap();
            within(&format!("kind {kind} after stack {s}"), &got, &want, &tol);
        }
    }
}

#[test]
fn phase_grad_is_the_phase_derivative() {
    phase_grad_case(QUAD);
}

/// Without the phase its segment is zeros, its rates' sums of zeros.
#[test]
fn phase_grad_is_zeros_without_the_phase() {
    phase_grad_case(0.0);
}

/// `phase_tangent` at `quad`, per entry kind, against the derivatives of `exp(i (t S_u + τ
/// S_v)) (X + t l_u + τ l_v + t² m_uu / 2 + t τ m_uv + τ² m_vv / 2)` at 0 by central
/// differences in f64: `Φ ψ` dephased, `S_u`, `S_v` the phase's angle along `u`'s and `v`'s
/// offsets of the measured rates. Planes past the stack's frames, and without CG `lv`, `muv`
/// and `mvv`, stay as they were.
fn phase_tangent_case(quad: f32, cg: bool) {
    let fx = fixture();
    let rates = phase_rates(&fx);
    let (slice, slice_h) = fx.field(fx.dims.obj_len(), 1);
    let mut bufs = BhPositions::zeroed(&fx.stream, &fx.dims, 1).unwrap();
    let ud: Vec<f32> = noise(2 * NFRAME, 2).iter().map(|d| 0.5 * d).collect();
    let prev: Vec<f32> = noise(2 * NFRAME, 3).iter().map(|d| 0.5 * d).collect();
    bufs.ud.copy_from_host(&fx.stream, &ud).unwrap();
    bufs.prev.copy_from_host(&fx.stream, &prev).unwrap();
    let off = BhOffsets {
        bufs: &mut bufs,
        order: &fx.order,
        scale: 1.0,
        step: 1.0,
    };
    let [u, v] = [4, 5].map(|s| [fx.field(fx.dims.obj_len(), s).0, fx.field(NPROBE * N, s).0]);
    let d = BhDisp {
        u: [&u[0], &u[1]],
        u_scale: 0.7,
        v: cg.then_some([&v[0], &v[1]]),
    };
    let q = f64::from(quad) / f64::from(QUAD);
    for (k, kind) in KINDS.into_iter().enumerate() {
        for (s, frames) in STACKS.iter().enumerate() {
            let seed = 100 * k as u64 + 10 * s as u64 + 10;
            let (e, e_h) = fx.field(entry_len(kind), seed);
            let fields = [1, 2, 3, 4, 5].map(|i| fx.field(FIELD, seed + i));
            let [mut lu, mut lv, muu, muv, mvv] = fields.each_ref().map(|f| f.0.clone_on(&fx));
            let mut m = [muu, muv, mvv];
            let last = BhLastSlice {
                slice: &slice,
                entry: entry(kind, &e),
                quad,
            };
            let mut t = BhTangentsMut {
                lu: &mut lu,
                lv: &mut lv,
                m: &mut m,
            };
            let at = fx.table.at(s);
            fx.bh
                .phase_tangent(&fx.ctx, at, last, &off, &d, &mut t)
                .unwrap();
            let x = exit(&fx, frames, &slice_h, &e_h, kind);
            let [lu0, lv0, muu0, muv0, mvv0] = fields.each_ref().map(|f| &f.1);
            let mut want = [lu0, lv0, muu0, muv0, mvv0].map(|w| w.clone());
            for j in 0..frames.len() * NMODE * N {
                let (kl, at) = (j / N, j % N);
                let pos = frames.start as usize + kl / NMODE;
                let frame = ORDER[pos] as usize;
                let angle = |dir: &[f32], p: usize| {
                    q * (f64::from(dir[2 * p]) * rates[0][at]
                        + f64::from(dir[2 * p + 1]) * rates[1][at])
                };
                let su = angle(&ud, pos);
                let sv = if cg { angle(&prev, frame) } else { 0.0 };
                let psi = |t: f64, w: f64| {
                    let taylor = x[j]
                        + lu0[j] * t
                        + lv0[j] * w
                        + muu0[j] * (t * t / 2.0)
                        + muv0[j] * (t * w)
                        + mvv0[j] * (w * w / 2.0);
                    C64::from_polar(1.0, t * su + w * sv) * taylor
                };
                let h = 1e-3;
                let p0 = psi(0.0, 0.0);
                want[0][j] = (psi(h, 0.0) - psi(-h, 0.0)) / (2.0 * h);
                want[2][j] = (psi(h, 0.0) - 2.0 * p0 + psi(-h, 0.0)) / (h * h);
                if cg {
                    want[1][j] = (psi(0.0, h) - psi(0.0, -h)) / (2.0 * h);
                    let cross = psi(h, h) - psi(h, -h) - psi(-h, h) + psi(-h, -h);
                    want[3][j] = cross / (4.0 * h * h);
                    want[4][j] = (psi(0.0, h) - 2.0 * p0 + psi(0.0, -h)) / (h * h);
                }
            }
            let got = [&lu, &lv, &m[0], &m[1], &m[2]].map(|b| fx.host(b));
            let names = ["lu", "lv", "muu", "muv", "mvv"];
            let written = frames.len() * NMODE * N;
            for b in 0..5 {
                let what = format!("kind {kind} stack {s} {}", names[b]);
                let moved = b == 0 || b == 2 || cg;
                if moved && quad != 0.0 {
                    close(&what, &got[b][..written], &want[b][..written], 1e-5);
                } else {
                    assert!(got[b][..written] == fields[b].1[..written], "{what} moved");
                }
                assert!(
                    got[b][written..] == fields[b].1[written..],
                    "{what}: past the frames"
                );
            }
        }
    }
}

trait CloneOn {
    fn clone_on(&self, fx: &Fixture) -> Self;
}

impl CloneOn for DeviceBuffer<C32> {
    fn clone_on(&self, fx: &Fixture) -> Self {
        let h = self.to_host_vec(&fx.stream).unwrap();
        DeviceBuffer::from_host(&fx.stream, &h).unwrap()
    }
}

#[test]
fn phase_tangent_is_the_phase_derivative() {
    phase_tangent_case(QUAD, false);
}

#[test]
fn phase_tangent_is_the_phase_derivative_cg() {
    phase_tangent_case(QUAD, true);
}

/// Without the phase the derivatives stay as they were (`==`, so ±0 alike).
#[test]
fn phase_tangent_leaves_the_derivatives_without_the_phase() {
    phase_tangent_case(0.0, true);
}

/// The step's displacement and the probes one `pos_second` case reads.
struct Moves {
    u: [DeviceBuffer<C32>; 2],
    v: [DeviceBuffer<C32>; 2],
    /// The estimate's probe, `u`'s and `v`'s parts, as the host holds them.
    probes: [Vec<C32>; 3],
    /// The estimate's offsets by epoch position.
    offsets: Vec<f32>,
    /// `u`'s offsets by epoch position, `v`'s by scan frame.
    ud: Vec<f32>,
    prev: Vec<f32>,
}

const U_SCALE: f32 = 0.7;

fn moves(fx: &Fixture) -> Moves {
    let obj = fx.dims.obj_len();
    let probes = [5, 2, 4].map(|s| c32s(&noise(2 * NPROBE * N, s)));
    let up = |p: &Vec<C32>| DeviceBuffer::from_host(&fx.stream, p).unwrap();
    Moves {
        u: [fx.field(obj, 1).0, up(&probes[1])],
        v: [fx.field(obj, 3).0, up(&probes[2])],
        offsets: noise(2 * NFRAME, 6),
        ud: noise(2 * NFRAME, 7).iter().map(|d| 0.3 * d).collect(),
        prev: noise(2 * NFRAME, 8).iter().map(|d| 0.3 * d).collect(),
        probes,
    }
}

/// `pos_spec`'s `(lu, lv)` of stack `s` along `mv`'s displacement, for the probes moved by
/// `offsets` (by epoch position) of the estimate's `probe`.
fn spec_at(
    fx: &Fixture,
    mv: &Moves,
    cg: bool,
    s: usize,
    offsets: &[f32],
    probe: &[C32],
) -> [Vec<C64>; 2] {
    let mut bufs = BhPositions::zeroed(&fx.stream, &fx.dims, 1).unwrap();
    bufs.ud.copy_from_host(&fx.stream, &mv.ud).unwrap();
    bufs.prev.copy_from_host(&fx.stream, &mv.prev).unwrap();
    let off = BhOffsets {
        bufs: &mut bufs,
        order: &fx.order,
        scale: 1.0,
        step: 1.0,
    };
    let d = BhDisp {
        u: [&mv.u[0], &mv.u[1]],
        u_scale: U_SCALE,
        v: cg.then_some([&mv.v[0], &mv.v[1]]),
    };
    let (ctx, dims) = (&fx.ctx, &fx.dims);
    let probe = DeviceBuffer::from_host(&fx.stream, probe).unwrap();
    let mut bufs = [0; 3].map(|_| SpectrumBuf::zeroed(&fx.stream, dims).unwrap());
    let [ba, bu, bv] = &mut bufs;
    let a = fx.kernels.probe_spectrum(ctx, dims, &probe, ba).unwrap();
    let au = fx.kernels.probe_spectrum(ctx, dims, &mv.u[1], bu).unwrap();
    let av = fx.kernels.probe_spectrum(ctx, dims, &mv.v[1], bv).unwrap();
    let offsets = fx.offsets(offsets);
    let shift = Shift {
        offsets: &offsets,
        spectrum: a,
    };
    let mvr = BhMovesRef::new(&d, shift, au, cg.then_some(av), &off).unwrap();
    let [mut lu, mut lv] = [0; 2].map(|_| fx.field(FIELD, 9).0);
    fx.bh
        .pos_spec(ctx, fx.table.at(s), &mvr, [&mut lu, &mut lv])
        .unwrap();
    [fx.host(&lu), fx.host(&lv)]
}

/// `pos_second` against central differences of `pos_spec` along each direction: the moved
/// probes' spectra `e(o + t d) (A + t a)` change to first order as `pos_spec` writes, so its
/// change along `(t_u, t_v)` is their second order. Without CG `muv` and `mvv` stay as they
/// were, as the planes past the stack's frames do.
fn pos_second_case(cg: bool) {
    let fx = fixture();
    let mv = moves(&fx);
    // `v`'s offsets by epoch position.
    let vd: Vec<f32> = (0..NFRAME)
        .flat_map(|p| {
            let f = ORDER[p] as usize;
            [mv.prev[2 * f], mv.prev[2 * f + 1]]
        })
        .collect();
    let h = 1e-2f32;
    let along = |dir: &[f32], part: &[C32], t: f32| {
        let o: Vec<f32> = (0..2 * NFRAME)
            .map(|i| mv.offsets[i] + t * dir[i])
            .collect();
        let p: Vec<C32> = (0..NPROBE * N)
            .map(|i| C32 {
                re: mv.probes[0][i].re + t * part[i].re,
                im: mv.probes[0][i].im + t * part[i].im,
            })
            .collect();
        (o, p)
    };
    let scaled: Vec<C32> = mv.probes[1]
        .iter()
        .map(|c| C32 {
            re: U_SCALE * c.re,
            im: U_SCALE * c.im,
        })
        .collect();
    let diff = |dir: &[f32], part: &[C32], s: usize, l: usize| -> Vec<C64> {
        let (op, pp) = along(dir, part, h);
        let (om, pm) = along(dir, part, -h);
        let (plus, minus) = (
            &spec_at(&fx, &mv, cg, s, &op, &pp)[l],
            &spec_at(&fx, &mv, cg, s, &om, &pm)[l],
        );
        (0..FIELD)
            .map(|i| (plus[i] - minus[i]) / (2.0 * f64::from(h)))
            .collect()
    };
    let mut bufs = BhPositions::zeroed(&fx.stream, &fx.dims, 1).unwrap();
    bufs.ud.copy_from_host(&fx.stream, &mv.ud).unwrap();
    bufs.prev.copy_from_host(&fx.stream, &mv.prev).unwrap();
    let off = BhOffsets {
        bufs: &mut bufs,
        order: &fx.order,
        scale: 1.0,
        step: 1.0,
    };
    let d = BhDisp {
        u: [&mv.u[0], &mv.u[1]],
        u_scale: U_SCALE,
        v: cg.then_some([&mv.v[0], &mv.v[1]]),
    };
    let (ctx, dims) = (&fx.ctx, &fx.dims);
    let probe = DeviceBuffer::from_host(&fx.stream, &mv.probes[0]).unwrap();
    let mut spectra = [0; 3].map(|_| SpectrumBuf::zeroed(&fx.stream, dims).unwrap());
    let [ba, bu, bv] = &mut spectra;
    let a = fx.kernels.probe_spectrum(ctx, dims, &probe, ba).unwrap();
    let au = fx.kernels.probe_spectrum(ctx, dims, &mv.u[1], bu).unwrap();
    let av = fx.kernels.probe_spectrum(ctx, dims, &mv.v[1], bv).unwrap();
    let offsets = fx.offsets(&mv.offsets);
    let shift = Shift {
        offsets: &offsets,
        spectrum: a,
    };
    let mvr = BhMovesRef::new(&d, shift, au, cg.then_some(av), &off).unwrap();
    for (s, frames) in STACKS.iter().enumerate() {
        let fields = [1, 2, 3].map(|i| fx.field(FIELD, 20 + 10 * s as u64 + i));
        let mut m = fields.each_ref().map(|f| f.0.clone_on(&fx));
        fx.bh.pos_second(ctx, fx.table.at(s), &mvr, &mut m).unwrap();
        let got = m.each_ref().map(|b| fx.host(b));
        let written = frames.len() * NMODE * N;
        let want = [
            Some(diff(&mv.ud, &scaled, s, 0)),
            cg.then(|| diff(&mv.ud, &scaled, s, 1)),
            cg.then(|| diff(&vd, &mv.probes[2], s, 1)),
        ];
        for (b, name) in ["muu", "muv", "mvv"].into_iter().enumerate() {
            let what = format!("stack {s} {name}");
            match &want[b] {
                Some(w) => close(&what, &got[b][..written], &w[..written], 1e-3),
                None => assert!(
                    got[b][..written] == fields[b].1[..written],
                    "{what} written"
                ),
            }
            assert!(
                got[b][written..] == fields[b].1[written..],
                "{what}: past the frames"
            );
        }
    }
}

#[test]
fn pos_second_is_the_change_of_pos_spec() {
    pos_second_case(false);
}

#[test]
fn pos_second_is_the_change_of_pos_spec_cg() {
    pos_second_case(true);
}

/// `slice_tangent` entered by the moved probes as by the waves that hold each view's: the
/// same values bit for bit.
fn moved_entry_case(cg: bool) {
    let fx = fixture();
    let obj = fx.dims.obj_len();
    let [u, v] = [1, 3].map(|s| [fx.field(obj, s).0, fx.field(NPROBE * N, s + 1).0]);
    let d = BhDisp {
        u: [&u[0], &u[1]],
        u_scale: U_SCALE,
        v: cg.then_some([&v[0], &v[1]]),
    };
    let slice = fx.field(obj, 5).0;
    for (s, frames) in STACKS.iter().enumerate() {
        let seed = 10 * s as u64 + 10;
        let moved = c32s(&noise(2 * entry_len(BH_ENTRY_MOVED), seed));
        let waves: Vec<C32> = (0..FIELD)
            .map(|i| {
                let (kl, at) = (i / N, i % N);
                let v = fx.views[(frames.start as usize * NMODE + kl).min(NFRAME * NMODE - 1)];
                moved[(kl / NMODE * NPROBE + v.probe as usize) * N + at]
            })
            .collect();
        let fields = [1, 2, 3, 4, 5].map(|i| fx.field(FIELD, seed + i).0);
        let run = |e: &[C32], kind: u32| {
            let e = DeviceBuffer::from_host(&fx.stream, e).unwrap();
            let [mut lu, mut lv, muu, muv, mvv] = fields.each_ref().map(|f| f.clone_on(&fx));
            let mut m = [muu, muv, mvv];
            let mut t = BhTangentsMut {
                lu: &mut lu,
                lv: &mut lv,
                m: &mut m,
            };
            let views = fx.order.views();
            fx.bh
                .slice_tangent(
                    &fx.ctx,
                    views,
                    fx.table.at(s),
                    (&slice, entry(kind, &e)),
                    &d,
                    &mut t,
                )
                .unwrap();
            [&lu, &lv, &m[0], &m[1], &m[2]].map(|b| fx.floats(b))
        };
        let (got, want) = (run(&moved, BH_ENTRY_MOVED), run(&waves, BH_ENTRY_WAVES));
        for (b, name) in ["lu", "lv", "muu", "muv", "mvv"].into_iter().enumerate() {
            assert!(got[b] == want[b], "stack {s} {name}");
        }
        assert_ne!(got[0], fx.floats(&fields[0]), "stack {s}: lu not written");
    }
}

#[test]
fn moved_entry_is_the_waves_holding_it() {
    moved_entry_case(false);
}

#[test]
fn moved_entry_is_the_waves_holding_it_cg() {
    moved_entry_case(true);
}

/// A segment past the partials' last is refused, by `partials` and by each kernel that
/// writes one, before anything is written.
#[test]
fn partials_refuse_a_segment_past_the_last() {
    let fx = fixture();
    assert!(BhPositions::zeroed(&fx.stream, &fx.dims, 0).is_err());
    let mut bufs = BhPositions::zeroed(&fx.stream, &fx.dims, 2).unwrap();
    assert_eq!(bufs.partials(1).unwrap().seg, 1);
    assert!(bufs.partials(2).is_err());
    let was = noise(bufs.grads.len(), 1);
    bufs.grads.copy_from_host(&fx.stream, &was).unwrap();
    let mut off = BhOffsets {
        bufs: &mut bufs,
        order: &fx.order,
        scale: 1.0,
        step: 1.0,
    };
    let (ctx, at) = (&fx.ctx, fx.table.at(0));
    let [r, w, probe] = [FIELD, 3 * NOBJ * NF_MAX * N, NPROBE * N].map(|n| fx.field(n, 2).0);
    let slice = fx.field(fx.dims.obj_len(), 3).0;
    let past = |what: &str, e: Result<(), Error>| match e {
        Err(Error::Shape(m)) if m == "segment 2 of 2" => {}
        e => panic!("{what}: {e:?}"),
    };
    let last = BhLastSlice {
        slice: &slice,
        entry: BhEntry::Probe(&probe),
        quad: QUAD,
    };
    past(
        "phase_grad",
        fx.bh.phase_grad(ctx, at, last, &r, (&mut off, 2)),
    );
    let mut spectrum = SpectrumBuf::zeroed(&fx.stream, &fx.dims).unwrap();
    let a = fx
        .kernels
        .probe_spectrum(ctx, &fx.dims, &probe, &mut spectrum)
        .unwrap();
    let offsets = fx.offsets(&[0.0; 2 * NFRAME]);
    let shift = Shift {
        offsets: &offsets,
        spectrum: a,
    };
    past(
        "pos_grad",
        fx.bh.pos_grad(ctx, at, &r, &shift, (&mut off, 2)),
    );
    let f = BhWinRef {
        r: &r,
        probe: &probe,
    };
    let mut patch = fx.field(NOBJ * NF_MAX * N, 4).0;
    let e = fx.bh.win_grad(ctx, at, f, (&w, 2), &mut patch, &mut off);
    past("win_grad", e);
    assert_eq!(off.bufs.grads.to_host_vec(&fx.stream).unwrap(), was);
}

/// `pos_dots` over partials of two segments: each position's `ud = scale 2π Σ_c grads` over
/// both segments' chunks, summed in f32 in order, and the offsets' pair `(Σ g ud, Σ g v_d)`;
/// positions past the batch's keep their `ud`.
#[test]
fn pos_dots_sums_every_segment() {
    let fx = fixture();
    let nseg = 2;
    let mut bufs = BhPositions::zeroed(&fx.stream, &fx.dims, nseg).unwrap();
    let grads = noise(bufs.grads.len(), 1);
    let (ud0, prev) = (noise(2 * NFRAME, 2), noise(2 * NFRAME, 3));
    bufs.grads.copy_from_host(&fx.stream, &grads).unwrap();
    bufs.ud.copy_from_host(&fx.stream, &ud0).unwrap();
    bufs.prev.copy_from_host(&fx.stream, &prev).unwrap();
    let scale = 0.37f32;
    let mut off = BhOffsets {
        bufs: &mut bufs,
        order: &fx.order,
        scale,
        step: 1.0,
    };
    let obj = fx.dims.obj_len();
    let [u, v] = [1, 3].map(|s| [fx.field(obj, s).0, fx.field(NPROBE * N, s + 1).0]);
    let d = BhDisp {
        u: [&u[0], &u[1]],
        u_scale: U_SCALE,
        v: Some([&v[0], &v[1]]),
    };
    let frames = 1..4u32;
    let batch = BhBatch {
        frames: frames.clone(),
        frac: 1.0,
        record: 0,
        positions: true,
    };
    let mut parts = BhParts::zeroed(&fx.stream, &fx.dims, 0).unwrap();
    fx.bh
        .pos_dots(&fx.ctx, &batch, &d, &mut off, &mut parts)
        .unwrap();
    let mut want = ud0.clone();
    let (mut dots, mut mag) = ([0.0f64; 2], [0.0f64; 2]);
    for p in frames.map(|p| p as usize) {
        for a in 0..2 {
            let mut g = 0.0f32;
            for c in 0..nseg * NCHUNK {
                g += grads[slot(p, 0, nseg, c) + a];
            }
            g *= 2.0 * std::f32::consts::PI;
            want[2 * p + a] = scale * g;
            let vd = prev[2 * ORDER[p] as usize + a];
            for (k, t) in [g * want[2 * p + a], g * vd].into_iter().enumerate() {
                dots[k] += f64::from(t);
                mag[k] += f64::from(t.abs());
            }
        }
    }
    assert_eq!(off.bufs.ud.to_host_vec(&fx.stream).unwrap(), want, "ud");
    let host = parts.buf().to_host_vec(&fx.stream).unwrap();
    let pair = &host[parts.pos()..parts.pos() + 2];
    let tol = mag.map(|m| 1e-6 * m);
    within("offsets' pair", pair, &dots, &tol);
}

/// Each kernel that writes partials, into segment 1 of two, writes the stack's rows as into
/// the single segment, bit for bit, and leaves segment 0 and the other stack's rows as they
/// were.
#[test]
fn a_segment_is_the_single_slot() {
    let fx = fixture();
    let (ctx, at) = (&fx.ctx, fx.table.at(0));
    let blk = NOBJ * NF_MAX * N;
    let fields = [(FIELD, 1), (FIELD, 2), (3 * blk, 3), (NPROBE * N, 4)];
    let [y, r, w, probe] = fields.map(|(n, s)| fx.field(n, s).0);
    let slice = fx.field(fx.dims.obj_len(), 5).0;
    let mut spectrum = SpectrumBuf::zeroed(&fx.stream, &fx.dims).unwrap();
    let a = fx
        .kernels
        .probe_spectrum(ctx, &fx.dims, &probe, &mut spectrum)
        .unwrap();
    let offsets = fx.offsets(&noise(2 * NFRAME, 6));
    let shift = Shift {
        offsets: &offsets,
        spectrum: a,
    };
    let mut patch = fx.field(blk, 7).0;
    let run = |nseg: usize, seg: usize, which: usize, patch: &mut DeviceBuffer<C32>| {
        let mut bufs = BhPositions::zeroed(&fx.stream, &fx.dims, nseg).unwrap();
        let was = noise(bufs.grads.len(), 8);
        bufs.grads.copy_from_host(&fx.stream, &was).unwrap();
        let mut off = BhOffsets {
            bufs: &mut bufs,
            order: &fx.order,
            scale: 1.0,
            step: 1.0,
        };
        match which {
            0 => fx.bh.pos_grad(ctx, at, &y, &shift, (&mut off, seg)),
            1 => {
                let f = BhWinRef {
                    r: &r,
                    probe: &probe,
                };
                fx.bh.win_grad(ctx, at, f, (&w, seg), patch, &mut off)
            }
            _ => {
                let last = BhLastSlice {
                    slice: &slice,
                    entry: BhEntry::Probe(&probe),
                    quad: QUAD,
                };
                fx.bh.phase_grad(ctx, at, last, &r, (&mut off, seg))
            }
        }
        .unwrap();
        (off.bufs.grads.to_host_vec(&fx.stream).unwrap(), was)
    };
    for (which, name) in ["pos_grad", "win_grad", "phase_grad"]
        .into_iter()
        .enumerate()
    {
        let (one, _) = run(1, 0, which, &mut patch);
        let (two, was) = run(2, 1, which, &mut patch);
        let row = 2 * NCHUNK;
        for pos in 0..NFRAME {
            let [seg0, seg1] = [0, 1].map(|s| &two[slot(pos, s, 2, 0)..][..row]);
            let [was0, was1] = [0, 1].map(|s| &was[slot(pos, s, 2, 0)..][..row]);
            let want = if STACKS[0].contains(&(pos as u32)) {
                &one[pos * row..][..row]
            } else {
                was1
            };
            assert!(seg1 == want, "{name}: position {pos}'s segment 1");
            assert!(seg0 == was0, "{name}: position {pos}'s segment 0 written");
        }
        assert_ne!(one, noise(one.len(), 8), "{name}: nothing written");
    }
}
