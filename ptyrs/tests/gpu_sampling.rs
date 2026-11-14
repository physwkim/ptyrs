//! The device full-object sampler, `ptyrs::gpu::sampling`, against the CPU one read through
//! `SliceWaves` under `Sampling::FullObject` with a probe of ones: odd object sides, corners
//! at both extremes, fractions, offsets that carry a window past the plane's edge, two slices,
//! two probe and two object modes, stacks of 3 and 2 frames in passes of 4 groups.
//! [`window_views_as_objects`] reads the windows' layout through the per-view kernels.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs::base::{C64, Geometry, Object, Probe, Stack, View};
use ptyrs::engine::{Optics, Sampling, SliceWaves, shift};
use ptyrs::gpu::sampling::{Modes, ObjectSampler, StackPoints, WindowBlock};
use ptyrs::gpu::{C32, Kernels, StackTable, ViewAddr, ViewDims, Views};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const OBJ: [usize; 2] = [23, 21];
const FRAME: [usize; 2] = [8, 6];
const NPROBE: usize = 2;
const NOBJ: usize = 2;
const NMODE: usize = NPROBE * NOBJ;
const NSLICE: u32 = 2;
const CHUNK: u32 = 4;
const NF_MAX: usize = 3;
const N: usize = FRAME[0] * FRAME[1];
const PLANE: usize = OBJ[0] * OBJ[1];

fn stacks() -> [Range<u32>; 2] {
    [0..3, 3..5]
}

/// Each frame's corner, its fraction and its offset.
const FRAMES: [([u32; 2], [f32; 2], [f32; 2]); 5] = [
    // Past the plane's first row and column.
    ([0, 0], [0.0, 0.0], [-2.6, 1.3]),
    // The last corner, past the last row and column.
    ([15, 15], [0.0, 0.0], [3.4, 2.2]),
    // The last corner a fraction allows.
    ([14, 14], [0.5, 0.75], [0.0, 0.0]),
    ([7, 3], [0.3, 0.0], [1.7, -3.9]),
    ([0, 14], [0.0, 0.25], [-0.4, 0.6]),
];

struct Gpu {
    stream: Arc<CudaStream>,
    ctx: ExecutionContext,
    views: Views,
    table: StackTable,
    offsets: DeviceBuffer<f32>,
}

impl Gpu {
    /// The scan on the device and its sampler.
    fn new() -> (Self, ObjectSampler) {
        let cuda = CudaContext::new(0).expect("CUDA device 0");
        let kernels = Kernels::load(&cuda).expect("load ptyrs-gpu kernels");
        let stream = cuda.default_stream();
        let dims = ViewDims {
            nframe: FRAMES.len() as u32,
            nmode: NMODE as u32,
            nprobe: NPROBE as u32,
            nobj: NOBJ as u32,
            frame: FRAME.map(|n| n as u32),
            obj: OBJ.map(|n| n as u32),
            quad_phase: 0.0,
        };
        let addr: Vec<ViewAddr> = (FRAMES.iter())
            .flat_map(|&([oy, ox], [fy, fx], _)| {
                (0..NMODE as u32).map(move |m| ViewAddr {
                    oy,
                    ox,
                    fy,
                    fx,
                    probe: m % NPROBE as u32,
                    obj: m / NPROBE as u32,
                })
            })
            .collect();
        let offsets: Vec<f32> = FRAMES.iter().flat_map(|f| f.2).collect();
        let sampler = ObjectSampler::new(&kernels, &stream, dims, NSLICE, CHUNK).unwrap();
        let gpu = Self {
            views: Views::new(&stream, dims, &addr).unwrap(),
            table: StackTable::new(&stream, &dims, &stacks()).unwrap(),
            offsets: up(&stream, &offsets),
            ctx: ExecutionContext::new(stream.clone()),
            stream,
        };
        (gpu, sampler)
    }

    fn at(&self, s: usize) -> StackPoints<'_> {
        StackPoints {
            views: &self.views,
            stack: self.table.at(s),
            offsets: &self.offsets,
        }
    }

    fn down<T: DeviceCopy>(&self, buf: &DeviceBuffer<T>) -> Vec<T> {
        buf.to_host_vec(&self.stream).unwrap()
    }
}

fn up<T: DeviceCopy>(stream: &CudaStream, host: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(stream, host).unwrap()
}

/// Deterministic values in `[-1, 1)`.
fn noise(len: usize, seed: u64) -> Vec<C32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    (0..len)
        .map(|_| C32 {
            re: next(),
            im: next(),
        })
        .collect()
}

fn c64(c: C32) -> C64 {
    C64::new(c.re as f64, c.im as f64)
}

/// Largest `|got - want|` over the largest `|want|`.
fn rel(got: &[C64], want: &[C64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = worst_of(want.iter().map(|w| w.norm()));
    worst_of(got.iter().zip(want).map(|(g, w)| (g - w).norm())) / scale
}

/// The CPU views of `frames`, frame-major, their positions `corner + fraction + offset +
/// nudge`, and the far field's one-slice optics.
fn cpu_scan(frames: Range<usize>, nudge: [f64; 2]) -> (Vec<View>, Vec<[f64; 2]>, Optics) {
    let views = (frames.clone())
        .flat_map(|f| {
            let [oy, ox] = FRAMES[f].0;
            (0..NMODE as u32).map(move |m| View {
                frame: f as u32,
                mode: m,
                probe: m % NPROBE as u32,
                obj: m / NPROBE as u32,
                oy,
                ox,
            })
        })
        .collect();
    let offsets = (FRAMES[frames].iter())
        .map(|&(_, fr, d)| [0, 1].map(|a| fr[a] as f64 + d[a] as f64 + nudge[a]))
        .collect();
    let geometry = Geometry::new(1e-10, 1.0, 75e-6, FRAME).unwrap();
    (views, offsets, Optics::far_field(&geometry))
}

fn ones() -> Probe {
    Probe::new(Stack::filled(
        [NPROBE, FRAME[0], FRAME[1]],
        C64::new(1.0, 0.0),
    ))
}

/// The CPU sampler's window of `object` (one slice) per view of `frames`, natural layout:
/// the exit waves of a probe of ones with the far field's phase divided out.
fn cpu_windows(object: &Object, frames: Range<usize>, nudge: [f64; 2]) -> Vec<C64> {
    let (views, offsets, optics) = cpu_scan(frames, nudge);
    let objects = std::slice::from_ref(object);
    let probe = ones();
    let w = SliceWaves::with_sampling(
        &optics,
        &views,
        objects,
        &probe,
        Some(&offsets),
        Sampling::FullObject,
    );
    let mut out = Vec::with_capacity(views.len() * N);
    for (k, psi) in w.psi().as_chunks::<N>().0.iter().enumerate() {
        let q = w.quad(0, k);
        out.extend((0..N).map(|i| psi[shift(i, FRAME[0], FRAME[1])] / q[i]));
    }
    out
}

/// The CPU sampler's adjoint of `patches`, a natural-layout window per view of `frames`,
/// onto `[nobj, nyo, nxo]`.
fn cpu_place(patches: &[C64], frames: Range<usize>) -> Vec<C64> {
    let (views, offsets, optics) = cpu_scan(frames, [0.0; 2]);
    let object = Object::new(Stack::filled([NOBJ, OBJ[0], OBJ[1]], C64::default()));
    let probe = ones();
    let w = SliceWaves::with_sampling(
        &optics,
        &views,
        std::slice::from_ref(&object),
        &probe,
        Some(&offsets),
        Sampling::FullObject,
    );
    let mut chi = vec![C64::default(); patches.len()];
    let planes = chi.as_chunks_mut::<N>().0.iter_mut();
    for (k, (c, p)) in planes.zip(patches.as_chunks::<N>().0).enumerate() {
        let q = w.quad(0, k);
        (0..N).for_each(|i| c[shift(i, FRAME[0], FRAME[1])] = p[i] / q[i].conj());
    }
    w.obj_adjoint(0, &chi).into_vec()
}

/// Slice `s` of a `[nslice, nobj, nyo, nxo]` object as a CPU object.
fn slice_object(obj: &[C32], s: usize) -> Object {
    let len = NOBJ * PLANE;
    let data = obj[s * len..][..len].iter().copied().map(c64).collect();
    Object::new(Stack::from_vec([NOBJ, OBJ[0], OBJ[1]], data).unwrap())
}

/// The plane of a `[nobj, nf_max, ny, nx]` block that view `k` of a stack reads: view `fl *
/// nmode + m` reads plane `m / nprobe * nf_max + fl`.
fn plane_of(k: usize) -> usize {
    k % NMODE / NPROBE * NF_MAX + k / NMODE
}

/// The planes of an all-modes block that the views of `nf` frames read, in view order.
fn per_view(block: &[C32], nf: usize) -> Vec<C64> {
    (0..nf * NMODE)
        .flat_map(|k| block[plane_of(k) * N..][..N].iter().copied().map(c64))
        .collect()
}

/// Per view of `nf` frames, its group's patch of a `[nobj, nf_max, ny, nx]` buffer on the
/// group's first view and zero on the others, whose `W^H` is the group's.
fn group_patches(patches: &[C32], nf: usize) -> Vec<C64> {
    (0..nf * NMODE)
        .flat_map(|k| {
            let first = k % NPROBE == 0;
            let plane = &patches[plane_of(k) * N..][..N];
            plane
                .iter()
                .map(move |&c| if first { c64(c) } else { C64::default() })
        })
        .collect()
}

/// The CPU values of the views that read a [`Modes::First`] block: each frame's first.
fn first_views(per_view: &[C64]) -> Vec<C64> {
    let frames = per_view.as_chunks::<{ NMODE * N }>().0;
    frames.iter().flat_map(|f| f[..N].to_vec()).collect()
}

const SENTINEL: C32 = C32 { re: 7.0, im: -3.0 };

/// Every plane of `win`'s `[modes, nf_max, ny, nx]` blocks still holds the sentinel but those
/// of the first `nf` frames in the `written` blocks.
fn untouched(win: &[C32], written: &[usize], nf: usize, modes: usize) -> bool {
    let len = modes * NF_MAX * N;
    win.chunks_exact(len).enumerate().all(|(b, block)| {
        block
            .as_chunks::<N>()
            .0
            .iter()
            .enumerate()
            .all(|(q, plane)| {
                (written.contains(&b) && q % NF_MAX < nf) || plane.iter().all(|&c| c == SENTINEL)
            })
    })
}

#[test]
fn windows_match_the_cpu_sampler_on_every_slice() {
    let (g, mut sampler) = Gpu::new();
    let host = noise(NSLICE as usize * NOBJ * PLANE, 3);
    let obj = up(&g.stream, &host);
    let mut spectra = DeviceBuffer::zeroed(&g.stream, host.len()).unwrap();
    sampler.spectra(&g.ctx, &obj, &mut spectra).unwrap();
    let len = NF_MAX * NOBJ * N;
    for (s, frames) in stacks().into_iter().enumerate() {
        let frames = frames.start as usize..frames.end as usize;
        let nf = frames.len();
        let mut win = up(&g.stream, &vec![SENTINEL; NSLICE as usize * len]);
        for slice in 0..NSLICE {
            let block = WindowBlock {
                slice,
                order: [0, 0],
                modes: Modes::All,
                block: slice,
            };
            sampler
                .windows(&g.ctx, g.at(s), &spectra, block, &mut win)
                .unwrap();
        }
        let got = g.down(&win);
        for slice in 0..NSLICE as usize {
            let want = cpu_windows(&slice_object(&host, slice), frames.clone(), [0.0; 2]);
            let e = rel(&per_view(&got[slice * len..], nf), &want);
            assert!(e < 2e-6, "stack {s} slice {slice}: {e:.2e}");
        }
        assert!(untouched(&got, &[0, 1], nf, NOBJ), "stack {s}");
    }
}

/// `∂^order W` by central differences of the CPU windows over the position.
fn cpu_derivative(object: &Object, frames: Range<usize>, order: [u32; 2]) -> Vec<C64> {
    let h = 1e-3;
    let at = |dy: f64, dx: f64| cpu_windows(object, frames.clone(), [dy, dx]);
    let comb = |terms: &[(f64, Vec<C64>)], div: f64| -> Vec<C64> {
        (0..terms[0].1.len())
            .map(|i| terms.iter().map(|(c, w)| *c * w[i]).sum::<C64>() / div)
            .collect()
    };
    match order {
        [1, 0] => comb(&[(1.0, at(h, 0.0)), (-1.0, at(-h, 0.0))], 2.0 * h),
        [0, 1] => comb(&[(1.0, at(0.0, h)), (-1.0, at(0.0, -h))], 2.0 * h),
        [2, 0] => comb(
            &[(1.0, at(h, 0.0)), (-2.0, at(0.0, 0.0)), (1.0, at(-h, 0.0))],
            h * h,
        ),
        [0, 2] => comb(
            &[(1.0, at(0.0, h)), (-2.0, at(0.0, 0.0)), (1.0, at(0.0, -h))],
            h * h,
        ),
        [1, 1] => comb(
            &[
                (1.0, at(h, h)),
                (-1.0, at(h, -h)),
                (-1.0, at(-h, h)),
                (1.0, at(-h, -h)),
            ],
            4.0 * h * h,
        ),
        _ => unreachable!(),
    }
}

#[test]
fn derivative_windows_match_cpu_differences() {
    let (g, mut sampler) = Gpu::new();
    let host = noise(NSLICE as usize * NOBJ * PLANE, 5);
    let obj = up(&g.stream, &host);
    let mut spectra = DeviceBuffer::zeroed(&g.stream, host.len()).unwrap();
    sampler.spectra(&g.ctx, &obj, &mut spectra).unwrap();
    let object = slice_object(&host, 1);
    let orders = [[1, 0], [0, 1], [2, 0], [1, 1], [0, 2]];
    for (s, frames) in stacks().into_iter().enumerate() {
        let frames = frames.start as usize..frames.end as usize;
        let nf = frames.len();
        for (modes, m) in [(Modes::All, NOBJ), (Modes::First, 1)] {
            let len = NF_MAX * m * N;
            let mut win = up(&g.stream, &vec![SENTINEL; orders.len() * len]);
            for (b, &order) in orders.iter().enumerate() {
                let block = WindowBlock {
                    slice: 1,
                    order,
                    modes,
                    block: b as u32,
                };
                sampler
                    .windows(&g.ctx, g.at(s), &spectra, block, &mut win)
                    .unwrap();
            }
            let got = g.down(&win);
            for (b, &order) in orders.iter().enumerate() {
                let want = cpu_derivative(&object, frames.clone(), order);
                let want = match modes {
                    Modes::All => want,
                    Modes::First => first_views(&want),
                };
                let views = match modes {
                    Modes::All => per_view(&got[b * len..], nf),
                    Modes::First => (got[b * len..][..nf * N].iter().copied().map(c64)).collect(),
                };
                let e = rel(&views, &want);
                assert!(e < 2e-5, "stack {s} {modes:?} order {order:?}: {e:.2e}");
            }
            let blocks: Vec<usize> = (0..orders.len()).collect();
            assert!(untouched(&got, &blocks, nf, m), "stack {s} {modes:?}");
        }
    }
}

#[test]
fn place_adds_the_cpu_adjoint_to_one_slice() {
    let (g, mut sampler) = Gpu::new();
    let obj0 = noise(NSLICE as usize * NOBJ * PLANE, 7);
    for (s, frames) in stacks().into_iter().enumerate() {
        let frames = frames.start as usize..frames.end as usize;
        let nf = frames.len();
        // Planes past the stack's frames hold values `place` must not read.
        let patches = noise(NOBJ * NF_MAX * N, 11 + s as u64);
        let mut obj = up(&g.stream, &obj0);
        sampler
            .place(&g.ctx, g.at(s), &up(&g.stream, &patches), 1, &mut obj)
            .unwrap();
        let got = g.down(&obj);
        let len = NOBJ * PLANE;
        assert_eq!(got[..len], obj0[..len], "stack {s}: slice 0 changed");
        let placed = cpu_place(&group_patches(&patches, nf), frames);
        let want: Vec<C64> = (obj0[len..].iter().zip(&placed))
            .map(|(&o, p)| c64(o) + p)
            .collect();
        let got: Vec<C64> = got[len..].iter().copied().map(c64).collect();
        let e = rel(&got, &want);
        assert!(e < 2e-6, "stack {s}: {e:.2e}");
    }
}

#[test]
fn place_is_the_adjoint_of_windows() {
    let (g, mut sampler) = Gpu::new();
    let x = noise(NSLICE as usize * NOBJ * PLANE, 13);
    let mut spectra = DeviceBuffer::zeroed(&g.stream, x.len()).unwrap();
    sampler
        .spectra(&g.ctx, &up(&g.stream, &x), &mut spectra)
        .unwrap();
    for s in 0..stacks().len() {
        let y = noise(NOBJ * NF_MAX * N, 17 + s as u64);
        let mut win = DeviceBuffer::zeroed(&g.stream, NF_MAX * NOBJ * N).unwrap();
        let block = WindowBlock {
            slice: 1,
            order: [0, 0],
            modes: Modes::All,
            block: 0,
        };
        sampler
            .windows(&g.ctx, g.at(s), &spectra, block, &mut win)
            .unwrap();
        let mut placed = DeviceBuffer::zeroed(&g.stream, x.len()).unwrap();
        sampler
            .place(&g.ctx, g.at(s), &up(&g.stream, &y), 1, &mut placed)
            .unwrap();
        // The windows past the stack's frames stay zero, and `place` reads no `y` there.
        let wx = g.down(&win);
        let lhs: C64 = (wx.iter().zip(&y))
            .map(|(&a, &b)| c64(a).conj() * c64(b))
            .sum();
        let len = NOBJ * PLANE;
        let rhs: C64 = (x[len..].iter().zip(&g.down(&placed)[len..]))
            .map(|(&a, &b)| c64(a).conj() * c64(b))
            .sum();
        let e = (lhs - rhs).norm() / lhs.norm();
        assert!(
            e < 2e-6,
            "stack {s}: <Wx, y> {lhs} <x, W^H y> {rhs}, {e:.2e}"
        );
    }
}

#[test]
fn illumination_matches_the_cpu_map() {
    let (g, mut sampler) = Gpu::new();
    let probe = noise(NPROBE * N, 19);
    let mut out = DeviceBuffer::zeroed(&g.stream, NOBJ * PLANE).unwrap();
    let (views, offsets) = (&g.views, &g.offsets);
    sampler
        .illumination(&g.ctx, views, offsets, &up(&g.stream, &probe), &mut out)
        .unwrap();
    let got = g.down(&out);
    let (views, offsets, _) = cpu_scan(0..FRAMES.len(), [0.0; 2]);
    let data = probe.iter().copied().map(c64).collect();
    let probe = Probe::new(Stack::from_vec([NPROBE, FRAME[0], FRAME[1]], data).unwrap());
    let want =
        Sampling::FullObject.illumination(&views, [NOBJ, OBJ[0], OBJ[1]], &probe, Some(&offsets));
    let want = want.as_slice();
    let scale = worst_of(want.iter().copied());
    let e = worst_of(got.iter().zip(want).map(|(&a, b)| (a as f64 - b).abs())) / scale;
    assert!(e < 2e-6, "{e:.2e}");
    assert!(got.iter().all(|&v| v >= 0.0));
}

/// The object passes on [`window_views`] over each stack's windows, cut from the object at
/// whole-pixel views in the sampler's layout, against the same passes on the scan's views:
/// the forward pass, the object sums placed back by the host and the probe sums. Frames of
/// two probe and two object modes in stacks of 3 and 2, the shorter leaving noise in its last
/// window rows.
mod window_views_as_objects {
    use std::ops::Range;
    use std::sync::Arc;

    use cuda_async::device_operation::ExecutionContext;
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
    use ptyrs::base::C64;
    use ptyrs::gpu::sampling::window_views;
    use ptyrs::gpu::{
        C32, Kernels, LsqmlKernels, StackTable, UpdateSums, ViewAddr, ViewDims, Views,
    };

    use super::worst_of;

    const NFRAME: usize = 5;
    const NPROBE: usize = 2;
    const NOBJ: usize = 2;
    const NMODE: usize = NPROBE * NOBJ;
    const NF_MAX: usize = 3;
    const STACKS: [Range<u32>; 2] = [0..3, 3..5];
    const SHAPE: [usize; 2] = [64, 64];
    const N: usize = SHAPE[0] * SHAPE[1];
    const OBJ: [usize; 2] = [96, 100];
    const PLANE: usize = OBJ[0] * OBJ[1];
    /// A stack's windows, `[nobj * nf_max, ny, nx]`.
    const WIN: usize = NOBJ * NF_MAX * N;
    /// A stack's waves, `[nf_max * nmode, ny, nx]`.
    const WAVES: usize = NF_MAX * NMODE * N;

    fn setup() -> (Arc<CudaStream>, ExecutionContext, Kernels, LsqmlKernels) {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
        let lsqml = LsqmlKernels::load(&kernels).expect("bind the LSQML kernels");
        let stream = ctx.default_stream();
        (
            stream.clone(),
            ExecutionContext::new(stream),
            kernels,
            lsqml,
        )
    }

    fn dims() -> ViewDims {
        ViewDims {
            nframe: NFRAME as u32,
            nmode: NMODE as u32,
            nprobe: NPROBE as u32,
            nobj: NOBJ as u32,
            frame: SHAPE.map(|n| n as u32),
            obj: OBJ.map(|n| n as u32),
            quad_phase: 0.0,
        }
    }

    /// Frame `f`'s corner.
    fn corner(f: usize) -> [usize; 2] {
        [3 + 5 * f, 30 - 4 * f]
    }

    fn views() -> Vec<ViewAddr> {
        (0..NFRAME)
            .flat_map(|f| {
                let [oy, ox] = corner(f).map(|c| c as u32);
                (0..NMODE as u32).map(move |m| ViewAddr {
                    oy,
                    ox,
                    fy: 0.0,
                    fx: 0.0,
                    probe: m % NPROBE as u32,
                    obj: m / NPROBE as u32,
                })
            })
            .collect()
    }

    /// Deterministic values in `[-1, 1)` per part.
    fn noise(len: usize, seed: u64) -> Vec<C64> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        (0..len).map(|_| C64::new(next(), next())).collect()
    }

    fn up(stream: &CudaStream, v: &[C64]) -> DeviceBuffer<C32> {
        let host: Vec<C32> = v
            .iter()
            .map(|c| C32 {
                re: c.re as f32,
                im: c.im as f32,
            })
            .collect();
        DeviceBuffer::from_host(stream, &host).unwrap()
    }

    fn down(stream: &CudaStream, b: &DeviceBuffer<C32>) -> Vec<C64> {
        let host = b.to_host_vec(stream).unwrap();
        host.iter()
            .map(|c| C64::new(c.re as f64, c.im as f64))
            .collect()
    }

    /// Object `[nobj, nyo, nxo]` pixel `(y, x)` of frame `f`'s window on object mode `o`.
    fn at(o: usize, f: usize, y: usize, x: usize) -> usize {
        let [oy, ox] = corner(f);
        o * PLANE + (oy + y) * OBJ[1] + ox + x
    }

    /// Stack `s`'s windows of `obj` in the sampler's layout, plane `o * nf_max + fl`; the rows
    /// of frames past the stack's hold noise.
    fn windows(obj: &[C64], s: usize) -> Vec<C64> {
        let r = &STACKS[s];
        let mut win = noise(WIN, 100 + s as u64);
        for o in 0..NOBJ {
            for fl in 0..r.len() {
                for i in 0..N {
                    let f = r.start as usize + fl;
                    win[(o * NF_MAX + fl) * N + i] = obj[at(o, f, i / SHAPE[1], i % SHAPE[1])];
                }
            }
        }
        win
    }

    /// The largest difference over the largest magnitude of `want`.
    fn rel(got: &[C64], want: &[C64]) -> f64 {
        let d = worst_of(got.iter().zip(want).map(|(g, w)| (g - w).norm()));
        d / want.iter().fold(0.0f64, |m, w| m.max(w.norm()))
    }

    #[test]
    fn window_views_need_the_stacks_to_partition_the_scan() {
        let (stream, ..) = setup();
        for stacks in [
            &[0..3, 4..5][..],
            &STACKS[..1],
            &[0..3, 3..3, 3..5],
            &[0..3, 3..6],
        ] {
            assert!(
                window_views(&stream, &dims(), stacks).is_err(),
                "{stacks:?} passed"
            );
        }
        let w = window_views(&stream, &dims(), &STACKS).unwrap();
        assert_eq!(w.dims().obj, [(NF_MAX * SHAPE[0]) as u32, SHAPE[1] as u32]);
    }

    #[test]
    fn forward_on_windows_is_forward_on_the_object() {
        let (stream, ctx, _, k) = setup();
        let d = dims();
        let (views, wviews) = (
            Views::new(&stream, d, &views()).unwrap(),
            window_views(&stream, &d, &STACKS).unwrap(),
        );
        let table = StackTable::new(&stream, &d, &STACKS).unwrap();
        let obj = noise(NOBJ * PLANE, 3);
        let probe = up(&stream, &noise(NPROBE * N, 5));
        let dev_obj = up(&stream, &obj);
        for (s, r) in STACKS.iter().enumerate() {
            let win = up(&stream, &windows(&obj, s));
            let mut want = DeviceBuffer::<C32>::zeroed(&stream, WAVES).unwrap();
            let mut got = DeviceBuffer::<C32>::zeroed(&stream, WAVES).unwrap();
            k.forward(&ctx, &views, table.at(s), &dev_obj, &probe, &mut want)
                .unwrap();
            k.forward(&ctx, &wviews, table.at(s), &win, &probe, &mut got)
                .unwrap();
            let len = r.len() * NMODE * N;
            let e = rel(&down(&stream, &got)[..len], &down(&stream, &want)[..len]);
            assert!(e < 1e-6, "stack {s}: max rel {e:e}");
        }
    }

    #[test]
    fn sums_on_windows_place_as_on_the_object() {
        let (stream, ctx, kernels, _) = setup();
        let d = dims();
        let (views, wviews) = (
            Views::new(&stream, d, &views()).unwrap(),
            window_views(&stream, &d, &STACKS).unwrap(),
        );
        let table = StackTable::new(&stream, &d, &STACKS).unwrap();
        let obj = noise(NOBJ * PLANE, 3);
        let (dev_obj, probe) = (up(&stream, &obj), up(&stream, &noise(NPROBE * N, 5)));
        let mut obj_sums = UpdateSums::zeroed(&stream, NOBJ * PLANE).unwrap();
        let mut probe_sums = UpdateSums::zeroed(&stream, NPROBE * N).unwrap();
        let mut wprobe_sums = UpdateSums::zeroed(&stream, NPROBE * N).unwrap();
        let (mut num, mut den) = (vec![C64::default(); NOBJ * PLANE], vec![0.0; NOBJ * PLANE]);
        for (s, r) in STACKS.iter().enumerate() {
            let psi = up(&stream, &noise(WAVES, 10 + s as u64));
            let win = up(&stream, &windows(&obj, s));
            let stack = table.at(s);
            kernels
                .psi_to_obj(&ctx, &views, stack, &psi, &probe, &mut obj_sums)
                .unwrap();
            kernels
                .psi_to_probe(&ctx, &views, stack, &psi, &dev_obj, &mut probe_sums)
                .unwrap();
            kernels
                .psi_to_probe(&ctx, &wviews, stack, &psi, &win, &mut wprobe_sums)
                .unwrap();
            let mut wsums = UpdateSums::zeroed(&stream, WIN).unwrap();
            kernels
                .psi_to_obj(&ctx, &wviews, stack, &psi, &probe, &mut wsums)
                .unwrap();
            let (wnum, wden) = (
                down(&stream, &wsums.num),
                wsums.den.to_host_vec(&stream).unwrap(),
            );
            for o in 0..NOBJ {
                for fl in 0..r.len() {
                    for i in 0..N {
                        let (w, f) = ((o * NF_MAX + fl) * N + i, r.start as usize + fl);
                        let a = at(o, f, i / SHAPE[1], i % SHAPE[1]);
                        num[a] += wnum[w];
                        den[a] += wden[w] as f64;
                    }
                }
            }
        }
        let e = rel(&num, &down(&stream, &obj_sums.num));
        assert!(e < 1e-5, "object num: max rel {e:e}");
        let want: Vec<C64> = (obj_sums.den.to_host_vec(&stream).unwrap())
            .iter()
            .map(|&v| C64::new(v as f64, 0.0))
            .collect();
        let got: Vec<C64> = den.iter().map(|&v| C64::new(v, 0.0)).collect();
        let e = rel(&got, &want);
        assert!(e < 1e-5, "object den: max rel {e:e}");
        let e = rel(
            &down(&stream, &wprobe_sums.num),
            &down(&stream, &probe_sums.num),
        );
        assert!(e < 1e-5, "probe num: max rel {e:e}");
        let real = |b: &DeviceBuffer<f32>| -> Vec<C64> {
            let h = b.to_host_vec(&stream).unwrap();
            h.iter().map(|&v| C64::new(v as f64, 0.0)).collect()
        };
        let e = rel(&real(&wprobe_sums.den), &real(&probe_sums.den));
        assert!(e < 1e-5, "probe den: max rel {e:e}");
    }
}
