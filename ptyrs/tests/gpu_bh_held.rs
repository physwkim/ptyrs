//! The kernels of a BH step with the offsets held, against those of the step that takes them:
//! `moved_spec` and `moved_far` against `pos_spec` and `pos_far` at zero offset directions,
//! where those kernels' rate terms are zeros, and `win_obj_grad` against `win_grad`'s patches.
//! Each must match bit for bit (`==`: ±0 alike, a NaN never) over whole buffers prefilled
//! alike, so what a kernel must leave as it is is checked with what it writes.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs::gpu::{
    BhDisp, BhFarRef, BhKernels, BhMovedFarRef, BhMovedRef, BhMovesRef, BhOffsets, BhParts,
    BhPositions, BhScan, BhWinRef, C32, EpochOrder, Kernels, MASKED, Shift, SpectrumBuf,
    StackTable, ViewAddr, ViewDims, Views,
};

const NFRAME: usize = 5;
const NMODE: usize = 4;
const NPROBE: usize = 2;
const NOBJ: usize = NMODE / NPROBE;
const NF_MAX: usize = 3;
/// The last stack is shorter than `NF_MAX`, so its last frame's slots stay as they were.
const STACKS: [Range<u32>; 2] = [0..3, 3..5];
/// The scan frame at each epoch position.
const ORDER: [u32; NFRAME] = [3, 0, 4, 2, 1];
/// Two chunks of `GAMMA_CHUNK` pixels per frame.
const SHAPE: [usize; 2] = [64, 64];
const N: usize = SHAPE[0] * SHAPE[1];
const OBJ: [usize; 2] = [96, 96];
/// Values of a stack's fields, `[nf_max * nmode, ny, nx]`.
const FIELD: usize = NF_MAX * NMODE * N;

struct Fixture {
    stream: Arc<CudaStream>,
    ctx: ExecutionContext,
    kernels: Kernels,
    bh: BhKernels,
    dims: ViewDims,
    table: StackTable,
    order: EpochOrder,
    /// The views `order` was set from.
    _views: Views,
}

fn fixture() -> Fixture {
    let cuda = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cuda).unwrap();
    let bh = BhKernels::load(&kernels).unwrap();
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
    let views = Views::new(&stream, dims, &addr).unwrap();
    let table = StackTable::new(&stream, &dims, &STACKS).unwrap();
    let mut order = EpochOrder::new(&kernels, &stream, &views).unwrap();
    order.set(&ctx, &kernels, &views, &ORDER).unwrap();
    Fixture {
        stream,
        ctx,
        kernels,
        bh,
        dims,
        table,
        order,
        _views: views,
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

impl Fixture {
    fn c32(&self, len: usize, seed: u64) -> DeviceBuffer<C32> {
        let host: Vec<C32> = noise(2 * len, seed)
            .chunks(2)
            .map(|p| C32 { re: p[0], im: p[1] })
            .collect();
        DeviceBuffer::from_host(&self.stream, &host).unwrap()
    }

    fn floats(&self, b: &DeviceBuffer<C32>) -> Vec<f32> {
        let host = b.to_host_vec(&self.stream).unwrap();
        host.iter().flat_map(|c| [c.re, c.im]).collect()
    }

    /// Counts in the scan's order, every seventh pixel masked.
    fn counts(&self) -> DeviceBuffer<u16> {
        let host: Vec<u16> = (0..NFRAME * N)
            .map(|i| {
                if i % 7 == 3 {
                    MASKED
                } else {
                    (i * 7919 % 1000) as u16
                }
            })
            .collect();
        DeviceBuffer::from_host(&self.stream, &host).unwrap()
    }

    fn disp<'a>(
        &self,
        u: [&'a DeviceBuffer<C32>; 2],
        v: [&'a DeviceBuffer<C32>; 2],
        cg: bool,
    ) -> BhDisp<'a> {
        BhDisp {
            u,
            u_scale: 0.7,
            v: cg.then_some(v),
        }
    }
}

/// Panics at the first value where `got != want`.
fn same(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!("{what}: value {i} is {:e}, want {:e}", got[i], want[i]);
    }
}

/// The step's displacements and the probes' spectra one case needs.
struct Disp {
    u: [DeviceBuffer<C32>; 2],
    v: [DeviceBuffer<C32>; 2],
    probe: DeviceBuffer<C32>,
    offsets: DeviceBuffer<f32>,
    /// The estimate's, `u`'s and `v`'s probe spectra.
    spectra: [SpectrumBuf; 3],
}

fn disp(fx: &Fixture) -> Disp {
    let obj = NOBJ * OBJ[0] * OBJ[1];
    let offsets = noise(2 * NFRAME, 6)
        .iter()
        .map(|d| 3.0 * d)
        .collect::<Vec<_>>();
    Disp {
        u: [fx.c32(obj, 1), fx.c32(NPROBE * N, 2)],
        v: [fx.c32(obj, 3), fx.c32(NPROBE * N, 4)],
        probe: fx.c32(NPROBE * N, 5),
        offsets: DeviceBuffer::from_host(&fx.stream, &offsets).unwrap(),
        spectra: [0; 3].map(|_| SpectrumBuf::zeroed(&fx.stream, &fx.dims).unwrap()),
    }
}

/// `pos_spec` and `moved_spec` per stack, with `cg` on `lv` too.
fn moved_spec_case(cg: bool) {
    let fx = fixture();
    let mut dp = disp(&fx);
    let [ba, bu, bv] = &mut dp.spectra;
    let (ctx, dims) = (&fx.ctx, &fx.dims);
    let a = fx.kernels.probe_spectrum(ctx, dims, &dp.probe, ba).unwrap();
    let au = fx.kernels.probe_spectrum(ctx, dims, &dp.u[1], bu).unwrap();
    let av = fx.kernels.probe_spectrum(ctx, dims, &dp.v[1], bv).unwrap();
    let av = cg.then_some(av);
    let d = fx.disp([&dp.u[0], &dp.u[1]], [&dp.v[0], &dp.v[1]], cg);
    let mut zero = BhPositions::zeroed(&fx.stream, dims, 1).unwrap();
    let off = BhOffsets {
        bufs: &mut zero,
        order: &fx.order,
        scale: 1.0,
        step: 1.0,
    };
    let shift = Shift {
        offsets: &dp.offsets,
        spectrum: a,
    };
    let pos = BhMovesRef::new(&d, shift, au, av, &off).unwrap();
    let held = BhMovedRef::new(&d, &fx.order, &dp.offsets, au, av).unwrap();
    for s in 0..STACKS.len() {
        let at = fx.table.at(s);
        let seed = 10 * s as u64;
        let fill = || [fx.c32(FIELD, seed + 1), fx.c32(FIELD, seed + 2)];
        let (mut want, mut got, was) = (fill(), fill(), fill());
        let [wu, wv] = &mut want;
        fx.bh
            .pos_spec(ctx, at, &pos, (wu, cg.then_some(wv)))
            .unwrap();
        let [gu, gv] = &mut got;
        fx.bh
            .moved_spec(ctx, at, &held, (gu, cg.then_some(gv)))
            .unwrap();
        let [wu, wv, gu, gv, lu0, lv0] =
            [&want[0], &want[1], &got[0], &got[1], &was[0], &was[1]].map(|b| fx.floats(b));
        same(&format!("stack {s} lu"), &gu, &wu);
        assert_ne!(gu, lu0, "stack {s}: lu not written");
        if cg {
            same(&format!("stack {s} lv"), &gv, &wv);
            assert_ne!(gv, lv0, "stack {s}: lv not written");
        }
    }
}

#[test]
fn moved_spec_is_pos_spec_without_rates() {
    moved_spec_case(false);
}

#[test]
fn moved_spec_is_pos_spec_without_rates_cg() {
    moved_spec_case(true);
}

/// `pos_far` and `moved_far` per stack onto the same prefilled parts.
fn moved_far_case(cg: bool) {
    let fx = fixture();
    let mut dp = disp(&fx);
    let [ba, bu, bv] = &mut dp.spectra;
    let (ctx, dims) = (&fx.ctx, &fx.dims);
    let a = fx.kernels.probe_spectrum(ctx, dims, &dp.probe, ba).unwrap();
    let au = fx.kernels.probe_spectrum(ctx, dims, &dp.u[1], bu).unwrap();
    let av = fx.kernels.probe_spectrum(ctx, dims, &dp.v[1], bv).unwrap();
    let d = fx.disp([&dp.u[0], &dp.u[1]], [&dp.v[0], &dp.v[1]], cg);
    let mut zero = BhPositions::zeroed(&fx.stream, dims, 1).unwrap();
    let off = BhOffsets {
        bufs: &mut zero,
        order: &fx.order,
        scale: 1.0,
        step: 1.0,
    };
    let shift = Shift {
        offsets: &dp.offsets,
        spectrum: a,
    };
    let pos = BhMovesRef::new(&d, shift, au, cg.then_some(av), &off).unwrap();
    let counts = fx.counts();
    let scan = BhScan {
        order: &fx.order,
        counts: &counts,
    };
    let parts = || {
        let mut p = BhParts::zeroed(&fx.stream, dims, 0).unwrap();
        let prefill = noise(p.buf().len(), 7);
        p.buf_mut().copy_from_host(&fx.stream, &prefill).unwrap();
        p
    };
    let (mut want, mut got) = (parts(), parts());
    for s in 0..STACKS.len() {
        let at = fx.table.at(s);
        let seed = 10 * s as u64;
        let [psi, lu, lv, y] = [1, 2, 3, 4].map(|k| fx.c32(FIELD, seed + k));
        let f = BhMovedFarRef {
            psi: &psi,
            lu: &lu,
            lv: cg.then_some(&lv),
            y: &y,
        };
        fx.bh
            .pos_far(ctx, &counts, at, &f, &pos, &mut want)
            .unwrap();
        let f = BhFarRef {
            psi: &psi,
            lu: &lu,
            lv: cg.then_some(&lv),
        };
        fx.bh.moved_far(ctx, scan, at, &f, &mut got).unwrap();
    }
    let host = |p: &BhParts| p.buf().to_host_vec(&fx.stream).unwrap();
    let (want, got) = (host(&want), host(&got));
    same("parts", &got, &want);
    assert_ne!(got, noise(got.len(), 7), "parts not written");
}

#[test]
fn moved_far_is_pos_far_without_rates() {
    moved_far_case(false);
}

#[test]
fn moved_far_is_pos_far_without_rates_cg() {
    moved_far_case(true);
}

/// `win_grad` and `win_obj_grad` per stack into patches prefilled alike.
#[test]
fn win_obj_grad_is_win_grad_patch() {
    let fx = fixture();
    let blk = NOBJ * NF_MAX * N;
    let mut zero = BhPositions::zeroed(&fx.stream, &fx.dims, 1).unwrap();
    let mut off = BhOffsets {
        bufs: &mut zero,
        order: &fx.order,
        scale: 1.0,
        step: 1.0,
    };
    let probe = fx.c32(NPROBE * N, 5);
    for s in 0..STACKS.len() {
        let at = fx.table.at(s);
        let seed = 10 * s as u64;
        let (r, win) = (fx.c32(FIELD, seed + 1), fx.c32(3 * blk, seed + 2));
        let f = BhWinRef {
            r: &r,
            probe: &probe,
        };
        let (mut want, mut got) = (fx.c32(blk, seed + 3), fx.c32(blk, seed + 3));
        fx.bh
            .win_grad(&fx.ctx, at, f, (&win, 0), &mut want, &mut off)
            .unwrap();
        fx.bh
            .win_obj_grad(&fx.ctx, &fx.dims, at, f, &mut got)
            .unwrap();
        let [want, got, was] = [&want, &got, &fx.c32(blk, seed + 3)].map(|b| fx.floats(b));
        same(&format!("stack {s} patch"), &got, &want);
        assert_ne!(got, was, "stack {s}: patch not written");
    }
}
