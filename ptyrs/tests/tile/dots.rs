//! `ptyrs::tile::TileCgDots` against the SIMT `ptyrs::gpu::GradKernels` dot products, bit for
//! bit, and `TileDots::dots` against f64 sums on the host, at the ML timing case's sizes: an
//! object of 636x632 values, two grid strides with the last block partial, and a probe of
//! 2x256x256, which leaves half the blocks without an element.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use cutile::tile_kernel::jit_compile_count;
use ptyrs::gpu::{BhDisp, C32, CG_BLOCKS, GradKernels, Kernels, ViewDims, gamma_parts};
use ptyrs::tile::{TileCgDots, TileDots};

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

fn flat(v: &[C32]) -> Vec<f32> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

fn up<T: DeviceCopy>(stream: &CudaStream, host: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(stream, host).unwrap()
}

fn down<T: DeviceCopy>(stream: &Arc<CudaStream>, buf: &DeviceBuffer<T>) -> Vec<T> {
    buf.to_host_vec(stream).unwrap()
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let bad = got
        .iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    if let Some(i) = bad {
        panic!("{what}: float {i} is {} (Tile), {} (SIMT)", got[i], want[i]);
    }
}

/// `gpu_time ml`'s scan: 400 frames of 256x256, 2 probe modes, one 636x632 object.
fn ml_dims() -> ViewDims {
    ViewDims {
        nframe: 400,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [256, 256],
        obj: [636, 632],
        quad_phase: 0.0,
    }
}

struct Gpu {
    stream: Arc<CudaStream>,
    ex: ExecutionContext,
    grad: GradKernels,
    dims: ViewDims,
    lens: [u32; 2],
}

fn gpu() -> Gpu {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    Gpu {
        ex: ExecutionContext::new(stream.clone()),
        grad: GradKernels::load(&k).unwrap(),
        lens: [636 * 632, 2 * 256 * 256],
        dims: ml_dims(),
        stream,
    }
}

/// One CG cycle's `beta_parts` (object into pairs `0 ..`, probe into `CG_BLOCKS ..`), then
/// the cycle's sums of `2 CG_BLOCKS` and of `gamma_parts` pairs, and the start's
/// `norm_parts` of both arrays added to nonzero parts and summed over `CG_BLOCKS`.
#[test]
fn cg_dots_match_simt_bitwise() {
    let g = gpu();
    let s = &g.stream;
    let tile = TileCgDots::new(s, &g.dims).unwrap();
    let parts0 = flat(&noise(2 * CG_BLOCKS as usize, 1, 3.0));
    let mut parts = [up(s, &parts0), up(s, &parts0)];
    let mut h = [Vec::new(), Vec::new()];
    let jit = jit_compile_count();
    for (i, n) in g.lens.into_iter().enumerate() {
        let gv = noise(n as usize, 10 + i as u64, 1.0);
        let hv = noise(n as usize, 20 + i as u64, 1.0);
        let gd = up(s, &gv);
        let mut hd = [up(s, &hv), up(s, &hv)];
        let first = i as u32 * CG_BLOCKS;
        g.grad
            .beta_parts(&g.ex, &gd, &mut hd[0], n, &mut parts[0], first)
            .unwrap();
        tile.beta_parts(&g.ex, &gd, &mut hd[1], n, &mut parts[1], first)
            .unwrap();
        let [hs, ht] = hd.map(|b| flat(&down(s, &b)));
        assert_bits("h after beta_parts", &ht, &hs);
        assert_bits("h = g", &hs, &flat(&gv));
        h[i] = gv;
    }
    let [ps, pt] = [&parts[0], &parts[1]].map(|b| down(s, b));
    assert_bits("beta_parts", &pt, &ps);
    assert!(ps.iter().all(|v| v.is_finite()) && ps[0] != parts0[0]);

    let gamma = gamma_parts(&g.dims).unwrap();
    let many = up(s, &flat(&noise(gamma as usize, 2, 5.0)));
    for (npart, src) in [(2 * CG_BLOCKS, &parts[0]), (gamma, &many)] {
        let mut pair = [up(s, &[0.0f32; 2]), up(s, &[0.0f32; 2])];
        g.grad.sum_parts(&g.ex, src, npart, &mut pair[0]).unwrap();
        tile.sum_parts(&g.ex, src, npart, &mut pair[1]).unwrap();
        let [want, got] = pair.map(|b| down(s, &b));
        assert_bits(&format!("sum_parts of {npart}"), &got, &want);
    }

    let mut norm = [up(s, &parts0), up(s, &parts0)];
    for (x, n) in h.iter().zip(g.lens) {
        let xd = up(s, x);
        g.grad.norm_parts(&g.ex, &xd, n, &mut norm[0]).unwrap();
        tile.norm_parts(&g.ex, &xd, n, &mut norm[1]).unwrap();
    }
    let [want, got] = [&norm[0], &norm[1]].map(|b| down(s, b));
    assert_bits("norm_parts", &got, &want);
    let mut pair = [up(s, &[0.0f32; 2]), up(s, &[0.0f32; 2])];
    g.grad
        .sum_parts(&g.ex, &norm[0], CG_BLOCKS, &mut pair[0])
        .unwrap();
    tile.sum_parts(&g.ex, &norm[0], CG_BLOCKS, &mut pair[1])
        .unwrap();
    let [want, got] = pair.map(|b| down(s, &b));
    assert_bits("sum_parts of CG_BLOCKS", &got, &want);
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

/// `dots` with and without `v`, the probe's `u` scaled, block by block against f64 sums of
/// the elements block `b` takes, `256 b + t + k 256 CG_BLOCKS`. A partial may differ from
/// its f64 sum by `depth u sum |terms|`, `depth` bounding the f32 roundings a term passes
/// through: two fmas per stride in its lane, 255 in a sum of 256 lanes in any order, and
/// the scale. A block without an element, or the second sum without `v`, must be 0.
#[test]
fn bh_dots_match_f64_host_sums() {
    let g = gpu();
    let s = &g.stream;
    let tile = TileDots::new(s, &g.dims).unwrap();
    let [u, v] = [30u64, 31].map(|seed| g.lens.map(|n| noise(n as usize, seed, 1.0)));
    let [ud, vd] = [&u, &v].map(|x| [up(s, &x[0]), up(s, &x[1])]);
    let scale = [1.0, 0.37f32];
    let stride = 256 * CG_BLOCKS as usize;
    let jit = jit_compile_count();
    for cg in [false, true] {
        let d = BhDisp {
            u: [&ud[0], &ud[1]],
            u_scale: scale[1],
            v: cg.then(|| [&vd[0], &vd[1]]),
        };
        let mut parts = up(s, &vec![f32::NAN; 4 * CG_BLOCKS as usize]);
        tile.dots(&g.ex, &d, &mut parts).unwrap();
        let got = down(s, &parts);
        for (i, n) in g.lens.map(|n| n as usize).into_iter().enumerate() {
            let depth = 2 * n.div_ceil(stride) + 255 + 1;
            let tol = |abs: f64| depth as f64 * f64::from(f32::EPSILON) / 2.0 * abs;
            for b in 0..CG_BLOCKS as usize {
                let (mut uu, mut uv, mut uv_abs) = (0.0f64, 0.0f64, 0.0f64);
                for e in (256 * b..n)
                    .step_by(stride)
                    .flat_map(|e0| e0..(e0 + 256).min(n))
                {
                    let (x, y) = (u[i][e], v[i][e]);
                    let [x0, x1, y0, y1] = [x.re, x.im, y.re, y.im].map(f64::from);
                    uu += x0 * x0 + x1 * x1;
                    uv += x0 * y0 + x1 * y1;
                    uv_abs += (x0 * y0).abs() + (x1 * y1).abs();
                }
                let uu = f64::from(scale[i]) * uu;
                let at = 2 * (i * CG_BLOCKS as usize + b);
                let [pu, pv] = [got[at], got[at + 1]].map(f64::from);
                assert!(
                    (pu - uu).abs() <= tol(uu),
                    "cg {cg}, array {i}, block {b}: <u, Pu> {pu}, f64 {uu}"
                );
                if cg {
                    assert!(
                        (pv - uv).abs() <= tol(uv_abs),
                        "array {i}, block {b}: <u, v> {pv}, f64 {uv}"
                    );
                } else {
                    assert_eq!(pv, 0.0, "array {i}, block {b}: <u, v> without v");
                }
            }
        }
    }
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

/// A length or pair count `new` did not compile is refused, not compiled.
#[test]
fn uncompiled_lengths_are_refused() {
    let g = gpu();
    let s = &g.stream;
    let tile = TileCgDots::new(s, &g.dims).unwrap();
    let n = g.lens[0] - 1;
    let x = up(s, &noise(n as usize, 3, 1.0));
    let mut h = up(s, &noise(n as usize, 4, 1.0));
    let mut parts = up(s, &vec![0.0f32; 4 * CG_BLOCKS as usize]);
    let mut pair = up(s, &[0.0f32; 2]);
    let jit = jit_compile_count();
    assert!(
        tile.beta_parts(&g.ex, &x, &mut h, n, &mut parts, 0)
            .is_err()
    );
    assert!(tile.norm_parts(&g.ex, &x, n, &mut parts).is_err());
    assert!(tile.sum_parts(&g.ex, &parts, 100, &mut pair).is_err());
    assert_eq!(jit_compile_count(), jit, "a refused launch compiled");
}
