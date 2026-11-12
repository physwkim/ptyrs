//! `ptyrs_tile::TileDots` against the SIMT `ptyrs_gpu::GradKernels` and `BhKernels` dot
//! products, bit for bit, at the ML timing case's sizes: an object of 636x632 values, two
//! grid strides with the last block partial, and a probe of 2x256x256, which leaves half the
//! blocks without an element.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use cutile::tile_kernel::jit_compile_count;
use ptyrs_gpu::{
    BhDisp, BhKernels, C32, CG_BLOCKS, GradKernels, Kernels, ViewDims, bh_parts, gamma_parts,
};
use ptyrs_tile::TileDots;

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
    bh: BhKernels,
    tile: TileDots,
    dims: ViewDims,
    lens: [u32; 2],
}

fn gpu() -> Gpu {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let dims = ml_dims();
    let tile = TileDots::new(&stream, &dims).unwrap();
    Gpu {
        ex: ExecutionContext::new(stream.clone()),
        grad: GradKernels::load(&k).unwrap(),
        bh: BhKernels::load(&k).unwrap(),
        tile,
        lens: [636 * 632, 2 * 256 * 256],
        dims,
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
        g.tile
            .beta_parts(&g.ex, &gd, &mut hd[1], n, &mut parts[1], first)
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
        g.tile.sum_parts(&g.ex, src, npart, &mut pair[1]).unwrap();
        let [want, got] = pair.map(|b| down(s, &b));
        assert_bits(&format!("sum_parts of {npart}"), &got, &want);
    }

    let mut norm = [up(s, &parts0), up(s, &parts0)];
    for (x, n) in h.iter().zip(g.lens) {
        let xd = up(s, x);
        g.grad.norm_parts(&g.ex, &xd, n, &mut norm[0]).unwrap();
        g.tile.norm_parts(&g.ex, &xd, n, &mut norm[1]).unwrap();
    }
    let [want, got] = [&norm[0], &norm[1]].map(|b| down(s, b));
    assert_bits("norm_parts", &got, &want);
    let mut pair = [up(s, &[0.0f32; 2]), up(s, &[0.0f32; 2])];
    g.grad
        .sum_parts(&g.ex, &norm[0], CG_BLOCKS, &mut pair[0])
        .unwrap();
    g.tile
        .sum_parts(&g.ex, &norm[0], CG_BLOCKS, &mut pair[1])
        .unwrap();
    let [want, got] = pair.map(|b| down(s, &b));
    assert_bits("sum_parts of CG_BLOCKS", &got, &want);
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

/// `BhKernels::dots` with and without `v`, the probe's `u` scaled.
#[test]
fn bh_dots_match_simt_bitwise() {
    let g = gpu();
    let s = &g.stream;
    let [u, v] = [0u64, 1].map(|j| g.lens.map(|n| up(s, &noise(n as usize, 30 + j, 1.0))));
    let jit = jit_compile_count();
    for cg in [false, true] {
        let d = BhDisp {
            u: [&u[0], &u[1]],
            u_scale: 0.37,
            v: cg.then(|| [&v[0], &v[1]]),
        };
        let sentinel = vec![f32::NAN; bh_parts(&g.dims)];
        let mut parts = [up(s, &sentinel), up(s, &sentinel)];
        g.bh.dots(&g.ex, &g.dims, &d, &mut parts[0]).unwrap();
        g.tile.dots(&g.ex, &d, &mut parts[1]).unwrap();
        let [want, got] = [&parts[0], &parts[1]].map(|b| down(s, b));
        assert_bits(&format!("dots, cg {cg}"), &got, &want);
        let pairs = &want[..4 * CG_BLOCKS as usize];
        assert!(pairs.iter().all(|x| x.is_finite()));
        assert_eq!(pairs.iter().skip(1).step_by(2).any(|&x| x != 0.0), cg);
    }
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

/// A length or pair count `new` did not compile is refused, not compiled.
#[test]
fn uncompiled_lengths_are_refused() {
    let g = gpu();
    let s = &g.stream;
    let n = g.lens[0] - 1;
    let x = up(s, &noise(n as usize, 3, 1.0));
    let mut h = up(s, &noise(n as usize, 4, 1.0));
    let mut parts = up(s, &vec![0.0f32; 4 * CG_BLOCKS as usize]);
    let mut pair = up(s, &[0.0f32; 2]);
    let jit = jit_compile_count();
    assert!(
        g.tile
            .beta_parts(&g.ex, &x, &mut h, n, &mut parts, 0)
            .is_err()
    );
    assert!(g.tile.norm_parts(&g.ex, &x, n, &mut parts).is_err());
    assert!(g.tile.sum_parts(&g.ex, &parts, 100, &mut pair).is_err());
    assert_eq!(jit_compile_count(), jit, "a refused launch compiled");
}
