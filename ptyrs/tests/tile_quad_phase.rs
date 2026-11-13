//! The quadratic phase of `ptyrs::tile::TileKernels::psi_to_obj` and `psi_to_probe` against
//! their SIMT counterparts over 512 x 512 frames, out to the corner's `r2 = 2 * 256^2`: one
//! view of unit object, probe and psi leaves `(cos a, -sin a)` of each pixel's phase `a` in
//! its sums. Tile and SIMT form the same angle over pi and reduce it exactly, so they differ
//! by the rounding of `cos` and `sin` alone, at any `quad_phase`.

use std::f64::consts::PI;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs::base::Geometry;
use ptyrs::gpu::{C32, Kernels, StackTable, UpdateSums, ViewAddr, ViewDims, Views};
use ptyrs::tile::TileKernels;

const N: usize = 512;

fn up<T: DeviceCopy>(stream: &CudaStream, host: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(stream, host).unwrap()
}

/// `num` of `s` as `(re, im)` pairs.
fn num(stream: &CudaStream, s: &UpdateSums) -> Vec<(f32, f32)> {
    let v = s.num.to_host_vec(stream).unwrap();
    v.iter().map(|c| (c.re, c.im)).collect()
}

/// Asserts per pixel that Tile and SIMT agree to `cos`/`sin` rounding and that both hold
/// `(cos a, -sin a)` up to the f32 rounding of the angle.
fn check(what: &str, quad: f32, tile: &[(f32, f32)], simt: &[(f32, f32)]) {
    let eps = f32::EPSILON as f64;
    let (mut vs_simt, mut vs_f64) = (0.0f64, 0.0f64);
    for (i, (t, s)) in tile.iter().zip(simt).enumerate() {
        let (dy, dx) = ((i / N) as f64 - 256.0, (i % N) as f64 - 256.0);
        let a = quad as f64 * (dy * dy + dx * dx);
        let want = (a.cos(), -a.sin());
        let d = (t.0 - s.0).abs().max((t.1 - s.1).abs()) as f64;
        assert!(d <= 4.0 * eps, "{what} pixel {i}: Tile {t:?}, SIMT {s:?}");
        for (who, g) in [("Tile", t), ("SIMT", s)] {
            let e = (g.0 as f64 - want.0).abs().max((g.1 as f64 - want.1).abs());
            let bound = 2.0 * eps * a + 4.0 * eps;
            assert!(e <= bound, "{what} {who} pixel {i}: {g:?}, want {want:?}");
            vs_f64 = vs_f64.max(e);
        }
        vs_simt = vs_simt.max(d);
    }
    println!("{what} quad {quad:e}: max |Tile - SIMT| {vs_simt:.2e}, max |- f64| {vs_f64:.2e}");
}

#[test]
fn quad_phase_matches_simt_at_512() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let one = C32 { re: 1.0, im: 0.0 };
    let unit = up(&stream, &vec![one; N * N]);
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [N, N]).unwrap();
    let px = g.pixel_size_object()[1];
    // The 512 bench's own phase, then phases of hundreds and thousands of radians at `r2`'s
    // largest, where the angle's rounding before reduction would show.
    let bench = (PI * px * px / (g.wavelength * g.distance)) as f32;
    for quad in [bench, 0.003, 0.05] {
        let d = ViewDims {
            nframe: 1,
            nmode: 1,
            nprobe: 1,
            nobj: 1,
            frame: [N as u32; 2],
            obj: [N as u32; 2],
            quad_phase: quad,
        };
        let view = ViewAddr {
            oy: 0,
            ox: 0,
            fy: 0.0,
            fx: 0.0,
            probe: 0,
            obj: 0,
        };
        let views = Views::new(&stream, d, &[view]).unwrap();
        let table = StackTable::new(&stream, &d, std::slice::from_ref(&(0..1))).unwrap();
        let tk = TileKernels::new(&stream, &views, &table).unwrap();
        let sums = || UpdateSums::zeroed(&stream, N * N).unwrap();

        let (mut simt, mut tile) = (sums(), sums());
        k.psi_to_obj(&ex, &views, table.at(0), &unit, &unit, &mut simt)
            .unwrap();
        tk.psi_to_obj(&ex, table.at(0), &unit, &unit, &mut tile)
            .unwrap();
        check(
            "psi_to_obj",
            quad,
            &num(&stream, &tile),
            &num(&stream, &simt),
        );

        let (mut simt, mut tile) = (sums(), sums());
        k.psi_to_probe(&ex, &views, table.at(0), &unit, &unit, &mut simt)
            .unwrap();
        tk.psi_to_probe(&ex, table.at(0), &unit, &unit, &mut tile)
            .unwrap();
        check(
            "psi_to_probe",
            quad,
            &num(&stream, &tile),
            &num(&stream, &simt),
        );
    }
}
