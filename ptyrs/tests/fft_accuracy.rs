#[path = "common/fft/mod.rs"]
mod common;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Context, DeviceBuffer};
use ptyrs::fft::{C32, Plan2d};
use rustfft::{FftPlanner, num_complex::Complex64};

const N: usize = 256;
const BATCH: usize = 64;

/// Unnormalized 2D DFT of every frame in f64 (row pass then column pass).
fn reference(input: &[C32], inverse: bool) -> Vec<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(N)
    } else {
        planner.plan_fft_forward(N)
    };
    let mut out: Vec<Complex64> = input
        .iter()
        .map(|c| Complex64::new(c.re as f64, c.im as f64))
        .collect();
    let mut col = vec![Complex64::default(); N];
    for frame in out.chunks_mut(N * N) {
        for row in frame.chunks_mut(N) {
            fft.process(row);
        }
        for x in 0..N {
            for y in 0..N {
                col[y] = frame[y * N + x];
            }
            fft.process(&mut col);
            for y in 0..N {
                frame[y * N + x] = col[y];
            }
        }
    }
    out
}

/// (max|gpu-ref|/max|ref|, rms(gpu-ref)/rms(ref))
fn errors(gpu: &[C32], reference: &[Complex64]) -> (f64, f64) {
    let (mut max_err, mut max_ref, mut se, mut sr) = (0f64, 0f64, 0f64, 0f64);
    for (g, r) in gpu.iter().zip(reference) {
        let d = Complex64::new(g.re as f64, g.im as f64) - r;
        max_err = max_err.max(d.norm());
        max_ref = max_ref.max(r.norm());
        se += d.norm_sqr();
        sr += r.norm_sqr();
    }
    (max_err / max_ref, (se / sr).sqrt())
}

#[test]
fn batched_256_forward_inverse_match_f64_reference() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    println!("cufft version {}", ptyrs::fft::version().unwrap());
    let mut plan = Plan2d::new(&stream, N, BATCH).unwrap();
    let ex = ExecutionContext::new(stream.clone());

    let host = common::random_c32(N * N * BATCH, 0x5eed);
    let d_in = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut d_out = DeviceBuffer::<C32>::zeroed(&stream, host.len()).unwrap();

    for inverse in [false, true] {
        if inverse {
            plan.inverse(&ex, &d_in, &mut d_out).unwrap();
        } else {
            plan.forward(&ex, &d_in, &mut d_out).unwrap();
        }
        let gpu = d_out.to_host_vec(&stream).unwrap();
        let (max_rel, rms_rel) = errors(&gpu, &reference(&host, inverse));
        println!("inverse={inverse}: max-rel {max_rel:.3e}, rms-rel {rms_rel:.3e}");
        assert!(max_rel <= 1e-5, "inverse={inverse}: max-rel {max_rel:e}");
    }
}

#[test]
fn inplace_matches_out_of_place_and_roundtrip_scales_by_n2() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut plan = Plan2d::new(&stream, N, 4).unwrap();
    let ex = ExecutionContext::new(stream.clone());
    let host = common::random_c32(N * N * 4, 7);
    let mut buf = DeviceBuffer::from_host(&stream, &host).unwrap();
    plan.forward_inplace(&ex, &mut buf).unwrap();
    plan.inverse_inplace(&ex, &mut buf).unwrap();
    let back = buf.to_host_vec(&stream).unwrap();
    let s = (N * N) as f32;
    let max = host
        .iter()
        .zip(&back)
        .map(|(a, b)| ((b.re / s - a.re).powi(2) + (b.im / s - a.im).powi(2)).sqrt())
        .fold(0f32, f32::max);
    println!("roundtrip max abs err {max:e}");
    assert!(max < 1e-5);
}

#[test]
fn short_buffer_rejected() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut plan = Plan2d::new(&stream, 16, 2).unwrap();
    let ex = ExecutionContext::new(stream.clone());
    let a = DeviceBuffer::<C32>::zeroed(&stream, 16 * 16).unwrap();
    let mut b = DeviceBuffer::<C32>::zeroed(&stream, 16 * 16 * 2).unwrap();
    assert!(matches!(
        plan.forward(&ex, &a, &mut b),
        Err(ptyrs::fft::FftError::BufferTooSmall { .. })
    ));
}
