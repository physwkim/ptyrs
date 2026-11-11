//! A plain host crate loads the embedded cubin and launches a SIMT kernel and an in-kernel
//! FFT kernel.

use std::f64::consts::PI;
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer};
use ptyrs_gpu::{C32, Kernels, Lines};

fn setup() -> (Arc<CudaContext>, Kernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    (ctx, kernels)
}

/// Deterministic values in [-1, 1).
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

/// f64 DFT of every line, same layout as the input.
fn dft_lines(data: &[C32], lines: Lines, inverse: bool) -> Vec<(f64, f64)> {
    let n = lines.n as usize;
    let at = |l: usize, j: usize| l * lines.line_stride as usize + j * lines.elem_stride as usize;
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut out = vec![(0.0, 0.0); data.len()];
    for l in 0..lines.count as usize {
        for k in 0..n {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..n {
                let x = data[at(l, j)];
                let phase = sign * 2.0 * PI * ((j * k) % n) as f64 / n as f64;
                let (s, c) = phase.sin_cos();
                re += x.re as f64 * c - x.im as f64 * s;
                im += x.re as f64 * s + x.im as f64 * c;
            }
            out[at(l, k)] = (re, im);
        }
    }
    out
}

/// sqrt(sum |got - want|^2 / sum |want|^2) over the elements `lines` covers.
fn rel_l2(got: &[C32], want: &[(f64, f64)], lines: Lines) -> f64 {
    let (mut err, mut norm) = (0.0, 0.0);
    for l in 0..lines.count as usize {
        for j in 0..lines.n as usize {
            let i = l * lines.line_stride as usize + j * lines.elem_stride as usize;
            let (dr, di) = (got[i].re as f64 - want[i].0, got[i].im as f64 - want[i].1);
            err += dr * dr + di * di;
            norm += want[i].0 * want[i].0 + want[i].1 * want[i].1;
        }
    }
    (err / norm).sqrt()
}

#[test]
fn simt_scale() {
    let (ctx, k) = setup();
    let stream = ctx.default_stream();
    let host = noise(1000, 1);
    let mut x = DeviceBuffer::from_host(&stream, &host).unwrap();
    k.scale_c32(&stream, &mut x, 999, 0.5).unwrap();
    let got = x.to_host_vec(&stream).unwrap();
    for (i, (g, h)) in got.iter().zip(&host).enumerate() {
        let s = if i < 999 { 0.5 } else { 1.0 };
        assert_eq!((g.re, g.im), (h.re * s, h.im * s), "element {i}");
    }
}

#[test]
fn block_fft_rows_out_of_place() {
    let (ctx, k) = setup();
    let stream = ctx.default_stream();
    for n in [128u32, 256] {
        for inverse in [false, true] {
            // 37 lines: the last block is partly empty.
            let lines = Lines {
                n,
                count: 37,
                line_stride: n,
                elem_stride: 1,
            };
            let host = noise((37 * n) as usize, u64::from(n) + u64::from(inverse));
            let input = DeviceBuffer::from_host(&stream, &host).unwrap();
            let mut output = DeviceBuffer::<C32>::zeroed(&stream, host.len()).unwrap();
            k.fft_lines(&stream, &input, &mut output, lines, inverse)
                .unwrap();
            let err = rel_l2(
                &output.to_host_vec(&stream).unwrap(),
                &dft_lines(&host, lines, inverse),
                lines,
            );
            println!("rows n={n} inverse={inverse}: rel L2 {err:.2e}");
            assert!(err < 2e-6, "n={n} inverse={inverse}: rel L2 {err:e}");
        }
    }
}

#[test]
fn block_fft_columns_in_place() {
    let (ctx, k) = setup();
    let stream = ctx.default_stream();
    for n in [128u32, 256] {
        let lines = Lines {
            n,
            count: n,
            line_stride: 1,
            elem_stride: n,
        };
        let host = noise((n * n) as usize, 7 * u64::from(n));
        let mut data = DeviceBuffer::from_host(&stream, &host).unwrap();
        k.fft_lines_inplace(&stream, &mut data, lines, false)
            .unwrap();
        let err = rel_l2(
            &data.to_host_vec(&stream).unwrap(),
            &dft_lines(&host, lines, false),
            lines,
        );
        println!("columns n={n}: rel L2 {err:.2e}");
        assert!(err < 2e-6, "n={n}: rel L2 {err:e}");
    }
}

#[test]
fn rejects_bad_shapes() {
    let (ctx, k) = setup();
    let stream = ctx.default_stream();
    let mut x = DeviceBuffer::<C32>::zeroed(&stream, 128 * 4).unwrap();
    let short = Lines {
        n: 128,
        count: 5,
        line_stride: 128,
        elem_stride: 1,
    };
    let overlapping = Lines {
        line_stride: 64,
        ..short
    };
    let unsupported = Lines {
        n: 64,
        count: 4,
        line_stride: 64,
        elem_stride: 1,
    };
    for lines in [short, overlapping, unsupported] {
        let err = k.fft_lines_inplace(&stream, &mut x, lines, false);
        assert!(
            matches!(err, Err(ptyrs_gpu::Error::Shape(_))),
            "{lines:?}: {err:?}"
        );
    }
    assert!(k.scale_c32(&stream, &mut x, 513, 1.0).is_err());
}
