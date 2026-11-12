//! `ptyrs_tile::llk_sums` against the host `ptyrs_gpu::LlkSums::from_partials` tree.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_gpu::{LlkPartial, LlkSums};

fn totals(s: &LlkSums) -> [u64; 4] {
    [s.poisson, s.gaussian, s.euclidean, s.icalc].map(f64::to_bits)
}

fn sequential(p: &[LlkPartial]) -> [u64; 4] {
    let s = p.iter().fold([0.0f64; 4], |s, p| {
        [
            s[0] + p.poisson as f64,
            s[1] + p.gaussian as f64,
            s[2] + p.euclidean as f64,
            s[3] + p.icalc as f64,
        ]
    });
    s.map(f64::to_bits)
}

fn same(v: f32) -> LlkPartial {
    LlkPartial {
        poisson: v,
        gaussian: v,
        euclidean: v,
        icalc: v,
    }
}

#[test]
fn from_partials_adds_adjacent_pairs() {
    // ((1e30 + 1) + (-1e30 + 1)) is 0; the sequential sum is 1.
    let p = [1e30, 1.0, -1e30, 1.0].map(same);
    assert_eq!(totals(&LlkSums::from_partials(&p)), [0.0f64.to_bits(); 4]);
    assert_eq!(sequential(&p), [1.0f64.to_bits(); 4]);
}

#[test]
fn llk_sums_equal_from_partials_bit_for_bit() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    // Terms spanning 1e-3 .. 1e7 with both signs, so the f64 totals depend on the order.
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let m = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        m * 10f32.powi(((s >> 8) % 11) as i32 - 3)
    };
    let host: Vec<LlkPartial> = (0..70_001)
        .map(|_| LlkPartial {
            poisson: next(),
            gaussian: next(),
            euclidean: next(),
            icalc: next(),
        })
        .collect();
    assert_ne!(
        totals(&LlkSums::from_partials(&host[..3000])),
        sequential(&host[..3000]),
        "the data does not tell the tree from the sequential sum"
    );
    let partials = DeviceBuffer::from_host(&stream, &host).unwrap();
    // One, two and three tree levels, at and next to the level boundaries.
    for n in [0usize, 1, 255, 256, 257, 3000, 65_536, 70_000] {
        let scratch_len = ptyrs_tile::llk_sums_scratch_len(n as u32);
        let run = || {
            let mut scratch = DeviceBuffer::<f64>::zeroed(&stream, scratch_len).unwrap();
            let mut out = DeviceBuffer::<f64>::zeroed(&stream, 4).unwrap();
            ptyrs_tile::llk_sums(&ex, &partials, n as u32, &mut scratch, &mut out).unwrap();
            out.to_host_vec(&stream).unwrap()
        };
        let got: Vec<u64> = run().iter().map(|v| v.to_bits()).collect();
        let want = totals(&LlkSums::from_partials(&host[..n]));
        assert_eq!(got, want, "n={n}: Tile totals differ from from_partials");
        let again: Vec<u64> = run().iter().map(|v| v.to_bits()).collect();
        assert_eq!(again, got, "n={n}: differs run to run");
    }
    assert_eq!(ptyrs_tile::llk_sums_scratch_len(256), 0);
    assert_eq!(ptyrs_tile::llk_sums_scratch_len(70_000), 4 * (274 + 2));
    let mut scratch = DeviceBuffer::<f64>::zeroed(&stream, 4 * (274 + 2)).unwrap();
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, 4).unwrap();
    assert!(ptyrs_tile::llk_sums(&ex, &partials, 70_002, &mut scratch, &mut out).is_err());
    let mut short = DeviceBuffer::<f64>::zeroed(&stream, 4 * (274 + 1)).unwrap();
    assert!(ptyrs_tile::llk_sums(&ex, &partials, 70_000, &mut short, &mut out).is_err());
}
