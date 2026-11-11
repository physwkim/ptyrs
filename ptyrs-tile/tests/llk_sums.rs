//! `ptyrs_tile::llk_sums` against the host `ptyrs_gpu::LlkSums::from_partials`.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_gpu::{LlkPartial, LlkSums};

fn totals(s: &LlkSums) -> [u64; 4] {
    [s.poisson, s.gaussian, s.euclidean, s.icalc].map(f64::to_bits)
}

#[test]
fn llk_sums_equal_from_partials_bit_for_bit() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream: Arc<CudaStream> = ctx.default_stream();
    // Terms spanning 1e-3 .. 1e7 with both signs, so the f64 totals depend on the order.
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let m = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        m * 10f32.powi(((s >> 8) % 11) as i32 - 3)
    };
    let host: Vec<LlkPartial> = (0..3000)
        .map(|_| LlkPartial {
            poisson: next(),
            gaussian: next(),
            euclidean: next(),
            icalc: next(),
        })
        .collect();
    let reversed: Vec<LlkPartial> = host[..2500].iter().rev().copied().collect();
    assert_ne!(
        totals(&LlkSums::from_partials(&reversed)),
        totals(&LlkSums::from_partials(&host[..2500])),
        "the data does not tell summation orders apart"
    );
    let partials = DeviceBuffer::from_host(&stream, &host).unwrap();
    for n in [2500usize, 1, 0] {
        let run = || {
            let mut out = DeviceBuffer::<f64>::zeroed(&stream, 4).unwrap();
            ptyrs_tile::llk_sums(&stream, &partials, n as u32, &mut out).unwrap();
            out.to_host_vec(&stream).unwrap()
        };
        let got: Vec<u64> = run().iter().map(|v| v.to_bits()).collect();
        let want = totals(&LlkSums::from_partials(&host[..n]));
        assert_eq!(got, want, "n={n}: Tile totals differ from from_partials");
        let again: Vec<u64> = run().iter().map(|v| v.to_bits()).collect();
        assert_eq!(again, got, "n={n}: differs run to run");
    }
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, 4).unwrap();
    assert!(ptyrs_tile::llk_sums(&stream, &partials, 3001, &mut out).is_err());
}
