//! The BH epoch order takes only a permutation of the frames, and holds the scan's views in it.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::CudaContext;
use ptyrs::gpu::{EpochOrder, Error, Kernels, ViewAddr, ViewDims, Views};

#[test]
fn epoch_order_sets_only_permutations() {
    let cuda = CudaContext::new(0).expect("CUDA device 0");
    let stream = cuda.new_stream().unwrap();
    let kernels = Kernels::load(&cuda).unwrap();
    let dims = ViewDims {
        nframe: 4,
        nmode: 1,
        nprobe: 1,
        nobj: 1,
        frame: [8, 8],
        obj: [24, 24],
        quad_phase: 0.0,
    };
    let addr: Vec<ViewAddr> = (0..4)
        .map(|f| ViewAddr {
            oy: 2 * f,
            ox: 3 * f,
            ..ViewAddr::default()
        })
        .collect();
    let src = Views::new(&stream, dims, &addr).unwrap();
    let mut order = EpochOrder::new(&kernels, &stream, &src).unwrap();
    let ctx = ExecutionContext::new(stream.clone());
    let views = |order: &EpochOrder| order.views().addr().to_host_vec(&stream).unwrap();
    for bad in [
        &[0, 1, 2][..],
        &[0, 1, 2, 3, 0],
        &[0, 1, 2, 2],
        &[0, 1, 2, 4],
    ] {
        let got = order.set(&ctx, &kernels, &src, bad);
        assert!(matches!(got, Err(Error::Shape(_))), "{bad:?}: {got:?}");
    }
    assert_eq!(views(&order), addr);
    let ids = [2, 0, 3, 1];
    order.set(&ctx, &kernels, &src, &ids).unwrap();
    assert_eq!(views(&order), ids.map(|f| addr[f as usize]));
}
