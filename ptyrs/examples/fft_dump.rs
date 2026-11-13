//! Writes the fixed-seed input and the GPU forward/inverse results as raw f32 (re,im interleaved)
//! into the directory given as `argv[1]`, for scripts/check_numpy.py.
#[path = "../tests/common/fft/mod.rs"]
mod common;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Context, DeviceBuffer};
use ptyrs::fft::{C32, Plan2d};

fn bytes(v: &[C32]) -> &[u8] {
    // SAFETY: C32 is repr(C) of two f32 with no padding.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("output dir"));
    std::fs::create_dir_all(&dir).unwrap();
    let (n, batch) = (256, 64);
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut plan = Plan2d::new(&stream, n, batch).unwrap();
    let ex = ExecutionContext::new(stream.clone());
    let host = common::random_c32(n * n * batch, 0x5eed);
    let d_in = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut d_out = DeviceBuffer::<C32>::zeroed(&stream, host.len()).unwrap();
    std::fs::write(dir.join("in.f32"), bytes(&host)).unwrap();
    plan.forward(&ex, &d_in, &mut d_out).unwrap();
    std::fs::write(
        dir.join("fwd.f32"),
        bytes(&d_out.to_host_vec(&stream).unwrap()),
    )
    .unwrap();
    plan.inverse(&ex, &d_in, &mut d_out).unwrap();
    std::fs::write(
        dir.join("inv.f32"),
        bytes(&d_out.to_host_vec(&stream).unwrap()),
    )
    .unwrap();
}
