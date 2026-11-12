//! M0 smoke: a cuda-oxide kernel built from a ptyrs workspace member runs on this GPU.
//!
//! Run with `cargo oxide run --arch=sm_89 --bin m0_smoke`; a plain cargo build embeds no
//! module to load.

use cuda_async::device_context::{init_device_contexts, with_cuda_context};
use cuda_async::device_operation::DeviceOperation;
use cuda_core::LaunchConfig;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::device_box::DeviceBox;
    use cuda_core::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    init_device_contexts(0, 1)?;
    // Not `kernels::load_async`: it finds the bundle through an anchor symbol that only the
    // cuda-oxide backend defines, so a plain cargo build of the workspace could not link.
    let module = cuda_host::load_cuda_module_from_async_context(0, |ctx| {
        let module = cuda_host::load_embedded_module(ctx, env!("CARGO_PKG_NAME"))?;
        kernels::from_module(module).map_err(cuda_host::EmbeddedModuleError::Driver)
    })?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let (a_dev, b_dev, mut c_dev) = with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        let num_bytes = N * mem::size_of::<f32>();
        unsafe {
            let a = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let b = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let c = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(a, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            memcpy_htod_async(b, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            stream.synchronize().unwrap();
            (
                DeviceBox::<[f32]>::from_raw_parts(a, N, stream.clone()),
                DeviceBox::<[f32]>::from_raw_parts(b, N, stream.clone()),
                DeviceBox::<[f32]>::from_raw_parts(c, N, stream.clone()),
            )
        }
    })?;

    // SAFETY: this is a 1D launch and `vecadd` guards its index against the
    // output length before writing.
    unsafe {
        module.vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }?
    .sync()?;

    let mut c_host = vec![0.0f32; N];
    with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}
