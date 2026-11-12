//! Device memory across the burn ↔ cuda-rust boundary (DESIGN §4, invariant 7).
//!
//! burn cannot take memory cuda-rust allocated: cubecl's CUDA storage frees and serves only
//! pointers it allocated itself, and no API adopts a foreign one. The other direction is free:
//! a burn tensor's buffer can be lent to cuda-rust without a copy. So the boundary is zero-copy
//! when burn owns the boundary buffer ([`lend`]), and costs one device-to-device copy when the
//! engine owns it ([`copy_in`]).
//!
//! Both sides run in the device's primary context; [`lend`] checks that instead of assuming it.
//! Synchronisation is by host waits (invariant 7): [`lend`] waits for burn's queued work, and the
//! caller waits on its cuda-rust stream before burn touches memory cuda-rust wrote.

use std::sync::{Arc, Mutex};

use burn::prelude::*;
use burn::tensor::DType;
use burn_cubecl::Cube;
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl_runtime::runtime::Runtime;
use cuda_core::{
    Context, DeviceAllocation, DeviceBuffer, DriverError, IntoResult, RawDeviceBuffer, Stream, sys,
};

type CudaServer = <CudaRuntime as Runtime>::Server;

#[derive(Debug, thiserror::Error)]
pub enum InteropError {
    #[error("tensor is not on burn's CUDA backend: {0}")]
    Backend(String),
    #[error("only f32 crosses the boundary, tensor is {0:?}")]
    DType(DType),
    #[error("burn: {0}")]
    Burn(String),
    #[error("CUDA driver: {0}")]
    Driver(#[from] DriverError),
    #[error("burn's buffer is not in the cuda-rust stream's context")]
    Context,
    #[error("shape {shape:?} holds {want} values, buffer has {have}")]
    Len {
        shape: Vec<usize>,
        want: usize,
        have: usize,
    },
}

/// A burn buffer held open for cuda-rust. `R` is cubecl's pinned resource.
struct BurnAllocation<R> {
    ptr: u64,
    bytes: usize,
    context: Arc<Context>,
    // The resource pins the allocation: cubecl never frees, reuses or moves a bound allocation.
    // The tensor keeps the handle shared, so burn never writes it in place while it is lent.
    _hold: Mutex<(CubeTensor, R)>,
}

// SAFETY: `ptr..ptr + bytes` lies in an allocation pinned by `_hold` for the owner's lifetime, in
// `context` (checked against the pointer's own context when the owner is built).
unsafe impl<R: Send + 'static> DeviceAllocation for BurnAllocation<R> {
    fn device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn context(&self) -> Arc<Context> {
        self.context.clone()
    }
}

/// Lends `tensor`'s memory to cuda-rust as a flat row-major `DeviceBuffer<f32>`, without a copy.
///
/// Every burn operation queued on the tensor has finished when this returns. The buffer aliases
/// the tensor: cuda-rust writes show in `tensor` once the caller has synchronised `stream`'s work.
/// A non-contiguous tensor is first made contiguous by burn, and the buffer aliases that copy.
pub fn lend<const D: usize>(
    tensor: &Tensor<D>,
    stream: &Stream,
) -> Result<DeviceBuffer<f32>, InteropError> {
    if tensor.dtype() != DType::F32 {
        return Err(InteropError::DType(tensor.dtype()));
    }
    let device = tensor.device();
    let mut cube = tensor
        .clone()
        .inner()
        .try_into_primitive::<Cube>()
        .map_err(|e| InteropError::Backend(format!("{e:?}")))?;
    if !cube.is_contiguous() || !cube.is_contiguous_buffer() {
        cube = into_contiguous(cube);
    }
    let len = cube.meta.shape().num_elements();
    let bytes = len * size_of::<f32>();
    let resource = cube
        .client
        .get_resource::<CudaServer>(cube.handle.clone())
        .map_err(|e| InteropError::Burn(format!("{e:?}")))?;
    let ptr = resource.resource().ptr;
    if (resource.resource().size as usize) < bytes {
        return Err(InteropError::Burn(format!(
            "resource holds {} bytes, tensor needs {bytes}",
            resource.resource().size
        )));
    }
    device
        .sync()
        .map_err(|e| InteropError::Burn(format!("{e:?}")))?;

    let context = stream.context().clone();
    let mut owner: sys::CUcontext = std::ptr::null_mut();
    // SAFETY: `owner` is a CUcontext-sized out value; `ptr` is a live device pointer.
    unsafe {
        sys::cuPointerGetAttribute(
            (&raw mut owner).cast(),
            sys::CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_CONTEXT,
            ptr,
        )
    }
    .result()?;
    if owner != context.cu_ctx() {
        return Err(InteropError::Context);
    }

    let allocation = BurnAllocation {
        ptr,
        bytes,
        context,
        _hold: Mutex::new((cube, resource)),
    };
    let storage = RawDeviceBuffer::foreign(Arc::new(allocation), bytes);
    // SAFETY: burn initialised all `len` f32 values (its queued work has finished) and the
    // allocation is cubecl's, so it is f32-aligned.
    Ok(unsafe { DeviceBuffer::from_storage(storage, len)? })
}

/// Copies an engine-owned buffer into a new burn tensor of `shape` on `device`: one
/// device-to-device copy on `stream`, waited for before returning.
pub fn copy_in<const D: usize>(
    buffer: &DeviceBuffer<f32>,
    shape: [usize; D],
    device: &Device,
    stream: &Stream,
) -> Result<Tensor<D>, InteropError> {
    let want: usize = shape.iter().product();
    if want != buffer.len() {
        return Err(InteropError::Len {
            shape: shape.to_vec(),
            want,
            have: buffer.len(),
        });
    }
    let tensor = Tensor::<D>::empty(shape, (device, DType::F32));
    let mut dst = lend(&tensor, stream)?;
    dst.copy_from_device_async(buffer, stream)?;
    stream.synchronize()?;
    Ok(tensor)
}
