//! Device memory across the burn ↔ cuda-rust boundary (DESIGN §4, invariant 7).
//!
//! burn cannot take memory cuda-rust allocated: cubecl's CUDA storage frees and serves only
//! pointers it allocated itself, and no API adopts a foreign one. The other direction is free:
//! a burn tensor's buffer can be lent to cuda-rust without a copy. So the boundary is zero-copy
//! when burn owns the boundary buffer ([`lend`] to read a tensor, [`lend_out`] to write a new
//! one), and costs one device-to-device copy when the engine owns it ([`copy_in`]).
//!
//! Both sides run in the device's primary context; lending checks that instead of assuming it.
//! Synchronisation is by host waits (invariant 7): lending waits for burn's queued work, and the
//! caller waits on its cuda-rust stream before burn touches memory cuda-rust wrote.

use std::ops::Deref;
use std::sync::{Arc, Mutex};

use burn::prelude::*;
use burn::tensor::DType;
use burn_cubecl::Cube;
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl_runtime::runtime::Runtime;
use cuda_core::{
    Context, DeviceAllocation, DeviceBuffer, DeviceCopy, DriverError, IntoResult, RawDeviceBuffer,
    Stream, sys,
};

use crate::gpu::C32;

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
    #[error("a complex tensor is [ny, nx, 2], got {0:?}")]
    Complex(Vec<usize>),
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

/// A value made of f32s, every bit pattern of which is valid: what burn's f32 memory is lent as.
trait F32s: DeviceCopy {}
impl F32s for f32 {}
impl F32s for C32 {}

/// A burn tensor's memory lent to cuda-rust to read, by [`lend`]: it derefs to the buffer but
/// gives no `&mut`, since a tensor burn first made contiguous is lent as that copy, which a write
/// would not reach. [`lend_out`] lends memory to write.
pub struct Lent<T: DeviceCopy>(DeviceBuffer<T>);

impl<T: DeviceCopy> Deref for Lent<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.0
    }
}

/// Lends `tensor`'s values to cuda-rust as a flat row-major `DeviceBuffer<f32>`, without a copy
/// when the tensor is row-major and through burn's contiguous copy when not.
///
/// Every burn operation queued on the tensor has finished when this returns.
pub fn lend<const D: usize>(
    tensor: &Tensor<D>,
    stream: &Stream,
) -> Result<Lent<f32>, InteropError> {
    lend_as(tensor, stream)
}

/// [`lend`] of a `[ny, nx, 2]` tensor, real and imaginary parts interleaved, as the `ny * nx`
/// complex values of the engine's layout.
pub fn lend_complex(tensor: &Tensor<3>, stream: &Stream) -> Result<Lent<C32>, InteropError> {
    if tensor.dims()[2] != 2 {
        return Err(InteropError::Complex(tensor.dims().to_vec()));
    }
    lend_as(tensor, stream)
}

fn lend_as<T: F32s, const D: usize>(
    tensor: &Tensor<D>,
    stream: &Stream,
) -> Result<Lent<T>, InteropError> {
    let cube = into_contiguous(primitive(tensor)?);
    Ok(Lent(alias(cube, &tensor.device(), stream)?))
}

/// A new f32 tensor of `shape` on `device`, uninitialised, and its memory lent to cuda-rust to
/// write: the writes show in the tensor once the caller has synchronised `stream`'s work.
pub fn lend_out<const D: usize>(
    shape: [usize; D],
    device: &Device,
    stream: &Stream,
) -> Result<(Tensor<D>, DeviceBuffer<f32>), InteropError> {
    lend_out_as(shape, device, stream)
}

/// [`lend_out`] of a `[ny, nx, 2]` tensor, lent as the `ny * nx` complex values of the engine's
/// layout, real and imaginary parts interleaved.
pub fn lend_out_complex(
    [ny, nx]: [usize; 2],
    device: &Device,
    stream: &Stream,
) -> Result<(Tensor<3>, DeviceBuffer<C32>), InteropError> {
    lend_out_as([ny, nx, 2], device, stream)
}

fn lend_out_as<T: F32s, const D: usize>(
    shape: [usize; D],
    device: &Device,
    stream: &Stream,
) -> Result<(Tensor<D>, DeviceBuffer<T>), InteropError> {
    // burn pads the rows of a fresh tensor of two or more dims; a flat one is row-major, and
    // reshaping a row-major tensor keeps its memory.
    let flat = Tensor::<1>::empty([shape.iter().product::<usize>()], (device, DType::F32));
    let buffer = alias(primitive(&flat)?, device, stream)?;
    Ok((flat.reshape(shape), buffer))
}

/// `tensor`'s cubecl tensor, if f32 on burn's CUDA backend.
fn primitive<const D: usize>(tensor: &Tensor<D>) -> Result<CubeTensor, InteropError> {
    if tensor.dtype() != DType::F32 {
        return Err(InteropError::DType(tensor.dtype()));
    }
    tensor
        .clone()
        .inner()
        .try_into_primitive::<Cube>()
        .map_err(|e| InteropError::Backend(format!("{e:?}")))
}

/// `cube`'s memory as a buffer, once burn's queued work on `device`, its device, has finished;
/// fails unless `cube` is laid out in rows, as the buffer would not hold its values.
fn alias<T: F32s>(
    cube: CubeTensor,
    device: &Device,
    stream: &Stream,
) -> Result<DeviceBuffer<T>, InteropError> {
    if cube.meta.is_tiled() || !cube.is_contiguous() {
        return Err(InteropError::Burn(format!(
            "a {:?} tensor not laid out in rows",
            cube.meta.shape()
        )));
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
    // SAFETY: any bits are a valid `T`, burn's queued writes have finished, and `bytes` holds
    // `bytes / size_of::<T>()` values (`from_storage` fails on a remainder or misalignment).
    Ok(unsafe { DeviceBuffer::from_storage(storage, bytes / size_of::<T>())? })
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
    let (tensor, mut dst) = lend_out(shape, device, stream)?;
    dst.copy_from_device_async(buffer, stream)?;
    stream.synchronize()?;
    Ok(tensor)
}
