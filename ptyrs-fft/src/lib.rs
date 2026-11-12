//! cuFFT host binding: batched 2D complex-to-complex plans, executed through the access
//! leases of a cuda-async `ExecutionContext`.

pub mod sys;

use std::{fmt, sync::Arc};

use cuda_async::device_operation::ExecutionContext;
use cuda_async::error::DeviceError;
use cuda_core::{Context, DeviceBuffer, DeviceCopy, RawDeviceBuffer, Stream};

/// Single-precision complex, layout-compatible with `cufftComplex` / `float2`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C32 {
    pub re: f32,
    pub im: f32,
}

// SAFETY: two f32s, no padding, all-zero is a valid value.
unsafe impl DeviceCopy for C32 {}

#[derive(Debug)]
pub enum FftError {
    /// Non-zero `cufftResult`.
    Cufft(i32),
    Driver(cuda_core::DriverError),
    /// A cuda-async error, e.g. a lease refused because another stream holds the buffer.
    Device(DeviceError),
    /// Buffer holds fewer than `n*n*batch` elements.
    BufferTooSmall {
        need: usize,
        have: usize,
    },
    /// A buffer, the plan and the execution context do not all belong to one context.
    ContextMismatch,
    /// `n` or `batch` is zero or does not fit a C int.
    BadShape,
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FftError {}

impl From<cuda_core::DriverError> for FftError {
    fn from(e: cuda_core::DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<DeviceError> for FftError {
    fn from(e: DeviceError) -> Self {
        Self::Device(e)
    }
}

fn check(r: sys::cufftResult) -> Result<(), FftError> {
    if r == sys::CUFFT_SUCCESS {
        Ok(())
    } else {
        Err(FftError::Cufft(r))
    }
}

/// cuFFT library version, e.g. 12500 for the 13.5 toolkit.
pub fn version() -> Result<i32, FftError> {
    let mut v = 0;
    check(unsafe { sys::cufftGetVersion(&mut v) })?;
    Ok(v)
}

/// Registers `reads` and `writes` with `ctx` before work touching them is enqueued on its
/// stream. Eagerly, each is leased until `ctx`'s work completes; while `ctx` records a
/// capture, the graph keeps the access and reacquires it on every replay. Every ptyrs launch
/// goes through this: [`Plan2d`] here, the kernels through `ptyrs_gpu::lease`.
pub fn lease(
    ctx: &ExecutionContext,
    reads: &[&RawDeviceBuffer],
    writes: &[&RawDeviceBuffer],
) -> Result<(), FftError> {
    let all = || {
        reads
            .iter()
            .map(|b| (b, false))
            .chain(writes.iter().map(|b| (b, true)))
    };
    if all().any(|(b, _)| b.context().as_ref() != ctx.context().as_ref()) {
        return Err(FftError::ContextMismatch);
    }
    for (buf, write) in all() {
        ctx.retain_buffer((*buf).clone(), write)?;
    }
    Ok(())
}

/// A cuFFT handle. Every submission and captured graph that runs the plan's kernels holds
/// it, since `cufftDestroy` frees the tables those kernels read.
struct Handle {
    raw: sys::cufftHandle,
    ctx: Arc<Context>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe { sys::cufftDestroy(self.raw) };
    }
}

/// Batched `n`×`n` C2C plan, in place or out of place.
///
/// Transforms are unnormalized in both directions (cuFFT convention): a forward
/// followed by an inverse scales the data by `n*n`. Each transform is enqueued on the
/// stream of the `ExecutionContext` it is given and leases its buffers and the plan's work
/// area there, like a `ptyrs_gpu` launch, so it is legal inside `CudaGraph::capture`: the
/// work area is allocated with the plan and nothing allocates at execution.
pub struct Plan2d {
    handle: Arc<Handle>,
    work: DeviceBuffer<u8>,
    n: usize,
    batch: usize,
}

impl Plan2d {
    /// Plans in `stream`'s context and allocates the work area there (synchronously).
    pub fn new(stream: &Stream, n: usize, batch: usize) -> Result<Self, FftError> {
        let (ni, bi) = match (i32::try_from(n), i32::try_from(batch)) {
            (Ok(n), Ok(b)) if n > 0 && b > 0 => (n, b),
            _ => return Err(FftError::BadShape),
        };
        // cuFFT runs on the runtime API, which uses the driver's current context; the primary
        // context cuda-core retains for this device is that context once bound.
        let ctx = stream.context().clone();
        ctx.bind_to_thread()?;
        let mut raw = 0;
        check(unsafe { sys::cufftCreate(&mut raw) })?;
        let handle = Arc::new(Handle { raw, ctx });
        check(unsafe { sys::cufftSetAutoAllocation(raw, 0) })?;
        let mut dims = [ni, ni];
        let mut work_size = 0;
        check(unsafe {
            sys::cufftMakePlanMany(
                raw,
                2,
                dims.as_mut_ptr(),
                std::ptr::null_mut(),
                1,
                ni * ni,
                std::ptr::null_mut(),
                1,
                ni * ni,
                sys::CUFFT_C2C,
                bi,
                &mut work_size,
            )
        })?;
        let work = DeviceBuffer::<u8>::zeroed(stream, work_size)?;
        check(unsafe { sys::cufftSetWorkArea(raw, work.cu_deviceptr() as *mut _) })?;
        Ok(Self {
            handle,
            work,
            n,
            batch,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn batch(&self) -> usize {
        self.batch
    }

    fn check_len(&self, b: &DeviceBuffer<C32>) -> Result<(), FftError> {
        let need = self.n * self.n * self.batch;
        if b.len() < need {
            return Err(FftError::BufferTooSmall {
                need,
                have: b.len(),
            });
        }
        Ok(())
    }

    /// Leases `input` (`None` for in place) and `output`, then enqueues on `ctx`'s stream.
    fn exec(
        &mut self,
        ctx: &ExecutionContext,
        input: Option<&DeviceBuffer<C32>>,
        output: &mut DeviceBuffer<C32>,
        dir: i32,
    ) -> Result<(), FftError> {
        let reads: &[&RawDeviceBuffer] = match input {
            Some(i) => {
                self.check_len(i)?;
                &[i.storage()]
            }
            None => &[],
        };
        self.check_len(output)?;
        lease(ctx, reads, &[output.storage(), self.work.storage()])?;
        ctx.retain(self.handle.clone())?;
        ctx.context().bind_to_thread()?;
        let o = output.cu_deviceptr() as *mut _;
        let i = input.map_or(o, |i| i.cu_deviceptr() as *mut _);
        // SAFETY: both buffers hold the plan's extent and, like the work area, are leased on
        // `ctx`'s stream; `&mut self` keeps the stream binding until the exec is enqueued.
        unsafe {
            check(sys::cufftSetStream(
                self.handle.raw,
                ctx.get_cuda_stream().cu_stream() as *mut _,
            ))?;
            check(sys::cufftExecC2C(self.handle.raw, i, o, dir))
        }
    }

    /// Out-of-place forward transform.
    pub fn forward(
        &mut self,
        ctx: &ExecutionContext,
        input: &DeviceBuffer<C32>,
        output: &mut DeviceBuffer<C32>,
    ) -> Result<(), FftError> {
        self.exec(ctx, Some(input), output, sys::CUFFT_FORWARD)
    }

    /// Out-of-place unnormalized inverse transform.
    pub fn inverse(
        &mut self,
        ctx: &ExecutionContext,
        input: &DeviceBuffer<C32>,
        output: &mut DeviceBuffer<C32>,
    ) -> Result<(), FftError> {
        self.exec(ctx, Some(input), output, sys::CUFFT_INVERSE)
    }

    /// In-place forward transform.
    pub fn forward_inplace(
        &mut self,
        ctx: &ExecutionContext,
        data: &mut DeviceBuffer<C32>,
    ) -> Result<(), FftError> {
        self.exec(ctx, None, data, sys::CUFFT_FORWARD)
    }

    /// In-place unnormalized inverse transform.
    pub fn inverse_inplace(
        &mut self,
        ctx: &ExecutionContext,
        data: &mut DeviceBuffer<C32>,
    ) -> Result<(), FftError> {
        self.exec(ctx, None, data, sys::CUFFT_INVERSE)
    }
}
