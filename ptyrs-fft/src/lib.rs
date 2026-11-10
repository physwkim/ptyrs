//! cuFFT host binding: batched 2D complex-to-complex plans on a cuda-core stream.

pub mod sys;

use std::{fmt, sync::Arc};

use cuda_core::{DeviceBuffer, DeviceCopy, Stream};

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
    /// Buffer holds fewer than `n*n*batch` elements.
    BufferTooSmall {
        need: usize,
        have: usize,
    },
    /// Buffer belongs to a different CUDA context than the plan's stream.
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

/// Batched `n`×`n` C2C plan, in place or out of place, bound to one stream.
///
/// Transforms are unnormalized in both directions (cuFFT convention): a forward
/// followed by an inverse scales the data by `n*n`. Calls are only enqueued on
/// the stream; the caller orders reads/writes of the buffers against it
/// (synchronize the stream or use events) — the cuda-core access tracker is
/// not consulted.
pub struct Plan2d {
    handle: sys::cufftHandle,
    stream: Arc<Stream>,
    n: usize,
    batch: usize,
}

// cuFFT handles may move between threads; concurrent use of one plan is not allowed (&mut-free
// exec takes &self but enqueues only, and the library documents plan use as thread-compatible).
unsafe impl Send for Plan2d {}

impl Plan2d {
    pub fn new(stream: &Arc<Stream>, n: usize, batch: usize) -> Result<Self, FftError> {
        let (ni, bi) = match (i32::try_from(n), i32::try_from(batch)) {
            (Ok(n), Ok(b)) if n > 0 && b > 0 => (n, b),
            _ => return Err(FftError::BadShape),
        };
        // cuFFT runs on the runtime API, which uses the driver's current context; the primary
        // context cuda-core retains for this device is that context once bound.
        stream.context().bind_to_thread()?;
        let mut dims = [ni, ni];
        let mut handle = 0;
        check(unsafe {
            sys::cufftPlanMany(
                &mut handle,
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
            )
        })?;
        let plan = Self {
            handle,
            stream: stream.clone(),
            n,
            batch,
        };
        check(unsafe { sys::cufftSetStream(handle, stream.cu_stream() as *mut _) })?;
        Ok(plan)
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn batch(&self) -> usize {
        self.batch
    }
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    fn check_buf(&self, b: &DeviceBuffer<C32>) -> Result<(), FftError> {
        let need = self.n * self.n * self.batch;
        if b.len() < need {
            return Err(FftError::BufferTooSmall {
                need,
                have: b.len(),
            });
        }
        if b.context().as_ref() != self.stream.context().as_ref() {
            return Err(FftError::ContextMismatch);
        }
        Ok(())
    }

    fn exec(
        &self,
        input: &DeviceBuffer<C32>,
        output: &mut DeviceBuffer<C32>,
        dir: i32,
    ) -> Result<(), FftError> {
        self.check_buf(input)?;
        self.check_buf(output)?;
        self.stream.context().bind_to_thread()?;
        check(unsafe {
            sys::cufftExecC2C(
                self.handle,
                input.cu_deviceptr() as *mut _,
                output.cu_deviceptr() as *mut _,
                dir,
            )
        })
    }

    /// Out-of-place forward transform, enqueued on the plan's stream.
    pub fn forward(
        &self,
        input: &DeviceBuffer<C32>,
        output: &mut DeviceBuffer<C32>,
    ) -> Result<(), FftError> {
        self.exec(input, output, sys::CUFFT_FORWARD)
    }

    /// Out-of-place unnormalized inverse transform, enqueued on the plan's stream.
    pub fn inverse(
        &self,
        input: &DeviceBuffer<C32>,
        output: &mut DeviceBuffer<C32>,
    ) -> Result<(), FftError> {
        self.exec(input, output, sys::CUFFT_INVERSE)
    }

    /// In-place forward transform.
    pub fn forward_inplace(&self, data: &mut DeviceBuffer<C32>) -> Result<(), FftError> {
        self.check_buf(data)?;
        self.stream.context().bind_to_thread()?;
        let p = data.cu_deviceptr() as *mut _;
        check(unsafe { sys::cufftExecC2C(self.handle, p, p, sys::CUFFT_FORWARD) })
    }

    /// In-place unnormalized inverse transform.
    pub fn inverse_inplace(&self, data: &mut DeviceBuffer<C32>) -> Result<(), FftError> {
        self.check_buf(data)?;
        self.stream.context().bind_to_thread()?;
        let p = data.cu_deviceptr() as *mut _;
        check(unsafe { sys::cufftExecC2C(self.handle, p, p, sys::CUFFT_INVERSE) })
    }
}

impl Drop for Plan2d {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        unsafe { sys::cufftDestroy(self.handle) };
    }
}
