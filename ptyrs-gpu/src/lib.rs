//! ptyrs-gpu: the ptyrs kernels, built, linked and loaded for any host crate.
//!
//! Depending on this crate is all a library, test or binary needs: its build script runs
//! `cargo oxide build` on `ptyrs-kernels`, links the device code with the cuFFT device-API
//! functions it calls and embeds the cubin ([`CUBIN`], built for [`ARCH`]; set
//! `PTYRS_CUDA_ARCH`, default `sm_89`, to change it). [`Kernels::load`] loads it into a
//! context.
//!
//! Every launch method checks buffer lengths against the shape it is given, registers each
//! buffer it touches with the cuda-async `ExecutionContext` it is given ([`lease`]), enqueues
//! on that context's stream and returns. No method allocates or synchronizes, so each is
//! legal inside `CudaGraph::capture`; [`record`] turns a sequence of them into the
//! `DeviceOp` to capture. Eagerly (`ExecutionContext::new(stream)`, or
//! `record(..).sync_on(&stream)`) the leases last until the context is dropped and its work
//! completes: dropping a buffer early is safe, and a conflicting access to it from another
//! stream fails with `CUDA_ERROR_NOT_READY` until then. A captured graph instead keeps the
//! accesses and reacquires them on every replay. Work on one stream runs in order.
//! `ptyrs_fft::Plan2d` transforms take the same context and lease the same way, so they mix
//! freely with the launches here, in a `record` closure or a capture.

use std::future::IntoFuture;
use std::sync::Arc;

use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig, RawDeviceBuffer,
};
use ptyrs_fft::FftError;
use ptyrs_kernels::kernels::LoadedModule;

mod ops;

pub use ops::{LlkPartial, LlkSums, MASKED, OBJ_TILE, UpdateSums, ViewAddr, ViewDims, Views};
pub use ptyrs_fft::C32;

/// The linked device code: every `ptyrs-kernels` kernel plus the cuFFT functions they call.
pub const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ptyrs_kernels.cubin"));

/// The architecture [`CUBIN`] was built for, e.g. `sm_89`.
pub const ARCH: &str = env!("PTYRS_GPU_ARCH");

/// Launch traits of one in-kernel FFT, read from libcufft when the cubin was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockFft {
    pub n: u32,
    /// Complex elements per thread.
    pub ept: u32,
    pub inverse: bool,
    pub threads_per_fft: u32,
    /// Dynamic shared memory per FFT, bytes.
    pub smem_per_fft: u32,
    pub suggested_ffts_per_block: u32,
}

include!(concat!(env!("OUT_DIR"), "/block_fft.rs"));

/// The traits of the `n`-point FFT in `CUBIN`, if it has one.
pub fn block_fft(n: u32, inverse: bool) -> Option<BlockFft> {
    BLOCK_FFTS
        .iter()
        .copied()
        .find(|f| f.n == n && f.inverse == inverse)
}

#[derive(Debug)]
pub enum Error {
    Driver(DriverError),
    /// A cuda-async error, e.g. a lease refused because another stream holds the buffer.
    Device(DeviceError),
    /// The device's compute capability is not the one [`CUBIN`] was built for.
    Arch {
        built: &'static str,
        device: (i32, i32),
    },
    /// A buffer, stream and module do not all belong to one context.
    ContextMismatch,
    /// A size is zero or out of range, a layout overlaps, or a buffer is too short.
    Shape(String),
    /// Non-zero `cufftResult` from a `ptyrs_fft::Plan2d` call.
    Cufft(i32),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e:?}"),
            Self::Device(e) => write!(f, "{e}"),
            Self::Arch { built, device } => write!(
                f,
                "kernels were built for {built}, device is sm_{}{}; rebuild with \
                 PTYRS_CUDA_ARCH=sm_{}{}",
                device.0, device.1, device.0, device.1
            ),
            Self::ContextMismatch => write!(f, "buffers, stream and kernels span contexts"),
            Self::Shape(what) => write!(f, "bad shape: {what}"),
            Self::Cufft(r) => write!(f, "cuFFT error {r}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<DriverError> for Error {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<DeviceError> for Error {
    fn from(e: DeviceError) -> Self {
        Self::Device(e)
    }
}

impl From<FftError> for Error {
    fn from(e: FftError) -> Self {
        match e {
            FftError::Cufft(r) => Self::Cufft(r),
            FftError::Driver(e) => Self::Driver(e),
            FftError::Device(e) => Self::Device(e),
            FftError::BufferTooSmall { need, have } => {
                Self::Shape(format!("FFT buffer holds {have} elements, needs {need}"))
            }
            FftError::ContextMismatch => Self::ContextMismatch,
            FftError::BadShape => Self::Shape("FFT side or batch is zero or exceeds i32".into()),
        }
    }
}

impl From<Error> for DeviceError {
    fn from(e: Error) -> Self {
        match e {
            Error::Device(e) => e,
            Error::Driver(e) => Self::Driver(e),
            e => Self::Launch(e.to_string()),
        }
    }
}

/// [`ptyrs_fft::lease`], the one access rule of every launch, with this crate's [`Error`].
pub fn lease(
    ctx: &ExecutionContext,
    reads: &[&RawDeviceBuffer],
    writes: &[&RawDeviceBuffer],
) -> Result<(), Error> {
    Ok(ptyrs_fft::lease(ctx, reads, writes)?)
}

/// Zeroes `buf` on `ctx`'s stream (a memset node under capture), leased like a launch.
pub fn zero<T: DeviceCopy>(ctx: &ExecutionContext, buf: &mut DeviceBuffer<T>) -> Result<(), Error> {
    if buf.num_bytes() == 0 {
        return Ok(());
    }
    lease(ctx, &[], &[buf.storage()])?;
    ctx.context().bind_to_thread()?;
    // SAFETY: the range is `buf`'s whole allocation, leased for writing on this stream.
    unsafe {
        cuda_core::memory::memset_d8_async(
            buf.cu_deviceptr(),
            0,
            buf.num_bytes(),
            ctx.get_cuda_stream().cu_stream(),
        )?
    };
    Ok(())
}

/// The `DeviceOp` that runs `f` on the context it executes on: what
/// `CudaGraph::capture` takes for a sequence of these launches, or what runs one eagerly
/// with `.sync_on(&stream)`. An error from `f` fails the op, and so the capture.
pub fn record<F>(f: F) -> Record<F>
where
    F: FnOnce(&ExecutionContext) -> Result<(), Error> + Send,
{
    Record(f)
}

/// See [`record`].
pub struct Record<F>(F);

impl<F> DeviceOp for Record<F>
where
    F: FnOnce(&ExecutionContext) -> Result<(), Error> + Send,
{
    type Output = ();

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        Ok((self.0)(ctx)?)
    }
}

impl<F> IntoFuture for Record<F>
where
    F: FnOnce(&ExecutionContext) -> Result<(), Error> + Send,
{
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => DeviceFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// `count` lines of `n` complex values: element `j` of line `l` is at
/// `l * line_stride + j * elem_stride` (complex units). Rows of a row-major `[h, n]` image
/// are `{ count: h, line_stride: n, elem_stride: 1 }`; its columns (with `h == n`) are
/// `{ count: n, line_stride: 1, elem_stride: n }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lines {
    pub n: u32,
    pub count: u32,
    pub line_stride: u32,
    pub elem_stride: u32,
}

impl Lines {
    /// Elements a buffer must hold. Errors unless every (line, element) has its own offset.
    fn extent(&self) -> Result<usize, Error> {
        let Self {
            n,
            count,
            line_stride: ls,
            elem_stride: es,
        } = *self;
        let (n, count, ls, es) = (n as u128, count as u128, ls as u128, es as u128);
        if n == 0 || count == 0 || ls == 0 || es == 0 || (ls < n * es && es < count * ls) {
            return Err(Error::Shape(format!("{self:?} is empty or overlaps")));
        }
        usize::try_from((count - 1) * ls + (n - 1) * es + 1)
            .map_err(|_| Error::Shape(format!("{self:?} exceeds the address space")))
    }
}

fn check_len<T: DeviceCopy>(what: &str, buf: &DeviceBuffer<T>, need: usize) -> Result<(), Error> {
    if buf.len() < need {
        return Err(Error::Shape(format!(
            "{what} holds {} elements, needs {need}",
            buf.len()
        )));
    }
    Ok(())
}

/// Device address of a buffer as the kernels take it (complex buffers as `*mut f32`).
fn dptr<T: DeviceCopy, U>(buf: &DeviceBuffer<T>) -> *mut U {
    buf.cu_deviceptr() as *mut U
}

/// The loaded kernels of [`CUBIN`] in one context.
#[derive(Clone, Debug)]
pub struct Kernels {
    module: LoadedModule,
    ctx: Arc<CudaContext>,
}

impl Kernels {
    /// Loads [`CUBIN`] into `ctx`, checking its device matches [`ARCH`].
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, Error> {
        let device = ctx.compute_capability()?;
        if format!("sm_{}{}", device.0, device.1) != ARCH {
            return Err(Error::Arch {
                built: ARCH,
                device,
            });
        }
        let module = ptyrs_kernels::kernels::from_module(ctx.load_module_from_image(CUBIN)?)?;
        Ok(Self {
            module,
            ctx: ctx.clone(),
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// [`lease`]s `reads` and `writes` with `ctx`, then runs `enqueue` on its stream.
    fn launch(
        &self,
        ctx: &ExecutionContext,
        reads: &[&RawDeviceBuffer],
        writes: &[&RawDeviceBuffer],
        enqueue: impl FnOnce(&CudaStream) -> Result<(), DriverError>,
    ) -> Result<(), Error> {
        if ctx.context().as_ref() != self.ctx.as_ref() {
            return Err(Error::ContextMismatch);
        }
        lease(ctx, reads, writes)?;
        Ok(enqueue(ctx.get_cuda_stream())?)
    }

    /// `x[i] *= s` for the first `n` values of `x`.
    pub fn scale_c32(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        n: u32,
        s: f32,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        if n == 0 {
            return Ok(());
        }
        self.launch(ctx, &[], &[x.storage()], |stream| {
            // SAFETY: `x` holds `n` values and the kernel guards its 1D index against `n`.
            unsafe {
                self.module
                    .scale_c32(stream, LaunchConfig::for_num_elems(n), dptr(x), n, s)
            }
        })
    }

    /// Unnormalized C2C FFT of every line of `lines` in `input`, written with the same layout
    /// to `output`; the forward kernel is `exp(-2 pi i jk / n)`. Each line is one in-kernel
    /// cuFFT call; [`BLOCK_FFTS`] lists the sizes `n` built in.
    pub fn fft_lines(
        &self,
        ctx: &ExecutionContext,
        input: &DeviceBuffer<C32>,
        output: &mut DeviceBuffer<C32>,
        lines: Lines,
        inverse: bool,
    ) -> Result<(), Error> {
        let need = lines.extent()?;
        check_len("input", input, need)?;
        check_len("output", output, need)?;
        let (launch, config) = fft_lines_launch(lines, inverse)?;
        self.launch(ctx, &[input.storage()], &[output.storage()], |stream| {
            // SAFETY: both buffers hold `lines`, whose lines do not overlap; `config` is the
            // launch shape the kernel documents.
            unsafe {
                launch(
                    &self.module,
                    stream,
                    config,
                    dptr(input),
                    dptr(output),
                    lines.count,
                    lines.line_stride,
                    lines.elem_stride,
                )
            }
        })
    }

    /// [`Self::fft_lines`] in place.
    pub fn fft_lines_inplace(
        &self,
        ctx: &ExecutionContext,
        data: &mut DeviceBuffer<C32>,
        lines: Lines,
        inverse: bool,
    ) -> Result<(), Error> {
        check_len("data", data, lines.extent()?)?;
        let (launch, config) = fft_lines_launch(lines, inverse)?;
        self.launch(ctx, &[], &[data.storage()], |stream| {
            // SAFETY: as in `fft_lines`; each thread reads its elements before the FFT's
            // block barrier and writes them after it.
            unsafe {
                launch(
                    &self.module,
                    stream,
                    config,
                    dptr(data),
                    dptr(data),
                    lines.count,
                    lines.line_stride,
                    lines.elem_stride,
                )
            }
        })
    }
}

type LinesLaunch = unsafe fn(
    &LoadedModule,
    &CudaStream,
    LaunchConfig,
    *const f32,
    *mut f32,
    u32,
    u32,
    u32,
) -> Result<(), DriverError>;

/// The line-FFT kernel for `lines.n` and its launch shape: blockDim = (threads per FFT,
/// FFTs per block), one line per FFT, dynamic shared memory from the cuFFT traits.
fn fft_lines_launch(lines: Lines, inverse: bool) -> Result<(LinesLaunch, LaunchConfig), Error> {
    let launch: LinesLaunch = match (lines.n, inverse) {
        (128, false) => LoadedModule::fft128_lines_fwd,
        (128, true) => LoadedModule::fft128_lines_inv,
        (256, false) => LoadedModule::fft256_lines_fwd,
        (256, true) => LoadedModule::fft256_lines_inv,
        _ => return Err(Error::Shape(format!("no {}-point line FFT", lines.n))),
    };
    let fft = block_fft(lines.n, inverse).expect("every line kernel's FFT is in BLOCK_FFTS");
    let per_block = fft.suggested_ffts_per_block;
    let config = LaunchConfig {
        grid_dim: (lines.count.div_ceil(per_block), 1, 1),
        block_dim: (fft.threads_per_fft, per_block, 1),
        shared_mem_bytes: fft.smem_per_fft * per_block,
    };
    Ok((launch, config))
}
