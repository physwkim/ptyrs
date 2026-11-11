//! ptyrs-gpu: the ptyrs kernels, built, linked and loaded for any host crate.
//!
//! Depending on this crate is all a library, test or binary needs: its build script runs
//! `cargo oxide build` on `ptyrs-kernels`, links the device code with the cuFFT device-API
//! functions it calls and embeds the cubin ([`CUBIN`], built for [`ARCH`]; set
//! `PTYRS_CUDA_ARCH`, default `sm_89`, to change it). [`Kernels::load`] loads it into a
//! context.
//!
//! Every launch method checks buffer lengths against the shape it is given, then enqueues on
//! `stream` and returns. Each buffer it touches is leased on that stream until the kernel
//! completes: dropping a buffer early is safe, and a conflicting access to it from another
//! stream fails with `CUDA_ERROR_NOT_READY` until then. Work on one stream runs in order.

use std::sync::Arc;

use cuda_core::{
    AllocationLease, CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig,
};
use ptyrs_kernels::kernels::LoadedModule;

mod ops;
mod rpie;

pub use ops::{LlkPartial, LlkSums, MASKED, OBJ_TILE, UpdateSums, ViewAddr, ViewDims, Views};
pub use ptyrs_fft::C32;
pub use rpie::{RpieKernels, Weight};

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
    /// The device's compute capability is not the one [`CUBIN`] was built for.
    Arch {
        built: &'static str,
        device: (i32, i32),
    },
    /// A buffer, stream and module do not all belong to one context.
    ContextMismatch,
    /// A size is zero or out of range, a layout overlaps, or a buffer is too short.
    Shape(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e:?}"),
            Self::Arch { built, device } => write!(
                f,
                "kernels were built for {built}, device is sm_{}{}; rebuild with \
                 PTYRS_CUDA_ARCH=sm_{}{}",
                device.0, device.1, device.0, device.1
            ),
            Self::ContextMismatch => write!(f, "buffers, stream and kernels span contexts"),
            Self::Shape(what) => write!(f, "bad shape: {what}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<DriverError> for Error {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
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

    /// Leases `reads` and `writes` on `stream`, runs `enqueue`, and keeps the leases until
    /// the work it enqueued completes.
    fn launch(
        &self,
        stream: &CudaStream,
        reads: &[&cuda_core::RawDeviceBuffer],
        writes: &[&cuda_core::RawDeviceBuffer],
        enqueue: impl FnOnce() -> Result<(), DriverError>,
    ) -> Result<(), Error> {
        if stream.context().as_ref() != self.ctx.as_ref() {
            return Err(Error::ContextMismatch);
        }
        let mut leases: Vec<AllocationLease> = Vec::with_capacity(reads.len() + writes.len());
        for (bufs, write) in [(reads, false), (writes, true)] {
            for buf in bufs {
                if buf.context().as_ref() != self.ctx.as_ref() {
                    return Err(Error::ContextMismatch);
                }
                leases.push(buf.acquire(stream, write, false)?);
            }
        }
        enqueue()?;
        for lease in leases {
            lease.retain_until_complete(stream)?;
        }
        Ok(())
    }

    /// `x[i] *= s` for the first `n` values of `x`.
    pub fn scale_c32(
        &self,
        stream: &CudaStream,
        x: &mut DeviceBuffer<C32>,
        n: u32,
        s: f32,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        if n == 0 {
            return Ok(());
        }
        self.launch(stream, &[], &[x.storage()], || {
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
        stream: &CudaStream,
        input: &DeviceBuffer<C32>,
        output: &mut DeviceBuffer<C32>,
        lines: Lines,
        inverse: bool,
    ) -> Result<(), Error> {
        let need = lines.extent()?;
        check_len("input", input, need)?;
        check_len("output", output, need)?;
        let (launch, config) = fft_lines_launch(lines, inverse)?;
        self.launch(stream, &[input.storage()], &[output.storage()], || {
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
        stream: &CudaStream,
        data: &mut DeviceBuffer<C32>,
        lines: Lines,
        inverse: bool,
    ) -> Result<(), Error> {
        check_len("data", data, lines.extent()?)?;
        let (launch, config) = fft_lines_launch(lines, inverse)?;
        self.launch(stream, &[], &[data.storage()], || {
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
