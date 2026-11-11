//! M0: a cuda-oxide kernel calls a cuFFT device-API block FFT (C2C f32, 128 and 256 points,
//! sm_89) through LTOIR linking, checked against an f64 DFT.
//!
//! Run with `cargo oxide run --emit-nvvm-ir --arch=sm_89 --bin m0_fft_device`. The embedded
//! NVVM IR becomes LTOIR here and nvJitLink links it with the LTOIR and fatbin pieces that
//! libcufft's device API (`cufft_device.h` 0.6.0) hands out for each FFT.

use cuda_artifact_finalizer::{CudaArch, FinalizationOptions, NvvmCompiler};
use cuda_core::artifacts::ArtifactPayloadKind;
use cuda_core::embedded::artifact_bundles_from_current_exe;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{device, kernel, shared::DynamicSharedArray, thread};
use cuda_host::cuda_module;
use libloading::Library;
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use std::error::Error;
use std::ffi::{c_char, c_int};

type Res<T> = Result<T, Box<dyn Error>>;

/// Complex elements per thread of every function linked here (all are EPT 16, BY_COMPONENT).
const EPT: usize = 16;

// Each symbol is cufftdx::database::detail::cufftdx_private_lto_function<600, ID, float, 1>(
//     float* rmem, void* smem, int dir, unsigned batchid, unsigned fftid, unsigned bpb),
// the entry libcufft generates per function. ID is `function_id` below, so the name pins
// size, direction, EPT 16, BY_COMPONENT and sm_89. Each identifier is the Itanium-mangled
// name the pieces export.
#[device]
unsafe extern "C" {
    fn _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy238926430336EfLj1EEEvPT1_Pvijjj(
        rmem: *mut f32,
        smem: *mut u8,
        dir: i32,
        batchid: u32,
        fftid: u32,
        bpb: u32,
    );
    fn _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy9035019452544EfLj1EEEvPT1_Pvijjj(
        rmem: *mut f32,
        smem: *mut u8,
        dir: i32,
        batchid: u32,
        fftid: u32,
        bpb: u32,
    );
    fn _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy238926430464EfLj1EEEvPT1_Pvijjj(
        rmem: *mut f32,
        smem: *mut u8,
        dir: i32,
        batchid: u32,
        fftid: u32,
        bpb: u32,
    );
    fn _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy9035019452672EfLj1EEEvPT1_Pvijjj(
        rmem: *mut f32,
        smem: *mut u8,
        dir: i32,
        batchid: u32,
        fftid: u32,
        bpb: u32,
    );
}

/// Batch index of this thread: blockDim = (threads per FFT, FFTs per block).
#[device]
fn batch_index() -> u32 {
    thread::blockIdx_x() * thread::blockDim_y() + thread::threadIdx_y()
}

/// Loads this thread's registers. Complex element `j` of batch `b` sits at
/// `b * batch_stride + j * elem_stride`, and thread `fftid` holds `j = fftid + i * n / EPT`.
#[device]
fn load_regs(
    input: &[f32],
    n: u32,
    nbatch: u32,
    batch_stride: u32,
    elem_stride: u32,
) -> [f32; 2 * EPT] {
    let mut r = [0.0f32; 2 * EPT];
    let b = batch_index();
    if b < nbatch {
        for i in 0..EPT {
            let j = thread::threadIdx_x() + i as u32 * (n / EPT as u32);
            let at = 2 * (b * batch_stride + j * elem_stride) as usize;
            r[2 * i] = input[at];
            r[2 * i + 1] = input[at + 1];
        }
    }
    r
}

/// Stores this thread's registers with the layout of [`load_regs`].
///
/// # Safety
///
/// `output` must cover every element of that layout for `nbatch` batches.
#[device]
unsafe fn store_regs(
    output: *mut f32,
    r: &[f32; 2 * EPT],
    n: u32,
    nbatch: u32,
    batch_stride: u32,
    elem_stride: u32,
) {
    let b = batch_index();
    if b < nbatch {
        for i in 0..EPT {
            let j = thread::threadIdx_x() + i as u32 * (n / EPT as u32);
            let at = 2 * (b * batch_stride + j * elem_stride) as usize;
            unsafe {
                *output.add(at) = r[2 * i];
                *output.add(at + 1) = r[2 * i + 1];
            }
        }
    }
}

// Threads past `nbatch` still call the FFT: it synchronizes the block on shared memory.
// The launch must give blockDim.x = n / EPT and bpb * smem-per-FFT bytes of dynamic shared memory.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fft128_fwd(input: &[f32], output: *mut f32, nbatch: u32, bstride: u32, estride: u32) {
        let mut r = load_regs(input, 128, nbatch, bstride, estride);
        unsafe {
            _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy238926430336EfLj1EEEvPT1_Pvijjj(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                0,
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_regs(output, &r, 128, nbatch, bstride, estride);
        }
    }

    #[kernel]
    pub fn fft128_inv(input: &[f32], output: *mut f32, nbatch: u32, bstride: u32, estride: u32) {
        let mut r = load_regs(input, 128, nbatch, bstride, estride);
        unsafe {
            _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy9035019452544EfLj1EEEvPT1_Pvijjj(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                1,
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_regs(output, &r, 128, nbatch, bstride, estride);
        }
    }

    #[kernel]
    pub fn fft256_fwd(input: &[f32], output: *mut f32, nbatch: u32, bstride: u32, estride: u32) {
        let mut r = load_regs(input, 256, nbatch, bstride, estride);
        unsafe {
            _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy238926430464EfLj1EEEvPT1_Pvijjj(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                0,
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_regs(output, &r, 256, nbatch, bstride, estride);
        }
    }

    #[kernel]
    pub fn fft256_inv(input: &[f32], output: *mut f32, nbatch: u32, bstride: u32, estride: u32) {
        let mut r = load_regs(input, 256, nbatch, bstride, estride);
        unsafe {
            _ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj600ELy9035019452672EfLj1EEEvPT1_Pvijjj(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                1,
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_regs(output, &r, 256, nbatch, bstride, estride);
        }
    }
}

/// Compute capability the extern symbols above were generated for.
const CAPABILITY: u32 = 89;

/// `function_id` of the cuFFT device API, as asserted by the entry source libcufft generates.
/// Single precision and C2C are zero, so their fields (bits 39 and 44) drop out.
const fn function_id(n: u64, ept: u64, transposition: u64, capability: u64, dir: u64) -> u64 {
    n | ept << 17 | (transposition - 1) << 24 | (capability * 10) << 28 | dir << 43
}

#[derive(Clone, Copy, Debug)]
enum Fft {
    Fwd128,
    Inv128,
    Fwd256,
    Inv256,
}

impl Fft {
    const ALL: [Fft; 4] = [Fft::Fwd128, Fft::Inv128, Fft::Fwd256, Fft::Inv256];

    fn n(self) -> u32 {
        match self {
            Fft::Fwd128 | Fft::Inv128 => 128,
            Fft::Fwd256 | Fft::Inv256 => 256,
        }
    }

    fn dir(self) -> u32 {
        match self {
            Fft::Fwd128 | Fft::Fwd256 => 0,
            Fft::Inv128 | Fft::Inv256 => 1,
        }
    }

    /// The ID spelled in this FFT's extern symbol.
    fn symbol_id(self) -> u64 {
        match self {
            Fft::Fwd128 => 238926430336,
            Fft::Inv128 => 9035019452544,
            Fft::Fwd256 => 238926430464,
            Fft::Inv256 => 9035019452672,
        }
    }

    /// Exponent sign of the unnormalized DFT this computes.
    fn sign(self) -> f64 {
        if self.dir() == 0 { -1.0 } else { 1.0 }
    }
}

// cufft_device.h enums.
const DESC_SIZE: c_int = 0;
const DESC_DIRECTION: c_int = 1;
const DESC_PRECISION: c_int = 2;
const DESC_TYPE: c_int = 3;
const DESC_SM: c_int = 4;
const DESC_SINGLE: i64 = 0;
const DESC_C2C: i64 = 0;
const FUNC_EPT: c_int = 0;
const FUNC_STORAGE: c_int = 1;
const FUNC_SMEM_PER_FFT: c_int = 2;
const FUNC_THREADS: c_int = 3;
const FUNC_SUGGESTED_BPB: c_int = 4;
const FUNC_TRANSPOSITION: c_int = 5;
const TRANSPOSITION_BY_COMPONENT: i64 = 2;
const CONTAINER_LTOIR: c_int = 0;
const CONTAINER_FATBIN: c_int = 1;
const DEVICE_API_VERSION: c_int = 600;

/// The cuFFT device-API entry points (`cufft_device.h`), resolved from libcufft at run time.
struct CufftDevice {
    get_version: unsafe extern "C" fn(*mut c_int) -> c_int,
    desc_create: unsafe extern "C" fn(*mut c_int) -> c_int,
    desc_destroy: unsafe extern "C" fn(c_int) -> c_int,
    desc_set: unsafe extern "C" fn(c_int, c_int, i64) -> c_int,
    dev_create: unsafe extern "C" fn(*mut c_int, usize, *mut c_int) -> c_int,
    dev_destroy: unsafe extern "C" fn(c_int) -> c_int,
    num_funcs: unsafe extern "C" fn(c_int, c_int, *mut usize) -> c_int,
    funcs: unsafe extern "C" fn(c_int, c_int, usize, *mut c_int) -> c_int,
    func_trait: unsafe extern "C" fn(c_int, c_int, c_int, *mut i64) -> c_int,
    db_size: unsafe extern "C" fn(c_int, *mut usize, c_int) -> c_int,
    db_str: unsafe extern "C" fn(c_int, usize, *mut c_char, c_int) -> c_int,
    num_ltoirs: unsafe extern "C" fn(c_int, *mut usize, c_int) -> c_int,
    ltoir_sizes: unsafe extern "C" fn(c_int, usize, *mut usize, c_int) -> c_int,
    ltoirs: unsafe extern "C" fn(c_int, usize, *mut *mut c_char, *mut c_int, c_int) -> c_int,
    // Keeps the function pointers above valid.
    _lib: Library,
}

/// One device function picked from libcufft: its traits, ID and code pieces.
struct DeviceFft {
    id: u64,
    ept: i64,
    storage: i64,
    smem_per_fft: i64,
    threads: i64,
    suggested_bpb: i64,
    transposition: i64,
    pieces: Vec<(InputType, Vec<u8>)>,
}

fn ck(call: &str, status: c_int) -> Res<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{call} returned cufftResult {status}").into())
    }
}

impl CufftDevice {
    fn load() -> Res<Self> {
        // SAFETY: loading libcufft runs no initialisers with preconditions, and every symbol
        // type below matches its declaration in cufft_device.h.
        unsafe {
            let lib = Library::new("libcufft.so.12").or_else(|_| Library::new("libcufft.so"))?;
            let api = Self {
                get_version: *lib.get(b"cufftDeviceGetVersion\0")?,
                desc_create: *lib.get(b"cufftDescriptionCreate\0")?,
                desc_destroy: *lib.get(b"cufftDescriptionDestroy\0")?,
                desc_set: *lib.get(b"cufftDescriptionSetTraitInt64\0")?,
                dev_create: *lib.get(b"cufftDeviceCreate\0")?,
                dev_destroy: *lib.get(b"cufftDeviceDestroy\0")?,
                num_funcs: *lib.get(b"cufftDeviceGetNumDeviceFunctions\0")?,
                funcs: *lib.get(b"cufftDeviceGetDeviceFunctions\0")?,
                func_trait: *lib.get(b"cufftDeviceGetDeviceFunctionTraitInt64\0")?,
                db_size: *lib.get(b"cufftDeviceGetDatabaseStrSize\0")?,
                db_str: *lib.get(b"cufftDeviceGetDatabaseStr\0")?,
                num_ltoirs: *lib.get(b"cufftDeviceGetNumLTOIRs\0")?,
                ltoir_sizes: *lib.get(b"cufftDeviceGetLTOIRSizes\0")?,
                ltoirs: *lib.get(b"cufftDeviceGetLTOIRs\0")?,
                _lib: lib,
            };
            let mut version = 0;
            ck("cufftDeviceGetVersion", (api.get_version)(&mut version))?;
            // The entry symbols carry the version as their first template argument.
            if version != DEVICE_API_VERSION {
                return Err(format!("cuFFT device API {version}, symbols expect 600").into());
            }
            Ok(api)
        }
    }

    /// Picks the EPT 16 / BY_COMPONENT function for one size and direction.
    fn query(&self, fft: Fft, capability: u32) -> Res<DeviceFft> {
        let mut desc = 0;
        ck("cufftDescriptionCreate", unsafe {
            (self.desc_create)(&mut desc)
        })?;
        let found = self.query_desc(desc, fft, capability);
        let destroyed = ck("cufftDescriptionDestroy", unsafe {
            (self.desc_destroy)(desc)
        });
        let found = found?;
        destroyed?;
        Ok(found)
    }

    fn query_desc(&self, desc: c_int, fft: Fft, capability: u32) -> Res<DeviceFft> {
        for (which, value) in [
            (DESC_SIZE, i64::from(fft.n())),
            (DESC_DIRECTION, i64::from(fft.dir())),
            (DESC_PRECISION, DESC_SINGLE),
            (DESC_TYPE, DESC_C2C),
            (DESC_SM, i64::from(capability * 10)),
        ] {
            ck("cufftDescriptionSetTraitInt64", unsafe {
                (self.desc_set)(desc, which, value)
            })?;
        }
        let (mut dev, mut descs) = (0, [desc]);
        ck("cufftDeviceCreate", unsafe {
            (self.dev_create)(&mut dev, 1, descs.as_mut_ptr())
        })?;
        let found = self.select(dev, desc);
        let destroyed = ck("cufftDeviceDestroy", unsafe { (self.dev_destroy)(dev) });
        let found = found?;
        destroyed?;
        Ok(found)
    }

    fn select(&self, dev: c_int, desc: c_int) -> Res<DeviceFft> {
        let mut count = 0;
        ck("cufftDeviceGetNumDeviceFunctions", unsafe {
            (self.num_funcs)(dev, desc, &mut count)
        })?;
        let mut handles = vec![0; count];
        ck("cufftDeviceGetDeviceFunctions", unsafe {
            (self.funcs)(dev, desc, count, handles.as_mut_ptr())
        })?;
        for func in handles {
            let get = |which| -> Res<i64> {
                let mut value = 0;
                ck("cufftDeviceGetDeviceFunctionTraitInt64", unsafe {
                    (self.func_trait)(dev, func, which, &mut value)
                })?;
                Ok(value)
            };
            if get(FUNC_EPT)? != EPT as i64
                || get(FUNC_TRANSPOSITION)? != TRANSPOSITION_BY_COMPONENT
            {
                continue;
            }
            return Ok(DeviceFft {
                id: self.database_id(dev, func)?,
                ept: get(FUNC_EPT)?,
                storage: get(FUNC_STORAGE)?,
                smem_per_fft: get(FUNC_SMEM_PER_FFT)?,
                threads: get(FUNC_THREADS)?,
                suggested_bpb: get(FUNC_SUGGESTED_BPB)?,
                transposition: get(FUNC_TRANSPOSITION)?,
                pieces: self.pieces(dev, func)?,
            });
        }
        Err("libcufft offers no EPT 16 / BY_COMPONENT function".into())
    }

    /// The function ID: field 7 of `block_fft_lto_implementation<...>` in the database string.
    fn database_id(&self, dev: c_int, func: c_int) -> Res<u64> {
        let mut size = 0;
        ck("cufftDeviceGetDatabaseStrSize", unsafe {
            (self.db_size)(dev, &mut size, func)
        })?;
        let mut buf = vec![0u8; size];
        ck("cufftDeviceGetDatabaseStr", unsafe {
            (self.db_str)(dev, size, buf.as_mut_ptr().cast(), func)
        })?;
        let db = String::from_utf8_lossy(&buf);
        let key = "block_fft_lto_implementation<";
        let args = db
            .split_once(key)
            .and_then(|(_, rest)| rest.split_once('>'))
            .ok_or("database string has no block_fft_lto_implementation")?
            .0;
        let id = args.split(',').nth(6).ok_or("database string too short")?;
        Ok(id.trim().parse()?)
    }

    fn pieces(&self, dev: c_int, func: c_int) -> Res<Vec<(InputType, Vec<u8>)>> {
        let mut count = 0;
        ck("cufftDeviceGetNumLTOIRs", unsafe {
            (self.num_ltoirs)(dev, &mut count, func)
        })?;
        let mut sizes = vec![0usize; count];
        ck("cufftDeviceGetLTOIRSizes", unsafe {
            (self.ltoir_sizes)(dev, count, sizes.as_mut_ptr(), func)
        })?;
        let mut bufs: Vec<Vec<u8>> = sizes.iter().map(|&size| vec![0u8; size]).collect();
        let mut ptrs: Vec<*mut c_char> = bufs.iter_mut().map(|b| b.as_mut_ptr().cast()).collect();
        let mut containers = vec![0; count];
        ck("cufftDeviceGetLTOIRs", unsafe {
            (self.ltoirs)(dev, count, ptrs.as_mut_ptr(), containers.as_mut_ptr(), func)
        })?;
        containers
            .into_iter()
            .zip(bufs)
            .map(|(container, bytes)| match container {
                CONTAINER_LTOIR => Ok((InputType::Ltoir, bytes)),
                CONTAINER_FATBIN => Ok((InputType::Fatbin, bytes)),
                other => Err(format!("unknown cufftDeviceCodeContainer {other}").into()),
            })
            .collect()
    }
}

/// Where the FFTs of one launch live, in complex elements (see [`load_regs`]).
#[derive(Clone, Copy)]
struct Layout {
    nbatch: u32,
    batch_stride: u32,
    elem_stride: u32,
}

struct Launcher<'a> {
    module: &'a kernels::LoadedModule,
    stream: &'a CudaStream,
}

impl Launcher<'_> {
    fn launch(
        &self,
        fft: Fft,
        dfft: &DeviceFft,
        bpb: u32,
        at: Layout,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let cfg = LaunchConfig {
            grid_dim: (at.nbatch.div_ceil(bpb), 1, 1),
            block_dim: (dfft.threads as u32, bpb, 1),
            shared_mem_bytes: dfft.smem_per_fft as u32 * bpb,
        };
        let out = output.cu_deviceptr() as *mut f32;
        let (m, s, e) = (self.module, self.stream, cfg);
        let (n, bs, es) = (at.nbatch, at.batch_stride, at.elem_stride);
        // SAFETY: blockDim.x and the dynamic shared memory come from the function's traits,
        // and the callers size `input` and `output` to cover `at`.
        unsafe {
            match fft {
                Fft::Fwd128 => m.fft128_fwd(s, e, input, out, n, bs, es),
                Fft::Inv128 => m.fft128_inv(s, e, input, out, n, bs, es),
                Fft::Fwd256 => m.fft256_fwd(s, e, input, out, n, bs, es),
                Fft::Inv256 => m.fft256_inv(s, e, input, out, n, bs, es),
            }
        }
    }

    /// `nbatch` contiguous FFTs, `bpb` per block.
    fn check_batched(&self, fft: Fft, dfft: &DeviceFft, nbatch: u32, bpb: u32) -> Res<f64> {
        let n = fft.n() as usize;
        let host = random_complex(n * nbatch as usize, 0x9e37_79b9 + n as u64);
        let input = DeviceBuffer::from_host(self.stream, &host)?;
        let mut output = DeviceBuffer::zeroed(self.stream, host.len())?;
        let at = Layout {
            nbatch,
            batch_stride: n as u32,
            elem_stride: 1,
        };
        self.launch(fft, dfft, bpb, at, &input, &mut output)?;
        let gpu = output.to_host_vec(self.stream)?;
        let reference = dft_lines(&widen(&host), n, nbatch as usize, n, 1, fft.sign());
        Ok(rel_err(&gpu, &reference))
    }

    /// An n x n frame: one launch over the rows, then one over the columns.
    fn check_2d(&self, fft: Fft, dfft: &DeviceFft) -> Res<f64> {
        let n = fft.n() as usize;
        let host = random_complex(n * n, 0x2545_f491 + n as u64);
        let input = DeviceBuffer::from_host(self.stream, &host)?;
        let mut rows = DeviceBuffer::zeroed(self.stream, host.len())?;
        let mut frame = DeviceBuffer::zeroed(self.stream, host.len())?;
        let (len, bpb) = (n as u32, dfft.suggested_bpb as u32);
        let by_row = Layout {
            nbatch: len,
            batch_stride: len,
            elem_stride: 1,
        };
        let by_col = Layout {
            nbatch: len,
            batch_stride: 1,
            elem_stride: len,
        };
        self.launch(fft, dfft, bpb, by_row, &input, &mut rows)?;
        self.launch(fft, dfft, bpb, by_col, &rows, &mut frame)?;
        let gpu = frame.to_host_vec(self.stream)?;
        let row_dft = dft_lines(&widen(&host), n, n, n, 1, fft.sign());
        let reference = dft_lines(&row_dft, n, n, 1, n, fft.sign());
        Ok(rel_err(&gpu, &reference))
    }
}

/// Interleaved complex values with both parts in [-1, 1), from xorshift64.
fn random_complex(len: usize, mut state: u64) -> Vec<f32> {
    (0..2 * len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn widen(x: &[f32]) -> Vec<[f64; 2]> {
    x.as_chunks::<2>()
        .0
        .iter()
        .map(|&[re, im]| [f64::from(re), f64::from(im)])
        .collect()
}

/// Unnormalized f64 DFT along `nlines` lines of `n` complex values; element `j` of line `l`
/// is `x[l * line_stride + j * elem_stride]`.
fn dft_lines(
    x: &[[f64; 2]],
    n: usize,
    nlines: usize,
    line_stride: usize,
    elem_stride: usize,
    sign: f64,
) -> Vec<[f64; 2]> {
    let twiddle: Vec<[f64; 2]> = (0..n)
        .map(|k| {
            let a = sign * std::f64::consts::TAU * k as f64 / n as f64;
            [a.cos(), a.sin()]
        })
        .collect();
    let mut y = x.to_vec();
    for l in 0..nlines {
        let at = |j: usize| l * line_stride + j * elem_stride;
        for k in 0..n {
            let mut acc = [0.0; 2];
            for j in 0..n {
                let (v, w) = (x[at(j)], twiddle[j * k % n]);
                acc[0] += v[0] * w[0] - v[1] * w[1];
                acc[1] += v[0] * w[1] + v[1] * w[0];
            }
            y[at(k)] = acc;
        }
    }
    y
}

/// max |gpu - ref| / max |ref|.
fn rel_err(gpu: &[f32], reference: &[[f64; 2]]) -> f64 {
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (&[re, im], r) in gpu.as_chunks::<2>().0.iter().zip(reference) {
        err = err.max((f64::from(re) - r[0]).hypot(f64::from(im) - r[1]));
        norm = norm.max(r[0].hypot(r[1]));
    }
    err / norm
}

const TOLERANCE: f64 = 1e-5;

fn main() -> Res<()> {
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|b| b.name == env!("CARGO_PKG_NAME"))
        .ok_or("no embedded artifact bundle")?;
    let nvvm_ir = bundle
        .payload(ArtifactPayloadKind::NvvmIr)
        .ok_or("no NVVM IR payload: run with `cargo oxide run --emit-nvvm-ir --arch=sm_89`")?;
    let arch: CudaArch = bundle.target.parse()?;
    if arch.capability() != CAPABILITY {
        return Err(format!("built for {}, the cuFFT symbols are sm_89", bundle.target).into());
    }
    let options = FinalizationOptions::new(arch.clone())
        .with_fma_contraction(bundle.compile_options.fma_contraction_enabled());
    let kernel_ltoir =
        NvvmCompiler::discover()?.compile_nvvm_ir_to_ltoir(&bundle.name, nvvm_ir, &options)?;

    let cufft = CufftDevice::load()?;
    let mut dffts = Vec::new();
    for fft in Fft::ALL {
        let d = cufft.query(fft, CAPABILITY)?;
        let want = function_id(
            u64::from(fft.n()),
            EPT as u64,
            TRANSPOSITION_BY_COMPONENT as u64,
            u64::from(CAPABILITY),
            u64::from(fft.dir()),
        );
        println!(
            "{fft:?}: ept={} storage={} threads={} smem/fft={} sugg_bpb={} transposition={} \
             pieces={} db_id={} formula_id={want} symbol_id={}",
            d.ept,
            d.storage,
            d.threads,
            d.smem_per_fft,
            d.suggested_bpb,
            d.transposition,
            d.pieces.len(),
            d.id,
            fft.symbol_id(),
        );
        if d.id != fft.symbol_id() || want != fft.symbol_id() {
            return Err(format!("{fft:?}: libcufft function is not the linked symbol").into());
        }
        dffts.push(d);
    }

    let nvjitlink = LibNvJitLink::load()?;
    let arch_option = format!("-arch={}", arch.sm());
    let mut linker = Linker::new(&nvjitlink, &[&arch_option, "-lto"])?;
    linker.add(InputType::Ltoir, &kernel_ltoir, &bundle.name)?;
    for (fft, d) in Fft::ALL.iter().zip(&dffts) {
        for (k, (kind, bytes)) in d.pieces.iter().enumerate() {
            linker.add(*kind, bytes, &format!("cufft_{fft:?}_{k}"))?;
        }
    }
    let cubin = linker.finish()?;
    println!(
        "linked kernel LTOIR + cuFFT pieces into a {} byte cubin",
        cubin.len()
    );

    let ctx = CudaContext::new(0)?;
    let module = kernels::from_module(ctx.load_module_from_image(&cubin)?)?;
    let stream = ctx.default_stream();
    let launcher = Launcher {
        module: &module,
        stream: &stream,
    };

    let mut failures = 0;
    let mut report = |what: String, err: f64| {
        let pass = err <= TOLERANCE;
        failures += usize::from(!pass);
        println!(
            "{what}: rel_err={err:.3e} {}",
            if pass { "PASS" } else { "FAIL" }
        );
    };
    for (&fft, d) in Fft::ALL.iter().zip(&dffts) {
        let bpb = d.suggested_bpb as u32;
        for (nbatch, bpb) in [(1001, bpb), (37, 1)] {
            let err = launcher.check_batched(fft, d, nbatch, bpb)?;
            report(format!("{fft:?} batch={nbatch} bpb={bpb}"), err);
        }
    }
    for (&fft, d) in Fft::ALL.iter().zip(&dffts).skip(2) {
        let err = launcher.check_2d(fft, d)?;
        report(format!("{fft:?} 2D 256x256 rows then columns"), err);
    }
    if failures > 0 {
        return Err(format!("{failures} checks over tolerance {TOLERANCE:e}").into());
    }
    Ok(())
}
