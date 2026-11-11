//! Builds ptyrs-kernels into one cubin for `PTYRS_CUDA_ARCH` (default `sm_89`).
//!
//! 1. A nested `cargo oxide build --emit-nvvm-ir` compiles ptyrs-kernels with the cuda-oxide
//!    backend into `<target>/oxide-device`, writing `ptyrs_kernels.{ll,options,target}` to
//!    `$OUT_DIR/device`.
//! 2. Every `ptyrs_cufft_c2c_f32_n{N}_e{EPT}_{fwd|inv}` the device code declares is defined in
//!    the IR as a call to the libcufft device-API function with those traits. Its ID and the
//!    API version come from libcufft's database string, not from constants here.
//! 3. libNVVM turns the IR into LTOIR; nvJitLink links it with the cuFFT LTOIR/fatbin pieces.
//!
//! Outputs: `$OUT_DIR/ptyrs_kernels.cubin`, `$OUT_DIR/block_fft.rs` (the FFT launch traits)
//! and `PTYRS_GPU_ARCH` for the crate.

use std::env;
use std::error::Error;
use std::ffi::{OsString, c_char, c_int};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use cuda_artifact_finalizer::{CudaArch, FinalizationOptions, NvvmCompiler};
use libloading::Library;
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use oxide_artifacts::{ArtifactCompileOptions, ArtifactDebugPolicy};

type Res<T> = Result<T, Box<dyn Error>>;

const DEVICE_PACKAGE: &str = "ptyrs-kernels";
const DEVICE_CRATE: &str = "ptyrs_kernels";
const FFT_PREFIX: &str = "ptyrs_cufft_c2c_f32_";
/// Stable entry signature: registers, shared memory, FFT index in block, thread index in
/// FFT, FFTs per block. The direction is in the name; the shim passes it to cuFFT.
const FFT_PARAMS: &str = "(float*, i8*, i32, i32, i32)";

/// Variables the nested cargo inherits. Everything else the outer cargo, clippy or nextest
/// set (RUSTC_WORKSPACE_WRAPPER, CARGO_TARGET_DIR, CARGO_ENCODED_RUSTFLAGS, jobserver, ...)
/// would redirect or break the device build.
const KEEP_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
    "LD_LIBRARY_PATH",
    "CUDA_TOOLKIT_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "LIBNVVM_PATH",
    "LIBNVJITLINK_PATH",
];

fn main() {
    if let Err(e) = run() {
        eprintln!("error: ptyrs-gpu build script: {e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace = manifest
        .parent()
        .ok_or("ptyrs-gpu has no parent directory")?;
    let kernels = workspace.join(DEVICE_PACKAGE);
    println!("cargo:rerun-if-changed={}", kernels.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        kernels.join("Cargo.toml").display()
    );
    println!("cargo:rerun-if-env-changed=PTYRS_CUDA_ARCH");

    let arch: CudaArch = env::var("PTYRS_CUDA_ARCH")
        .unwrap_or_else(|_| "sm_89".to_string())
        .parse()?;
    let out = PathBuf::from(env::var("OUT_DIR")?);
    let nvvm_dir = out.join("device");
    fs::create_dir_all(&nvvm_dir)?;
    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("target"))
        .join("oxide-device");

    build_device_crate(workspace, &target_dir, &nvvm_dir, &arch)?;
    let (ir, options) = read_device_artifact(&nvvm_dir, &arch)?;

    let cufft = CufftDevice::load()?;
    let (ir, ffts) = define_block_ffts(&ir, &cufft, arch.capability())?;
    let ltoir = NvvmCompiler::discover()?.compile_nvvm_ir_to_ltoir(
        DEVICE_CRATE,
        ir.as_bytes(),
        &options,
    )?;

    let nvjitlink = LibNvJitLink::load()?;
    let link_options = [
        format!("-arch={}", arch.sm()),
        "-lto".to_string(),
        format!("-fma={}", u8::from(options.allow_fma_contraction())),
    ];
    let link_refs: Vec<&str> = link_options.iter().map(String::as_str).collect();
    let mut linker = Linker::new(&nvjitlink, &link_refs)?;
    linker.add(InputType::Ltoir, &ltoir, &format!("{DEVICE_CRATE}.ltoir"))?;
    for fft in &ffts {
        for (k, (kind, bytes)) in fft.pieces.iter().enumerate() {
            linker.add(*kind, bytes, &format!("{}_{k}", fft.name))?;
        }
    }
    fs::write(out.join("ptyrs_kernels.cubin"), linker.finish()?)?;
    fs::write(out.join("block_fft.rs"), block_fft_table(&ffts))?;
    println!("cargo:rustc-env=PTYRS_GPU_ARCH={}", arch.sm());
    Ok(())
}

/// Runs `cargo oxide build` on the device crate. The backend's options are part of the
/// device crate's Cargo fingerprint (cargo-oxide hashes every `CUDA_OXIDE_*` variable,
/// `CUDA_OXIDE_PTX_DIR` included), so a no-op build means `nvvm_dir` already holds the IR
/// of the current sources for this architecture.
fn build_device_crate(
    workspace: &Path,
    target_dir: &Path,
    nvvm_dir: &Path,
    arch: &CudaArch,
) -> Res<()> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let mut cmd = Command::new(cargo);
    cmd.current_dir(workspace).env_clear();
    for key in KEEP_ENV {
        if let Some(value) = env::var_os(key) {
            cmd.env(key, value);
        }
    }
    cmd.env("CUDA_OXIDE_PTX_DIR", nvvm_dir)
        .args(["oxide", "build", "--emit-nvvm-ir", "--arch"])
        .arg(arch.sm())
        .args(["--device-codegen-crate", DEVICE_CRATE, "--cargo-target-dir"])
        .arg(target_dir)
        // Release: the legacy NVVM IR dialect for pre-Blackwell targets cannot carry the
        // debug metadata a dev-profile build attaches.
        .args(["--", "-p", DEVICE_PACKAGE, "--lib", "--release"]);
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!(
            "`cargo oxide build` of {DEVICE_PACKAGE} failed ({}):\n{}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(())
}

/// Reads the IR and its compile options, checking the backend built it for `arch`.
fn read_device_artifact(dir: &Path, arch: &CudaArch) -> Res<(String, FinalizationOptions)> {
    let target = fs::read_to_string(dir.join(format!("{DEVICE_CRATE}.target")))?;
    let built = target.lines().next().unwrap_or_default();
    if built != arch.sm() {
        return Err(format!(
            "device IR was built for `{built}`, expected `{}`",
            arch.sm()
        )
        .into());
    }
    let sidecar = fs::read_to_string(dir.join(format!("{DEVICE_CRATE}.options")))?;
    let compile = ArtifactCompileOptions::from_sidecar_text(&sidecar)?;
    if compile.debug_policy() != ArtifactDebugPolicy::None {
        return Err("device debug info is not carried through the cuFFT link".into());
    }
    let options = FinalizationOptions::new(arch.clone())
        .with_fma_contraction(compile.fma_contraction_enabled());
    let ir = fs::read_to_string(dir.join(format!("{DEVICE_CRATE}.ll")))?;
    Ok((ir, options))
}

/// One block FFT the device code calls.
struct BlockFft {
    /// The stable entry name, without the leading `@`.
    name: String,
    n: u32,
    ept: u32,
    inverse: bool,
    threads: i64,
    smem_per_fft: i64,
    suggested_bpb: i64,
    pieces: Vec<(InputType, Vec<u8>)>,
}

/// Replaces each `declare` of a stable FFT entry with a definition that calls the libcufft
/// function, and returns the functions whose pieces must be linked.
fn define_block_ffts(
    ir: &str,
    cufft: &CufftDevice,
    capability: u32,
) -> Res<(String, Vec<BlockFft>)> {
    let mut out = String::with_capacity(ir.len() + 1024);
    let mut ffts = Vec::new();
    for line in ir.lines() {
        let Some(rest) = line
            .strip_prefix("declare void @")
            .filter(|rest| rest.starts_with(FFT_PREFIX))
        else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let (name, params) = rest
            .split_once('(')
            .map(|(name, params)| (name, format!("({params}")))
            .ok_or_else(|| format!("malformed declaration `{line}`"))?;
        if params != FFT_PARAMS {
            return Err(format!("`{name}` must take {FFT_PARAMS}, IR has `{line}`").into());
        }
        let (n, ept, inverse) = parse_fft_name(name)?;
        let found = cufft.query(n, ept, inverse, capability)?;
        if found.threads != i64::from(n / ept) {
            // The device code places element `t + i * N / EPT` in thread `t`.
            return Err(format!("`{name}` runs on {} threads, not N / EPT", found.threads).into());
        }
        let symbol = format!(
            "_ZN7cufftdx8database6detail28cufftdx_private_lto_functionILj{}ELy{}EfLj1EEEvPT1_Pvijjj",
            found.version, found.id
        );
        let dir = u8::from(inverse);
        writeln!(
            out,
            "declare void @{symbol}(float*, i8*, i32, i32, i32, i32)"
        )?;
        writeln!(
            out,
            "define internal void @{name}(float* %r, i8* %s, i32 %b, i32 %f, i32 %p) alwaysinline {{\n  \
             call void @{symbol}(float* %r, i8* %s, i32 {dir}, i32 %b, i32 %f, i32 %p)\n  \
             ret void\n}}"
        )?;
        ffts.push(BlockFft {
            name: name.to_string(),
            n,
            ept,
            inverse,
            threads: found.threads,
            smem_per_fft: found.smem_per_fft,
            suggested_bpb: found.suggested_bpb,
            pieces: found.pieces,
        });
    }
    Ok((out, ffts))
}

/// `ptyrs_cufft_c2c_f32_n{N}_e{EPT}_{fwd|inv}` -> (N, EPT, inverse).
fn parse_fft_name(name: &str) -> Res<(u32, u32, bool)> {
    let bad = || format!("`{name}` is not {FFT_PREFIX}n<N>_e<EPT>_<fwd|inv>");
    let rest = name.strip_prefix(FFT_PREFIX).ok_or_else(bad)?;
    let mut parts = rest.split('_');
    let (Some(n), Some(ept), Some(dir), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(bad().into());
    };
    let n = n
        .strip_prefix('n')
        .and_then(|v| v.parse().ok())
        .ok_or_else(bad)?;
    let ept = ept
        .strip_prefix('e')
        .and_then(|v| v.parse().ok())
        .ok_or_else(bad)?;
    let inverse = match dir {
        "fwd" => false,
        "inv" => true,
        _ => return Err(bad().into()),
    };
    Ok((n, ept, inverse))
}

fn block_fft_table(ffts: &[BlockFft]) -> String {
    let mut s = String::from("pub const BLOCK_FFTS: &[BlockFft] = &[\n");
    for f in ffts {
        writeln!(
            s,
            "    BlockFft {{ n: {}, ept: {}, inverse: {}, threads_per_fft: {}, smem_per_fft: {}, \
             suggested_ffts_per_block: {} }},",
            f.n, f.ept, f.inverse, f.threads, f.smem_per_fft, f.suggested_bpb
        )
        .expect("writing to a String cannot fail");
    }
    s.push_str("];\n");
    s
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
const FUNC_SMEM_PER_FFT: c_int = 2;
const FUNC_THREADS: c_int = 3;
const FUNC_SUGGESTED_BPB: c_int = 4;
const FUNC_TRANSPOSITION: c_int = 5;
/// Thread `t` of an FFT holds elements `t + i * N / EPT`, the layout the device code uses.
const TRANSPOSITION_BY_COMPONENT: i64 = 2;
const CONTAINER_LTOIR: c_int = 0;
const CONTAINER_FATBIN: c_int = 1;

/// The cuFFT device-API entry points (`cufft_device.h`), resolved from libcufft.
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

/// One libcufft device function: its entry-symbol parameters, launch traits and code.
struct DeviceFunction {
    version: u64,
    id: u64,
    threads: i64,
    smem_per_fft: i64,
    suggested_bpb: i64,
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
        let toolkit = env::var_os("CUDA_TOOLKIT_PATH").map(PathBuf::from);
        let candidates = toolkit
            .iter()
            .map(|root| root.join("lib64").join("libcufft.so"))
            .chain([
                PathBuf::from("libcufft.so.12"),
                PathBuf::from("libcufft.so"),
            ]);
        // SAFETY: loading libcufft runs no initialisers with preconditions, and every symbol
        // type below matches its declaration in cufft_device.h.
        unsafe {
            let lib = candidates
                .into_iter()
                .find_map(|path| Library::new(path).ok())
                .ok_or("cannot load libcufft (set CUDA_TOOLKIT_PATH)")?;
            Ok(Self {
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
            })
        }
    }

    /// The single-precision C2C function of size `n`, `ept` elements per thread,
    /// BY_COMPONENT layout, for compute capability `capability` (e.g. 89).
    fn query(&self, n: u32, ept: u32, inverse: bool, capability: u32) -> Res<DeviceFunction> {
        let mut desc = 0;
        ck("cufftDescriptionCreate", unsafe {
            (self.desc_create)(&mut desc)
        })?;
        let found = self.query_desc(desc, n, ept, inverse, capability);
        let destroyed = ck("cufftDescriptionDestroy", unsafe {
            (self.desc_destroy)(desc)
        });
        let found = found?;
        destroyed?;
        Ok(found)
    }

    fn query_desc(
        &self,
        desc: c_int,
        n: u32,
        ept: u32,
        inverse: bool,
        capability: u32,
    ) -> Res<DeviceFunction> {
        for (which, value) in [
            (DESC_SIZE, i64::from(n)),
            (DESC_DIRECTION, i64::from(inverse)),
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
        let found = self.select(dev, desc, ept);
        let destroyed = ck("cufftDeviceDestroy", unsafe { (self.dev_destroy)(dev) });
        let found =
            found.map_err(|e| format!("C2C f32 n={n} inverse={inverse} sm_{capability}: {e}"))?;
        destroyed?;
        Ok(found)
    }

    fn select(&self, dev: c_int, desc: c_int, ept: u32) -> Res<DeviceFunction> {
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
            if get(FUNC_EPT)? != i64::from(ept)
                || get(FUNC_TRANSPOSITION)? != TRANSPOSITION_BY_COMPONENT
            {
                continue;
            }
            let (version, id) = self.entry_template_args(dev, func)?;
            return Ok(DeviceFunction {
                version,
                id,
                threads: get(FUNC_THREADS)?,
                smem_per_fft: get(FUNC_SMEM_PER_FFT)?,
                suggested_bpb: get(FUNC_SUGGESTED_BPB)?,
                pieces: self.pieces(dev, func)?,
            });
        }
        Err(format!("libcufft offers no EPT {ept} / BY_COMPONENT function").into())
    }

    /// The entry symbol's template arguments: the function ID and API version, arguments 6
    /// and 7 (from 0) of `block_fft_lto_implementation<...>` in the database string. The
    /// version must also be the one `cufftDeviceGetVersion` reports.
    fn entry_template_args(&self, dev: c_int, func: c_int) -> Res<(u64, u64)> {
        let mut size = 0;
        ck("cufftDeviceGetDatabaseStrSize", unsafe {
            (self.db_size)(dev, &mut size, func)
        })?;
        let mut buf = vec![0u8; size];
        ck("cufftDeviceGetDatabaseStr", unsafe {
            (self.db_str)(dev, size, buf.as_mut_ptr().cast(), func)
        })?;
        let db = String::from_utf8_lossy(&buf);
        let args = db
            .split_once("block_fft_lto_implementation<")
            .and_then(|(_, rest)| rest.split_once('>'))
            .ok_or("database string has no block_fft_lto_implementation<...>")?
            .0;
        let fields: Vec<&str> = args.split(',').map(str::trim).collect();
        let (Some(id), Some(version)) = (fields.get(6), fields.get(7)) else {
            return Err(format!("unexpected block_fft_lto_implementation<{args}>").into());
        };
        let (id, version): (u64, u64) = (id.parse()?, version.parse()?);
        let mut api = 0;
        ck("cufftDeviceGetVersion", unsafe {
            (self.get_version)(&mut api)
        })?;
        if u64::try_from(api)? != version {
            return Err(format!("database names API {version}, libcufft reports {api}").into());
        }
        Ok((version, id))
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
