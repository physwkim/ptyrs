//! No kernel in [`CUBIN`] uses local memory: a per-thread array indexed outside an
//! `#[unroll]` loop or by a non-constant may be placed there, slowing its kernel.

use std::ffi::CStr;
use std::ptr;

use cuda_bindings as sys;
use cuda_core::CudaContext;
use ptyrs::gpu::CUBIN;

/// Each kernel of `module` with its local memory per thread, in bytes.
fn local_sizes(module: sys::CUmodule) -> Vec<(String, i32)> {
    let check = |r: sys::CUresult| assert_eq!(r, sys::cudaError_enum_CUDA_SUCCESS);
    // SAFETY: `module` is loaded and outlives the calls, and each array is sized from the
    // count the driver reports.
    unsafe {
        let mut n = 0;
        check(sys::cuModuleGetFunctionCount(&mut n, module));
        let mut funcs = vec![ptr::null_mut(); n as usize];
        check(sys::cuModuleEnumerateFunctions(
            funcs.as_mut_ptr(),
            n,
            module,
        ));
        funcs
            .into_iter()
            .map(|f| {
                let (mut name, mut local) = (ptr::null(), 0);
                check(sys::cuFuncGetName(&mut name, f));
                check(sys::cuFuncGetAttribute(
                    &mut local,
                    sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    f,
                ));
                (CStr::from_ptr(name).to_string_lossy().into_owned(), local)
            })
            .collect()
    }
}

#[test]
fn no_kernel_uses_local_memory() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let module = ctx.load_module_from_image(CUBIN).expect("load CUBIN");
    let sizes = local_sizes(module.cu_module());
    assert!(!sizes.is_empty());
    let local: Vec<_> = sizes.into_iter().filter(|&(_, bytes)| bytes != 0).collect();
    assert!(local.is_empty(), "local bytes per thread: {local:?}");
}
