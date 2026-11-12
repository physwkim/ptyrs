use cuda_async::device_operation::ExecutionContext;
use cuda_async::error::DeviceError;
use cuda_core::{Context, DeviceBuffer, DriverError};
use ptyrs::fft::{C32, FftError, Plan2d};

fn not_ready(r: Result<(), FftError>) -> bool {
    let e = cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_READY;
    matches!(r, Err(FftError::Device(DeviceError::Driver(DriverError(c)))) if c == e)
}

#[test]
fn exec_leases_its_buffers_and_work_area_until_its_context_is_done() {
    let ctx = Context::new(0).unwrap();
    let (s1, s2) = (ctx.new_stream().unwrap(), ctx.new_stream().unwrap());
    let len = 64 * 64 * 2;
    let buf = || DeviceBuffer::<C32>::zeroed(&s1, len).unwrap();
    let (a, mut b, mut c, mut d) = (buf(), buf(), buf(), buf());
    let mut plan = Plan2d::new(&s1, 64, 2).unwrap();
    let mut other = Plan2d::new(&s2, 64, 2).unwrap();

    let first = ExecutionContext::new(s1.clone());
    plan.forward(&first, &a, &mut b).unwrap();
    let on_s2 = || ExecutionContext::new(s2.clone());
    assert!(not_ready(other.forward(&on_s2(), &b, &mut c)), "read of b");
    assert!(not_ready(other.forward(&on_s2(), &c, &mut b)), "write of b");
    other.forward(&on_s2(), &a, &mut c).unwrap();
    assert!(
        not_ready(plan.forward(&on_s2(), &c, &mut d)),
        "plan's work area"
    );

    s1.synchronize().unwrap();
    drop(first);
    plan.forward(&on_s2(), &c, &mut d).unwrap();
    s2.synchronize().unwrap();
}
