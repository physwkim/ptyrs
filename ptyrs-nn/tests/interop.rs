use burn::prelude::*;
use burn::tensor::DType;
use cuda_core::{Context, DeviceBuffer};
use ptyrs_nn::interop::{InteropError, copy_in, lend};

fn values(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| scale * i as f32 - 3.5).collect()
}

fn host(t: &Tensor<2>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().unwrap()
}

#[test]
fn lend_aliases_burn_memory() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let device = Device::cuda(0);
    let a = values(48, 0.25);
    // A computed tensor, so lend has to wait for burn's queued kernel.
    let t = Tensor::<2>::from_data(TensorData::new(a.clone(), [6, 8]), &device) * 2.0;
    let mut lent = lend(&t, &stream).unwrap();
    let want: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
    assert_eq!(lent.to_host_vec(&stream).unwrap(), want);

    // A cuda-rust write through the lent buffer shows in the burn tensor: same memory.
    let b = values(48, -0.5);
    let src = DeviceBuffer::from_host(&stream, &b).unwrap();
    lent.copy_from_device_async(&src, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(host(&t), b);
}

#[test]
fn lend_makes_a_strided_view_row_major() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let device = Device::cuda(0);
    let a = values(12, 1.0);
    let t = Tensor::<2>::from_data(TensorData::new(a.clone(), [3, 4]), &device).swap_dims(0, 1);
    let lent = lend(&t, &stream).unwrap();
    let want: Vec<f32> = (0..4)
        .flat_map(|j| (0..3).map(move |i| (i, j)))
        .map(|(i, j)| a[i * 4 + j])
        .collect();
    assert_eq!(lent.to_host_vec(&stream).unwrap(), want);
}

#[test]
fn lend_rejects_non_f32() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let t = Tensor::<2>::zeros([2, 2], &Device::cuda(0)).cast(DType::F16);
    assert!(matches!(
        lend(&t, &stream),
        Err(InteropError::DType(DType::F16))
    ));
}

#[test]
fn copy_in_copies_engine_memory() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let device = Device::cuda(0);
    let a = values(48, 0.75);
    let mut engine = DeviceBuffer::from_host(&stream, &a).unwrap();
    let t = copy_in::<2>(&engine, [6, 8], &device, &stream).unwrap();
    assert_eq!(host(&t), a);

    // A later engine write leaves the burn tensor alone: it is a copy.
    let b = values(48, 9.0);
    let src = DeviceBuffer::from_host(&stream, &b).unwrap();
    engine.copy_from_device_async(&src, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(host(&t), a);
    // And burn keeps computing on it.
    let doubled: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
    assert_eq!(host(&(t * 2.0)), doubled);
}

#[test]
fn copy_in_rejects_wrong_length() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let engine = DeviceBuffer::from_host(&stream, &values(10, 1.0)).unwrap();
    assert!(matches!(
        copy_in::<2>(&engine, [3, 4], &Device::cuda(0), &stream),
        Err(InteropError::Len {
            want: 12,
            have: 10,
            ..
        })
    ));
}
