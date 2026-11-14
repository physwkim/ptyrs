#![warn(clippy::disallowed_methods)] // clippy.toml: CUDA devices only through ptyrs::nn::cuda
use burn::prelude::*;
use burn::tensor::DType;
use cuda_core::{Context, DeviceBuffer};
use ptyrs::nn::interop::{InteropError, copy_in, lend, lend_out};

fn values(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| scale * i as f32 - 3.5).collect()
}

fn host(t: &Tensor<2>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().unwrap()
}

#[test]
fn lend_reads_burn_memory() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let device = ptyrs::nn::cuda(0);
    let a = values(48, 0.25);
    // A computed tensor, so lend has to wait for burn's queued kernel.
    let t = Tensor::<2>::from_data(TensorData::new(a.clone(), [6, 8]), &device) * 2.0;
    let lent = lend(&t, &stream).unwrap();
    let want: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
    assert_eq!(lent.to_host_vec(&stream).unwrap(), want);
}

#[test]
fn lend_out_writes_show_in_its_tensor() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let device = ptyrs::nn::cuda(0);
    // burn pads the rows of two values of a fresh tensor of this shape.
    let (t, mut lent) = lend_out([16, 16, 2], &device, &stream).unwrap();
    assert_eq!(t.dims(), [16, 16, 2]);
    let b = values(512, -0.5);
    let src = DeviceBuffer::from_host(&stream, &b).unwrap();
    lent.copy_from_device_async(&src, &stream).unwrap();
    stream.synchronize().unwrap();
    drop(lent);
    let doubled: Vec<f32> = b.iter().map(|v| v * 2.0).collect();
    assert_eq!((t * 2.0).to_data().try_to_vec::<f32>().unwrap(), doubled);
}

#[test]
fn lend_makes_a_strided_view_row_major() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let device = ptyrs::nn::cuda(0);
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
    let t = Tensor::<2>::zeros([2, 2], &ptyrs::nn::cuda(0)).cast(DType::F16);
    assert!(matches!(
        lend(&t, &stream),
        Err(InteropError::DType(DType::F16))
    ));
}

#[test]
fn copy_in_copies_engine_memory() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let device = ptyrs::nn::cuda(0);
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
fn copy_in_fills_a_tensor_burn_lays_out_in_padded_rows() {
    // A fresh [16, 16, 2] tensor of burn's has its rows of two values padded.
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let a = values(512, 0.5);
    let engine = DeviceBuffer::from_host(&stream, &a).unwrap();
    let t = copy_in::<3>(&engine, [16, 16, 2], &ptyrs::nn::cuda(0), &stream).unwrap();
    assert_eq!(t.to_data().try_to_vec::<f32>().unwrap(), a);
}

#[test]
fn copy_in_rejects_wrong_length() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let engine = DeviceBuffer::from_host(&stream, &values(10, 1.0)).unwrap();
    assert!(matches!(
        copy_in::<2>(&engine, [3, 4], &ptyrs::nn::cuda(0), &stream),
        Err(InteropError::Len {
            want: 12,
            have: 10,
            ..
        })
    ));
}
