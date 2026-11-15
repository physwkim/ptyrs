//! [`Dip`] with pty-chi's deep image prior network, against pty-chi on the same start: the
//! default UNet of pty-chi's `DIPPlanarObject` rebuilt in burn and loaded from the weights,
//! input code and start object `refs/py/compare_dip_ptychi.py init` exports, then trained by
//! Adam (torch's epsilon) on the engine's Poisson LLK with the truth probe fixed. Fails if the
//! network's start object differs from pty-chi's by more than 1e-10 (f64) or 1e-2 (f32) of its
//! largest modulus. Writes `<out_dir>/ptyrs_<dtype>.tsv` (iteration, LLK after it with 0 the start,
//! synchronised seconds of the step and of its loss, host copies included) and the last object
//! to `<out_dir>/ptyrs_<dtype>_object.h5`.
//!
//! `f64` runs the network in f64 on the CPU engine's f64 loss (`Ptycho`), the curve against
//! pty-chi in float64; `f32` runs it in f32 on `GpuPtycho`, both on the GPU, for the time per
//! iteration against pty-chi in float32.
//!
//! `cargo run --release -p ptyrs --features nn --example nn_dip_ptychi -- <data_dir> <out_dir>
//! [f64|f32] [iterations=100] [lr=1e-3] [name=fermat_clean]`
#![warn(clippy::disallowed_methods)] // clippy.toml: CUDA devices only through ptyrs::nn::cuda

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use burn::module::{Param, RunningState};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::activation::relu;
use burn::tensor::module::max_pool2d;
use burn::tensor::ops::MaxPoolOptions;
use cuda_core::CudaContext;
use ptyrs::base::{C64, Object, Probe, Scan, Stack};
use ptyrs::engine::{GpuOptions, GpuPtycho, Ptycho};
use ptyrs::io::{read_cxi, read_modes, write_modes};
use ptyrs::nn::dip::{Dip, LossError, ObjectLoss, Prior};
use rust_hdf5::H5File;

/// pty-chi's `DoubleConv`: (3×3 convolution without bias, batch normalisation, ReLU) twice. On
/// an autodiff device the normalisation takes the batch statistics, as torch's in train mode.
#[derive(Module, Debug)]
struct DoubleConv {
    c0: Conv2d,
    n1: BatchNorm,
    c3: Conv2d,
    n4: BatchNorm,
}

impl DoubleConv {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = relu(self.n1.forward(self.c0.forward(x)));
        relu(self.n4.forward(self.c3.forward(x)))
    }
}

/// pty-chi's `UNet` (`reconstructors/nn/models/unet.py`) at its defaults, bilinear upsampling
/// and skip connections: `inc`, `down1..4` (2× max pool then a [`DoubleConv`]), `up1..4`
/// ([`upsample`] beside the skip, then a [`DoubleConv`]) and the 1×1 `outc`.
#[derive(Module, Debug)]
struct UNet {
    inc: DoubleConv,
    down: Vec<DoubleConv>,
    up: Vec<DoubleConv>,
    outc: Conv2d,
}

impl UNet {
    fn raw(&self, x: Tensor<4>) -> Tensor<4> {
        let mut skips = vec![self.inc.forward(x)];
        for d in &self.down {
            let last = skips.last().expect("inc's output").clone();
            skips.push(d.forward(max_pool2d(last, MaxPoolOptions::new([2, 2]))));
        }
        let mut x = skips.pop().expect("down4's output");
        for (u, s) in self.up.iter().zip(skips.into_iter().rev()) {
            let [_, _, h, w] = s.dims();
            let x_up = upsample(upsample(x, 3, w), 2, h);
            x = u.forward(Tensor::cat(vec![s, x_up], 1));
        }
        self.outc.forward(x)
    }
}

/// pty-chi's `DIPPlanarObject.process_net_output`: channel 0 the amplitude and 1 the phase, both
/// raw.
impl Prior for UNet {
    fn object(&self, code: Tensor<4>) -> Tensor<3> {
        let raw = self.raw(code);
        let [_, _, ny, nx] = raw.dims();
        let amp = raw.clone().narrow(1, 0, 1).reshape([ny, nx]);
        let phase = raw.narrow(1, 1, 1).reshape([ny, nx]);
        Tensor::stack(
            vec![amp.clone() * phase.clone().cos(), amp * phase.sin()],
            2,
        )
    }
}

/// `x` twice as long along `dim` by torch's `Upsample(scale_factor=2, mode='bilinear',
/// align_corners=True)` on that axis, then zero-padded to `m` as `Up.forward`'s `F.pad`, the
/// odd pixel at the end. Applied to the width and then the height, it rounds as torch's kernel.
fn upsample(x: Tensor<4>, dim: usize, m: usize) -> Tensor<4> {
    let n = x.dims()[dim];
    assert!(n >= 2 && m >= 2 * n, "upsample {n} to {m}");
    let lead = (m - 2 * n) / 2;
    let scale = (n - 1) as f64 / (2 * n - 1) as f64;
    // Padding pixels read sample 0 at weight 0.
    let (mut i0, mut i1) = (vec![0_i64; m], vec![0_i64; m]);
    let (mut w0, mut w1) = (vec![0.0; m], vec![0.0; m]);
    for o in 0..2 * n {
        let src = scale * o as f64;
        let k = src as usize;
        let lambda = src - k as f64;
        i0[lead + o] = k as i64;
        i1[lead + o] = (k + 1).min(n - 1) as i64;
        w0[lead + o] = 1.0 - lambda;
        w1[lead + o] = lambda;
    }
    let (device, dtype) = (x.device(), x.dtype());
    let index = |i: Vec<i64>| Tensor::<1, Int>::from_data(TensorData::new(i, [m]), &device);
    let mut shape = [1; 4];
    shape[dim] = m;
    let weight = |w: Vec<f64>| {
        Tensor::<1>::from_data(
            TensorData::new(w, [m]).convert_dtype(dtype),
            (&device, dtype),
        )
        .reshape(shape)
    };
    x.clone().select(dim, index(i0)) * weight(w0) + x.select(dim, index(i1)) * weight(w1)
}

/// The tensors of `dip_init.h5`, by torch `state_dict` name.
struct Weights {
    file: H5File,
    device: Device,
    dtype: DType,
}

impl Weights {
    fn tensor<const D: usize>(&self, name: &str) -> Result<Tensor<D>, LossError> {
        let ds = self.file.dataset(name)?;
        let shape: [usize; D] = ds
            .shape()
            .try_into()
            .map_err(|s| format!("/{name} has shape {s:?}, expected {D} dimensions"))?;
        let data = TensorData::new(ds.read_raw::<f64>()?, shape).convert_dtype(self.dtype);
        Ok(Tensor::from_data(data, (&self.device, self.dtype)))
    }

    fn conv(&self, name: &str, bias: bool) -> Result<Conv2d, LossError> {
        let weight: Tensor<4> = self.tensor(&format!("{name}.weight"))?;
        let [cout, cin, kh, kw] = weight.dims();
        let (ph, pw) = (kh / 2, kw / 2);
        let mut c = Conv2dConfig::new([cin, cout], [kh, kw])
            .with_padding(PaddingConfig2d::Explicit(ph, pw, ph, pw))
            .with_bias(bias)
            .init(&self.device);
        c.weight = Param::from_tensor(weight);
        if bias {
            c.bias = Some(Param::from_tensor(self.tensor(&format!("{name}.bias"))?));
        }
        Ok(c)
    }

    /// torch's `BatchNorm2d` at its defaults, which are burn's: epsilon 1e-5, momentum 0.1.
    fn norm(&self, name: &str) -> Result<BatchNorm, LossError> {
        let gamma: Tensor<1> = self.tensor(&format!("{name}.weight"))?;
        let mut n = BatchNormConfig::new(gamma.dims()[0]).init(&self.device);
        n.gamma = Param::from_tensor(gamma);
        n.beta = Param::from_tensor(self.tensor(&format!("{name}.bias"))?);
        n.running_mean = RunningState::new(self.tensor(&format!("{name}.running_mean"))?);
        n.running_var = RunningState::new(self.tensor(&format!("{name}.running_var"))?);
        Ok(n)
    }

    fn double_conv(&self, name: &str) -> Result<DoubleConv, LossError> {
        Ok(DoubleConv {
            c0: self.conv(&format!("{name}.double_conv.0"), false)?,
            n1: self.norm(&format!("{name}.double_conv.1"))?,
            c3: self.conv(&format!("{name}.double_conv.3"), false)?,
            n4: self.norm(&format!("{name}.double_conv.4"))?,
        })
    }

    fn unet(&self) -> Result<UNet, LossError> {
        Ok(UNet {
            inc: self.double_conv("inc")?,
            down: (1..=4)
                .map(|k| self.double_conv(&format!("down{k}.maxpool_conv.1")))
                .collect::<Result<_, _>>()?,
            up: (1..=4)
                .map(|k| self.double_conv(&format!("up{k}.conv")))
                .collect::<Result<_, _>>()?,
            outc: self.conv("outc", true)?,
        })
    }
}

/// `[ny, nx, 2]` on the host, real and imaginary parts interleaved.
fn host(t: &Tensor<3>) -> Result<Vec<f64>, LossError> {
    t.to_data()
        .convert::<f64>()
        .try_into_vec::<f64>()
        .map_err(|e| format!("{e:?}").into())
}

fn to_object(t: &Tensor<3>) -> Result<Object, LossError> {
    let [ny, nx, _] = t.dims();
    let data = host(t)?.chunks(2).map(|z| C64::new(z[0], z[1])).collect();
    Ok(Object::new(Stack::from_vec([1, ny, nx], data)?))
}

/// The gradient of the normalised LLK from that of its sum, on `like`'s device and dtype.
fn to_tensor(g: &Stack<C64>, nb_obs: f64, like: &Tensor<3>) -> Tensor<3> {
    let g: Vec<f64> = g
        .as_slice()
        .iter()
        .flat_map(|c| [c.re / nb_obs, c.im / nb_obs])
        .collect();
    Tensor::from_data(
        TensorData::new(g, like.shape()).convert_dtype(like.dtype()),
        (&like.device(), like.dtype()),
    )
}

/// The engine's normalised Poisson LLK on the CPU in f64.
struct CpuLoss<'a> {
    scan: &'a Scan,
    probe: &'a Probe,
    nb_obs: f64,
}

impl ObjectLoss for CpuLoss<'_> {
    fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError> {
        let p = Ptycho::new(self.scan, to_object(object)?, self.probe.clone())?;
        let g = to_tensor(&p.obj_grad(), self.nb_obs, object);
        Ok((p.llk().poisson, g))
    }
}

/// The same on the GPU, whose engine computes in f32.
struct GpuLoss {
    ptycho: GpuPtycho,
    nb_obs: f64,
}

impl ObjectLoss for GpuLoss {
    fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError> {
        self.ptycho.set_object(&to_object(object)?)?;
        let llk = self.ptycho.llk()?.poisson;
        let [g, _] = self.ptycho.ml_grads()?;
        Ok((llk, to_tensor(&g, self.nb_obs, object)))
    }
}

/// `loss` and the seconds of its last call, from the object's device synchronised: the forward
/// that made the object is not counted.
struct Timed<L> {
    loss: L,
    secs: f64,
}

impl<L: ObjectLoss> ObjectLoss for Timed<L> {
    fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError> {
        sync(&object.device())?;
        let t = Instant::now();
        let r = self.loss.gradient(object);
        self.secs = t.elapsed().as_secs_f64();
        r
    }
}

fn sync(device: &Device) -> Result<(), LossError> {
    device.sync().map_err(|e| format!("{e:?}").into())
}

/// One row per iteration of [`train`], 0 the start.
struct Row {
    llk: f64,
    secs: f64,
    loss_secs: f64,
}

/// `n` steps of `dip` on `device` down `loss`; the last object.
fn train(
    dip: &mut Dip<UNet>,
    device: &Device,
    loss: impl ObjectLoss,
    n: usize,
    lr: f64,
    rows: &mut Vec<Row>,
) -> Result<Tensor<3>, LossError> {
    let mut loss = Timed { loss, secs: 0.0 };
    let mut row = |llk, secs, loss_secs| {
        println!("{:4} {llk:.10e} {secs:.3} {loss_secs:.3}", rows.len());
        rows.push(Row {
            llk,
            secs,
            loss_secs,
        });
    };
    let (mut secs, mut loss_secs) = (0.0, 0.0);
    for _ in 0..n {
        let t = Instant::now();
        let llk = dip.step(&mut loss, lr)?;
        row(llk, secs, loss_secs);
        if !llk.is_finite() {
            return Ok(dip.object());
        }
        sync(device)?;
        (secs, loss_secs) = (t.elapsed().as_secs_f64(), loss.secs);
    }
    let last = dip.object();
    row(loss.gradient(&last)?.0, secs, loss_secs);
    Ok(last)
}

/// Largest start-object difference from pty-chi's, relative to its largest modulus: f32
/// convolutions may round operands to TF32 on either side.
fn start_tol(dtype: DType) -> f64 {
    if dtype == DType::F64 { 1e-10 } else { 1e-2 }
}

fn main() -> Result<(), LossError> {
    let usage = "usage: nn_dip_ptychi <data_dir> <out_dir> [f64|f32] [iterations] [lr] [name]";
    let mut args = std::env::args().skip(1);
    let data_dir = args.next().ok_or(usage)?;
    let out_dir = args.next().ok_or(usage)?;
    let tag = args.next().unwrap_or_else(|| "f64".into());
    let dtype = match tag.as_str() {
        "f64" => DType::F64,
        "f32" => DType::F32,
        _ => return Err(usage.into()),
    };
    let n: usize = args.next().map_or(Ok(100), |a| a.parse())?;
    let lr: f64 = args.next().map_or(Ok(1e-3), |a| a.parse())?;
    let name = args.next().unwrap_or_else(|| "fermat_clean".into());
    let (data_dir, out_dir) = (Path::new(&data_dir), Path::new(&out_dir));

    let scan = read_cxi(data_dir.join(format!("{name}.cxi")))?;
    let (_, probe) = read_modes(data_dir.join(format!("{name}_truth.h5")))?;
    let nb_obs = scan.counts.nb_obs() as f64;
    let device = ptyrs::nn::cuda(0).autodiff();
    let w = Weights {
        file: H5File::open(out_dir.join("dip_init.h5"))?,
        device: device.clone(),
        dtype,
    };
    let net = w.unet()?;
    let code: Tensor<4> = w.tensor("nn_input")?;
    let object0 = w.file.dataset("object0")?.read_raw::<f64>()?;
    let adam = AdamConfig::new().with_epsilon(1e-8);
    let mut dip = Dip::with_prior(net, code, &adam)?;

    let start = host(&dip.object())?;
    if start.len() != object0.len() {
        return Err(format!(
            "start object of {} values, pty-chi's {}",
            start.len(),
            object0.len()
        )
        .into());
    }
    let d = start
        .iter()
        .zip(&object0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let size = object0
        .chunks(2)
        .map(|z| z[0].hypot(z[1]))
        .fold(0.0, f64::max);
    println!(
        "start object against pty-chi's: max difference / max modulus {:.2e}",
        d / size
    );
    // f64::max drops NaN, so a non-finite start is caught here, not by d.
    if start.iter().any(|v| !v.is_finite()) || d > start_tol(dtype) * size {
        return Err("start objects differ".into());
    }

    println!("ptyrs DIP {tag} lr {lr}: iteration, Poisson LLK, s, loss s");
    let mut rows = Vec::with_capacity(n + 1);
    let last = if dtype == DType::F64 {
        let loss = CpuLoss {
            scan: &scan,
            probe: &probe,
            nb_obs,
        };
        train(&mut dip, &device, loss, n, lr, &mut rows)?
    } else {
        let ctx = CudaContext::new(0)?;
        let start = to_object(&dip.object())?;
        let ptycho = GpuPtycho::new(&ctx, &scan, start, probe.clone(), GpuOptions::default())?;
        train(
            &mut dip,
            &device,
            GpuLoss { ptycho, nb_obs },
            n,
            lr,
            &mut rows,
        )?
    };

    let mut f = BufWriter::new(File::create(out_dir.join(format!("ptyrs_{tag}.tsv")))?);
    writeln!(f, "iter\tllk\tseconds\tloss_seconds")?;
    for (i, r) in rows.iter().enumerate() {
        writeln!(f, "{i}\t{:?}\t{:?}\t{:?}", r.llk, r.secs, r.loss_secs)?;
    }
    f.flush()?;
    write_modes(
        out_dir.join(format!("ptyrs_{tag}_object.h5")),
        &to_object(&last)?,
        &probe,
    )?;
    Ok(())
}
