//! Deep image prior (DESIGN §4): a convolutional network maps a fixed random code to the object's
//! amplitude and phase, and the engine's object gradient trains its weights.
//!
//! With `g` the gradient of the reconstruction loss `L` with respect to the object at the
//! network's current output, held constant, the weight gradient of `⟨O(θ), g⟩` is exactly
//! `∂L/∂θ`; burn needs no complex numbers or FFTs. `O(θ)` is formed from the amplitude and phase
//! outputs inside the graph, so the engine, behind [`ObjectLoss`], sends its own object gradient
//! and the polar chain rule is autodiff's.

use std::f64::consts::PI;

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::tensor::activation::tanh;
use burn::tensor::module::{interpolate, max_pool2d};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, MaxPoolOptions};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::NnError;
use crate::ptychonn::Block;

/// Shape of a DIP network.
#[derive(Config, Debug)]
pub struct DipConfig {
    /// Object pixels `[ny, nx]`.
    pub shape: [usize; 2],
    /// Channels of the first stage; stage `k` has `width << k`.
    #[config(default = 16)]
    pub width: usize,
    /// Stages, each halving the image.
    #[config(default = 4)]
    pub depth: usize,
    /// Channels of the fixed input code.
    #[config(default = 8)]
    pub code: usize,
    /// Seed of the input code.
    #[config(default = 0)]
    pub seed: u64,
}

/// U-Net: `depth` encoder stages with skip connections into a mirror decoder, then one 3×3
/// convolution to the raw amplitude and phase channels.
#[derive(Module, Debug)]
struct DipNet {
    down: Vec<Block>,
    bottom: Block,
    up: Vec<Block>,
    out: Conv2d,
}

impl DipNet {
    fn new(c: &DipConfig, device: &Device) -> Self {
        let w = |k: usize| c.width << k;
        let top = w(c.depth - 1);
        let down = (0..c.depth)
            .map(|k| Block::new(if k == 0 { c.code } else { w(k - 1) }, w(k), device))
            .collect();
        // up[k] takes the deeper features upsampled beside stage k's skip.
        let up = (0..c.depth)
            .map(|k| {
                let deeper = if k + 1 == c.depth { top } else { w(k + 1) };
                Block::new(deeper + w(k), w(k), device)
            })
            .collect();
        let out = Conv2dConfig::new([w(0), 2], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        Self {
            down,
            bottom: Block::new(top, top, device),
            up,
            out,
        }
    }

    fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        let mut skips = Vec::with_capacity(self.down.len());
        for b in &self.down {
            let s = b.forward(x);
            x = max_pool2d(s.clone(), MaxPoolOptions::new([2, 2]));
            skips.push(s);
        }
        x = self.bottom.forward(x);
        for (b, s) in self.up.iter().zip(skips).rev() {
            let [_, _, h, w] = s.dims();
            let x_up = interpolate(
                x,
                InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([h, w]),
            );
            x = b.forward(Tensor::cat(vec![x_up, s], 1));
        }
        self.out.forward(x)
    }
}

/// Whatever an [`ObjectLoss`] fails with.
pub type LossError = Box<dyn std::error::Error + Send + Sync>;

/// The reconstruction loss `L(O)` that trains a [`Dip`]: the engine's, or a stand-in.
pub trait ObjectLoss {
    /// `L` at `object` and its gradient `[∂L/∂Re O, ∂L/∂Im O]`, twice the Wirtinger gradient
    /// `∂L/∂O*`. Both tensors are `[ny, nx, 2]` with real and imaginary parts interleaved, the
    /// engine's complex f32 layout that [`crate::interop`] moves; the gradient takes `object`'s
    /// device and dtype.
    fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError>;
}

/// A DIP network, its fixed input code and its Adam state.
pub struct Dip {
    shape: [usize; 2],
    net: DipNet,
    code: Tensor<4>,
    optim: ModuleOptimizer,
}

impl Dip {
    /// A fresh network of `config` on `device`; autodiff is enabled on it here. The code is
    /// uniform in `[0, 0.1)` (Ulyanov et al. 2018) on the object grid rounded up to a multiple of
    /// `2^depth`, and the output is cropped back.
    pub fn new(config: &DipConfig, device: &Device) -> Result<Self, NnError> {
        let [ny, nx] = config.shape;
        if ny == 0 || nx == 0 || config.width == 0 || config.depth == 0 || config.code == 0 {
            return Err(NnError::Invalid(format!("DIP config {config:?}")));
        }
        let granule = 1 << config.depth;
        let (cy, cx) = (
            ny.div_ceil(granule) * granule,
            nx.div_ceil(granule) * granule,
        );
        let ad = device.clone().autodiff();
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let code: Vec<f32> = (0..config.code * cy * cx)
            .map(|_| 0.1 * rng.r#gen::<f32>())
            .collect();
        Ok(Self {
            shape: config.shape,
            net: DipNet::new(config, &ad),
            code: Tensor::from_data(TensorData::new(code, [1, config.code, cy, cx]), &ad),
            optim: AdamConfig::new().init(),
        })
    }

    /// The network's current object, `[ny, nx, 2]` as [`ObjectLoss`] takes it: amplitude in
    /// `(0, 2)`, phase in `(-π, π)`.
    pub fn object(&self) -> Tensor<3> {
        self.forward().inner()
    }

    /// One Adam step of size `lr` down `loss`, which is returned at the object before the step.
    pub fn step(&mut self, loss: &mut impl ObjectLoss, lr: f64) -> Result<f64, NnError> {
        let (value, grads) = self.weight_grads(loss)?;
        self.net = self.optim.step(lr, self.net.clone(), grads);
        Ok(value)
    }

    /// The object in the weights' autodiff graph.
    fn forward(&self) -> Tensor<3> {
        let [ny, nx] = self.shape;
        let raw = self.net.forward(self.code.clone());
        // 2·sigmoid(x), as 1 + tanh(x/2): burn's sigmoid computes in f32 whatever the dtype.
        let amp = tanh(
            raw.clone()
                .slice([0..1, 0..1, 0..ny, 0..nx])
                .reshape([ny, nx])
                * 0.5,
        ) + 1.0;
        let phase = tanh(raw.slice([0..1, 1..2, 0..ny, 0..nx]).reshape([ny, nx])) * PI;
        Tensor::stack(
            vec![amp.clone() * phase.clone().cos(), amp * phase.sin()],
            2,
        )
    }

    /// `loss` at the current object and its weight gradient, that of `⟨O(θ), g⟩`.
    fn weight_grads(&self, loss: &mut impl ObjectLoss) -> Result<(f64, GradientsParams), NnError> {
        let object = self.forward();
        let (value, g) = loss
            .gradient(&object.clone().inner())
            .map_err(NnError::Loss)?;
        if g.dims() != object.dims() || g.dtype() != object.dtype() {
            return Err(NnError::Invalid(format!(
                "gradient {:?} {:?} for object {:?} {:?}",
                g.dims(),
                g.dtype(),
                object.dims(),
                object.dtype()
            )));
        }
        let surrogate = (object * g.inner()).sum();
        Ok((
            value,
            GradientsParams::from_grads(surrogate.backward(), &self.net),
        ))
    }
}

#[cfg(test)]
mod tests {
    use burn::module::{ModuleMapper, ModuleVisitor, Param};
    use burn::tensor::DType;

    use super::*;

    /// Stand-in for the engine: a loss on the complex object with a closed-form gradient,
    /// `F(O) = Σ w(|O|² − t)² + a·Re O + b·Im O`, nonlinear in `O` like an intensity fit.
    #[derive(Clone)]
    struct StandIn {
        w: Vec<f64>,
        t: Vec<f64>,
        a: Vec<f64>,
        b: Vec<f64>,
    }

    impl StandIn {
        fn new(n: usize, seed: u64) -> Self {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut draw =
                |lo: f64, hi: f64| -> Vec<f64> { (0..n).map(|_| rng.gen_range(lo..hi)).collect() };
            Self {
                w: draw(0.5, 1.0),
                t: draw(0.5, 1.5),
                a: draw(-0.5, 0.5),
                b: draw(-0.5, 0.5),
            }
        }

        fn terms(o: &[f64]) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
            o.chunks(2).enumerate().map(|(i, z)| (i, z[0], z[1]))
        }

        fn loss(&self, o: &[f64]) -> f64 {
            Self::terms(o)
                .map(|(i, re, im)| {
                    let r = re * re + im * im - self.t[i];
                    self.w[i] * r * r + self.a[i] * re + self.b[i] * im
                })
                .sum()
        }

        /// `[∂F/∂Re O, ∂F/∂Im O]`, interleaved.
        fn grad(&self, o: &[f64]) -> Vec<f64> {
            Self::terms(o)
                .flat_map(|(i, re, im)| {
                    let k = 4.0 * self.w[i] * (re * re + im * im - self.t[i]);
                    [k * re + self.a[i], k * im + self.b[i]]
                })
                .collect()
        }
    }

    impl ObjectLoss for StandIn {
        fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError> {
            let o = host64(object.clone());
            Ok((self.loss(&o), like(object, self.grad(&o))))
        }
    }

    /// `F(O) = ½|O − T|²`, gradient `O − T`.
    struct Fit(Vec<f64>);

    impl ObjectLoss for Fit {
        fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError> {
            let o = host64(object.clone());
            let d: Vec<f64> = o.iter().zip(&self.0).map(|(o, t)| o - t).collect();
            let value = 0.5 * d.iter().map(|d| d * d).sum::<f64>();
            Ok((value, like(object, d)))
        }
    }

    /// A gradient with the object's axes swapped.
    struct Transposed;

    impl ObjectLoss for Transposed {
        fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError> {
            Ok((0.0, object.clone().swap_dims(0, 1)))
        }
    }

    /// `values` as a tensor of `object`'s shape, device and dtype.
    fn like(object: &Tensor<3>, values: Vec<f64>) -> Tensor<3> {
        Tensor::from_data(
            TensorData::new(values, object.shape()),
            (&object.device(), object.dtype()),
        )
    }

    /// Each float parameter's gradient (`None` without one), in visiting order.
    struct Collect<'a> {
        grads: &'a GradientsParams,
        out: Vec<Option<Vec<f64>>>,
    }

    impl ModuleVisitor for Collect<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let g = self.grads.get::<D>(param.id).map(host64);
            self.out.push(g);
        }
    }

    /// Adds `step` to the `target`-th float parameter in visiting order.
    struct Shift {
        target: usize,
        step: Vec<f64>,
        seen: usize,
    }

    impl ModuleMapper for Shift {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            self.seen += 1;
            if self.seen - 1 != self.target {
                return param;
            }
            let step = self.step.clone();
            param.map(|t| {
                let d =
                    Tensor::from_data(TensorData::new(step, t.shape()), (&t.device(), DType::F64));
                t + d
            })
        }
    }

    /// Casts every float tensor to f64, keeping which ones take a gradient.
    struct ToF64;

    impl ModuleMapper for ToF64 {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            param.map(|t| {
                let grad = t.is_require_grad();
                t.cast(DType::F64).detach().set_require_grad(grad)
            })
        }
    }

    fn host64<const D: usize>(t: Tensor<D>) -> Vec<f64> {
        t.into_data()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .unwrap()
    }

    /// `dip` with every float tensor in f64.
    fn dip64(config: &DipConfig, device: &Device) -> Dip {
        let mut dip = Dip::new(config, device).unwrap();
        dip.net = dip.net.map(&mut ToF64);
        dip.code = dip.code.cast(DType::F64);
        dip
    }

    fn loss_at(
        dip: &Dip,
        value: &mut impl FnMut(&[f64]) -> f64,
        target: usize,
        step: Vec<f64>,
    ) -> f64 {
        let mut shift = Shift {
            target,
            step,
            seen: 0,
        };
        let net = dip.net.clone().map(&mut shift);
        let shifted = Dip {
            shape: dip.shape,
            net,
            code: dip.code.clone(),
            optim: AdamConfig::new().init(),
        };
        value(&host64(shifted.object()))
    }

    /// Checks the weight gradient `dip` takes from `loss` against central differences of
    /// `value`, the same loss on the host object, along a random direction in each parameter
    /// tensor. Returns the number of tensors checked.
    fn check_weight_grads(
        dip: &Dip,
        loss: &mut impl ObjectLoss,
        mut value: impl FnMut(&[f64]) -> f64,
    ) -> usize {
        let (_, grads) = dip.weight_grads(loss).unwrap();
        let mut collect = Collect {
            grads: &grads,
            out: Vec::new(),
        };
        dip.net.visit(&mut collect);

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut checked = 0;
        for (target, grad) in collect.out.iter().enumerate() {
            let Some(grad) = grad else { continue };
            // Along a random unit direction over this parameter tensor.
            let d: Vec<f64> = grad.iter().map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let ad: f64 = grad.iter().zip(&d).map(|(g, v)| g * v / norm).sum();
            let h = 1e-5;
            let step = |s: f64| d.iter().map(|v| s * h * v / norm).collect();
            let fd = (loss_at(dip, &mut value, target, step(1.0))
                - loss_at(dip, &mut value, target, step(-1.0)))
                / (2.0 * h);
            assert!(
                (fd - ad).abs() < 1e-6 * (1.0 + ad.abs()),
                "parameter {target}: autodiff {ad:e}, finite difference {fd:e}"
            );
            checked += 1;
        }
        checked
    }

    #[test]
    fn weight_gradients_match_finite_differences() {
        let device = Device::cuda(0);
        device.seed(5);
        let shape = [20, 28];
        let config = DipConfig::new(shape)
            .with_width(4)
            .with_depth(2)
            .with_code(3);
        // In f64: along a weight direction the loss's slope changes on a 1e-3 scale, below which
        // f32 differences are noise.
        let dip = dip64(&config, &device);
        let f = StandIn::new(shape[0] * shape[1], 1);
        // Weight and bias of 11 convolutions, scale and shift of 10 batch norms.
        assert_eq!(check_weight_grads(&dip, &mut f.clone(), |o| f.loss(o)), 42);
    }

    #[test]
    fn steps_fit_a_target_object() {
        let device = Device::cuda(0);
        device.seed(5);
        let shape = [24, 40];
        let config = DipConfig::new(shape).with_width(8).with_depth(2);
        let mut dip = Dip::new(&config, &device).unwrap();
        let mut fit = Fit(crate::data::random_object(shape, 3)
            .as_slice()
            .iter()
            .flat_map(|z| [z.re, z.im])
            .collect());
        let start = dip.step(&mut fit, 1e-2).unwrap();
        for _ in 0..99 {
            dip.step(&mut fit, 1e-2).unwrap();
        }
        let (end, _) = fit.gradient(&dip.object()).unwrap();
        assert!(end < 0.1 * start, "loss {start} -> {end}");

        assert!(matches!(
            dip.step(&mut Transposed, 1e-2),
            Err(NnError::Invalid(_))
        ));
    }
}
