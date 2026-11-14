//! The PtychoNN network (Cherukara et al. 2020): one convolutional encoder over the diffraction
//! amplitude and two decoders, one for the object amplitude and one for its phase.

use std::f64::consts::PI;

use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::module::{interpolate, max_pool2d};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, MaxPoolOptions};

/// Two 3×3 same-padded convolutions, each followed by batch normalisation and ReLU. Without the
/// normalisation training sits for tens of epochs at the mean patch: the object shows only as a
/// small modulation of the probe's far field.
#[derive(Module, Debug)]
pub(crate) struct Block {
    a: Conv2d,
    na: BatchNorm,
    b: Conv2d,
    nb: BatchNorm,
}

impl Block {
    pub(crate) fn new(cin: usize, cout: usize, device: &Device) -> Self {
        let conv = |cin, cout| {
            Conv2dConfig::new([cin, cout], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        Self {
            a: conv(cin, cout),
            na: BatchNormConfig::new(cout).init(device),
            b: conv(cout, cout),
            nb: BatchNormConfig::new(cout).init(device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = relu(self.na.forward(self.a.forward(x)));
        relu(self.nb.forward(self.b.forward(x)))
    }

    fn fold(&self) -> [Conv2d; 2] {
        [fold(&self.a, &self.na), fold(&self.b, &self.nb)]
    }
}

/// `c` followed by `n` at its running statistics as one convolution.
fn fold(c: &Conv2d, n: &BatchNorm) -> Conv2d {
    let [co] = n.gamma.shape().dims();
    let s = n.gamma.val() / (n.running_var.value() + n.epsilon).sqrt();
    let bias = c.bias.as_ref().expect("Block's convolutions carry a bias");
    Conv2d {
        weight: Param::from_tensor(c.weight.val() * s.clone().reshape([co, 1, 1, 1])),
        bias: Some(Param::from_tensor(
            (bias.val() - n.running_mean.value()) * s + n.beta.val(),
        )),
        ..c.clone()
    }
}

/// One stage of [`run`]: a [`Block`], or its two convolutions with the normalisation folded in.
trait Stage {
    fn forward(&self, x: Tensor<4>) -> Tensor<4>;
}

impl Stage for Block {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        Block::forward(self, x)
    }
}

impl Stage for [Conv2d; 2] {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(self[1].forward(relu(self[0].forward(x))))
    }
}

/// The decoder's stages, each followed by a nearest 2× upsampling, then its output convolution.
fn decode<S: Stage>(stages: &[S], out: &Conv2d, mut x: Tensor<4>) -> Tensor<4> {
    for s in stages {
        let [_, _, h, w] = x.dims();
        x = interpolate(
            s.forward(x),
            InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([2 * h, 2 * w]),
        );
    }
    out.forward(x)
}

/// The network over its stages: the encoder's, each followed by a 2× max pool, then each
/// decoder's with its output convolution and range map.
fn run<S: Stage>(
    encoder: &[S],
    amp: (&[S], &Conv2d),
    phase: (&[S], &Conv2d),
    mut x: Tensor<4>,
) -> (Tensor<4>, Tensor<4>) {
    for s in encoder {
        x = max_pool2d(s.forward(x), MaxPoolOptions::new([2, 2]));
    }
    let a = sigmoid(decode(amp.0, amp.1, x.clone())) * 2.0;
    let p = tanh(decode(phase.0, phase.1, x)) * PI;
    (a, p)
}

/// Upsampling decoder ending in one output channel (before its range map).
#[derive(Module, Debug)]
struct Decoder {
    blocks: Vec<Block>,
    out: Conv2d,
}

impl Decoder {
    fn new(width: usize, depth: usize, device: &Device) -> Self {
        let top = width << (depth - 1);
        let blocks = (0..depth)
            .map(|k| Block::new(top >> k, (top >> k) / 2, device))
            .collect();
        let out = Conv2dConfig::new([top >> depth, 1], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        Self { blocks, out }
    }

    fn fold(&self) -> (Vec<[Conv2d; 2]>, Conv2d) {
        (
            self.blocks.iter().map(Block::fold).collect(),
            self.out.clone(),
        )
    }
}

/// PtychoNN with `depth` encoder stages of `width << k` channels, each halving the image, and
/// two mirror decoders. Input `[batch, channels, n, n]` with `n` a multiple of `2^depth`; output object
/// amplitude in `(0, 2)` (a flat object is 1) and phase in `(-π, π)`, each `[batch, 1, n, n]`.
#[derive(Module, Debug)]
pub struct PtychoNN {
    encoder: Vec<Block>,
    amp: Decoder,
    phase: Decoder,
}

impl PtychoNN {
    pub fn new(channels: usize, width: usize, depth: usize, device: &Device) -> Self {
        assert!(width >= 2 && depth >= 1, "width {width}, depth {depth}");
        let encoder = (0..depth)
            .map(|k| {
                Block::new(
                    if k == 0 { channels } else { width << (k - 1) },
                    width << k,
                    device,
                )
            })
            .collect();
        Self {
            encoder,
            amp: Decoder::new(width, depth, device),
            phase: Decoder::new(width, depth, device),
        }
    }

    /// The image side must be a multiple of this.
    pub fn granule(&self) -> usize {
        1 << self.encoder.len()
    }

    pub fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        run(
            &self.encoder,
            (&self.amp.blocks, &self.amp.out),
            (&self.phase.blocks, &self.phase.out),
            x,
        )
    }

    /// The inference copy of the network at its batch-norm running statistics.
    pub(crate) fn fold(&self) -> Folded {
        Folded {
            encoder: self.encoder.iter().map(Block::fold).collect(),
            amp: self.amp.fold(),
            phase: self.phase.fold(),
        }
    }
}

/// [`PtychoNN`] at inference, each batch normalisation folded into the convolution before it.
/// On a device without autodiff batch normalisation applies its fixed running statistics, so
/// the copy gives the network's output to rounding without the normalisation's kernels.
#[derive(Debug)]
pub(crate) struct Folded {
    encoder: Vec<[Conv2d; 2]>,
    amp: (Vec<[Conv2d; 2]>, Conv2d),
    phase: (Vec<[Conv2d; 2]>, Conv2d),
}

impl Folded {
    pub(crate) fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        run(
            &self.encoder,
            (&self.amp.0, &self.amp.1),
            (&self.phase.0, &self.phase.1),
            x,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::RunningState;
    use burn::tensor::Distribution;

    #[test]
    fn folded_network_gives_the_networks_output() {
        let device = Device::cuda(0);
        let mut net = PtychoNN::new(2, 4, 2, &device);
        let draw = |n, lo, hi| Tensor::<1>::random([n], Distribution::Uniform(lo, hi), &device);
        for b in net
            .encoder
            .iter_mut()
            .chain(&mut net.amp.blocks)
            .chain(&mut net.phase.blocks)
        {
            for n in [&mut b.na, &mut b.nb] {
                let [c] = n.gamma.shape().dims();
                n.gamma = Param::from_tensor(draw(c, 0.5, 2.0));
                n.beta = Param::from_tensor(draw(c, -1.0, 1.0));
                n.running_mean = RunningState::new(draw(c, -1.0, 1.0));
                n.running_var = RunningState::new(draw(c, 0.2, 3.0));
            }
        }
        let x = Tensor::<4>::random([3, 2, 16, 16], Distribution::Normal(0.0, 1.0), &device);
        let (a, p) = net.forward(x.clone());
        let (fa, fp) = net.fold().forward(x);
        for (want, got) in [(a, fa), (p, fp)] {
            let d = (want - got).abs().max().into_scalar::<f32>();
            // Autotune may pick tensor-core convolutions, which round f32 operands to TF32's
            // 11-bit significand, so the scaled weights round apart from the unscaled ones.
            assert!(d < 2e-3, "folded output differs by {d}");
        }
    }
}
