//! The PtychoNN network (Cherukara et al. 2020): one convolutional encoder over the diffraction
//! amplitude and two decoders, one for the object amplitude and one for its phase.

use std::f64::consts::PI;

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

    fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        for b in &self.blocks {
            let [_, _, h, w] = x.dims();
            x = interpolate(
                b.forward(x),
                InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([2 * h, 2 * w]),
            );
        }
        self.out.forward(x)
    }
}

/// PtychoNN with `depth` encoder stages of `width << k` channels, each halving the image, and
/// two mirror decoders. Input `[batch, 1, n, n]` with `n` a multiple of `2^depth`; output object
/// amplitude in `(0, 2)` (a flat object is 1) and phase in `(-π, π)`, each `[batch, 1, n, n]`.
#[derive(Module, Debug)]
pub struct PtychoNN {
    encoder: Vec<Block>,
    amp: Decoder,
    phase: Decoder,
}

impl PtychoNN {
    pub fn new(width: usize, depth: usize, device: &Device) -> Self {
        assert!(width >= 2 && depth >= 1, "width {width}, depth {depth}");
        let encoder = (0..depth)
            .map(|k| {
                Block::new(
                    if k == 0 { 1 } else { width << (k - 1) },
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

    pub fn forward(&self, mut x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        for b in &self.encoder {
            x = max_pool2d(b.forward(x), MaxPoolOptions::new([2, 2]));
        }
        let amp = sigmoid(self.amp.forward(x.clone())) * 2.0;
        let phase = tanh(self.phase.forward(x)) * PI;
        (amp, phase)
    }
}
