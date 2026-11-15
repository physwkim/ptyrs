//! 2D convolution on channels-last activations, with the weight gradient of a 3×3 same-padded
//! convolution as one CUDA kernel split over bands of image rows. burn takes that gradient as a
//! matmul whose output, `co × 9·ci` (64 × 576 at a U-Net's top level), is a few dozen cubes at
//! most while its inner dimension is every pixel of the batch, so most of the GPU idles; here each
//! cube sums its own band and one more kernel adds the bands. The kernel takes deep levels too,
//! where burn's matmul was faster while it ran f32 on TF32 tensor cores; that put the weight
//! gradient 1e-4 off, and for some shapes (2 to 64 channels on 8 × 8 pixels) off by most of its
//! magnitude. vendor/cubek-matmul keeps f32 off them, and the two have not been timed since.

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::ops::{FloatTensorOps, ModuleOps};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Autodiff, Backend, Dispatch, backend_extension};
use burn::module::{
    Content, Devices, DisplaySettings, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};
use burn::nn::conv::Conv2d;
use burn::nn::{BatchNorm, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::ops::ConvOptions;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::prelude::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

use crate::nn::grid::grid;

/// A convolution of the networks, whose only forward is [`Conv::forward`]. Its record is the
/// [`Conv2d`]'s it wraps, field for field, so saved networks read the same.
#[derive(Clone, Debug)]
pub struct Conv(Conv2d);

impl From<Conv2d> for Conv {
    fn from(c: Conv2d) -> Self {
        Self(c)
    }
}

impl Conv {
    /// The convolution on channels-last `x`, `[batch, h, w, channels]`: `Conv2d::forward` on the
    /// channels-first view, permuted back. burn's CUDA convolutions run channels-last and take
    /// and give that memory as a permuted view, so neither permutation copies.
    ///
    /// On an autodiff device a 3×3 convolution of stride 1, dilation 1, one group and one pixel
    /// of padding on every side takes its output and its input and bias gradients from the same
    /// burn calls, and its weight gradient from `weight_grad`; anything else is burn's throughout.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let c = &self.0;
        if !(x.device().is_autodiff() && is_3x3_same(c)) {
            return c.forward(x.permute([0, 3, 1, 2])).permute([0, 2, 3, 1]);
        }
        let (x, weight) = (x.into_dispatch(), c.weight.val().into_dispatch());
        Tensor::<4>::from_dispatch(match &c.bias {
            None => <Dispatch as ConvOps>::conv3x3(x, weight),
            Some(b) => <Dispatch as ConvOps>::conv3x3_bias(x, weight, b.val().into_dispatch()),
        })
    }

    /// The weight, `[out channels, in channels, kh, kw]`.
    pub fn weight(&self) -> &Param<Tensor<4>> {
        &self.0.weight
    }

    pub fn bias(&self) -> Option<&Param<Tensor<1>>> {
        self.0.bias.as_ref()
    }

    /// This convolution followed by `n` at its running statistics as one convolution.
    pub(crate) fn fold(&self, n: &BatchNorm) -> Self {
        let c = &self.0;
        let [co] = n.gamma.shape().dims();
        let s = n.gamma.val() / (n.running_var.value() + n.epsilon).sqrt();
        let bias = c
            .bias
            .as_ref()
            .expect("a folded convolution carries a bias");
        Self(Conv2d {
            weight: Param::from_tensor(c.weight.val() * s.clone().reshape([co, 1, 1, 1])),
            bias: Some(Param::from_tensor(
                (bias.val() - n.running_mean.value()) * s + n.beta.val(),
            )),
            ..c.clone()
        })
    }
}

impl Module for Conv {
    fn collect_devices(&self, devices: Devices) -> Devices {
        self.0.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        Self(self.0.fork(device))
    }

    fn to_device(self, device: &Device) -> Self {
        Self(self.0.to_device(device))
    }

    fn train(self) -> Self {
        Self(self.0.train())
    }

    fn valid(&self) -> Self {
        Self(self.0.valid())
    }

    fn materialize(self) -> Self {
        Self(self.0.materialize())
    }

    fn num_params(&self) -> usize {
        Module::num_params(&self.0)
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        self.0.visit(visitor)
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        Self(self.0.map(mapper))
    }
}

impl ModuleDisplayDefault for Conv {
    fn content(&self, content: Content) -> Option<Content> {
        self.0.content(content)
    }

    fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

impl ModuleDisplay for Conv {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        self.0.custom_settings()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        self.0.custom_content(content)
    }
}

/// Whether `c` is a 3×3 convolution of stride 1, dilation 1 and one group, padded by one pixel on
/// every side, which is what `Same` padding comes to for it.
fn is_3x3_same(c: &Conv2d) -> bool {
    c.kernel_size == [3, 3]
        && c.stride == [1, 1]
        && c.dilation == [1, 1]
        && c.groups == 1
        && matches!(
            c.padding,
            PaddingConfig2d::Same | PaddingConfig2d::Explicit(1, 1, 1, 1)
        )
}

/// The options `Conv2d::forward` hands burn for a convolution [`is_3x3_same`] admits.
fn three_by_three() -> ConvOptions<2> {
    ConvOptions::new_with_padding([1, 1], [(1, 1), (1, 1)], [1, 1], 1)
}

/// A 3×3 same-padded convolution of stride 1 on channels-last `x`, without and with a bias.
#[backend_extension(Cube, Autodiff)]
trait ConvOps: Backend {
    fn conv3x3(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self>;

    fn conv3x3_bias(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

impl ConvOps for CubeBackend {
    fn conv3x3(x: CubeTensor, weight: CubeTensor) -> CubeTensor {
        forward(x, weight, None)
    }

    fn conv3x3_bias(x: CubeTensor, weight: CubeTensor, bias: CubeTensor) -> CubeTensor {
        forward(x, weight, Some(bias))
    }
}

/// `x` `[n, h, w, c]` as the `[n, c, h, w]` view burn's convolutions take, and back.
fn first(x: CubeTensor) -> CubeTensor {
    CubeBackend::float_permute(x, &[0, 3, 1, 2])
}

fn last(x: CubeTensor) -> CubeTensor {
    CubeBackend::float_permute(x, &[0, 2, 3, 1])
}

fn forward(x: CubeTensor, weight: CubeTensor, bias: Option<CubeTensor>) -> CubeTensor {
    last(CubeBackend::conv2d(
        first(x),
        weight,
        bias,
        three_by_three(),
    ))
}

/// What the backward reads: the input as the forward took it, the weight and the bias.
#[derive(Clone, Debug)]
struct Saved {
    x: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
}

impl Saved {
    /// The convolution's output, and what its backward keeps.
    fn forward(x: CubeTensor, weight: CubeTensor, bias: Option<CubeTensor>) -> (Self, CubeTensor) {
        let y = forward(x.clone(), weight.clone(), bias.clone());
        (Self { x, weight, bias }, y)
    }

    /// The gradients of the input, weight and bias the parents ask for, from the output's `dy`.
    fn backward(self, dy: CubeTensor, want: [bool; 3]) -> [Option<CubeTensor>; 3] {
        let (x, dy_first) = (first(self.x.clone()), first(dy.clone()));
        let dx = want[0].then(|| {
            last(CubeBackend::conv2d_x_backward(
                x.clone(),
                self.weight.clone(),
                dy_first.clone(),
                three_by_three(),
            ))
        });
        let dw = want[1].then(|| weight_grad(self.x, dy));
        let db = match (want[2], self.bias) {
            (true, Some(b)) => Some(CubeBackend::conv2d_bias_backward(x, b, dy_first)),
            _ => None,
        };
        [dx, dw, db]
    }
}

#[derive(Debug)]
struct Conv3x3NoBias;

impl Backward<CubeBackend, 2> for Conv3x3NoBias {
    type State = Saved;

    fn backward(self, ops: Ops<Saved, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
        let [px, pw] = ops.parents;
        let dy = grads.consume::<CubeBackend>(&ops.node);
        let [dx, dw, _] = ops.state.backward(dy, [px.is_some(), pw.is_some(), false]);
        for (parent, grad) in [(px, dx), (pw, dw)] {
            if let (Some(node), Some(grad)) = (parent, grad) {
                grads.register::<CubeBackend>(node.id, grad);
            }
        }
    }
}

#[derive(Debug)]
struct Conv3x3WithBias;

impl Backward<CubeBackend, 3> for Conv3x3WithBias {
    type State = Saved;

    fn backward(self, ops: Ops<Saved, 3>, grads: &mut Gradients, _: &mut Checkpointer) {
        let [px, pw, pb] = ops.parents;
        let dy = grads.consume::<CubeBackend>(&ops.node);
        let want = [px.is_some(), pw.is_some(), pb.is_some()];
        let [dx, dw, db] = ops.state.backward(dy, want);
        for (parent, grad) in [(px, dx), (pw, dw), (pb, db)] {
            if let (Some(node), Some(grad)) = (parent, grad) {
                grads.register::<CubeBackend>(node.id, grad);
            }
        }
    }
}

impl<C: CheckpointStrategy> ConvOps for Autodiff<CubeBackend, C> {
    fn conv3x3(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        let guards = [x.node(), weight.node()];
        let (state, y) = Saved::forward(x.into_primitive(), weight.into_primitive(), None);
        match Conv3x3NoBias
            .prepare::<C>(guards)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(state, y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }

    fn conv3x3_bias(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let guards = [x.node(), weight.node(), bias.node()];
        let (state, y) = Saved::forward(
            x.into_primitive(),
            weight.into_primitive(),
            Some(bias.into_primitive()),
        );
        match Conv3x3WithBias
            .prepare::<C>(guards)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(state, y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
}

/// Input channels per cube, one per lane of a warp.
const LANES: usize = 32;
/// Warps per cube.
const WARPS: usize = 8;
/// Output channels per unit, so a cube takes `PER · WARPS` of them.
const PER: usize = 8;
/// Pixels of a row a cube takes.
const TILE: usize = 32;
/// Pixels whose sums one pass of a unit's loop unrolls.
const UNROLL: usize = 4;
/// At most how many cubes one weight gradient spreads over.
const CUBES: usize = 512;

/// The weight gradient `[co, ci, 3, 3]` of a 3×3 same-padded convolution of stride 1 from its
/// channels-last input `x`, `[n, h, w, ci]`, and output gradient `dy`, `[n, h, w, co]`.
///
/// Cube `a + across·(s + bands·b)` sums input channels `a·LANES ..` against output channels
/// `b·PER·WARPS ..` over band `s`, `TILE` columns of a run of rows of one image, into row `s` of a
/// `[bands, co·ci·9]` tensor, and a second kernel adds the bands in order, so the result does not
/// depend on scheduling.
fn weight_grad(x: CubeTensor, dy: CubeTensor) -> CubeTensor {
    let (x, dy) = (into_contiguous(x), into_contiguous(dy));
    let [n, h, w, ci] = x.meta.shape().dims();
    let [_, _, _, co] = dy.meta.shape().dims();
    let (across, down) = (ci.div_ceil(LANES), co.div_ceil(PER * WARPS));
    let strips = w.div_ceil(TILE);
    let rows = h.div_ceil((CUBES / (across * down * n * strips)).clamp(1, h));
    let blocks = h.div_ceil(rows);
    let bands = n * blocks * strips;
    let cubes = across * bands * down;
    let len = co * ci * 9;
    let dtype = dtype_to_storage_type(x.dtype);
    let client = x.client.clone();
    let empty = |shape: Shape| {
        empty_device_contiguous_dtype(client.clone(), x.device.clone(), shape, x.dtype)
    };
    let parts = empty([bands, len].into());
    let out = empty([co, ci, 3, 3].into());
    let u = |v: usize| u32::try_from(v).expect("a weight gradient of fewer than 2³² elements");
    // SAFETY: every index the kernels form is below its tensor's length: pixels are masked against
    // the image, channels against `ci` and `co`, bands against `n` images and elements against
    // `len`.
    unsafe {
        kernel::weight_parts::launch_unchecked(
            &client,
            grid(&client, cubes),
            CubeDim::new_2d(LANES as u32, WARPS as u32),
            x.into_tensor_arg(),
            dy.into_tensor_arg(),
            parts.clone().into_tensor_arg(),
            u(h),
            u(w),
            u(ci),
            u(co),
            u(rows),
            u(blocks),
            u(strips),
            u(across),
            u(bands),
            u(cubes),
            TILE,
            LANES,
            WARPS,
            PER,
            UNROLL,
            dtype,
        );
        kernel::sum_bands::launch_unchecked(
            &client,
            grid(&client, len.div_ceil(256)),
            CubeDim::new_1d(256),
            parts.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            u(len),
            u(bands),
            dtype,
        );
    }
    out
}

mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    /// One band's partial weight gradient. For each row of its band the cube stages the strip of
    /// `dy` for its output channels, and keeps the strip of `x` for its input channels, one pixel
    /// wider on each side, for that row and the rows above and below in a ring of three, so each
    /// row of `x` is read once. Unit `(l, k)` holds input channel `l` against `per` output
    /// channels at all nine taps. Reads for the next row go to registers before the current row's
    /// sums, so they overlap.
    #[cube(launch_unchecked)]
    pub fn weight_parts<F: Float>(
        x: &Tensor<F>,
        dy: &Tensor<F>,
        parts: &mut Tensor<F>,
        h: u32,
        w: u32,
        ci: u32,
        co: u32,
        rows: u32,
        blocks: u32,
        strips: u32,
        across: u32,
        bands: u32,
        cubes: u32,
        #[comptime] tile: usize,
        #[comptime] lanes: usize,
        #[comptime] warps: usize,
        #[comptime] per: usize,
        #[comptime] unroll: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let (h, w, ci, co) = (h as usize, w as usize, ci as usize, co as usize);
        let units = comptime!(lanes * warps);
        let outs = comptime!(per * warps);
        let xrow = comptime!((tile + 2) * lanes);
        let gsize = comptime!(tile * outs);
        let xloads = comptime!(xrow.div_ceil(units));
        let gloads = comptime!(gsize.div_ceil(units));
        let mut gs = Shared::<[F]>::new_slice(gsize);
        let mut xs = Shared::<[F]>::new_slice(comptime!(3 * xrow));
        let mut acc = Array::<F>::new(comptime!(per * 9));
        #[unroll]
        for k in 0..comptime!(per * 9) {
            acc[k] = F::new(0.0_f32);
        }
        let mut xr = Array::<F>::new(xloads);
        let mut gr = Array::<F>::new(gloads);
        let mut gv = Array::<F>::new(per);
        let (l, k0) = (UNIT_POS_X as usize, UNIT_POS_Y as usize * per);
        let unit = UNIT_POS as usize;
        let (across, bands) = (across as usize, bands as usize);
        let ci0 = CUBE_POS % across * lanes;
        let band = CUBE_POS / across % bands;
        let co0 = CUBE_POS / across / bands * outs;
        let (strips, blocks) = (strips as usize, blocks as usize);
        let x0 = (band % strips) * tile;
        let image = band / strips / blocks;
        let y0 = (band / strips % blocks) * rows as usize;
        let mut y1 = y0 + rows as usize;
        if y1 > h {
            y1 = h;
        }
        // Rows of `x` are counted from one above the image, so row `s` is the image's `s - 1`.
        let first = image * h;
        let reads = y1 - y0 + 2;
        // Step `t` stages the row of `x` read at step `t - 1` and, from the fourth, the row of
        // `dy` read with it; reads row `y0 + t` of `x` and, from the third, image row `y0 + t - 2`
        // of `dy`; and, from the fourth, sums image row `y0 + t - 3`.
        let mut t = 0;
        while t <= reads {
            if t >= 1 {
                let slot = ((y0 + t - 1) % 3) * xrow;
                #[unroll]
                for i in 0..xloads {
                    let e = unit + i * units;
                    if e < xrow {
                        xs[slot + e] = xr[i];
                    }
                }
            }
            if t >= 3 {
                #[unroll]
                for i in 0..gloads {
                    let e = unit + i * units;
                    if e < gsize {
                        gs[e] = gr[i];
                    }
                }
            }
            sync_cube();
            if t < reads {
                let s = y0 + t;
                #[unroll]
                for i in 0..xloads {
                    let e = unit + i * units;
                    let (j, c) = (e / lanes, e % lanes);
                    let mut v = F::new(0.0_f32);
                    if e < xrow && s >= 1 && s <= h && x0 + j >= 1 && x0 + j <= w && ci0 + c < ci {
                        v = x[((first + s - 1) * w + x0 + j - 1) * ci + ci0 + c];
                    }
                    xr[i] = v;
                }
                if t >= 2 {
                    let row = first + y0 + t - 2;
                    #[unroll]
                    for i in 0..gloads {
                        let e = unit + i * units;
                        let (p, o) = (e / outs, e % outs);
                        let mut v = F::new(0.0_f32);
                        if e < gsize && x0 + p < w && co0 + o < co {
                            v = dy[(row * w + x0 + p) * co + co0 + o];
                        }
                        gr[i] = v;
                    }
                }
            }
            if t >= 3 {
                // Pixels past the row's end have a zero gradient, so they add exact zeros.
                let mut pb = 0;
                while pb < tile {
                    #[unroll]
                    for q in 0..unroll {
                        let p = pb + q;
                        #[unroll]
                        for k in 0..per {
                            gv[k] = gs[p * outs + k0 + k];
                        }
                        #[unroll]
                        for ky in 0..3 {
                            let base = ((y0 + t - 3 + ky) % 3) * xrow + l;
                            #[unroll]
                            for kx in 0..3 {
                                let v = xs[base + (p + kx) * lanes];
                                #[unroll]
                                for k in 0..per {
                                    acc[k * 9 + ky * 3 + kx] += gv[k] * v;
                                }
                            }
                        }
                    }
                    pb += unroll;
                }
            }
            sync_cube();
            t += 1;
        }
        let c = ci0 + l;
        if c < ci {
            let at = band * co * ci * 9;
            #[unroll]
            for k in 0..per {
                let o = co0 + k0 + k;
                if o < co {
                    #[unroll]
                    for t in 0..9 {
                        parts[at + (o * ci + c) * 9 + t] = acc[k * 9 + t];
                    }
                }
            }
        }
    }

    /// `out[i]`, the sum of column `i` of the `[bands, len]` partials in band order.
    #[cube(launch_unchecked)]
    pub fn sum_bands<F: Float>(
        parts: &Tensor<F>,
        out: &mut Tensor<F>,
        len: u32,
        bands: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        let i = ABSOLUTE_POS;
        if i < len as usize {
            let mut acc = F::new(0.0_f32);
            let mut s = 0;
            while s < bands as usize {
                acc += parts[s * len as usize + i];
                s += 1;
            }
            out[i] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::nn::conv::Conv2dConfig;
    use burn::tensor::ops::PadMode;
    use burn::tensor::{DType, Distribution};

    /// A convolution's sizes: batch, height, width, input and output channels, kernel side, and
    /// whether it has a bias, whether its input asks for a gradient and whether that input is a
    /// strided view.
    #[derive(Clone, Copy, Debug)]
    struct Case {
        n: usize,
        h: usize,
        w: usize,
        ci: usize,
        co: usize,
        k: usize,
        bias: bool,
        x_grad: bool,
        strided: bool,
    }

    const fn case(n: usize, h: usize, w: usize, ci: usize, co: usize) -> Case {
        Case {
            n,
            h,
            w,
            ci,
            co,
            k: 3,
            bias: true,
            x_grad: true,
            strided: false,
        }
    }

    /// The input (channels-first when strided), weight, bias and output weighting, in f32.
    fn inputs(c: Case, device: &Device) -> [Tensor<4>; 4] {
        device.seed((c.n * 1009 + c.h * 101 + c.w * 11 + c.ci * 7 + c.co) as u64);
        let normal = Distribution::Normal(0.2, 1.0);
        let x = if c.strided {
            [c.n, c.ci, c.h, c.w]
        } else {
            [c.n, c.h, c.w, c.ci]
        };
        let f32 = |t: Tensor<4>| t.cast(DType::F32).detach();
        [
            f32(Tensor::random(x, normal, device)),
            f32(Tensor::random([c.co, c.ci, c.k, c.k], normal, device)),
            f32(Tensor::random([1, 1, 1, c.co], normal, device)),
            f32(Tensor::random([c.n, c.h, c.w, c.co], normal, device)),
        ]
    }

    fn host<const D: usize>(t: Tensor<D>) -> Vec<f64> {
        t.into_data()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .unwrap()
    }

    /// The output and the input, weight and bias gradients of `Σ y·g`, `y` the convolution of
    /// `inputs` in `dtype` through [`Conv::forward`] if `fused`, else through `Conv2d::forward`. A
    /// gradient not asked for is empty.
    fn run(c: Case, inputs: &[Tensor<4>; 4], dtype: DType, fused: bool) -> [Vec<f64>; 4] {
        let [x, weight, bias, g] = inputs.clone().map(|t| t.cast(dtype));
        let mut conv2d = Conv2dConfig::new([c.ci, c.co], [c.k, c.k])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(c.bias)
            .init(&x.device());
        conv2d.weight = Param::from_tensor(weight);
        conv2d.bias = c.bias.then(|| Param::from_tensor(bias.reshape([c.co])));
        let x = if c.x_grad { x.require_grad() } else { x };
        let view = if c.strided {
            x.clone().permute([0, 2, 3, 1])
        } else {
            x.clone()
        };
        let y = if fused {
            Conv(conv2d.clone()).forward(view)
        } else {
            conv2d
                .forward(view.permute([0, 3, 1, 2]))
                .permute([0, 2, 3, 1])
        };
        let grads = (y.clone() * g).sum().backward();
        let dx = x.grad(&grads).map(host).unwrap_or_default();
        let dw = host(conv2d.weight.val().grad(&grads).unwrap());
        let db = conv2d
            .bias
            .map(|b| host(b.val().grad(&grads).unwrap()))
            .unwrap_or_default();
        [host(y.inner()), dx, dw, db]
    }

    fn max_error(a: &[f64], b: &[f64]) -> (f64, f64) {
        assert_eq!(a.len(), b.len());
        let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let err = a
            .iter()
            .zip(b)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        (err, scale)
    }

    /// [`Conv::forward`] against burn's path: the output and the input and bias gradients are the
    /// same burn calls, so bit for bit; the weight gradient to `1e-12` of its largest magnitude in
    /// f64, and to `2e-5` of the f64 one in f32.
    fn agrees(c: Case) {
        let device = crate::nn::cuda(0).autodiff();
        let inputs = inputs(c, &device);
        for dtype in [DType::F64, DType::F32] {
            // burn's autotune may pick its kernel on the first call of a shape; pick it first.
            run(c, &inputs, dtype, false);
            let [y, dx, dw, db] = run(c, &inputs, dtype, true);
            let [ry, rdx, rdw, rdb] = run(c, &inputs, dtype, false);
            assert_eq!(y, ry, "{c:?} {dtype:?}: output");
            assert_eq!(dx, rdx, "{c:?} {dtype:?}: input gradient");
            assert_eq!(db, rdb, "{c:?} {dtype:?}: bias gradient");
            assert_eq!(dx.is_empty(), !c.x_grad);
            assert_eq!(db.is_empty(), !c.bias);
            let (reference, tol) = match dtype {
                DType::F64 => (rdw, 1e-12),
                _ => (run(c, &inputs, DType::F64, false)[2].clone(), 2e-5),
            };
            let (err, scale) = max_error(&dw, &reference);
            assert!(
                err <= tol * scale,
                "{c:?} {dtype:?}: weight gradient off by {err:e} of {scale:e}"
            );
        }
    }

    #[test]
    fn small_images_and_channels() {
        // Narrower than a tile, fewer channels than a cube's lanes and output channels.
        agrees(case(2, 9, 13, 3, 5));
        // On TF32 tensor cores burn 0.22.1's own f32 weight gradient here was off by most of its
        // magnitude.
        agrees(case(1, 8, 8, 2, 64));
        agrees(case(2, 1, 1, 1, 1));
        agrees(Case {
            bias: false,
            ..case(3, 17, 33, 8, 2)
        });
    }

    #[test]
    fn several_channel_blocks_and_tiles() {
        // 40 input and 70 output channels take two blocks each; 70 pixels take three tiles.
        agrees(case(1, 5, 70, 40, 70));
    }

    #[test]
    fn short_last_bands() {
        // 301 rows an image over 256 cubes: bands of two rows, the last of each image one.
        agrees(Case {
            x_grad: false,
            ..case(2, 301, 7, 4, 4)
        });
    }

    /// 65536 one-pixel images are as many bands, and 4194241 output channels 65536 blocks of 64,
    /// each count past the 65535 cubes CUDA launches along a grid's Y or Z.
    #[test]
    fn more_than_65535_bands_or_channel_blocks() {
        f32_weight_gradient(case(65_536, 1, 1, 1, 1), 2e-5);
        let c = Case {
            x_grad: false,
            ..case(1, 1, 1, 1, PER * WARPS * 65_535 + 1)
        };
        f32_weight_gradient(c, 2e-5);
    }

    #[test]
    fn strided_input() {
        agrees(Case {
            strided: true,
            ..case(2, 11, 37, 6, 9)
        });
    }

    /// The f64 weight gradient of `Σ y·g`, `y` the `k × k` same convolution of the channels-last
    /// `x`: for each kernel offset, `gᵀ` times `x` shifted by it, a product per image row summed
    /// over rows. burn's f64 convolution would unfold `x` nine-fold, 6.7 GB on pty-chi's 852²
    /// 128-channel one.
    fn weight_gradient(c: Case, x: Tensor<4>, g: Tensor<4>) -> Vec<f64> {
        let p = c.k / 2;
        let rows = c.n * c.h;
        let x = x
            .cast(DType::F64)
            .pad([(p, p), (p, p), (0, 0)], PadMode::Constant(0.0));
        let gt = g
            .cast(DType::F64)
            .reshape([rows, c.w, c.co])
            .swap_dims(1, 2);
        let offsets = (0..c.k * c.k)
            .map(|o| {
                let (ky, kx) = (o / c.k, o % c.k);
                let shifted = x
                    .clone()
                    .slice([0..c.n, ky..ky + c.h, kx..kx + c.w])
                    .reshape([rows, c.w, c.ci]);
                gt.clone().matmul(shifted).sum_dim(0).reshape([c.co, c.ci])
            })
            .collect();
        host(Tensor::<2>::stack::<3>(offsets, 2))
    }

    /// The f32 weight gradient through [`Conv::forward`] against the f64 one, to `tol` of its
    /// largest magnitude.
    fn f32_weight_gradient(c: Case, tol: f64) {
        let device = crate::nn::cuda(0).autodiff();
        let inputs = inputs(c, &device);
        let reference = weight_gradient(c, inputs[0].clone(), inputs[3].clone());
        let [_, _, dw, _] = run(c, &inputs, DType::F32, true);
        let (err, scale) = max_error(&dw, &reference);
        assert!(
            err <= tol * scale,
            "{c:?}: f32 weight gradient off by {err:e} of {scale:e}"
        );
    }

    /// The `[side, in, out]` of each distinct convolution of a PtychoNN on `side`² images of
    /// `images` channels, as `PtychoNN::new` builds it: two per encoder stage, two per decoder
    /// stage, then the output's.
    fn ptychonn(images: usize, width: usize, depth: usize, side: usize) -> Vec<[usize; 3]> {
        let mut convs = Vec::new();
        let mut s = side;
        for k in 0..depth {
            let cin = if k == 0 { images } else { width << (k - 1) };
            convs.extend([[s, cin, width << k], [s, width << k, width << k]]);
            s /= 2;
        }
        let top = width << (depth - 1);
        for k in 0..depth {
            let (cin, cout) = (top >> k, (top >> k) / 2);
            convs.extend([[s, cin, cout], [s, cout, cout]]);
            s *= 2;
        }
        convs.push([side, top >> depth, 1]);
        convs.sort();
        convs.dedup();
        convs
    }

    /// The same for a DIP U-Net on a `side`² image, as `DipNet::new` builds it.
    fn dip(code: usize, width: usize, depth: usize, side: usize) -> Vec<[usize; 3]> {
        let w = |k: usize| width << k;
        let top = w(depth - 1);
        let mut convs = vec![[side >> depth, top, top], [side, w(0), 2]];
        for k in 0..depth {
            let s = side >> k;
            let deeper = if k + 1 == depth { top } else { w(k + 1) };
            let cin = if k == 0 { code } else { w(k - 1) };
            convs.extend([[s, cin, w(k)], [s, deeper + w(k), w(k)], [s, w(k), w(k)]]);
        }
        convs.sort();
        convs.dedup();
        convs
    }

    /// Every convolution PtychoNN runs as `nn_train_ptychonn` trains it: one or two 64² images
    /// (256² frames binned by 4), width 32, depth 3, batches of 32.
    #[test]
    fn ptychonn_convolutions() {
        let mut convs = [ptychonn(1, 32, 3, 64), ptychonn(2, 32, 3, 64)].concat();
        convs.sort();
        convs.dedup();
        for [side, ci, co] in convs {
            f32_weight_gradient(case(32, side, side, ci, co), 2e-5);
        }
    }

    /// Every convolution a DIP runs at its defaults (width 16, depth 4, code 8) on a 320² image.
    #[test]
    fn dip_convolutions() {
        for [side, ci, co] in dip(8, 16, 4, 320) {
            f32_weight_gradient(case(1, side, side, ci, co), 2e-5);
        }
    }

    /// Every convolution of pty-chi's U-Net as `nn_dip_ptychi` runs it on the 852² comparison
    /// object from 32 code channels; its 3×3 convolutions have no bias, and its 1×1 output
    /// convolution is burn's throughout.
    #[test]
    fn ptychi_unet_convolutions() {
        for [side, ci, co] in [
            [852, 32, 64],
            [852, 64, 64],
            [426, 64, 128],
            [426, 128, 128],
            [213, 128, 256],
            [213, 256, 256],
            [106, 256, 512],
            [106, 512, 512],
            [53, 512, 512],
            [106, 1024, 512],
            [106, 512, 256],
            [213, 512, 256],
            [213, 256, 128],
            [426, 256, 128],
            [426, 128, 64],
            [852, 128, 64],
        ] {
            let c = Case {
                bias: false,
                ..case(1, side, side, ci, co)
            };
            f32_weight_gradient(c, 2e-5);
        }
        let outc = Case {
            k: 1,
            ..case(1, 852, 852, 64, 2)
        };
        f32_weight_gradient(outc, 2e-5);
    }

    #[test]
    fn only_3x3_same_takes_the_kernel() {
        let device = crate::nn::cuda(0);
        let admits = |config: Conv2dConfig| is_3x3_same(&config.init(&device));
        let base = || Conv2dConfig::new([2, 2], [3, 3]);
        assert!(admits(base().with_padding(PaddingConfig2d::Same)));
        assert!(admits(
            base().with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        ));
        assert!(!admits(base().with_padding(PaddingConfig2d::Valid)));
        assert!(!admits(
            base()
                .with_padding(PaddingConfig2d::Same)
                .with_dilation([2, 2])
        ));
        assert!(!admits(
            Conv2dConfig::new([2, 2], [1, 1]).with_padding(PaddingConfig2d::Same)
        ));
    }
}
