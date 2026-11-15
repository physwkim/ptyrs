//! 2D convolution on channels-last activations. A 3×3 same-padded convolution runs on kernels of
//! its own: `conv3x3` gives its output, and its input gradient as the convolution of the output
//! gradient by the weight `arrange` flips and transposes; a cube of it holds 8 × 32 pixels against
//! up to 64 output channels, staging the input a few channels at a time with a border of one. Its
//! weight and bias gradients are one kernel split over bands of image rows. burn takes the first as a
//! matmul whose output, `co × 9·ci` (64 × 576 at a U-Net's top level), is a few dozen cubes at
//! most while its inner dimension is every pixel of the batch, so most of the GPU idles; here each
//! cube sums its own band and one more kernel adds the bands. The kernel takes deep levels too,
//! where burn's matmul was faster while it ran f32 on TF32 tensor cores; that put the weight
//! gradient 1e-4 off, and for some shapes (2 to 64 channels on 8 × 8 pixels) off by most of its
//! magnitude. vendor/cubek-matmul keeps f32 off them; timed so, the kernel is 1.25 to 1.6 times as
//! fast as burn at the deep levels, and burn comes within 6% of it only on PtychoNN's 8 × 8 pixels.

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::FloatTensor;
use burn::backend::{Autodiff, Backend, Dispatch, backend_extension};
use burn::module::{
    Content, Devices, DisplaySettings, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};
use burn::nn::conv::Conv2d;
use burn::nn::{BatchNorm, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::DType;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::prelude::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

use crate::nn::grid::{check_elements, grid};

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
    /// The convolution on channels-last `x`, `[batch, h, w, channels]`.
    ///
    /// A 3×3 convolution of stride 1, dilation 1, one group and one pixel of padding on every
    /// side, in f32 or f64, takes its output and input gradient from `conv3x3`, its weight
    /// gradient from `weight_grad` and its bias gradient from burn; those kernels sum in the
    /// activation's float type, too narrow in f16 or bf16. Anything else is `Conv2d::forward` on
    /// the channels-first view, permuted back; burn's CUDA convolutions run channels-last and take
    /// and give that memory as a permuted view, so neither permutation copies.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let c = &self.0;
        let wide = matches!(x.dtype(), DType::F32 | DType::F64);
        if !(is_3x3_same(c) && wide) {
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

/// The convolution of `x`, `[n, h, w, ci]`, by `weight`, `[co, ci, 3, 3]`, plus `bias`.
fn forward(x: CubeTensor, weight: CubeTensor, bias: Option<CubeTensor>) -> CubeTensor {
    conv3x3(x, arrange(weight, false), bias)
}

/// The input gradient `[n, h, w, ci]` of the convolution by `weight` from its output's `dy`,
/// `[n, h, w, co]`: the convolution of `dy` by `weight` turned half a turn, channels swapped.
fn input_grad(dy: CubeTensor, weight: CubeTensor) -> CubeTensor {
    conv3x3(dy, arrange(weight, true), None)
}

/// What the backward reads: the input as the forward took it, and the weight.
#[derive(Clone, Debug)]
struct Saved {
    x: CubeTensor,
    weight: CubeTensor,
}

impl Saved {
    /// The convolution's output, and what its backward keeps.
    fn forward(x: CubeTensor, weight: CubeTensor, bias: Option<CubeTensor>) -> (Self, CubeTensor) {
        let y = forward(x.clone(), weight.clone(), bias);
        (Self { x, weight }, y)
    }

    /// The gradients of the input, weight and bias the parents ask for, from the output's `dy`.
    /// The bias gradient comes from the weight gradient's kernel, so it alone costs both.
    fn backward(self, dy: CubeTensor, want: [bool; 3]) -> [Option<CubeTensor>; 3] {
        let dx = want[0].then(|| input_grad(dy.clone(), self.weight));
        let (dw, db) = if want[1] || want[2] {
            let (dw, db) = weight_grad(self.x, dy, want[2]);
            (want[1].then_some(dw), db)
        } else {
            (None, None)
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

/// A kernel's size argument. Past [`check_elements`] it is below 2³² unless a tensor is empty.
fn u(v: usize) -> u32 {
    u32::try_from(v).expect("a convolution size below 2³²")
}

/// Input channels per cube, one per lane of a warp.
const LANES: usize = 32;
/// Warps per cube.
const WARPS: usize = 8;
/// Output channels per unit, so a cube takes `PER · WARPS` of them.
const PER: usize = 8;
/// Pixels whose sums one pass of a unit's loop unrolls.
const UNROLL: usize = 4;
/// At most how many cubes one weight gradient spreads over.
const CUBES: usize = 512;

/// The weight gradient `[co, ci, 3, 3]` of a 3×3 same-padded convolution of stride 1 from its
/// channels-last input `x`, `[n, h, w, ci]`, and output gradient `dy`, `[n, h, w, co]`, and, if
/// `bias`, the bias gradient `[co]`.
///
/// Cube `a + across·(s + bands·b)` sums input channels `a·LANES ..` against output channels
/// `b·PER·WARPS ..` over band `s`, `LANES` columns of a run of rows of one image, into row `s` of a
/// `[bands, co·ci·9]` tensor, followed by the band's sums of `dy` from the cubes of `a = 0` if
/// `bias`; a second kernel adds the bands in order, so the result does not depend on scheduling.
fn weight_grad(x: CubeTensor, dy: CubeTensor, bias: bool) -> (CubeTensor, Option<CubeTensor>) {
    let [n, h, w, ci] = x.meta.shape().dims();
    let [dn, dh, dw, co] = dy.meta.shape().dims();
    // The kernel reads `dy` at `x`'s pixels.
    assert_eq!(
        [dn, dh, dw],
        [n, h, w],
        "an output gradient over the input's pixels"
    );
    // `dw` and `db`, checked before their counts are multiplied out.
    check_elements(&[x.meta.shape(), dy.meta.shape(), &[co, ci, 9], &[co]]);
    // The cubes of `a = 0` sum the bias gradient, so they are there even for no input channels.
    let (across, down) = (ci.div_ceil(LANES).max(1), co.div_ceil(PER * WARPS));
    let strips = w.div_ceil(LANES);
    // A band holds at least one row; no pixels make no bands, and no output channels no cubes.
    let split = (CUBES / (across * down * n * strips).max(1)).clamp(1, h.max(1));
    let rows = h.div_ceil(split).max(1);
    let blocks = h.div_ceil(rows);
    let bands = n * blocks * strips;
    let cubes = across * bands * down;
    let lw = co * ci * 9;
    let len = lw + if bias { co } else { 0 };
    check_elements(&[&[bands, len]]);
    let (x, dy) = (into_contiguous(x), into_contiguous(dy));
    let dtype = dtype_to_storage_type(x.dtype);
    let client = x.client.clone();
    let empty = |shape: Shape| {
        empty_device_contiguous_dtype(client.clone(), x.device.clone(), shape, x.dtype)
    };
    let parts = empty([bands, len].into());
    let dw = empty([co, ci, 3, 3].into());
    let db = bias.then(|| empty([co].into()));
    // SAFETY: every index the kernels form is below its tensor's length: pixels are masked against
    // the image, channels against `ci` and `co`, bands against `n` images and elements against
    // `lw` and `len`.
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
            u(len),
            u(rows),
            u(blocks),
            u(strips),
            u(across),
            u(bands),
            u(cubes),
            LANES,
            WARPS,
            PER,
            UNROLL,
            bias,
            dtype,
        );
        kernel::sum_bands::launch_unchecked(
            &client,
            grid(&client, len.div_ceil(256)),
            CubeDim::new_1d(256),
            parts.into_tensor_arg(),
            dw.clone().into_tensor_arg(),
            db.clone().map(|t| t.into_tensor_arg()).into(),
            u(lw),
            u(len),
            u(bands),
            dtype,
        );
    }
    (dw, db)
}

/// Pixel rows a cube of `conv3x3` takes.
const ROWS: usize = 8;
/// Pixel columns a cube of `conv3x3` takes, one per lane of a warp.
const COLS: usize = 32;
/// At most how many warps a cube of `conv3x3` has, each for `PER` output channels.
const GROUPS: usize = 8;

/// The 3×3 same-padded convolution of stride 1 of channels-last `x`, `[n, h, w, a]`, by `wk`,
/// `[a, 9, b]` (input channel, tap `3·ky + kx`, output channel), plus `bias`, `[b]`.
///
/// Cube `t + tiles·s` takes tile `t`, `ROWS × COLS` pixels of one image, against output channels
/// `s·PER·groups ..`, `groups` the fewest warps that hold every output channel, at most
/// `GROUPS`. It stages as many input channels a pass as it has warps, so each unit stages about
/// eleven pixels a pass whatever the warps, and no more than 32 bytes of them a pixel, which keeps
/// an f64 cube within 48 KiB of shared memory.
fn conv3x3(x: CubeTensor, wk: CubeTensor, bias: Option<CubeTensor>) -> CubeTensor {
    let [n, h, w, a] = x.meta.shape().dims();
    let [_, _, b] = wk.meta.shape().dims();
    // `wk` is `arrange`'s, which checked it.
    let bias_shape = bias.as_ref().map_or(&[][..], |t| &t.meta.shape()[..]);
    check_elements(&[x.meta.shape(), bias_shape, &[n, h, w, b]]);
    let x = into_contiguous(x);
    let groups = b.div_ceil(PER).next_power_of_two().min(GROUPS);
    let chunk = groups.min(32 / x.dtype.size());
    let (across, down) = (w.div_ceil(COLS), h.div_ceil(ROWS));
    let tiles = n * down * across;
    let cubes = tiles * b.div_ceil(PER * groups);
    let client = x.client.clone();
    let y = empty_device_contiguous_dtype(
        client.clone(),
        x.device.clone(),
        [n, h, w, b].into(),
        x.dtype,
    );
    let dtype = dtype_to_storage_type(x.dtype);
    // SAFETY: every index the kernel forms is below its tensor's length: halo pixels are masked
    // against the image, channels against `a` and `b`, and stores against the image and `b`.
    unsafe {
        kernel::conv3x3::launch_unchecked(
            &client,
            grid(&client, cubes),
            CubeDim::new_2d(COLS as u32, groups as u32),
            x.into_tensor_arg(),
            wk.into_tensor_arg(),
            bias.map(|b| b.into_tensor_arg()).into(),
            y.clone().into_tensor_arg(),
            u(h),
            u(w),
            u(a),
            u(b),
            u(across),
            u(down),
            u(tiles),
            u(cubes),
            ROWS,
            COLS,
            groups,
            PER,
            chunk,
            dtype,
        );
    }
    y
}

/// `weight`, `[co, ci, 3, 3]`, as the `[ci, 9, co]` `conv3x3` reads for the forward, or, `flip`ped,
/// as the `[co, 9, ci]` of the input gradient, whose tap `t` is the forward's `8 - t`.
fn arrange(weight: CubeTensor, flip: bool) -> CubeTensor {
    let [co, ci, _, _] = weight.meta.shape().dims();
    check_elements(&[weight.meta.shape(), &[co, ci, 9]]);
    let weight = into_contiguous(weight);
    let shape = if flip { [co, 9, ci] } else { [ci, 9, co] };
    let client = weight.client.clone();
    let wk = empty_device_contiguous_dtype(
        client.clone(),
        weight.device.clone(),
        shape.into(),
        weight.dtype,
    );
    let len = co * ci * 9;
    let dtype = dtype_to_storage_type(weight.dtype);
    // SAFETY: the kernel reads and writes element `i < len` of tensors of `len` elements.
    unsafe {
        kernel::arrange::launch_unchecked(
            &client,
            grid(&client, len.div_ceil(256)),
            CubeDim::new_1d(256),
            weight.into_tensor_arg(),
            wk.clone().into_tensor_arg(),
            u(ci),
            u(co),
            flip,
            dtype,
        );
    }
    wk
}

mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    /// One band's partial weight gradient. For each row of its band the cube stages the strip of
    /// `dy` for its output channels, and keeps the strip of `x` for its input channels, one pixel
    /// wider on each side, for that row and the rows above and below in a ring of three, so each
    /// row of `x` is read once. Unit `(l, k)` holds input channel `l` against `per` output
    /// channels at all nine taps, and, if `bias` and the cube's input channels are the first, the
    /// sum of `dy` at pixel `l` of the strip for them. Reads for the next row go to registers
    /// before the current row's sums, so they overlap.
    #[cube(launch_unchecked)]
    pub fn weight_parts<F: Float>(
        x: &Tensor<F>,
        dy: &Tensor<F>,
        parts: &mut Tensor<F>,
        h: u32,
        w: u32,
        ci: u32,
        co: u32,
        len: u32,
        rows: u32,
        blocks: u32,
        strips: u32,
        across: u32,
        bands: u32,
        cubes: u32,
        #[comptime] lanes: usize,
        #[comptime] warps: usize,
        #[comptime] per: usize,
        #[comptime] unroll: usize,
        #[comptime] bias: bool,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let (h, w, ci, co) = (h as usize, w as usize, ci as usize, co as usize);
        let units = comptime!(lanes * warps);
        let outs = comptime!(per * warps);
        // A strip is as wide as a warp.
        let tile = lanes;
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
        let mut bacc = Array::<F>::new(per);
        #[unroll]
        for k in 0..per {
            bacc[k] = F::new(0.0_f32);
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
                if bias {
                    #[unroll]
                    for k in 0..per {
                        bacc[k] += gs[l * outs + k0 + k];
                    }
                }
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
        let at = band * len as usize;
        // Every cube sums `dy`; those of the first input channels store the sums.
        if bias {
            #[unroll]
            for k in 0..per {
                gs[l * outs + k0 + k] = bacc[k];
            }
            sync_cube();
            if ci0 == 0 && unit < outs && co0 + unit < co {
                let mut sum = F::new(0.0_f32);
                #[unroll]
                for j in 0..lanes {
                    sum += gs[j * outs + unit];
                }
                parts[at + co * ci * 9 + co0 + unit] = sum;
            }
        }
        let c = ci0 + l;
        if c < ci {
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

    /// The sum of column `i` of the `[bands, len]` partials in band order, into `dw[i]` for the
    /// first `lw` columns and into `db` for the rest.
    #[cube(launch_unchecked)]
    pub fn sum_bands<F: Float>(
        parts: &Tensor<F>,
        dw: &mut Tensor<F>,
        db: ComptimeOption<&mut Tensor<F>>,
        lw: u32,
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
            let lw = lw as usize;
            if i < lw {
                dw[i] = acc;
            } else {
                #[comptime]
                if let ComptimeOption::Some(db) = db {
                    db[i - lw] = acc;
                }
            }
        }
    }

    /// The convolution of channels-last `x` by `wk` plus `bias`, into `y`. Unit `(j, g)` holds
    /// column `j` of the cube's tile, every row of it, against output channels `g·per ..` of the
    /// cube's. For each `chunk` input channels the cube stages the tile's pixels with a border of
    /// one, and their weights; per channel and `kx` a unit reads its column's `rows + 2` pixels
    /// once for all three `ky`. Reads for the next chunk go to registers before the current
    /// chunk's sums, so they overlap.
    #[cube(launch_unchecked)]
    pub fn conv3x3<F: Float>(
        x: &Tensor<F>,
        wk: &Tensor<F>,
        bias: ComptimeOption<&Tensor<F>>,
        y: &mut Tensor<F>,
        h: u32,
        w: u32,
        a: u32,
        b: u32,
        across: u32,
        down: u32,
        tiles: u32,
        cubes: u32,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
        #[comptime] groups: usize,
        #[comptime] per: usize,
        #[comptime] chunk: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let (h, w, a, b) = (h as usize, w as usize, a as usize, b as usize);
        let units = comptime!(cols * groups);
        let outs = comptime!(per * groups);
        let span = comptime!(cols + 2);
        let plane = comptime!((rows + 2) * span);
        let xsize = comptime!(chunk * plane);
        let wsize = comptime!(chunk * 9 * outs);
        let xloads = comptime!(xsize.div_ceil(units));
        let wloads = comptime!(wsize.div_ceil(units));
        let mut xs = Shared::<[F]>::new_slice(xsize);
        let mut ws = Shared::<[F]>::new_slice(wsize);
        let mut acc = Array::<F>::new(comptime!(rows * per));
        #[unroll]
        for k in 0..comptime!(rows * per) {
            acc[k] = F::new(0.0_f32);
        }
        let mut xr = Array::<F>::new(xloads);
        let mut wr = Array::<F>::new(wloads);
        let mut xv = Array::<F>::new(comptime!(rows + 2));
        let mut wv = Array::<F>::new(per);
        let (j, k0) = (UNIT_POS_X as usize, UNIT_POS_Y as usize * per);
        let unit = UNIT_POS as usize;
        let (across, down) = (across as usize, down as usize);
        let tile = CUBE_POS % tiles as usize;
        let x0 = (tile % across) * cols;
        let y0 = (tile / across % down) * rows;
        let image = tile / across / down;
        let o0 = CUBE_POS / tiles as usize * outs;
        let chunks = a.div_ceil(chunk);
        // Step `t` stages the chunk read at step `t - 1`, reads chunk `t` and sums chunk `t - 1`.
        let mut t = 0;
        while t <= chunks {
            if t >= 1 {
                #[unroll]
                for i in 0..xloads {
                    let e = unit + i * units;
                    if e < xsize {
                        xs[(e % chunk) * plane + e / chunk] = xr[i];
                    }
                }
                #[unroll]
                for i in 0..wloads {
                    let e = unit + i * units;
                    if e < wsize {
                        ws[e] = wr[i];
                    }
                }
            }
            sync_cube();
            if t < chunks {
                let c0 = t * chunk;
                #[unroll]
                for i in 0..xloads {
                    let e = unit + i * units;
                    let (q, c) = (e / chunk, e % chunk);
                    // Pixel `(r, s)` of the bordered tile is the image's `(y0 + r - 1, x0 + s - 1)`.
                    let (r, s) = (q / span, q % span);
                    let mut v = F::new(0.0_f32);
                    if e < xsize
                        && y0 + r >= 1
                        && y0 + r <= h
                        && x0 + s >= 1
                        && x0 + s <= w
                        && c0 + c < a
                    {
                        v = x[((image * h + y0 + r - 1) * w + x0 + s - 1) * a + c0 + c];
                    }
                    xr[i] = v;
                }
                #[unroll]
                for i in 0..wloads {
                    let e = unit + i * units;
                    let (c, tap, o) = (e / comptime!(9 * outs), e / outs % 9, e % outs);
                    let mut v = F::new(0.0_f32);
                    if e < wsize && c0 + c < a && o0 + o < b {
                        v = wk[((c0 + c) * 9 + tap) * b + o0 + o];
                    }
                    wr[i] = v;
                }
            }
            if t >= 1 {
                let mut c = 0;
                while c < chunk {
                    #[unroll]
                    for kx in 0..3 {
                        #[unroll]
                        for r in 0..comptime!(rows + 2) {
                            xv[r] = xs[c * plane + r * span + j + kx];
                        }
                        #[unroll]
                        for ky in 0..3 {
                            #[unroll]
                            for k in 0..per {
                                wv[k] = ws[(c * 9 + ky * 3 + kx) * outs + k0 + k];
                            }
                            #[unroll]
                            for r in 0..rows {
                                #[unroll]
                                for k in 0..per {
                                    acc[r * per + k] += xv[r + ky] * wv[k];
                                }
                            }
                        }
                    }
                    c += 1;
                }
            }
            sync_cube();
            t += 1;
        }
        let col = x0 + j;
        if col < w {
            #[unroll]
            for r in 0..rows {
                let row = y0 + r;
                if row < h {
                    let at = ((image * h + row) * w + col) * b;
                    #[unroll]
                    for k in 0..per {
                        let o = o0 + k0 + k;
                        if o < b {
                            let v = acc[r * per + k];
                            #[comptime]
                            let v = match bias {
                                ComptimeOption::Some(bias) => v + bias[o],
                                ComptimeOption::None => v,
                            };
                            y[at + o] = v;
                        }
                    }
                }
            }
        }
    }

    /// `wk[i]`, the weight `[co, ci, 3, 3]` as `[ci, 9, co]`, or `flip`ped as `[co, 9, ci]` with
    /// tap `t` the weight's `8 - t`.
    #[cube(launch_unchecked)]
    pub fn arrange<F: Float>(
        weight: &Tensor<F>,
        wk: &mut Tensor<F>,
        ci: u32,
        co: u32,
        #[comptime] flip: bool,
        #[define(F)] _dtype: ElemType,
    ) {
        let (ci, co) = (ci as usize, co as usize);
        let i = ABSOLUTE_POS;
        if i < ci * co * 9 {
            if flip {
                let (c, t, o) = (i % ci, i / ci % 9, i / ci / 9);
                wk[i] = weight[(o * ci + c) * 9 + 8 - t];
            } else {
                let (o, t, c) = (i % co, i / co % 9, i / co / 9);
                wk[i] = weight[(o * ci + c) * 9 + t];
            }
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

    use crate::worst::worst_of;

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
            let y = Conv(conv2d.clone()).forward(view.clone());
            let off = Conv(conv2d.valid()).forward(view.clone().inner());
            assert_eq!(
                host(off),
                host(y.clone().inner()),
                "{c:?}: forward off autodiff"
            );
            y
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
        let err = worst_of(a.iter().zip(b).map(|(a, b)| (a - b).abs()));
        (err, scale)
    }

    /// A NaN gradient is off by NaN, which no tolerance passes.
    #[test]
    fn a_nan_is_the_worst_error() {
        let (err, scale) = max_error(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0]);
        assert!(err.is_nan(), "{err:e} of {scale:e}");
    }

    /// That `a` is within `tol` of `b`'s largest magnitude of `b`.
    fn close(a: &[f64], b: &[f64], tol: f64, what: impl std::fmt::Display) {
        let (err, scale) = max_error(a, b);
        assert!(err <= tol * scale, "{what} off by {err:e} of {scale:e}");
    }

    /// The f64 output, input gradient and bias gradient of `Σ y·g`, `y` the `k × k` same
    /// convolution of `inputs`: per kernel offset, the input shifted by it times that offset's
    /// weights transposed, plus the bias; `g` shifted the other way times the weights; and `g`
    /// summed over pixels. The input gradient is laid out as the input is.
    fn reference(c: Case, inputs: &[Tensor<4>; 4]) -> [Vec<f64>; 3] {
        let [x, weight, bias, g] = inputs.clone().map(|t| t.cast(DType::F64));
        let x = if c.strided {
            x.permute([0, 2, 3, 1])
        } else {
            x
        };
        let (p, pixels) = (c.k / 2, c.n * c.h * c.w);
        let db = host(g.clone().reshape([pixels, c.co]).sum_dim(0));
        let pad = |t: Tensor<4>| t.pad([(p, p), (p, p), (0, 0)], PadMode::Constant(0.0));
        let (x, g) = (pad(x), pad(g));
        let shifted = |t: &Tensor<4>, ky: usize, kx: usize| {
            let [.., channels] = t.dims();
            t.clone()
                .slice([0..c.n, ky..ky + c.h, kx..kx + c.w])
                .reshape([pixels, channels])
        };
        let tap = |ky: usize, kx: usize| {
            weight
                .clone()
                .slice([0..c.co, 0..c.ci, ky..ky + 1, kx..kx + 1])
                .reshape([c.co, c.ci])
        };
        let offsets = || (0..c.k).flat_map(|ky| (0..c.k).map(move |kx| (ky, kx)));
        let y = offsets()
            .map(|(ky, kx)| shifted(&x, ky, kx).matmul(tap(ky, kx).transpose()))
            .reduce(|a, b| a + b)
            .unwrap();
        let y = if c.bias {
            y + bias.reshape([1, c.co])
        } else {
            y
        };
        let k = c.k - 1;
        let dx = offsets()
            .map(|(ky, kx)| shifted(&g, k - ky, k - kx).matmul(tap(ky, kx)))
            .reduce(|a, b| a + b)
            .unwrap()
            .reshape([c.n, c.h, c.w, c.ci]);
        let dx = if c.strided {
            dx.permute([0, 3, 1, 2])
        } else {
            dx
        };
        [host(y), host(dx), db]
    }

    /// The largest error of an f32 output or gradient against the f64 one, relative to the f64
    /// one's largest magnitude.
    const F32: f64 = 2e-5;

    /// [`Conv::forward`] against [`reference`] and against burn's path: the output, input
    /// gradient and bias gradient to `1e-12` of their largest magnitude in f64 and to [`F32`] in
    /// f32, and the weight gradient to `1e-12` of burn's in f64 and to [`F32`] of burn's f64 one
    /// in f32.
    fn agrees(c: Case) {
        let device = crate::nn::cuda(0).autodiff();
        let inputs = inputs(c, &device);
        let [ry, rdx, rdb] = reference(c, &inputs);
        for dtype in [DType::F64, DType::F32] {
            // burn's autotune may pick its kernel on the first call of a shape; pick it first.
            run(c, &inputs, dtype, false);
            let [y, dx, dw, db] = run(c, &inputs, dtype, true);
            let [_, _, rdw, _] = run(c, &inputs, dtype, false);
            let tol = match dtype {
                DType::F64 => 1e-12,
                _ => F32,
            };
            close(&y, &ry, tol, format!("{c:?} {dtype:?}: output"));
            assert_eq!(dx.is_empty(), !c.x_grad);
            if c.x_grad {
                close(&dx, &rdx, tol, format!("{c:?} {dtype:?}: input gradient"));
            }
            assert_eq!(db.is_empty(), !c.bias);
            if c.bias {
                close(&db, &rdb, tol, format!("{c:?} {dtype:?}: bias gradient"));
            }
            let reference = match dtype {
                DType::F64 => rdw,
                _ => run(c, &inputs, DType::F64, false)[2].clone(),
            };
            close(
                &dw,
                &reference,
                tol,
                format!("{c:?} {dtype:?}: weight gradient"),
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

    /// `conv3x3`'s tiles of 8 × 32 pixels, its passes of 1 to 8 input channels, its cubes of 8 to
    /// 64 output channels: a tile exactly, two by two, one short of one and one past one or two;
    /// input channels one past a pass, and one or two past two blocks of output channels.
    #[test]
    fn tile_and_pass_edges() {
        agrees(case(1, 8, 32, 8, 8));
        agrees(case(2, 16, 64, 16, 16));
        agrees(case(1, 7, 31, 9, 17));
        agrees(case(1, 9, 33, 65, 33));
        agrees(case(1, 17, 65, 130, 129));
    }

    #[test]
    fn short_last_bands() {
        // 301 rows an image over 256 cubes: bands of two rows, the last of each image one.
        agrees(Case {
            x_grad: false,
            ..case(2, 301, 7, 4, 4)
        });
    }

    /// 65536 one-pixel images are as many bands and tiles, and 4194241 output channels 65536
    /// blocks of 64, each count past the 65535 cubes CUDA launches along a grid's Y or Z.
    #[test]
    fn more_than_65535_bands_tiles_or_channel_blocks() {
        f32_against_f64(case(65_536, 1, 1, 1, 1));
        f32_against_f64(Case {
            x_grad: false,
            ..case(1, 1, 1, 1, PER * WARPS * 65_535 + 1)
        });
    }

    /// The weight gradient over no pixels is zero, and over no channels empty; the bias gradient
    /// is the sum of `dy` over pixels whatever the input channels. burn's own convolution backward
    /// fails at these sizes, so `weight_grad` is called directly.
    #[test]
    fn weight_gradient_at_zero_sizes() {
        let device = crate::nn::cuda(0);
        let cube = |shape: [usize; 4]| {
            Tensor::<4>::ones(shape, (&device, DType::F32))
                .try_into_primitive::<CubeBackend>()
                .unwrap()
        };
        for [n, h, w, ci, co] in [
            [0, 5, 6, 2, 3],
            [2, 0, 6, 2, 3],
            [2, 5, 0, 2, 3],
            [2, 5, 6, 0, 3],
            [2, 5, 6, 2, 0],
        ] {
            let (dw, db) = weight_grad(cube([n, h, w, ci]), cube([n, h, w, co]), true);
            let dw = Tensor::<4>::from_primitive::<CubeBackend>(dw);
            let db = Tensor::<1>::from_primitive::<CubeBackend>(db.unwrap());
            assert_eq!(dw.dims(), [co, ci, 3, 3]);
            assert_eq!(host(dw), vec![0.0; co * ci * 9], "{:?}", [n, h, w, ci, co]);
            let pixels = (n * h * w) as f64;
            assert_eq!(host(db), vec![pixels; co], "{:?}", [n, h, w, ci, co]);
        }
    }

    /// A tensor of `shape` that is one element seen through zero strides, so a launch site's check
    /// meets it at its size without allocating it.
    fn broadcast<const D: usize>(shape: [usize; D]) -> CubeTensor {
        Tensor::<D>::ones([1; D], (&crate::nn::cuda(0), DType::F32))
            .expand(shape)
            .try_into_primitive::<CubeBackend>()
            .unwrap()
    }

    /// A tensor of `shape`, which holds no elements.
    fn empty<const D: usize>(shape: [usize; D]) -> CubeTensor {
        Tensor::<D>::empty(shape, (&crate::nn::cuda(0), DType::F32))
            .try_into_primitive::<CubeBackend>()
            .unwrap()
    }

    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn convolution_input_of_2_32_elements() {
        forward(
            broadcast([1, 1 << 15, 1 << 16, 2]),
            broadcast([1, 2, 3, 3]),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn convolution_output_of_2_32_elements() {
        forward(
            broadcast([1, 1 << 15, 1 << 16, 1]),
            broadcast([2, 1, 3, 3]),
            None,
        );
    }

    /// Over no pixels and no input channels only the bias of 2³² channels is over.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn convolution_bias_of_2_32_elements() {
        conv3x3(
            empty([0, 1, 1, 0]),
            empty([0, 9, 1 << 32]),
            Some(broadcast([1 << 32])),
        );
    }

    /// 2²⁶ input channels on 64 pixels: the output gradient, weight gradient and partials fit.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn weight_gradient_input_of_2_32_elements() {
        weight_grad(
            broadcast([1, 1, 64, 1 << 26]),
            broadcast([1, 1, 64, 1]),
            false,
        );
    }

    /// 2²⁶ output channels on 64 pixels: the input, weight and bias gradients and partials fit.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn weight_gradient_output_gradient_of_2_32_elements() {
        weight_grad(
            broadcast([1, 1, 64, 1]),
            broadcast([1, 1, 64, 1 << 26]),
            true,
        );
    }

    /// 2¹⁶ by 2¹⁶ channels over no pixels: only the weight gradient is over, and its count is
    /// checked before it is multiplied out.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn weight_gradient_of_more_than_2_32_elements() {
        weight_grad(empty([0, 1, 1, 1 << 16]), empty([0, 1, 1, 1 << 16]), false);
    }

    /// 2³² output channels from no input channels over no pixels: only the bias gradient is over.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn bias_gradient_of_2_32_elements() {
        weight_grad(empty([0, 1, 1, 0]), empty([0, 1, 1, 1 << 32]), true);
    }

    /// The kernel reads the output gradient at the input's pixels, so it refuses one over others.
    #[test]
    #[should_panic(expected = "an output gradient over the input's pixels")]
    fn weight_gradient_over_other_pixels() {
        weight_grad(broadcast([1, 1, 2, 1]), broadcast([1, 1, 1, 1]), false);
    }

    /// 9·2²⁸ weights over two bands of 32 columns: the input, output gradient and weight gradient
    /// fit, the partials do not.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn weight_gradient_partials_of_more_than_2_32_elements() {
        let side = [1, 1, 64, 1 << 14];
        weight_grad(broadcast(side), broadcast(side), false);
    }

    /// 2¹⁶ · 7282 · 9 weights, just past 2³².
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn arranged_weight_of_more_than_2_32_elements() {
        arrange(broadcast([1 << 16, 7282, 3, 3]), false);
    }

    /// f16 and bf16 take burn's path, which sums wider than the activation, so they match it bit
    /// for bit.
    #[test]
    fn half_precision_takes_burns_path() {
        let c = case(2, 33, 40, 8, 16);
        let device = crate::nn::cuda(0).autodiff();
        let inputs = inputs(c, &device);
        for dtype in [DType::F16, DType::BF16] {
            // burn's autotune may pick its kernel on the first call of a shape; pick it first.
            run(c, &inputs, dtype, false);
            let (fused, burn) = (run(c, &inputs, dtype, true), run(c, &inputs, dtype, false));
            for (name, (a, b)) in ["y", "dx", "dw", "db"].iter().zip(fused.iter().zip(&burn)) {
                assert!(a == b, "{dtype:?}: {name} differs");
            }
        }
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

    /// The f32 output and input, weight and bias gradients through [`Conv::forward`] against the
    /// f64 ones, each to [`F32`] of its largest magnitude.
    fn f32_against_f64(c: Case) {
        let device = crate::nn::cuda(0).autodiff();
        let inputs = inputs(c, &device);
        let [y, dx, dw, db] = run(c, &inputs, DType::F32, true);
        let [ry, rdx, rdb] = reference(c, &inputs);
        close(&y, &ry, F32, format!("{c:?}: f32 output"));
        if c.x_grad {
            close(&dx, &rdx, F32, format!("{c:?}: f32 input gradient"));
        }
        if c.bias {
            close(&db, &rdb, F32, format!("{c:?}: f32 bias gradient"));
        }
        let rdw = weight_gradient(c, inputs[0].clone(), inputs[3].clone());
        close(&dw, &rdw, F32, format!("{c:?}: f32 weight gradient"));
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
            f32_against_f64(case(32, side, side, ci, co));
        }
    }

    /// Every convolution a DIP runs at its defaults (width 16, depth 4, code 8) on a 320² image.
    #[test]
    fn dip_convolutions() {
        for [side, ci, co] in dip(8, 16, 4, 320) {
            f32_against_f64(case(1, side, side, ci, co));
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
            f32_against_f64(c);
        }
        let outc = Case {
            k: 1,
            ..case(1, 852, 852, 64, 2)
        };
        f32_against_f64(outc);
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
