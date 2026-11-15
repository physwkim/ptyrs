//! Batch normalisation followed by ReLU on channels-last activations, as one CUDA operation when
//! the normalisation trains. burn records a batch norm at batch statistics as one autodiff op but
//! runs its forward and backward as a dozen reductions, broadcasts and elementwise kernels, plus
//! two more for the ReLU; here the forward is three kernels and the backward two, each a pass over
//! the `[rows, channels]` activation.

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::FloatTensor;
use burn::backend::{Autodiff, AutodiffBackend, Backend, Dispatch, backend_extension};
use burn::nn::BatchNorm;
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::activation::relu;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim, InputScalar};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

use crate::nn::grid::{check_elements, grid};

/// `relu(n.forward(x))` for channels-last `x`, whose last axis is `n`'s channels. It runs on the
/// `[rows, channels]` view of `x`, which costs none of the transposing copy a channels-first view
/// does.
///
/// On an autodiff device with `n` training and `x` in f32 or f64, the batch statistics, the
/// running-statistics update and the gradients are burn's; elsewhere this is burn's own path. The
/// kernels sum in the activation's float type, which in f16 or bf16 loses the batch sums.
pub fn norm_relu<const D: usize>(n: &BatchNorm, x: Tensor<D>) -> Tensor<D> {
    let dims = x.dims();
    let cols = dims[D - 1];
    let x = x.reshape([dims[..D - 1].iter().product(), cols]);
    let device = x.device();
    let wide = matches!(x.dtype(), DType::F32 | DType::F64);
    let y = if device.is_autodiff() && n.training.is_enabled() && wide {
        // burn's batch_norm_train checks; the kernels read γ and β at every channel.
        assert_eq!(n.gamma.dims(), [cols], "invalid batch norm gamma shape");
        assert_eq!(n.beta.dims(), [cols], "invalid batch norm beta shape");
        let (y, mean, var) = <Dispatch as NormReluOps>::norm_relu(
            x.into_dispatch(),
            n.gamma.val().into_dispatch(),
            n.beta.val().into_dispatch(),
            n.epsilon,
        );
        let update = |state: &burn::module::RunningState<Tensor<1>>, stat| {
            let running = state
                .value_sync()
                .to_device(&device)
                .mul_scalar(1.0 - n.momentum)
                .add(
                    Tensor::<1>::from_dispatch(stat)
                        .detach()
                        .mul_scalar(n.momentum),
                );
            state.update(running.detach());
        };
        update(&n.running_mean, mean);
        update(&n.running_var, var);
        Tensor::<2>::from_dispatch(y)
    } else {
        relu(n.forward(x))
    };
    y.reshape(dims)
}

/// The fused training forward: the output and the batch mean and biased variance.
#[backend_extension(Cube, Autodiff)]
trait NormReluOps: Backend {
    fn norm_relu(
        x: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        epsilon: f64,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}

impl NormReluOps for CubeBackend {
    fn norm_relu(
        x: CubeTensor,
        gamma: CubeTensor,
        beta: CubeTensor,
        epsilon: f64,
    ) -> (CubeTensor, CubeTensor, CubeTensor) {
        let s = forward(x, gamma, beta, epsilon);
        (s.y, s.mean, s.var)
    }
}

/// What the backward reads: the input and γ as the forward packed them, the output (ReLU's mask)
/// and the statistics.
#[derive(Clone, Debug)]
struct Saved {
    x: CubeTensor,
    gamma: CubeTensor,
    y: CubeTensor,
    mean: CubeTensor,
    var: CubeTensor,
    epsilon: f64,
}

#[derive(Debug)]
struct NormReluBackward;

impl Backward<CubeBackend, 3> for NormReluBackward {
    type State = Saved;

    fn backward(self, ops: Ops<Saved, 3>, grads: &mut Gradients, _: &mut Checkpointer) {
        let [px, pg, pb] = ops.parents;
        let dy = grads.consume::<CubeBackend>(&ops.node);
        let s = ops.state;
        let [dx, dg, db] = backward(s.x, s.gamma, s.y, s.mean, s.var, s.epsilon, dy);
        for (parent, grad) in [(px, dx), (pg, dg), (pb, db)] {
            if let Some(node) = parent {
                grads.register::<CubeBackend>(node.id, grad);
            }
        }
    }
}

impl<C: CheckpointStrategy> NormReluOps for Autodiff<CubeBackend, C> {
    fn norm_relu(
        x: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        epsilon: f64,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let guards = [x.node(), gamma.node(), beta.node()];
        // `backward` finds x and γ as `forward` packed them, so it copies nothing.
        let s = forward(
            x.into_primitive(),
            gamma.into_primitive(),
            beta.into_primitive(),
            epsilon,
        );
        let (y, mean, var) = (s.y.clone(), s.mean.clone(), s.var.clone());
        let y = match NormReluBackward
            .prepare::<C>(guards)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(s, y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        };
        (y, Self::from_inner(mean), Self::from_inner(var))
    }
}

/// Threads per cube, `tx` channels by `UNITS / tx` rows.
const UNITS: usize = 256;

/// How the `[rows, cols]` activation splits over cubes: cube `g + groups·b` takes channels
/// `b·tx .. b·tx + tx` of rows `g·chunk .. (g + 1)·chunk`, and its partial sums are row `g` of a
/// `[groups, cols]` tensor. About 512 cubes fill the GPU once; each reduction over the partials
/// runs in a fixed order, so every cube that totals them gets the same value.
struct Grid {
    rows: usize,
    cols: usize,
    chunk: usize,
    groups: usize,
    cubes: usize,
    count: CubeCount,
    dim: CubeDim,
}

impl Grid {
    /// The grid over `x`, `[rows, cols]`, for launches that also read `inputs`, once
    /// [`check_elements`] passed every tensor they read or write: `x`, `inputs`, the `[rows, cols]`
    /// output and the `[cols]` statistics. None over no channels, where they have nothing to write.
    /// The `[groups, cols]` partials need no check of their own: `groups` is at most `rows`, or 1
    /// over no rows.
    fn new(x: &CubeTensor, inputs: &[&CubeTensor]) -> Option<Self> {
        let [rows, cols] = x.meta.shape().dims();
        let (out, stats) = ([rows, cols], [cols]);
        let shapes: Vec<&[usize]> = [x]
            .into_iter()
            .chain(inputs.iter().copied())
            .map(|t| &t.meta.shape()[..])
            .chain([&out[..], &stats[..]])
            .collect();
        check_elements(&shapes);
        if cols == 0 {
            return None;
        }
        let (tx, chunk, groups) = layout(rows, cols);
        let cubes = groups * cols.div_ceil(tx);
        Some(Self {
            rows,
            cols,
            chunk,
            groups,
            cubes,
            count: grid(&x.client, cubes),
            dim: CubeDim::new_2d(tx as u32, (UNITS / tx) as u32),
        })
    }

    /// `rows`, `cols`, `chunk`, `groups` and `cubes` as the kernels take them. The `[cols]`
    /// statistics bound `cols`, and over rows `x` bounds `rows`, `chunk` (at most `rows` or 2048)
    /// and `cubes` (at most `rows · cols`), or over none `cubes` is `cols`' blocks.
    fn sizes(&self) -> [u32; 5] {
        [self.rows, self.cols, self.chunk, self.groups, self.cubes]
            .map(|n| u32::try_from(n).expect("a norm_relu size below 2³²"))
    }
}

/// A [`Grid`] over `[rows, cols]`, `cols` at least 1: its cubes' width `tx` in channels, and its
/// `chunk` and `groups`.
fn layout(rows: usize, cols: usize) -> (usize, usize, usize) {
    let tx = cols.next_power_of_two().min(32);
    let blocks = cols.div_ceil(tx);
    let chunk = rows
        .div_ceil((512 / blocks.max(1)).max(1))
        .max(UNITS / tx * 8);
    // Group 0 writes each channel's statistics and its dγ and dβ, so it exists over no rows.
    let groups = rows.div_ceil(chunk).max(1);
    (tx, chunk, groups)
}

/// An uninitialised tensor of `shape` on `like`'s device, in its float type.
fn empty(like: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        shape.into(),
        like.dtype,
    )
}

fn forward(x: CubeTensor, gamma: CubeTensor, beta: CubeTensor, epsilon: f64) -> Saved {
    let Some(g) = Grid::new(&x, &[&gamma, &beta]) else {
        let rows = x.meta.shape()[0];
        let (y, mean, var) = (empty(&x, [rows, 0]), empty(&x, [0]), empty(&x, [0]));
        return Saved {
            x,
            gamma,
            y,
            mean,
            var,
            epsilon,
        };
    };
    let (x, gamma, beta) = (
        into_contiguous(x),
        into_contiguous(gamma),
        into_contiguous(beta),
    );
    let dtype = dtype_to_storage_type(x.dtype);
    let client = x.client.clone();
    let sums = empty(&x, [g.groups, g.cols]);
    let squares = empty(&x, [g.groups, g.cols]);
    let y = empty(&x, [g.rows, g.cols]);
    let mean = empty(&x, [g.cols]);
    let var = empty(&x, [g.cols]);
    let [rows, cols, chunk, groups, cubes] = g.sizes();
    // SAFETY: every index the kernels form is below its tensor's length: rows and channels are
    // masked against `rows` and `cols`, partial rows against `groups`.
    unsafe {
        kernel::sum_parts::launch_unchecked(
            &client,
            g.count.clone(),
            g.dim,
            x.clone().into_tensor_arg(),
            sums.clone().into_tensor_arg(),
            rows,
            cols,
            chunk,
            groups,
            cubes,
            UNITS,
            dtype,
        );
        kernel::square_parts::launch_unchecked(
            &client,
            g.count.clone(),
            g.dim,
            x.clone().into_tensor_arg(),
            sums.clone().into_tensor_arg(),
            squares.clone().into_tensor_arg(),
            rows,
            cols,
            chunk,
            groups,
            cubes,
            UNITS,
            dtype,
        );
        kernel::apply::launch_unchecked(
            &client,
            g.count,
            g.dim,
            x.clone().into_tensor_arg(),
            gamma.clone().into_tensor_arg(),
            beta.into_tensor_arg(),
            sums.into_tensor_arg(),
            squares.into_tensor_arg(),
            y.clone().into_tensor_arg(),
            mean.clone().into_tensor_arg(),
            var.clone().into_tensor_arg(),
            rows,
            cols,
            chunk,
            groups,
            cubes,
            InputScalar::new(epsilon, dtype),
            UNITS,
            dtype,
        );
    }
    Saved {
        x,
        gamma,
        y,
        mean,
        var,
        epsilon,
    }
}

fn backward(
    x: CubeTensor,
    gamma: CubeTensor,
    y: CubeTensor,
    mean: CubeTensor,
    var: CubeTensor,
    epsilon: f64,
    dy: CubeTensor,
) -> [CubeTensor; 3] {
    let Some(g) = Grid::new(&x, &[&gamma, &y, &mean, &var, &dy]) else {
        let rows = x.meta.shape()[0];
        return [empty(&x, [rows, 0]), empty(&x, [0]), empty(&x, [0])];
    };
    let (x, gamma, dy) = (
        into_contiguous(x),
        into_contiguous(gamma),
        into_contiguous(dy),
    );
    let dtype = dtype_to_storage_type(x.dtype);
    let client = x.client.clone();
    let shifts = empty(&x, [g.groups, g.cols]);
    let scales = empty(&x, [g.groups, g.cols]);
    let dx = empty(&x, [g.rows, g.cols]);
    let dgamma = empty(&x, [g.cols]);
    let dbeta = empty(&x, [g.cols]);
    let [rows, cols, chunk, groups, cubes] = g.sizes();
    let eps = InputScalar::new(epsilon, dtype);
    // SAFETY: as in `forward`; `y`, `mean` and `var` are the packed tensors it wrote.
    unsafe {
        kernel::grad_parts::launch_unchecked(
            &client,
            g.count.clone(),
            g.dim,
            x.clone().into_tensor_arg(),
            y.clone().into_tensor_arg(),
            dy.clone().into_tensor_arg(),
            mean.clone().into_tensor_arg(),
            var.clone().into_tensor_arg(),
            shifts.clone().into_tensor_arg(),
            scales.clone().into_tensor_arg(),
            rows,
            cols,
            chunk,
            groups,
            cubes,
            eps,
            UNITS,
            dtype,
        );
        kernel::grad_apply::launch_unchecked(
            &client,
            g.count,
            g.dim,
            x.into_tensor_arg(),
            gamma.into_tensor_arg(),
            y.into_tensor_arg(),
            dy.into_tensor_arg(),
            mean.into_tensor_arg(),
            var.into_tensor_arg(),
            shifts.into_tensor_arg(),
            scales.into_tensor_arg(),
            dx.clone().into_tensor_arg(),
            dgamma.clone().into_tensor_arg(),
            dbeta.clone().into_tensor_arg(),
            rows,
            cols,
            chunk,
            groups,
            cubes,
            eps,
            UNITS,
            dtype,
        );
    }
    [dx, dgamma, dbeta]
}

mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    /// This cube's group of rows.
    #[cube]
    fn group(groups: u32) -> usize {
        CUBE_POS % groups as usize
    }

    // Sizes are 32 bits, so an index past its bound may wrap below it: the helpers below add to
    // an index at most the distance to its bound.

    /// This unit's channel, or `cols` for a lane past the last one.
    #[cube]
    fn channel(cols: u32, groups: u32) -> usize {
        let cols = cols as usize;
        // The cube's first channel, below `cols`.
        let first = CUBE_POS / groups as usize * CUBE_DIM_X as usize;
        first + (UNIT_POS_X as usize).min(cols - first)
    }

    /// This cube's rows, `start..end`.
    #[cube]
    fn span(rows: u32, chunk: u32, groups: u32) -> (usize, usize) {
        let rows = rows as usize;
        // At most `rows`, which it equals only over no rows.
        let start = group(groups) * chunk as usize;
        (start, start + (chunk as usize).min(rows - start))
    }

    /// This unit's first row of `start..end`, or `end`.
    #[cube]
    fn first_row(start: usize, end: usize) -> usize {
        start + (UNIT_POS_Y as usize).min(end - start)
    }

    /// This unit's row after `r`, of `..end`, or `end`.
    #[cube]
    fn next_row(r: usize, end: usize) -> usize {
        r + (CUBE_DIM_Y as usize).min(end - r)
    }

    /// The sum of `v` over the cube's rows of units, given to every unit of the column.
    #[cube]
    fn sum_y<F: Float>(shared: &mut Shared<[F]>, v: F) -> F {
        let tx = UNIT_POS_X as usize;
        let at = UNIT_POS_Y as usize * CUBE_DIM_X as usize + tx;
        shared[at] = v;
        sync_cube();
        let mut half = CUBE_DIM_Y as usize / 2;
        while half > 0 {
            if (UNIT_POS_Y as usize) < half {
                let o = shared[at + half * CUBE_DIM_X as usize];
                shared[at] += o;
            }
            sync_cube();
            half /= 2;
        }
        let out = shared[tx];
        sync_cube();
        out
    }

    /// burn's ReLU: 0 where `z <= 0`, else `z`, so a NaN stays NaN.
    #[cube]
    fn relu<F: Float>(z: F) -> F {
        select(z <= F::new(0.0_f32), F::new(0.0_f32), z)
    }

    /// burn's ReLU backward: 0 where the output `y <= 0`, else `dy`, so a NaN output passes `dy`.
    #[cube]
    fn relu_grad<F: Float>(y: F, dy: F) -> F {
        select(y <= F::new(0.0_f32), F::new(0.0_f32), dy)
    }

    /// Channel `c`'s total of the `[groups, cols]` partials.
    #[cube]
    fn total<F: Float>(
        shared: &mut Shared<[F]>,
        parts: &Tensor<F>,
        c: usize,
        cols: u32,
        groups: u32,
    ) -> F {
        let mut acc = F::new(0.0_f32);
        if c < cols as usize {
            let mut g = first_row(0, groups as usize);
            while g < groups as usize {
                acc += parts[g * cols as usize + c];
                g = next_row(g, groups as usize);
            }
        }
        sum_y::<F>(shared, acc)
    }

    #[cube(launch_unchecked)]
    pub fn sum_parts<F: Float>(
        x: &Tensor<F>,
        sums: &mut Tensor<F>,
        rows: u32,
        cols: u32,
        chunk: u32,
        groups: u32,
        cubes: u32,
        #[comptime] units: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let mut shared = Shared::new_slice(units);
        let c = channel(cols, groups);
        let (start, end) = span(rows, chunk, groups);
        let mut acc = F::new(0.0_f32);
        if c < cols as usize {
            let mut r = first_row(start, end);
            while r < end {
                acc += x[r * cols as usize + c];
                r = next_row(r, end);
            }
        }
        let s = sum_y::<F>(&mut shared, acc);
        if c < cols as usize && UNIT_POS_Y == 0 {
            sums[group(groups) * cols as usize + c] = s;
        }
    }

    #[cube(launch_unchecked)]
    pub fn square_parts<F: Float>(
        x: &Tensor<F>,
        sums: &Tensor<F>,
        squares: &mut Tensor<F>,
        rows: u32,
        cols: u32,
        chunk: u32,
        groups: u32,
        cubes: u32,
        #[comptime] units: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let mut shared = Shared::new_slice(units);
        let c = channel(cols, groups);
        let (start, end) = span(rows, chunk, groups);
        let mean = total::<F>(&mut shared, sums, c, cols, groups) / F::cast_from(rows);
        let mut acc = F::new(0.0_f32);
        if c < cols as usize {
            let mut r = first_row(start, end);
            while r < end {
                let d = x[r * cols as usize + c] - mean;
                acc += d * d;
                r = next_row(r, end);
            }
        }
        let s = sum_y::<F>(&mut shared, acc);
        if c < cols as usize && UNIT_POS_Y == 0 {
            squares[group(groups) * cols as usize + c] = s;
        }
    }

    #[cube(launch_unchecked)]
    pub fn apply<F: Float>(
        x: &Tensor<F>,
        gamma: &Tensor<F>,
        beta: &Tensor<F>,
        sums: &Tensor<F>,
        squares: &Tensor<F>,
        y: &mut Tensor<F>,
        mean_out: &mut Tensor<F>,
        var_out: &mut Tensor<F>,
        rows: u32,
        cols: u32,
        chunk: u32,
        groups: u32,
        cubes: u32,
        eps: InputScalar,
        #[comptime] units: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let mut shared = Shared::new_slice(units);
        let c = channel(cols, groups);
        let (start, end) = span(rows, chunk, groups);
        let n = F::cast_from(rows);
        let mean = total::<F>(&mut shared, sums, c, cols, groups) / n;
        let var = total::<F>(&mut shared, squares, c, cols, groups) / n;
        if c < cols as usize {
            // burn's order: ((x - mean) / std) * gamma + beta.
            let std = F::sqrt(var + eps.get::<F>());
            let (g, b) = (gamma[c], beta[c]);
            let mut r = first_row(start, end);
            while r < end {
                let i = r * cols as usize + c;
                let z = (x[i] - mean) / std * g + b;
                y[i] = relu::<F>(z);
                r = next_row(r, end);
            }
            if group(groups) == 0 && UNIT_POS_Y == 0 {
                mean_out[c] = mean;
                var_out[c] = var;
            }
        }
    }

    /// Partials of `Σ dz` and `Σ dz·x̂`, where `dz` is the gradient through the ReLU.
    #[cube(launch_unchecked)]
    pub fn grad_parts<F: Float>(
        x: &Tensor<F>,
        y: &Tensor<F>,
        dy: &Tensor<F>,
        mean: &Tensor<F>,
        var: &Tensor<F>,
        shifts: &mut Tensor<F>,
        scales: &mut Tensor<F>,
        rows: u32,
        cols: u32,
        chunk: u32,
        groups: u32,
        cubes: u32,
        eps: InputScalar,
        #[comptime] units: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let mut shared = Shared::new_slice(units);
        let c = channel(cols, groups);
        let (start, end) = span(rows, chunk, groups);
        let mut db = F::new(0.0_f32);
        let mut dg = F::new(0.0_f32);
        if c < cols as usize {
            let m = mean[c];
            let rstd = F::new(1.0_f32) / F::sqrt(var[c] + eps.get::<F>());
            let mut r = first_row(start, end);
            while r < end {
                let i = r * cols as usize + c;
                let dz = relu_grad::<F>(y[i], dy[i]);
                db += dz;
                dg += dz * ((x[i] - m) * rstd);
                r = next_row(r, end);
            }
        }
        let db = sum_y::<F>(&mut shared, db);
        let dg = sum_y::<F>(&mut shared, dg);
        if c < cols as usize && UNIT_POS_Y == 0 {
            let at = group(groups) * cols as usize + c;
            shifts[at] = db;
            scales[at] = dg;
        }
    }

    /// `dx = γ·rstd·(dz - mean(dz) - x̂·mean(dz·x̂))`, with `dγ` and `dβ` from the first row of
    /// cubes.
    #[cube(launch_unchecked)]
    pub fn grad_apply<F: Float>(
        x: &Tensor<F>,
        gamma: &Tensor<F>,
        y: &Tensor<F>,
        dy: &Tensor<F>,
        mean: &Tensor<F>,
        var: &Tensor<F>,
        shifts: &Tensor<F>,
        scales: &Tensor<F>,
        dx: &mut Tensor<F>,
        dgamma: &mut Tensor<F>,
        dbeta: &mut Tensor<F>,
        rows: u32,
        cols: u32,
        chunk: u32,
        groups: u32,
        cubes: u32,
        eps: InputScalar,
        #[comptime] units: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        if CUBE_POS >= cubes as usize {
            terminate!();
        }
        let mut shared = Shared::new_slice(units);
        let c = channel(cols, groups);
        let (start, end) = span(rows, chunk, groups);
        let db = total::<F>(&mut shared, shifts, c, cols, groups);
        let dg = total::<F>(&mut shared, scales, c, cols, groups);
        if c < cols as usize {
            let n = F::cast_from(rows);
            let m = mean[c];
            let rstd = F::new(1.0_f32) / F::sqrt(var[c] + eps.get::<F>());
            let (mean_db, mean_dg) = (db / n, dg / n);
            let scale = gamma[c] * rstd;
            let mut r = first_row(start, end);
            while r < end {
                let i = r * cols as usize + c;
                let dz = relu_grad::<F>(y[i], dy[i]);
                let xhat = (x[i] - m) * rstd;
                dx[i] = scale * (dz - mean_db - xhat * mean_dg);
                r = next_row(r, end);
            }
            if group(groups) == 0 && UNIT_POS_Y == 0 {
                dgamma[c] = dg;
                dbeta[c] = db;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{Param, RunningState};
    use burn::nn::BatchNormConfig;
    use burn::tensor::{DType, Distribution, TensorData};

    use crate::worst::worst_of;

    fn values(n: usize, lo: f64, hi: f64, device: &Device, dtype: DType) -> Tensor<1> {
        Tensor::<1>::random([n], Distribution::Uniform(lo, hi), device)
            .cast(dtype)
            .detach()
    }

    fn host(t: Tensor<1>) -> Vec<f64> {
        t.into_data()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .unwrap()
    }

    /// A training batch norm with the given parameters and running state, over as many channels.
    fn batch_norm(p: &[Tensor<1>; 4]) -> BatchNorm {
        // burn's config takes no zero channels; every tensor it makes is replaced.
        let mut n = BatchNormConfig::new(1).init(&p[0].device());
        n.gamma = Param::from_tensor(p[0].clone());
        n.beta = Param::from_tensor(p[1].clone());
        n.running_mean = RunningState::new(p[2].clone());
        n.running_var = RunningState::new(p[3].clone());
        n
    }

    /// Batch-norm parameters and running state over `cols` channels, an input `x` and the weights
    /// `w` of the loss `Σ y·w`, seeded by the sizes. A `transposed` input is a strided view rather
    /// than packed rows.
    fn inputs(
        rows: usize,
        cols: usize,
        dtype: DType,
        transposed: bool,
    ) -> ([Tensor<1>; 4], Tensor<2>, Tensor<2>) {
        let device = crate::nn::cuda(0).autodiff();
        device.seed(rows as u64 * 31 + cols as u64);
        let p = [
            values(cols, 0.5, 1.5, &device, dtype),
            values(cols, -0.5, 0.5, &device, dtype),
            values(cols, -1.0, 1.0, &device, dtype),
            values(cols, 0.5, 2.0, &device, dtype),
        ];
        let normal = Distribution::Normal(0.3, 2.0);
        let x = if transposed {
            Tensor::<2>::random([cols, rows], normal, &device).transpose()
        } else {
            Tensor::<2>::random([rows, cols], normal, &device)
        };
        let x = x.cast(dtype).detach();
        let w = Tensor::<2>::random([rows, cols], Distribution::Uniform(-1.0, 1.0), &device)
            .cast(dtype)
            .detach();
        (p, x, w)
    }

    /// The fused path against `relu(n.forward(x))`: output, running statistics and the gradients
    /// of `Σ y·w` with respect to `x`, γ and β, each NaN where burn's is and otherwise within
    /// `tol` of its largest magnitude.
    fn compare(p: &[Tensor<1>; 4], x: &Tensor<2>, w: &Tensor<2>, tol: f64) {
        let [rows, cols] = x.dims();
        let run = |fused: bool| {
            let n = batch_norm(p);
            let x = x.clone().require_grad();
            let y = if fused {
                norm_relu(&n, x.clone())
            } else {
                relu(n.forward(x.clone()))
            };
            let grads = (y.clone() * w.clone()).sum().backward();
            let flat = |t: Tensor<2>| t.reshape([rows * cols]);
            // `value` would give the running state from before this thread's update.
            [
                host(flat(y.inner())),
                host(n.running_mean.value_sync()),
                host(n.running_var.value_sync()),
                host(flat(x.grad(&grads).unwrap())),
                host(n.gamma.val().grad(&grads).unwrap()),
                host(n.beta.val().grad(&grads).unwrap()),
            ]
        };
        let (fused, reference) = (run(true), run(false));
        let names = ["y", "running mean", "running var", "dx", "dgamma", "dbeta"];
        for ((name, a), b) in names.iter().zip(&fused).zip(&reference) {
            let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let err = worst_of(
                a.iter()
                    .zip(b)
                    .map(|(a, b)| match a.is_nan() && b.is_nan() {
                        true => 0.0,
                        false => (a - b).abs(),
                    }),
            );
            assert!(
                err <= tol * scale,
                "{name} ({rows}×{cols}, {:?}): max error {err:e} of {scale:e}",
                x.dtype()
            );
        }
    }

    fn agrees(rows: usize, cols: usize, dtype: DType, tol: f64, transposed: bool) {
        let (p, x, w) = inputs(rows, cols, dtype, transposed);
        compare(&p, &x, &w, tol);
    }

    #[test]
    fn fused_matches_burn_in_f64() {
        // 24 channels leave masked lanes in a 32-wide cube; 40 take two channel blocks.
        for (rows, cols) in [(1013, 24), (1013, 40), (37, 3), (70_001, 8)] {
            agrees(rows, cols, DType::F64, 1e-12, false);
        }
    }

    #[test]
    fn fused_matches_burn_in_f32() {
        for (rows, cols) in [(1013, 24), (1013, 40), (37, 3), (70_001, 8)] {
            agrees(rows, cols, DType::F32, 2e-5, false);
        }
    }

    /// 2097121 channels are 65536 blocks of 32, past the 65535 cubes CUDA launches along a grid's
    /// Y.
    #[test]
    fn fused_matches_burn_past_65535_channel_blocks() {
        agrees(2, 32 * 65_535 + 1, DType::F32, 2e-5, false);
    }

    fn cube<const D: usize>(shape: [usize; D], device: &Device) -> CubeTensor {
        Tensor::<D>::ones(shape, (device, DType::F64))
            .try_into_primitive::<CubeBackend>()
            .unwrap()
    }

    fn read<const D: usize>(t: CubeTensor) -> Vec<f64> {
        let t = Tensor::<D>::from_primitive::<CubeBackend>(t);
        t.into_data()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .unwrap()
    }

    /// The kernels at zero sizes: over no rows each channel's mean and variance are 0/0, NaN as
    /// burn's, and its dγ and dβ zero; over no channels, even of 2³² rows, every output is empty
    /// and nothing launches. burn's own elementwise kernels fail on an empty matrix, so the
    /// kernels are called directly.
    #[test]
    fn kernels_at_zero_sizes() {
        let device = crate::nn::cuda(0);
        for (rows, cols) in [(0, 8), (8, 0), (0, 0), (1 << 32, 0)] {
            let x = cube([rows, cols], &device);
            let gamma = cube([cols], &device);
            let s = forward(x, gamma, cube([cols], &device), 1e-5);
            let (y, mean, var) = (s.y.clone(), s.mean.clone(), s.var.clone());
            let dy = cube([rows, cols], &device);
            let [dx, dgamma, dbeta] = backward(s.x, s.gamma, s.y, s.mean, s.var, 1e-5, dy);
            let at = format!("{rows}×{cols}");
            let nan = |v: Vec<f64>| v.len() == cols && v.iter().all(|v| v.is_nan());
            assert_eq!(read::<2>(y).len(), rows * cols, "{at}: y");
            assert_eq!(read::<2>(dx).len(), rows * cols, "{at}: dx");
            assert!(nan(read::<1>(mean)), "{at}: mean");
            assert!(nan(read::<1>(var)), "{at}: var");
            assert_eq!(read::<1>(dgamma), vec![0.0; cols], "{at}: dgamma");
            assert_eq!(read::<1>(dbeta), vec![0.0; cols], "{at}: dbeta");
        }
    }

    /// A tensor of `shape` that is one element seen through zero strides, so the launch check
    /// meets it at its size without allocating it.
    fn broadcast<const D: usize>(shape: [usize; D]) -> CubeTensor {
        Tensor::<D>::ones([1; D], (&crate::nn::cuda(0), DType::F32))
            .expand(shape)
            .try_into_primitive::<CubeBackend>()
            .unwrap()
    }

    /// An activation of 2³² elements, refused before it is packed; γ and β fit.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn an_activation_of_2_32_elements() {
        let side = 1 << 16;
        forward(
            broadcast([side, side]),
            broadcast([side]),
            broadcast([side]),
            1e-5,
        );
    }

    /// The backward checks its tensors too, before it packs `x`, γ or `dy`.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn an_activation_gradient_of_2_32_elements() {
        let (side, a) = (1 << 16, broadcast([1 << 16, 1 << 16]));
        let c = || broadcast([side]);
        backward(a.clone(), c(), a.clone(), c(), c(), 1e-5, a);
    }

    /// 2³² channels over no rows: only γ, β and the statistics are over.
    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn statistics_of_2_32_channels() {
        let x = Tensor::<2>::empty([0, 1usize << 32], (&crate::nn::cuda(0), DType::F32))
            .try_into_primitive::<CubeBackend>()
            .unwrap();
        forward(x, broadcast([1 << 32]), broadcast([1 << 32]), 1e-5);
    }

    /// The kernels' `span` of group `g`, in 32 bits with every operation checked.
    fn span(rows: u32, chunk: u32, g: u32) -> (u32, u32) {
        let start = g.checked_mul(chunk).unwrap();
        let end = start.checked_add(chunk.min(rows.checked_sub(start).unwrap()));
        (start, end.unwrap())
    }

    /// The kernels' `next_row`, and their `first_row` for unit `step`, checked as [`span`] is.
    fn next_row(r: u32, end: u32, step: u32) -> u32 {
        r.checked_add(step.min(end.checked_sub(r).unwrap()))
            .unwrap()
    }

    /// One channel of 2³² - 511, 2³² - 255 and 2³² - 1 rows, where the last group's `start + chunk`
    /// and, from 2³² - 255, a unit's step past its last row wrapped past 2³². The kernels' row
    /// arithmetic is run here on the host: no f16 tensor of that many elements fits the 8190 MiB a
    /// buffer may take on the GPUs here. Each group's span follows the last, and the last group's
    /// units visit its rows once each and stop at its end.
    #[test]
    fn rows_near_2_32_stay_in_32_bits() {
        for rows in [u32::MAX - 510, u32::MAX - 254, u32::MAX] {
            let (tx, chunk, groups) = layout(rows as usize, 1);
            let ty = (UNITS / tx) as u32;
            let [chunk, groups] = [chunk, groups].map(|v| u32::try_from(v).unwrap());
            let mut next = 0;
            for g in 0..groups {
                let (start, end) = span(rows, chunk, g);
                assert_eq!(start, next);
                next = end;
            }
            assert_eq!(next, rows);
            let (start, end) = span(rows, chunk, groups - 1);
            assert_eq!(start.checked_add(chunk), None);
            let (mut seen, mut wrapped) = (0, false);
            for y in 0..ty {
                let mut r = next_row(start, end, y);
                while r < end {
                    seen += 1;
                    wrapped |= r.checked_add(ty).is_none();
                    r = next_row(r, end, ty);
                }
                assert_eq!(r, end);
            }
            assert_eq!(seen, end - start);
            assert_eq!(wrapped, rows > u32::MAX - 256);
        }
    }

    /// Through `norm_relu`, an activation of no channels stays empty, and over no rows the running
    /// statistics take the NaN batch statistics.
    #[test]
    fn norm_relu_at_zero_sizes() {
        let device = crate::nn::cuda(0).autodiff();
        for (rows, cols) in [(4, 0), (0, 8)] {
            let ones = || Tensor::<1>::ones([cols], (&device, DType::F64));
            let n = batch_norm(&[ones(), ones(), ones(), ones()]);
            let y = norm_relu(&n, Tensor::<2>::ones([rows, cols], (&device, DType::F64)));
            assert_eq!(y.dims(), [rows, cols]);
            let mean = host(n.running_mean.value_sync());
            assert!(
                mean.len() == cols && mean.iter().all(|v| v.is_nan()),
                "{mean:?}"
            );
        }
    }

    /// f16 and bf16 take burn's path, which sums wider than the activation, so they match it bit
    /// for bit.
    #[test]
    fn half_precision_takes_burns_path() {
        for dtype in [DType::F16, DType::BF16] {
            let (p, x, w) = inputs(1013, 24, dtype, false);
            // The first call may autotune burn's kernels; the second runs those it picked.
            compare(&p, &x, &w, 1e-2);
            compare(&p, &x, &w, 0.0);
        }
    }

    #[test]
    fn fused_matches_burn_on_a_strided_input() {
        agrees(1013, 8, DType::F64, 1e-12, true);
    }

    /// A NaN activation stays NaN and passes its gradient, as through burn's ReLU: a NaN in β
    /// makes one channel's activations NaN on finite statistics, one in `x` its channel's
    /// statistics too.
    #[test]
    fn nan_activations_pass_as_through_burn() {
        for (dtype, tol) in [(DType::F64, 1e-12), (DType::F32, 2e-5)] {
            let (mut p, x, w) = inputs(1013, 24, dtype, false);
            let device = x.device();
            let nan = Tensor::<1>::full([1], f64::NAN, &device).cast(dtype);
            p[1] = p[1].clone().slice_assign(1..2, nan.clone());
            let x = x.slice_assign([5..6, 3..4], nan.reshape([1, 1]));
            compare(&p, &x, &w, tol);
        }
    }

    /// A γ or β of another width than `x`'s channels is refused as burn refuses it, before the
    /// kernels read past it and the running statistics take `x`'s width.
    #[test]
    #[should_panic(expected = "invalid batch norm gamma shape")]
    fn refuses_a_gamma_of_another_width() {
        let device = crate::nn::cuda(0).autodiff();
        let n = BatchNormConfig::new(1).init(&device);
        norm_relu(&n, Tensor::<2>::ones([8, 16], &device));
    }

    #[test]
    #[should_panic(expected = "invalid batch norm beta shape")]
    fn refuses_a_beta_of_another_width() {
        let device = crate::nn::cuda(0).autodiff();
        let mut n = BatchNormConfig::new(16).init(&device);
        n.beta = Param::from_tensor(Tensor::<1>::zeros([1], &device));
        norm_relu(&n, Tensor::<2>::ones([8, 16], &device));
    }

    #[test]
    fn inference_is_burns_path() {
        let device = crate::nn::cuda(0);
        let dtype = DType::F32;
        let p = [
            values(5, 0.5, 1.5, &device, dtype),
            values(5, -0.5, 0.5, &device, dtype),
            values(5, -1.0, 1.0, &device, dtype),
            values(5, 0.5, 2.0, &device, dtype),
        ];
        let n = batch_norm(&p);
        let x = Tensor::<4>::random([2, 3, 4, 5], Distribution::Normal(0.0, 1.0), &device);
        let a = norm_relu(&n, x.clone());
        let b = relu(n.forward(x.reshape([24, 5]))).reshape([2, 3, 4, 5]);
        let (a, b): (TensorData, TensorData) = (a.into_data(), b.into_data());
        assert_eq!(a, b);
    }
}
