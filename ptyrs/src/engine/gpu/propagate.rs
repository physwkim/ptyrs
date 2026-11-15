//! The propagator of an [`ExitStage`] on whole planes, for the passes that transform a stack
//! through its cuFFT plan: BH's, rPIE's and unfused ML's.

use crate::fft::Plan2d;
use crate::gpu::{
    BandedTransfer, BhKernels, C32, Error as GpuError, ExitStage, StackRef, ViewDims,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::DeviceBuffer;

/// The propagator `F V` of an [`ExitStage`] on a stack's whole planes, through the stack's
/// plan and at the far field's norm, `√(ny nx)` times a unitary.
#[derive(Clone, Copy)]
pub(super) struct Propagate<'a> {
    pub kernels: &'a BhKernels,
    pub stage: &'a ExitStage,
    pub dims: ViewDims,
}

impl Propagate<'_> {
    /// `buf` to the detector plane.
    pub fn forward(
        &self,
        ctx: &ExecutionContext,
        plan: &mut Plan2d,
        stack: StackRef<'_>,
        buf: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        self.run(ctx, plan, stack, buf, false)
    }

    /// `src` to the detector plane into `dst`, as [`Self::forward`] of a copy.
    pub fn forward_from(
        &self,
        ctx: &ExecutionContext,
        plan: &mut Plan2d,
        stack: StackRef<'_>,
        src: &DeviceBuffer<C32>,
        dst: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        match self.stage {
            ExitStage::None => plan.forward(ctx, src, dst)?,
            ExitStage::Reverse => plan.inverse(ctx, src, dst)?,
            ExitStage::Transfer(tf) | ExitStage::Banded(BandedTransfer { tf, .. }) => {
                plan.forward(ctx, src, dst)?;
                self.kernels
                    .transfer(ctx, &self.dims, stack, tf, dst, false)?;
                plan.inverse_inplace(ctx, dst)?;
            }
        }
        Ok(())
    }

    /// The adjoint of [`Self::forward`].
    pub fn adjoint(
        &self,
        ctx: &ExecutionContext,
        plan: &mut Plan2d,
        stack: StackRef<'_>,
        buf: &mut DeviceBuffer<C32>,
    ) -> Result<(), GpuError> {
        self.run(ctx, plan, stack, buf, true)
    }

    fn run(
        &self,
        ctx: &ExecutionContext,
        plan: &mut Plan2d,
        stack: StackRef<'_>,
        buf: &mut DeviceBuffer<C32>,
        adjoint: bool,
    ) -> Result<(), GpuError> {
        match self.stage {
            // The reversal after a forward transform is the unnormalized inverse.
            ExitStage::None if !adjoint => plan.forward_inplace(ctx, buf)?,
            ExitStage::Reverse if adjoint => plan.forward_inplace(ctx, buf)?,
            ExitStage::None | ExitStage::Reverse => plan.inverse_inplace(ctx, buf)?,
            ExitStage::Transfer(tf) | ExitStage::Banded(BandedTransfer { tf, .. }) => {
                plan.forward_inplace(ctx, buf)?;
                self.kernels
                    .transfer(ctx, &self.dims, stack, tf, buf, adjoint)?;
                plan.inverse_inplace(ctx, buf)?;
            }
        }
        Ok(())
    }
}
