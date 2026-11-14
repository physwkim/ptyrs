//! The exit waves DM and RAAR carry between cycles, in the formats the kernels read them
//! (DESIGN B5).

use cuda_core::{DeviceBuffer, RawDeviceBuffer};
use ptyrs_kernels::pixel::{PsiPtr, psi_f32};

use super::{C32, Error, dptr};

/// A stack's exit waves as the kernels read and write them through [`PsiPtr`].
pub trait Waves {
    /// Complex values held.
    fn values(&self) -> usize;
    /// The allocation a launch leases.
    fn raw(&self) -> &RawDeviceBuffer;
    /// The address and format the kernels take.
    fn ptr(&self) -> PsiPtr;
}

impl Waves for DeviceBuffer<C32> {
    fn values(&self) -> usize {
        self.len()
    }

    fn raw(&self) -> &RawDeviceBuffer {
        self.storage()
    }

    fn ptr(&self) -> PsiPtr {
        psi_f32(dptr(self))
    }
}

pub(super) fn check_waves(what: &str, waves: &dyn Waves, need: usize) -> Result<(), Error> {
    if waves.values() < need {
        return Err(Error::Shape(format!(
            "{what} holds {} elements, needs {need}",
            waves.values()
        )));
    }
    Ok(())
}
