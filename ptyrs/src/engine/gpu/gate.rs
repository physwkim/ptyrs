//! The gate of [`GpuPtycho`]'s calls.
//!
//! [`GpuPtycho`]: super::GpuPtycho

use super::device::Device;
use crate::engine::EngineError;

/// The [`Device`] behind the gate of [`GpuPtycho`]'s calls: aborted from the first failure
/// of a call through [`Self::change`] on. A call's checks, with what it builds before it
/// changes anything, run through the gate too, after its refusal, so an aborted engine
/// answers [`EngineError::Aborted`] whatever the arguments.
///
/// [`GpuPtycho`]: super::GpuPtycho
pub(super) struct Gated {
    dev: Device,
    aborted: bool,
}

impl Gated {
    /// `dev`, open.
    pub(super) fn new(dev: Device) -> Self {
        Self {
            dev,
            aborted: false,
        }
    }

    /// `f`, which may leave the device part-changed when it fails, on what `check` built:
    /// aborted while `f` runs, so its failure leaves the gate aborted.
    pub(super) fn change<T, R>(
        &mut self,
        check: impl FnOnce(&Device) -> Result<T, EngineError>,
        f: impl FnOnce(&mut Device, T) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let built = check(self.checked()?)?;
        self.aborted = true;
        let r = f(&mut self.dev, built)?;
        self.aborted = false;
        Ok(r)
    }

    /// `f`, which leaves the device's state as it found it, failed or not, after `check`: it
    /// never aborts.
    pub(super) fn read<R>(
        &mut self,
        check: impl FnOnce(&Device) -> Result<(), EngineError>,
        f: impl FnOnce(&mut Device) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        check(self.checked()?)?;
        f(&mut self.dev)
    }

    /// The device, refused once a call through [`Self::change`] has failed.
    pub(super) fn checked(&self) -> Result<&Device, EngineError> {
        if self.aborted {
            return Err(EngineError::Aborted);
        }
        Ok(&self.dev)
    }

    /// The device unchecked, for a read-back, which also runs once aborted.
    pub(super) fn unchecked(&self) -> &Device {
        &self.dev
    }

    /// The device past the gate, for a test that drives it directly.
    #[cfg(test)]
    pub(super) fn ungated(&mut self) -> &mut Device {
        &mut self.dev
    }
}
