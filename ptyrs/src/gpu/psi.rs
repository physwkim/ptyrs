//! The exit waves DM and RAAR carry between cycles, in the formats the kernels read them
//! (DESIGN B5).

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, RawDeviceBuffer};
use ptyrs_kernels::pixel::{PsiPtr, half_to_f32, psi_f32};

use super::{C32, Error, dptr};

/// A stack's exit waves as the kernels read and write them through [`PsiPtr`].
pub trait Waves {
    /// Complex values held.
    fn values(&self) -> usize;
    /// The allocation a launch leases.
    fn raw(&self) -> &RawDeviceBuffer;
    /// The address the kernels take, and the f16 scale.
    fn ptr(&self) -> PsiPtr;
    /// Whether the values are f16 pairs: which build of a wave kernel reads and writes them.
    fn half(&self) -> bool;
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

    fn half(&self) -> bool {
        false
    }
}

/// The scale of [`HalfWaves`] for frames whose counts sum to `max_frame_sum` at most: the
/// largest power of two `s` with `s sqrt(max_frame_sum) <= 2^15`, half the largest finite
/// f16, capped at `2^15` (DESIGN B5). A projected frame's waves hold its sum on the unmasked
/// pixels, so one component is within `sqrt` of it there; the factor of two is for DM's and
/// RAAR's combinations.
pub fn half_scale(max_frame_sum: u64) -> f32 {
    // `ceil(log2(sum))`, 0 for a sum of 0 or 1.
    let log2 = u64::BITS - (max_frame_sum.max(1) - 1).leading_zeros();
    // `s^2 sum <= 2^30` with `s = 2^k`.
    2f32.powi((30 - log2 as i32).div_euclid(2))
}

/// A stack's exit waves as one packed f16 pair per complex value, the real part in the low
/// half, each holding the value times a power of two.
pub struct HalfWaves {
    buf: DeviceBuffer<u32>,
    scale: f32,
}

impl HalfWaves {
    /// `len` zero values, to be stored times `scale`, a power of two.
    pub fn zeroed(stream: &Arc<CudaStream>, len: usize, scale: f32) -> Result<Self, Error> {
        if !scale.is_normal() || scale < 0.0 || scale.to_bits() & 0x007f_ffff != 0 {
            return Err(Error::Shape(format!(
                "f16 exit waves need a power-of-two scale, not {scale}"
            )));
        }
        Ok(Self {
            buf: DeviceBuffer::zeroed(stream, len)?,
            scale,
        })
    }

    /// The values held, unscaled.
    pub fn to_host_vec(&self, stream: &Arc<CudaStream>) -> Result<Vec<C32>, Error> {
        let unscale = 1.0 / self.scale;
        let packed = self.buf.to_host_vec(stream)?;
        Ok(packed
            .into_iter()
            .map(|p| C32 {
                re: half_to_f32(p & 0xffff) * unscale,
                im: half_to_f32(p >> 16) * unscale,
            })
            .collect())
    }
}

impl Waves for HalfWaves {
    fn values(&self) -> usize {
        self.buf.len()
    }

    fn raw(&self) -> &RawDeviceBuffer {
        self.buf.storage()
    }

    fn ptr(&self) -> PsiPtr {
        PsiPtr {
            ptr: dptr(&self.buf),
            scale: self.scale,
            unscale: 1.0 / self.scale,
        }
    }

    fn half(&self) -> bool {
        true
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every half against its value from the fields, in f64.
    #[test]
    fn half_to_f32_is_exact() {
        for h in 0..=u16::MAX as u32 {
            let (e, m) = ((h >> 10) & 0x1f, (h & 0x3ff) as f64);
            let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let got = half_to_f32(h);
            match e {
                0 => assert_eq!(got as f64, sign * m * 2f64.powi(-24), "{h:#06x}"),
                0x1f if m == 0.0 => assert_eq!(got as f64, sign * f64::INFINITY, "{h:#06x}"),
                0x1f => assert!(got.is_nan(), "{h:#06x}"),
                _ => {
                    let want = sign * (1024.0 + m) * 2f64.powi(e as i32 - 25);
                    assert_eq!(got as f64, want, "{h:#06x}");
                }
            }
            assert_eq!(got.is_sign_negative(), h & 0x8000 != 0, "{h:#06x}");
        }
    }

    /// `s` is the largest power of two with `s^2 sum <= 2^30`, on both sides of each step.
    #[test]
    fn half_scale_is_the_largest_power_of_two_in_range() {
        assert_eq!(half_scale(0), 32768.0);
        assert_eq!(half_scale(1), 32768.0);
        assert_eq!(half_scale(2), 16384.0);
        assert_eq!(half_scale(4), 16384.0);
        assert_eq!(half_scale(5), 8192.0);
        assert_eq!(half_scale(1 << 30), 1.0);
        assert_eq!(half_scale((1 << 30) + 1), 0.5);
        assert_eq!(half_scale(1 << 32), 0.5);
        assert_eq!(half_scale(u64::MAX), 2f32.powi(-17));
        for sum in [3u64, 1000, 65_535 * 4096, 1 << 40, (1 << 40) + 1] {
            let s = half_scale(sum) as f64;
            assert!(s * s * sum as f64 <= 2f64.powi(30), "{sum}");
            assert!(
                s == 32768.0 || 4.0 * s * s * sum as f64 > 2f64.powi(30),
                "{sum}"
            );
        }
    }
}
