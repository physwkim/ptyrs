//! The LLK ring: per-frame LLK partials of successive records, each stamped with its record's
//! sequence number, in a device ring the host downloads after the epochs that wrote it.
//! Record `k` lives in slot `k % nslot`; only `ring_advance` moves the cursor that numbers it.
//! Each frame's unmasked pixels, counted once by `ring_count`, ride along with its partials;
//! the same count adds the frame to each of its unmasked pixels' frame counts.

use cuda_host::cuda_module;

use crate::pixel::LlkPartial;

/// The count of a frame `ring_count` has not counted.
pub const UNCOUNTED: u32 = u32::MAX;

/// Frame `f`'s partials of one record, at `slot * nframe + f`; `seq` is the cursor when it was
/// written, so a slot written for another record, or not written, shows a stamp other than
/// the record's.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingRecord {
    pub llk: LlkPartial,
    /// The frame's unmasked pixels when the partials were pushed, or [`UNCOUNTED`].
    pub nobs: u32,
    /// Zero: `seq`'s alignment, held by a field so the record has no padding bytes.
    pub pad: u32,
    pub seq: u64,
}

// SAFETY: repr(C) of four f32, two u32, then a u64 at offset 24, no padding; every bit
// pattern is a valid value.
unsafe impl cuda_core::DeviceCopy for RingRecord {}

#[cuda_module]
pub mod ring_kernels {
    use crate::pixel::*;
    use crate::*;
    use cuda_device::SharedArray;

    use super::{RingRecord, UNCOUNTED};

    /// For each frame `f = f0 + fl`, `fl < nf`, not counted yet (`nobs[f] == UNCOUNTED`):
    /// `nobs[f]` = its unmasked pixels in `counts`, and `pixel_frames[p] += 1` for each of
    /// them. A frame already counted is left as it is, so each frame is in `pixel_frames` at
    /// most once. One block of `PIXEL_THREADS` per frame, `nf_max` blocks.
    ///
    /// # Safety
    ///
    /// `counts` `nframe * ny * nx`, `nobs` `nframe`, `pixel_frames` `ny * nx`, `ctl` one
    /// entry.
    #[kernel]
    pub unsafe fn ring_count(
        counts: *const u16,
        nobs: *mut u32,
        pixel_frames: *mut u32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
        static mut RED: SharedArray<u32, 256> = SharedArray::UNINIT;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let fl = thread::blockIdx_x() as usize;
        let f = f0 as usize + fl;
        // The whole block leaves together, before its first barrier; thread 0 writes
        // `nobs[f]` only after every thread has passed one.
        if fl >= nf as usize || unsafe { *nobs.add(f) } != UNCOUNTED {
            return;
        }
        let n = g.ny as usize * g.nx as usize;
        let t = thread::threadIdx_x() as usize;
        let mut acc = 0u32;
        let mut p = t;
        while p < n {
            if unsafe { *counts.add(f * n + p) } != MASKED {
                acc += 1;
                unsafe { DeviceAtomicU32::from_ptr(pixel_frames.add(p)) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
            p += PIXEL_THREADS as usize;
        }
        let s = (&raw mut RED).cast::<u32>();
        unsafe { *s.add(t) = acc };
        let mut half = PIXEL_THREADS as usize / 2;
        while half > 0 {
            thread::sync_threads();
            if t < half {
                unsafe { *s.add(t) += *s.add(t + half) };
            }
            half /= 2;
        }
        if t == 0 {
            unsafe { *nobs.add(f) = *s };
        }
    }

    /// Copies the stack's partials `partials[fl]`, `fl < nf`, with the frame's `nobs`, to
    /// frame `f0 + fl` of slot `*cursor % nslot`, stamped `*cursor`. One thread per frame,
    /// `nf_max` threads.
    ///
    /// # Safety
    ///
    /// `partials` `nf_max` records, `nobs` `nframe`, `ring` `nslot * nframe`, `cursor` and
    /// `ctl` one entry each; `nslot >= 1`.
    #[kernel]
    pub unsafe fn llk_push(
        partials: *const LlkPartial,
        nobs: *const u32,
        ring: *mut RingRecord,
        cursor: *const u64,
        ctl: *const StackCtl,
        g: StackGeom,
        nslot: u32,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let fl = global_index();
        if fl >= nf as usize {
            return;
        }
        let f = f0 as usize + fl;
        unsafe {
            let seq = *cursor;
            let slot = (seq % nslot as u64) as usize;
            *ring.add(slot * g.nframe as usize + f) = RingRecord {
                llk: *partials.add(fl),
                nobs: *nobs.add(f),
                pad: 0,
                seq,
            };
        }
    }

    /// `*cursor += 1` (wrapping): ends a record. One thread.
    ///
    /// # Safety
    ///
    /// `cursor` one entry.
    #[kernel]
    pub unsafe fn ring_advance(cursor: *mut u64) {
        if global_index() == 0 {
            unsafe { *cursor = (*cursor).wrapping_add(1) };
        }
    }
}
