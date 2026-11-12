//! The LLK ring: per-frame LLK partials of successive records, each stamped with its record's
//! sequence number, in a device ring the host downloads after the epochs that wrote it.
//! Record `k` lives in slot `k % nslot`; only `ring_advance` moves the cursor that numbers it.

use cuda_host::cuda_module;

use crate::pixel::LlkPartial;

/// Frame `f`'s partials of one record, at `slot * nframe + f`; `seq` is the cursor when it was
/// written, so a slot written for another record, or not written, shows a stamp other than
/// the record's.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingRecord {
    pub llk: LlkPartial,
    pub seq: u64,
}

// SAFETY: repr(C) of four f32 then a u64 at offset 16, no padding; every bit pattern is a
// valid value.
unsafe impl cuda_core::DeviceCopy for RingRecord {}

#[cuda_module]
pub mod ring_kernels {
    use crate::pixel::*;
    use crate::*;

    use super::RingRecord;

    /// Copies the stack's partials `partials[fl]`, `fl < nf`, to frame `f0 + fl` of slot
    /// `*cursor % nslot`, stamped `*cursor`. One thread per frame, `nf_max` threads.
    ///
    /// # Safety
    ///
    /// `partials` `nf_max` records, `ring` `nslot * nframe`, `cursor` and `ctl` one entry
    /// each; `nslot >= 1`.
    #[kernel]
    pub unsafe fn llk_push(
        partials: *const LlkPartial,
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
        unsafe {
            let seq = *cursor;
            let slot = (seq % nslot as u64) as usize;
            *ring.add(slot * g.nframe as usize + f0 as usize + fl) = RingRecord {
                llk: *partials.add(fl),
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
