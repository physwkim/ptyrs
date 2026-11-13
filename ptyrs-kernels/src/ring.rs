//! The LLK ring: per-frame LLK partials of successive records, each stamped with its record's
//! sequence number, in a device ring the host downloads after the epochs that wrote it.
//! Record `k` lives in slot `k % nslot`; only `ring_advance` moves the cursor that numbers it.
//! Each frame's unmasked pixels, counted once by `ring_count`, ride along with its partials;
//! the same count adds the frame to each of its unmasked pixels' frame counts.

use cuda_host::cuda_module;

use crate::pixel::{LlkPartial, PIXEL_THREADS};

/// The count of a frame `ring_count` has not counted.
pub const UNCOUNTED: u32 = u32::MAX;

/// Frames per block of `ring_count`, which adds them to a pixel's frame count in one add.
pub const COUNT_FRAMES: u32 = 8;

/// One warp count per warp and frame of a `ring_count` block, then a flag per frame.
const COUNT_RED: usize = ((PIXEL_THREADS / 32 + 1) * COUNT_FRAMES) as usize;

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

    use super::{COUNT_FRAMES, COUNT_RED, RingRecord, UNCOUNTED};

    /// For each frame `f` of `f0 .. f0 + nf` not counted yet (`nobs[f] == UNCOUNTED`):
    /// `nobs[f]` = its unmasked pixels in `counts`, and `pixel_frames[p] += 1` for each of
    /// them. A frame already counted is left as it is, so each frame is in `pixel_frames` at
    /// most once. Block `b` takes pixels `c * PIXEL_THREADS ..`, `c = b % nchunk`, of the
    /// `COUNT_FRAMES` frames from `f0 + b / nchunk * COUNT_FRAMES`, and adds each frame's
    /// count over them to `partial[f]` as `1 << 32 | count`; the add that brings the high
    /// word to `nchunk` writes `nobs[f]` and zeroes `partial[f]`. Each block reads `nobs[f]`
    /// once, before its add, so every block of `f` sees the value from before the launch.
    ///
    /// # Safety
    ///
    /// `counts` `nframe * n`, `nobs` and `partial` `nframe`, `partial` zero at uncounted
    /// frames, `pixel_frames` `n`, `f0 + nf <= nframe`, `n + PIXEL_THREADS` within `u32`;
    /// grid `(nchunk * ceil(nf / COUNT_FRAMES), 1, 1)`, `nchunk = ceil(n / PIXEL_THREADS)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn ring_count(
        counts: *const u16,
        nobs: *mut u32,
        partial: *mut u64,
        pixel_frames: *mut u32,
        f0: u32,
        nf: u32,
        n: u32,
    ) {
        use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};
        use cuda_device::warp;
        const F: usize = COUNT_FRAMES as usize;
        const W: usize = (PIXEL_THREADS / 32) as usize;
        static mut RED: SharedArray<u32, COUNT_RED> = SharedArray::UNINIT;
        let nchunk = n.div_ceil(PIXEL_THREADS);
        let b = thread::blockIdx_x();
        let fa = f0 + b / nchunk * COUNT_FRAMES;
        let nb = (f0 + nf - fa).min(COUNT_FRAMES) as usize;
        let t = thread::threadIdx_x();
        let p = (b % nchunk) * PIXEL_THREADS + t;
        let s = (&raw mut RED).cast::<u32>();
        // `s[W * F + i]`: frame `fa + i` is in the block and was not counted before.
        let open = unsafe { s.add(W * F) };
        if t < COUNT_FRAMES {
            let new = (t as usize) < nb && unsafe { *nobs.add((fa + t) as usize) } == UNCOUNTED;
            unsafe { *open.add(t as usize) = new as u32 };
        }
        thread::sync_threads();
        // Every load before the first vote, so a thread has all its frames in flight.
        // Both loops unrolled, so `v` stays in registers rather than local memory.
        let mut v = [MASKED; F];
        #[unroll]
        for i in 0..COUNT_FRAMES {
            let i = i as usize;
            if p < n && unsafe { *open.add(i) } != 0 {
                let f = fa as usize + i;
                v[i] = unsafe { *counts.add(f * n as usize + p as usize) };
            }
        }
        let w = (t / 32) as usize;
        let mut acc = 0u32;
        #[unroll]
        for i in 0..COUNT_FRAMES {
            let i = i as usize;
            let unmasked = v[i] != MASKED;
            acc += unmasked as u32;
            let k = warp::popc(unmasked);
            if warp::lane_id() == 0 {
                unsafe { *s.add(w * F + i) = k };
            }
        }
        if acc > 0 {
            unsafe { DeviceAtomicU32::from_ptr(pixel_frames.add(p as usize)) }
                .fetch_add(acc, AtomicOrdering::Relaxed);
        }
        thread::sync_threads();
        let i = t as usize;
        if i < F && unsafe { *open.add(i) } != 0 {
            let f = fa as usize + i;
            let mut k = 0u32;
            for w in 0..W {
                k += unsafe { *s.add(w * F + i) };
            }
            let at = unsafe { DeviceAtomicU64::from_ptr(partial.add(f)) };
            let old = at.fetch_add(1 << 32 | k as u64, AtomicOrdering::Relaxed);
            if (old >> 32) as u32 == nchunk - 1 {
                unsafe {
                    *nobs.add(f) = old as u32 + k;
                    *partial.add(f) = 0;
                }
            }
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
