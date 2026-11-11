//! Pull-only halo exchange and the fixed-tree scalar reduction (DESIGN B3).
//!
//! Each region stores its tiles in slots, own tiles first, then halo tiles, each ascending.
//! Peers never write into another region (invariant 5): a region pulls its halo slots from
//! the owners' memory, and an owner reads the per-frame contributions that neighbours computed
//! for its tiles directly from their memory, adding them in [`Partition::tile_frames`] order.
//! Global scalars are per-tile partials summed through one pairwise tree over the tile index,
//! padded to a power of two, so neither order depends on the GPU count (invariant 8).

use std::ffi::c_void;
use std::sync::Arc;

use cuda_core::{Context, DeviceBuffer, Function, Stream, launch_kernel_on_stream};

use crate::{DistError, Partition};

/// Where one region keeps its tiles and frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    nown: usize,
    slots: Vec<u32>,
    slot_of_tile: Vec<i32>,
    frames: Vec<u32>,
    frame_local: Vec<i32>,
}

impl Layout {
    /// Layout of `region`. Fails if a frame the region computes touches a tile it does not
    /// hold, which would make its kernels read outside their slots.
    pub fn new(part: &Partition, region: usize) -> Result<Self, DistError> {
        let r = part.regions().get(region).ok_or_else(|| {
            DistError::Partition(format!("region {region} of {}", part.nregion()))
        })?;
        let slots: Vec<u32> = r.own.iter().chain(&r.halo).copied().collect();
        if slots.len() > i32::MAX as usize || part.nframe() > i32::MAX as usize {
            return Err(DistError::Partition(
                "more than i32::MAX slots or frames".into(),
            ));
        }
        let mut slot_of_tile = vec![-1i32; part.grid().len()];
        for (s, &t) in slots.iter().enumerate() {
            slot_of_tile[t as usize] = s as i32;
        }
        let mut frame_local = vec![-1i32; part.nframe()];
        for (l, &f) in r.frames.iter().enumerate() {
            frame_local[f as usize] = l as i32;
            let c = part.corners()[f as usize];
            if let Some(t) = part
                .grid()
                .footprint(c, part.frame())
                .find(|&t| slot_of_tile[t] < 0)
            {
                return Err(DistError::Partition(format!(
                    "region {region} computes frame {f} but does not hold tile {t}"
                )));
            }
        }
        Ok(Self {
            nown: r.own.len(),
            slots,
            slot_of_tile,
            frames: r.frames.clone(),
            frame_local,
        })
    }

    /// Layouts of every region of `part`, in region order.
    pub fn all(part: &Partition) -> Result<Vec<Self>, DistError> {
        (0..part.nregion()).map(|r| Self::new(part, r)).collect()
    }

    /// Owned tiles occupy slots `0..nown`.
    pub fn nown(&self) -> usize {
        self.nown
    }

    pub fn nhalo(&self) -> usize {
        self.slots.len() - self.nown
    }

    /// Tile held in each slot: own tiles ascending, then halo tiles ascending.
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    /// Slot of each grid tile, or -1 where this region does not hold it.
    pub fn slot_of_tile(&self) -> &[i32] {
        &self.slot_of_tile
    }

    /// Frames this region computes, ascending; frame `frames[l]` is stored at local index `l`.
    pub fn frames(&self) -> &[u32] {
        &self.frames
    }

    /// Local index of each global frame, or -1 where another region computes it.
    pub fn frame_local(&self) -> &[i32] {
        &self.frame_local
    }

    /// Owner addresses of this region's halo slots, from [`tile_sources`].
    pub fn halo_sources(&self, tile_src: &[u64]) -> Vec<u64> {
        self.slots[self.nown..]
            .iter()
            .map(|&t| tile_src[t as usize])
            .collect()
    }
}

/// Address of every tile in its owner's memory: `base[r] + slot * stride`, where `base[r]` is
/// region `r`'s slot array (or any per-own-tile array, such as scalar partials).
pub fn tile_sources(part: &Partition, layouts: &[Layout], base: &[u64], stride: u64) -> Vec<u64> {
    part.owner()
        .iter()
        .enumerate()
        .map(|(t, &o)| {
            let o = o as usize;
            base[o] + layouts[o].slot_of_tile[t] as u64 * stride
        })
        .collect()
}

/// Address of every frame's contribution in the memory of the region that computes it:
/// `base[r] + local * stride`.
pub fn frame_sources(part: &Partition, layouts: &[Layout], base: &[u64], stride: u64) -> Vec<u64> {
    (0..part.nframe())
        .map(|f| {
            let r = part.computed_by(f);
            base[r] + layouts[r].frame_local[f] as u64 * stride
        })
        .collect()
}

/// Length of the [`Exchange::tree_sum`] scratch for `nleaf` leaves of `k` components.
pub fn tree_scratch_len(nleaf: usize, k: usize) -> usize {
    nleaf.next_power_of_two() * k
}

/// Host twin of [`Exchange::tree_sum`]: `leaves` holds `k` components per leaf, leaf-major.
pub fn tree_sum_host(leaves: &[f64], k: usize) -> Vec<f64> {
    let n = leaves.len() / k;
    let mut s = vec![0.0; tree_scratch_len(n, k)];
    s[..n * k].copy_from_slice(&leaves[..n * k]);
    let mut half = s.len() / k / 2;
    while half > 0 {
        for i in 0..half * k {
            s[i] += s[i + half * k];
        }
        half /= 2;
    }
    s.truncate(k);
    s
}

/// The exchange kernels, loaded into one context.
pub struct Exchange {
    halo_pull: Function,
    tree_sum: Function,
}

const TREE_THREADS: u32 = 1024;

impl Exchange {
    pub fn load(ctx: &Arc<Context>) -> Result<Self, DistError> {
        let m = ctx.load_module_from_ptx_src(PTX)?;
        Ok(Self {
            halo_pull: m.load_function("halo_pull")?,
            tree_sum: m.load_function("tree_sum")?,
        })
    }

    /// Copies halo slot `j` (`words` 32-bit words) from `srcs[j]` to `dst + j * words * 4`.
    ///
    /// # Safety
    /// `dst` must hold `srcs.len() * words` words on the stream's device, and every `srcs[j]`
    /// must address `words` words on this device or a peer with access enabled. The owners
    /// must not write those words until this launch completes ([`crate::Gpu::fence`]).
    pub unsafe fn halo_pull(
        &self,
        stream: &Stream,
        dst: u64,
        srcs: &DeviceBuffer<u64>,
        words: u32,
    ) -> Result<(), DistError> {
        let nhalo = u32::try_from(srcs.len())
            .map_err(|_| DistError::Partition("more than u32::MAX halo slots".into()))?;
        if nhalo == 0 || words == 0 {
            return Ok(());
        }
        let (mut dst, mut src, mut words) = (dst, srcs.cu_deviceptr(), words);
        let mut params = [
            (&raw mut dst).cast::<c_void>(),
            (&raw mut src).cast::<c_void>(),
            (&raw mut words).cast::<c_void>(),
        ];
        // SAFETY: the caller guarantees the extents; the kernel reads `src[0..nhalo]`.
        unsafe {
            launch_kernel_on_stream(
                &self.halo_pull,
                (nhalo, 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut params,
            )?
        };
        Ok(())
    }

    /// Sums the `k` f64 components of the leaves `leaves[0..n]` through one pairwise tree
    /// over `n.next_power_of_two()` leaf positions (missing leaves are zero) and writes the
    /// `k` sums to `out`. One block; bit-identical to [`tree_sum_host`].
    ///
    /// # Safety
    /// Every `leaves[i]` must address `k` f64 on this device or a peer with access enabled,
    /// written before this launch on every device ([`crate::Gpu::fence`]); `out` must hold
    /// `k` f64 on this device.
    pub unsafe fn tree_sum(
        &self,
        stream: &Stream,
        leaves: &DeviceBuffer<u64>,
        k: u32,
        scratch: &DeviceBuffer<f64>,
        out: u64,
    ) -> Result<(), DistError> {
        let n = leaves.len();
        let len = tree_scratch_len(n, k as usize);
        if n == 0 || k == 0 || k > TREE_THREADS || len > u32::MAX as usize {
            return Err(DistError::Partition(format!(
                "tree_sum over {n} leaves of {k} components"
            )));
        }
        if scratch.len() < len {
            return Err(DistError::Partition(format!(
                "tree_sum scratch holds {} f64, needs {len}",
                scratch.len()
            )));
        }
        let (mut lv, mut nleaf, mut pow2, mut k) = (
            leaves.cu_deviceptr(),
            n as u32,
            n.next_power_of_two() as u32,
            k,
        );
        let (mut sc, mut out) = (scratch.cu_deviceptr(), out);
        let mut params = [
            (&raw mut lv).cast::<c_void>(),
            (&raw mut nleaf).cast::<c_void>(),
            (&raw mut pow2).cast::<c_void>(),
            (&raw mut k).cast::<c_void>(),
            (&raw mut sc).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
        ];
        // SAFETY: extents checked above; the caller guarantees the leaf and out addresses.
        unsafe {
            launch_kernel_on_stream(
                &self.tree_sum,
                (1, 1, 1),
                (TREE_THREADS, 1, 1),
                0,
                stream,
                &mut params,
            )?
        };
        Ok(())
    }
}

/// `.rn` on every float op keeps ptxas from contracting to FMA, so the host twins match bits.
const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// dst[j * words + i] = src[j][i] for halo slot j = blockIdx.x.
.visible .entry halo_pull(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_words
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [p_dst];
    ld.param.u64 %rd2, [p_src];
    ld.param.u32 %r1, [p_words];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ntid.x;
    mul.wide.u32 %rd3, %r2, 8;
    add.s64 %rd3, %rd2, %rd3;
    ld.global.u64 %rd4, [%rd3];
    cvta.to.global.u64 %rd4, %rd4;
    mul.wide.u32 %rd5, %r2, %r1;
    shl.b64 %rd5, %rd5, 2;
    add.s64 %rd5, %rd1, %rd5;
    mov.u32 %r5, %r3;
HP_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra HP_DONE;
    mul.wide.u32 %rd6, %r5, 4;
    add.s64 %rd7, %rd4, %rd6;
    ld.global.u32 %r6, [%rd7];
    add.s64 %rd8, %rd5, %rd6;
    st.global.u32 [%rd8], %r6;
    add.u32 %r5, %r5, %r4;
    bra HP_LOOP;
HP_DONE:
    ret;
}

// scratch[l * k + c] = leaf l < nleaf ? leaves[l][c] : 0 for l < pow2; then for
// half = pow2 / 2 .. 1: scratch[i] += scratch[i + half * k] for i < half * k; out = scratch[0..k].
.visible .entry tree_sum(
    .param .u64 p_leaves,
    .param .u32 p_nleaf,
    .param .u32 p_pow2,
    .param .u32 p_k,
    .param .u64 p_scratch,
    .param .u64 p_out
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f64 %fd<4>;

    ld.param.u64 %rd1, [p_leaves];
    ld.param.u32 %r1, [p_nleaf];
    ld.param.u32 %r2, [p_pow2];
    ld.param.u32 %r3, [p_k];
    ld.param.u64 %rd2, [p_scratch];
    ld.param.u64 %rd3, [p_out];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r2, %r3;
    mov.u32 %r7, %r4;
TS_LOAD:
    setp.ge.u32 %p1, %r7, %r6;
    @%p1 bra TS_LOADED;
    div.u32 %r8, %r7, %r3;
    mul.lo.u32 %r9, %r8, %r3;
    sub.u32 %r9, %r7, %r9;
    mov.f64 %fd1, 0d0000000000000000;
    setp.lt.u32 %p2, %r8, %r1;
    @!%p2 bra TS_STORE;
    mul.wide.u32 %rd4, %r8, 8;
    add.s64 %rd4, %rd1, %rd4;
    ld.global.u64 %rd5, [%rd4];
    cvta.to.global.u64 %rd5, %rd5;
    mul.wide.u32 %rd6, %r9, 8;
    add.s64 %rd5, %rd5, %rd6;
    ld.global.f64 %fd1, [%rd5];
TS_STORE:
    mul.wide.u32 %rd7, %r7, 8;
    add.s64 %rd7, %rd2, %rd7;
    st.global.f64 [%rd7], %fd1;
    add.u32 %r7, %r7, %r5;
    bra TS_LOAD;
TS_LOADED:
    barrier.sync 0;
    shr.u32 %r10, %r2, 1;
TS_LEVEL:
    setp.eq.u32 %p1, %r10, 0;
    @%p1 bra TS_DONE;
    mul.lo.u32 %r11, %r10, %r3;
    mul.wide.u32 %rd9, %r11, 8;
    mov.u32 %r7, %r4;
TS_PAIR:
    setp.ge.u32 %p2, %r7, %r11;
    @%p2 bra TS_PAIRED;
    mul.wide.u32 %rd8, %r7, 8;
    add.s64 %rd8, %rd2, %rd8;
    add.s64 %rd10, %rd8, %rd9;
    ld.global.f64 %fd2, [%rd8];
    ld.global.f64 %fd3, [%rd10];
    add.rn.f64 %fd2, %fd2, %fd3;
    st.global.f64 [%rd8], %fd2;
    add.u32 %r7, %r7, %r5;
    bra TS_PAIR;
TS_PAIRED:
    barrier.sync 0;
    shr.u32 %r10, %r10, 1;
    bra TS_LEVEL;
TS_DONE:
    setp.ge.u32 %p1, %r4, %r3;
    @%p1 bra TS_END;
    mul.wide.u32 %rd11, %r4, 8;
    add.s64 %rd12, %rd2, %rd11;
    ld.global.f64 %fd1, [%rd12];
    add.s64 %rd13, %rd3, %rd11;
    st.global.f64 [%rd13], %fd1;
TS_END:
    ret;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::enable_all_pairs;
    use cuda_core::IntoResult;

    /// GPU `a` pulls slots straight out of GPU `b`'s memory, for every ordered pair.
    #[test]
    fn halo_pull_reads_every_peer() {
        let ctxs: Vec<_> = (0..4).map(|o| Context::new(o).unwrap()).collect();
        enable_all_pairs(&ctxs).unwrap();
        const WORDS: usize = 4096;
        let mut bad = Vec::new();
        for b in &ctxs {
            let sb = b.new_stream().unwrap();
            let host: Vec<u32> = (0..2 * WORDS as u32).map(|i| i ^ 0x5a5a_0000).collect();
            let src = DeviceBuffer::from_host(&sb, &host).unwrap();
            for a in &ctxs {
                let sa = a.new_stream().unwrap();
                let ex = Exchange::load(a).unwrap();
                // Two slots in reverse order, so the table indexing is checked too.
                let p = src.cu_deviceptr();
                let table = DeviceBuffer::from_host(&sa, &[p + 4 * WORDS as u64, p]).unwrap();
                let dst = DeviceBuffer::<u32>::zeroed(&sa, 2 * WORDS).unwrap();
                unsafe {
                    ex.halo_pull(&sa, dst.cu_deviceptr(), &table, WORDS as u32)
                        .unwrap()
                };
                let got = dst.to_host_vec(&sa).unwrap();
                let want: Vec<u32> = host[WORDS..]
                    .iter()
                    .chain(&host[..WORDS])
                    .copied()
                    .collect();
                if got != want {
                    bad.push((a.ordinal(), b.ordinal(), got[0]));
                }
            }
            unsafe { cuda_core::sys::cuCtxSynchronize().result().unwrap() };
        }
        assert!(bad.is_empty(), "(reader, owner, first word): {bad:x?}");
    }
}
