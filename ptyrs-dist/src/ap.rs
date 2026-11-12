//! PyNX AP on 1, 2 or 4 GPUs, bit-identical whatever the GPU count (DESIGN §7 M4).
//!
//! The scan is cut once into [`NLEAF`] leaf regions and a run on `g` GPUs merges them
//! ([`Partition::merge`]), so every order fixed per leaf is the same on 1, 2 and 4 GPUs:
//! exit waves are computed by their leaf's owner in stacks of `stack` frames per leaf; each
//! object pixel adds its views in (leaf, frame) order, a region holding its own leaves' frames
//! and, copied from their owners, the other leaves' frames that touch its tiles; the probe sums
//! are one per leaf, added through a fixed pair tree; the regulariser's `max(den)` is an exact
//! max over regions; the LLK is [`LlkSums::from_partials`] over frames by global index.
//!
//! Per cycle, with [`Gpu::fence`] between the steps: A exit waves, LLK, projection and the
//! leaf probe sums of the own stacks; B gather of foreign exit waves, object sums and their
//! max, probe pair tree; C object and probe updates, then each owner packs the updated rows
//! its neighbours read; D each receiver pulls them by copy engine and unpacks its halo.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Context, DeviceBuffer};
use ptyrs_core::{AddrTable, View};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    C32, Kernels, LlkPartial, LlkSums, StackTable, UpdateSums, ViewAddr, ViewDims, Views,
    view_tiles,
};

use crate::exchange::{self, Pull, Run, Runs};
use crate::{DEFAULT_TILE, DistError, Exchange, Gpu, Partition, Scheme, Workers};

/// Leaf regions every run is cut from; a run uses 1, 2 or 4 GPUs.
pub const NLEAF: usize = 4;

/// One far-field scan with its starting estimate.
pub struct ApProblem {
    /// Views with integer offsets, frame-major.
    pub addr: AddrTable,
    /// `[nframe, ny, nx]` in detector order; [`ptyrs_gpu::MASKED`] marks a pixel to ignore.
    pub counts: Vec<u16>,
    /// `[nobj, nyo, nxo]`.
    pub object: Vec<C32>,
    /// `[nprobe, ny, nx]`.
    pub probe: Vec<C32>,
    /// [`ViewDims::quad_phase`].
    pub quad_phase: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApConfig {
    pub scheme: Scheme,
    /// Frames per stack; every GPU transforms `stack * nmode` planes per FFT.
    pub stack: usize,
    /// Object update with this inertia; `None` keeps the object fixed.
    pub obj_inertia: Option<f32>,
    /// Probe update with this inertia; `None` keeps the probe fixed.
    pub probe_inertia: Option<f32>,
    /// Every `calc_llk`-th cycle of a [`DistAp::run`] and its last record the LLK of the
    /// estimate before the update; 0 never does.
    pub calc_llk: usize,
}

/// Frames `frames` (local indices) of leaf `leaf`.
struct Stack {
    leaf: usize,
    frames: Range<u32>,
}

/// One region's share, fixed at setup.
struct RegionPlan {
    dims: ViewDims,
    views: Vec<ViewAddr>,
    /// Global frame of each local frame, in (leaf, frame) order.
    global: Vec<u32>,
    stacks: Vec<Stack>,
    /// Stacks of the own leaves, ascending.
    own: Vec<usize>,
    /// First stack of each leaf with local frames.
    leaf_stack0: [Option<usize>; NLEAF],
    /// Halo tiles some local view reads.
    halo_read: Vec<u32>,
}

/// The object rows region `recv` reads from region `owner`'s tiles, `[byte offset, words]`
/// each. `owner` packs them from word `send_at` of its send buffer on; `recv` pulls them to
/// word `recv_at` of its receive buffer and unpacks them into its object.
struct HaloSeg {
    recv: usize,
    owner: usize,
    rows: Vec<[u64; 2]>,
    words: u64,
    send_at: u64,
    recv_at: u64,
}

struct Plan {
    run: Partition,
    /// Leaves per region.
    k: usize,
    /// Global frames of each leaf, ascending.
    leaf_frames: Vec<Vec<u32>>,
    stack: usize,
    obj_inertia: Option<f32>,
    probe_inertia: Option<f32>,
    calc_llk: usize,
    regions: Vec<RegionPlan>,
    /// By receiver, then owner.
    halo: Vec<HaloSeg>,
}

impl Plan {
    fn new(
        addr: &AddrTable,
        cfg: &ApConfig,
        quad_phase: f32,
        ngpu: usize,
    ) -> Result<Self, DistError> {
        let bad = |m: String| Err(DistError::Partition(m));
        let [ny, nx] = addr.frame_shape();
        if ny != nx {
            return bad(format!("frame {ny}x{nx} is not square"));
        }
        if cfg.stack == 0 {
            return bad("stack of 0 frames".into());
        }
        let leaves = Partition::from_addr(addr, DEFAULT_TILE, cfg.scheme, NLEAF)?;
        let run = leaves.merge(ngpu)?;
        let k = NLEAF / ngpu;
        let u = |v: usize| u32::try_from(v).map_err(|_| DistError::Partition(format!("{v} > u32")));
        let [nyo, nxo] = addr.obj_shape();
        let dims = ViewDims {
            nframe: 0,
            nmode: u(addr.nmode())?,
            nprobe: u(addr.nprobe())?,
            nobj: u(addr.nobj())?,
            frame: [u(ny)?, u(nx)?],
            obj: [u(nyo)?, u(nxo)?],
            quad_phase,
        };
        let grid = run.grid();
        if dims.tiles().map(|t| t as usize) != grid.dims() {
            return bad(format!(
                "kernel tiles {:?} differ from partition tiles {:?}",
                dims.tiles(),
                grid.dims()
            ));
        }
        let view_addr = |v: &View| ViewAddr {
            oy: v.oy,
            ox: v.ox,
            fy: 0.0,
            fx: 0.0,
            probe: v.probe,
            obj: v.obj,
        };
        // Tiles each frame's views read, by the kernels' rule.
        let tiles: Vec<Vec<usize>> = (0..addr.nframe())
            .map(|f| {
                let mut t: Vec<usize> = addr
                    .frame_views(f)
                    .iter()
                    .flat_map(|v| view_tiles(&dims, &view_addr(v)))
                    .map(|[ty, tx]| grid.id(ty as usize, tx as usize))
                    .collect();
                t.sort_unstable();
                t.dedup();
                t
            })
            .collect();
        let leaf_frames: Vec<Vec<u32>> =
            leaves.regions().iter().map(|g| g.frames.clone()).collect();
        let owner = run.owner();
        let mut regions = Vec::with_capacity(ngpu);
        for (r, region) in run.regions().iter().enumerate() {
            let mut global = Vec::new();
            let mut stacks = Vec::new();
            let mut own = Vec::new();
            let mut leaf_stack0 = [None; NLEAF];
            for (l, all) in leaf_frames.iter().enumerate() {
                let mine = l / k == r;
                let frames: Vec<u32> = if mine {
                    all.clone()
                } else {
                    all.iter()
                        .copied()
                        .filter(|&f| tiles[f as usize].iter().any(|&t| owner[t] as usize == r))
                        .collect()
                };
                if frames.is_empty() {
                    continue;
                }
                leaf_stack0[l] = Some(stacks.len());
                for chunk in frames.chunks(cfg.stack) {
                    let start = u(global.len())?;
                    global.extend_from_slice(chunk);
                    if mine {
                        own.push(stacks.len());
                    }
                    stacks.push(Stack {
                        leaf: l,
                        frames: start..u(global.len())?,
                    });
                }
            }
            if global.is_empty() {
                return bad(format!("region {r} of {ngpu} has no frames"));
            }
            let mut zone = vec![0u8; grid.len()];
            region.own.iter().for_each(|&t| zone[t as usize] = 1);
            region.halo.iter().for_each(|&t| zone[t as usize] = 2);
            let mut read = vec![false; grid.len()];
            for &f in &global {
                for &t in &tiles[f as usize] {
                    match zone[t] {
                        1 => {}
                        2 => read[t] = true,
                        _ => {
                            return bad(format!(
                                "frame {f} reads tile {t}, outside region {r}'s own and halo tiles"
                            ));
                        }
                    }
                }
            }
            regions.push(RegionPlan {
                dims: ViewDims {
                    nframe: u(global.len())?,
                    ..dims
                },
                views: global
                    .iter()
                    .flat_map(|&f| addr.frame_views(f as usize).iter().map(view_addr))
                    .collect(),
                halo_read: (0..grid.len())
                    .filter(|&t| read[t])
                    .map(|t| t as u32)
                    .collect(),
                global,
                stacks,
                own,
                leaf_stack0,
            });
        }
        let halo = halo_segs(&run, &regions, dims.nobj as usize);
        Ok(Self {
            run,
            k,
            leaf_frames,
            stack: cfg.stack,
            obj_inertia: cfg.obj_inertia,
            probe_inertia: cfg.probe_inertia,
            calc_llk: cfg.calc_llk,
            regions,
            halo,
        })
    }

    fn dims(&self) -> &ViewDims {
        &self.regions[0].dims
    }

    fn n(&self) -> usize {
        let [ny, nx] = self.dims().frame;
        ny as usize * nx as usize
    }

    /// Probe elements.
    fn np(&self) -> usize {
        self.dims().nprobe as usize * self.n()
    }

    /// 64-bit words (one [`C32`] each) of one frame's exit waves.
    fn plane(&self) -> usize {
        self.dims().nmode as usize * self.n()
    }

    /// Exit-wave gather runs of region `r`: every foreign local frame from its owner's stack.
    fn gather(&self, r: usize, ptrs: &[Ptrs]) -> Vec<Run> {
        let rp = &self.regions[r];
        let bytes = (self.plane() * size_of::<C32>()) as u64;
        let mut runs = Vec::new();
        for (s, st) in rp.stacks.iter().enumerate() {
            let o = st.leaf / self.k;
            if o == r {
                continue;
            }
            let o0 = self.regions[o].leaf_stack0[st.leaf].expect("owner holds its leaf");
            for j in st.frames.clone() {
                let f = rp.global[j as usize];
                let i = self.leaf_frames[st.leaf]
                    .binary_search(&f)
                    .expect("foreign frame of its leaf");
                runs.push(Run {
                    dst: ptrs[r].psi[s] + (j - st.frames.start) as u64 * bytes,
                    src: ptrs[o].psi[o0 + i / self.stack] + (i % self.stack) as u64 * bytes,
                    len: self.plane() as u64,
                });
            }
        }
        runs
    }

    /// Words of region `r`'s halo send and receive buffers.
    fn halo_words(&self, r: usize) -> [u64; 2] {
        let sum = |f: &dyn Fn(&HaloSeg) -> bool| -> u64 {
            self.halo.iter().filter(|s| f(s)).map(|s| s.words).sum()
        };
        [sum(&|s| s.owner == r), sum(&|s| s.recv == r)]
    }

    /// `[object byte offset, buffer word, words]` of every row of the segments `pick` selects,
    /// laid out in the buffer from word `at(segment)` on.
    fn halo_rows(
        &self,
        pick: impl Fn(&HaloSeg) -> bool,
        at: fn(&HaloSeg) -> u64,
    ) -> impl Iterator<Item = [u64; 3]> {
        self.halo
            .iter()
            .filter(move |s| pick(s))
            .flat_map(move |s| {
                let mut w = at(s);
                s.rows.iter().map(move |&[off, len]| {
                    w += len;
                    [off, w - len, len]
                })
            })
    }

    /// Runs copying region `r`'s object rows that others read into its send buffer.
    fn halo_pack(&self, r: usize, obj: u64, send: u64) -> Vec<Run> {
        self.halo_rows(|s| s.owner == r, |s| s.send_at)
            .map(|[off, w, len]| Run {
                dst: send + w * 8,
                src: obj + off,
                len,
            })
            .collect()
    }

    /// Runs copying region `r`'s receive buffer into the halo rows of its object.
    fn halo_unpack(&self, r: usize, obj: u64, recv: u64) -> Vec<Run> {
        self.halo_rows(|s| s.recv == r, |s| s.recv_at)
            .map(|[off, w, len]| Run {
                dst: obj + off,
                src: recv + w * 8,
                len,
            })
            .collect()
    }

    /// Copy-engine pulls of region `r`'s receive buffer from the owners' send buffers.
    fn halo_pulls(&self, r: usize, ptrs: &[Ptrs]) -> Vec<Pull> {
        self.halo
            .iter()
            .filter(|s| s.recv == r)
            .map(|s| {
                let o = &ptrs[s.owner];
                Pull {
                    dst: ptrs[r].halo_recv + s.recv_at * 8,
                    src: o.halo_send + s.send_at * 8,
                    bytes: (s.words * 8) as usize,
                    from: o.ctx.clone(),
                }
            })
            .collect()
    }
}

/// The halo segments of `regions`: each pixel row of consecutive read halo tiles with one
/// owner, every object mode; receive buffers hold the owners in order, send buffers the
/// receivers.
fn halo_segs(run: &Partition, regions: &[RegionPlan], nobj: usize) -> Vec<HaloSeg> {
    let grid = run.grid();
    let [nyo, nxo] = grid.obj();
    let tile = grid.tile();
    let owner = run.owner();
    let n = regions.len();
    let mut send_at = vec![0; n];
    let mut segs = Vec::new();
    for (r, rp) in regions.iter().enumerate() {
        let mut rows = vec![Vec::new(); n];
        let read = &rp.halo_read;
        for m in 0..nobj {
            let mut i = 0;
            while i < read.len() {
                let t0 = read[i] as usize;
                let [ty, tx0] = grid.coords(t0);
                let o = owner[t0] as usize;
                let mut tx1 = tx0;
                while let Some(&t) = read.get(i + 1) {
                    let t = t as usize;
                    if grid.coords(t) != [ty, tx1 + 1] || owner[t] as usize != o {
                        break;
                    }
                    tx1 += 1;
                    i += 1;
                }
                i += 1;
                let (x0, x1) = (tx0 * tile, ((tx1 + 1) * tile).min(nxo));
                for y in ty * tile..((ty + 1) * tile).min(nyo) {
                    let off = (((m * nyo + y) * nxo + x0) * size_of::<C32>()) as u64;
                    rows[o].push([off, (x1 - x0) as u64]);
                }
            }
        }
        let mut recv_at = 0;
        for (o, rows) in rows.into_iter().enumerate() {
            let words: u64 = rows.iter().map(|w| w[1]).sum();
            if words == 0 {
                continue;
            }
            segs.push(HaloSeg {
                recv: r,
                owner: o,
                rows,
                words,
                send_at: send_at[o],
                recv_at,
            });
            send_at[o] += words;
            recv_at += words;
        }
    }
    segs
}

/// Device addresses a region exposes to its peers.
struct Ptrs {
    ctx: Arc<Context>,
    psi: Vec<u64>,
    halo_send: u64,
    halo_recv: u64,
    max_bits: u64,
    /// `(num, den)` of each own leaf's probe sums.
    leaf: Vec<(u64, u64)>,
}

/// What a region's exchange reads, built once all regions exist.
struct Links {
    gather: Runs,
    halo: Vec<Pull>,
    max_srcs: DeviceBuffer<u64>,
    /// `(num, den)` of every leaf's probe sums, by leaf.
    leaf: [(u64, u64); NLEAF],
}

/// Probe pair tree over the leaf sums: `node[0] = L0 + L1`, `node[1] = L2 + L3`,
/// `root = node[0] + node[1]`.
struct Tree {
    node: [(DeviceBuffer<C32>, DeviceBuffer<f32>); 2],
    root: UpdateSums,
}

/// A region's side of the halo exchange: it packs what others read of its tiles into `send`
/// and unpacks what it pulled from the owners into `recv`.
struct Halo {
    send: DeviceBuffer<u64>,
    recv: DeviceBuffer<u64>,
    pack: Runs,
    unpack: Runs,
}

struct State {
    k: Kernels,
    ex: Exchange,
    fft: Plan2d,
    views: Views,
    table: StackTable,
    counts: DeviceBuffer<u16>,
    obj: DeviceBuffer<C32>,
    halo: Halo,
    probe: DeviceBuffer<C32>,
    /// Per local stack.
    psi: Vec<DeviceBuffer<C32>>,
    /// Per own stack.
    llk: Vec<DeviceBuffer<LlkPartial>>,
    obj_sums: Option<UpdateSums>,
    /// Per own leaf, when the probe updates.
    leaf_sums: Vec<UpdateSums>,
    tree: Option<Tree>,
    gmax: DeviceBuffer<u32>,
    links: Option<Links>,
}

/// Host data the workers copy from at setup.
struct Data {
    counts: Vec<u16>,
    object: Vec<C32>,
    probe: Vec<C32>,
}

impl State {
    fn new(gpu: &mut Gpu, plan: &Plan, data: &Data) -> Result<Self, DistError> {
        let rp = &plan.regions[gpu.region()];
        let stream = gpu.stream();
        let n = plan.n();
        let counts: Vec<u16> = rp
            .global
            .iter()
            .flat_map(|&f| &data.counts[f as usize * n..][..n])
            .copied()
            .collect();
        let ranges: Vec<Range<u32>> = rp.stacks.iter().map(|s| s.frames.clone()).collect();
        let nk = plan.stack * rp.dims.nmode as usize;
        let zeroed = |len| DeviceBuffer::zeroed(stream, len);
        let probe_sums = plan.probe_inertia.is_some();
        let np = plan.np();
        let obj = DeviceBuffer::from_host(stream, &data.object)?;
        let halo = {
            let r = gpu.region();
            let [send, recv] = plan
                .halo_words(r)
                .map(|w| DeviceBuffer::<u64>::zeroed(stream, (w as usize).max(1)));
            let (send, recv) = (send?, recv?);
            let o = obj.cu_deviceptr();
            Halo {
                pack: Runs::new(stream, &plan.halo_pack(r, o, send.cu_deviceptr()))?,
                unpack: Runs::new(stream, &plan.halo_unpack(r, o, recv.cu_deviceptr()))?,
                send,
                recv,
            }
        };
        Ok(Self {
            k: Kernels::load(gpu.ctx())?,
            ex: Exchange::load(gpu.ctx())?,
            fft: Plan2d::new(stream, rp.dims.frame[0] as usize, nk)
                .map_err(ptyrs_gpu::Error::from)?,
            views: Views::new(stream, rp.dims, &rp.views)?,
            table: StackTable::new(stream, &rp.dims, &ranges)?,
            counts: DeviceBuffer::from_host(stream, &counts)?,
            obj,
            halo,
            probe: DeviceBuffer::from_host(stream, &data.probe)?,
            psi: (0..rp.stacks.len())
                .map(|_| zeroed(nk * n))
                .collect::<Result<_, _>>()?,
            llk: (0..rp.own.len())
                .map(|_| DeviceBuffer::zeroed(stream, plan.stack))
                .collect::<Result<_, _>>()?,
            obj_sums: plan
                .obj_inertia
                .map(|_| UpdateSums::zeroed(stream, data.object.len()))
                .transpose()?,
            leaf_sums: (0..if probe_sums { plan.k } else { 0 })
                .map(|_| UpdateSums::zeroed(stream, np))
                .collect::<Result<_, _>>()?,
            tree: probe_sums
                .then(|| -> Result<Tree, DistError> {
                    let node = || -> Result<_, DistError> {
                        Ok((zeroed(np)?, DeviceBuffer::zeroed(stream, np)?))
                    };
                    Ok(Tree {
                        node: [node()?, node()?],
                        root: UpdateSums::zeroed(stream, np)?,
                    })
                })
                .transpose()?,
            gmax: DeviceBuffer::zeroed(stream, 1)?,
            links: None,
        })
    }

    fn ptrs(&self) -> Ptrs {
        Ptrs {
            ctx: self.obj.context().clone(),
            psi: self.psi.iter().map(|b| b.cu_deviceptr()).collect(),
            halo_send: self.halo.send.cu_deviceptr(),
            halo_recv: self.halo.recv.cu_deviceptr(),
            max_bits: self
                .obj_sums
                .as_ref()
                .map_or(0, |s| s.max_bits.cu_deviceptr()),
            leaf: self
                .leaf_sums
                .iter()
                .map(|s| (s.num.cu_deviceptr(), s.den.cu_deviceptr()))
                .collect(),
        }
    }

    /// One AP cycle; with `due`, the LLK partials of the own frames as `(global frame, sums)`.
    fn cycle(
        &mut self,
        gpu: &mut Gpu,
        plan: &Plan,
        due: bool,
    ) -> Result<Vec<(u32, LlkPartial)>, DistError> {
        let r = gpu.region();
        let rp = &plan.regions[r];
        let stream = gpu.stream().clone();
        let ctx = ExecutionContext::new(stream.clone());
        let Self {
            k,
            ex,
            fft,
            views,
            table,
            counts,
            obj,
            halo,
            probe,
            psi,
            llk,
            obj_sums,
            leaf_sums,
            tree,
            gmax,
            links,
        } = self;
        let links = links.as_ref().expect("DistAp::new installs the links");
        let dims = *views.dims();
        let np = plan.np() as u32;

        // A: own stacks.
        for s in leaf_sums.iter_mut() {
            ptyrs_gpu::zero(&ctx, &mut s.num)?;
            ptyrs_gpu::zero(&ctx, &mut s.den)?;
        }
        for (i, &s) in rp.own.iter().enumerate() {
            let at = table.at(s);
            let psi = &mut psi[s];
            k.obj_probe_to_psi(&ctx, views, at, obj, probe, psi)?;
            fft.forward_inplace(&ctx, psi)
                .map_err(ptyrs_gpu::Error::from)?;
            if due {
                k.llk(&ctx, &dims, at, counts, psi, &mut llk[i])?;
            }
            k.apply_amplitude(&ctx, &dims, at, counts, psi)?;
            fft.inverse_inplace(&ctx, psi)
                .map_err(ptyrs_gpu::Error::from)?;
            if let Some(sums) = leaf_sums.get_mut(rp.stacks[s].leaf - r * plan.k) {
                k.psi_to_probe(&ctx, views, at, psi, obj, sums)?;
            }
        }
        gpu.fence()?;

        // B: foreign exit waves, object sums and their max, probe pair tree.
        if let Some(sums) = obj_sums.as_mut() {
            // SAFETY: the runs address live exit-wave stacks, written before the fence and
            // rewritten by their owners only after the next.
            unsafe { ex.copy_runs(&stream, &links.gather)? };
            for (s, psi) in psi.iter().enumerate() {
                k.psi_to_obj(&ctx, views, table.at(s), psi, probe, sums)?;
            }
            k.obj_den_max(&ctx, views, sums)?;
        }
        if let Some(t) = tree.as_mut() {
            let l = &links.leaf;
            let [(n0, d0), (n1, d1)] = &mut t.node;
            let (nn, nd) = (2 * np, np);
            // SAFETY: every leaf buffer holds the probe's `np` sums, written before the fence
            // and zeroed by its owner only after the next; the nodes and root are local.
            unsafe {
                ex.add_f32(&stream, n0.cu_deviceptr(), l[0].0, l[1].0, nn)?;
                ex.add_f32(&stream, d0.cu_deviceptr(), l[0].1, l[1].1, nd)?;
                ex.add_f32(&stream, n1.cu_deviceptr(), l[2].0, l[3].0, nn)?;
                ex.add_f32(&stream, d1.cu_deviceptr(), l[2].1, l[3].1, nd)?;
                let root = &t.root;
                let (rn, rd) = (root.num.cu_deviceptr(), root.den.cu_deviceptr());
                ex.add_f32(&stream, rn, n0.cu_deviceptr(), n1.cu_deviceptr(), nn)?;
                ex.add_f32(&stream, rd, d0.cu_deviceptr(), d1.cu_deviceptr(), nd)?;
            }
        }
        gpu.fence()?;

        // C: updates.
        if let (Some(sums), Some(inertia)) = (obj_sums.as_mut(), plan.obj_inertia) {
            // SAFETY: every source is a region's `max_bits` word, written before the fence and
            // rewritten only after the next.
            unsafe { ex.max_u32(&stream, gmax.cu_deviceptr(), &links.max_srcs)? };
            k.normalize_obj(&ctx, views, obj, sums, gmax, inertia)?;
            // SAFETY: the runs address this region's object and send buffer, which its peers
            // last pulled before this cycle's first fence.
            unsafe { ex.copy_runs(&stream, &halo.pack)? };
        }
        if let (Some(t), Some(inertia)) = (tree.as_mut(), plan.probe_inertia) {
            k.update_from_sums(&ctx, probe, &mut t.root, np, inertia)?;
        }
        gpu.fence()?;

        // D: halo pull from the owners' send buffers; the unpack overwrites what
        // `normalize_obj` left in the halo tiles it read.
        if obj_sums.is_some() {
            // SAFETY: each pull reads the part of an owner's send buffer packed for this
            // region before the fence, repacked only after the next cycle's second fence; the
            // receive buffer and the object are this region's and touched by this stream only.
            unsafe {
                exchange::pull(&stream, &links.halo)?;
                ex.copy_runs(&stream, &halo.unpack)?;
            }
        }
        drop(ctx);

        let mut out = Vec::new();
        if due {
            for (i, &s) in rp.own.iter().enumerate() {
                let f = &rp.stacks[s].frames;
                let host = llk[i].to_host_vec(&stream)?;
                out.extend(
                    rp.global[f.start as usize..f.end as usize]
                        .iter()
                        .zip(host)
                        .map(|(&g, p)| (g, p)),
                );
            }
        }
        Ok(out)
    }
}

/// What one region computes and exchanges each cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionLoad {
    /// Frames whose exit waves it computes.
    pub computed: usize,
    /// Other regions' frames whose exit waves it copies for its object sums.
    pub copied: usize,
    /// Halo tiles it pulls after the object update.
    pub halo_tiles: usize,
}

/// AP over 1, 2 or 4 GPUs, one region each.
pub struct DistAp {
    workers: Workers<State>,
    plan: Arc<Plan>,
    cycle: usize,
    history: Vec<(usize, LlkSums)>,
}

impl DistAp {
    /// Uploads `problem` to `ordinals` (1, 2 or 4 GPUs, region `r` on `ordinals[r]`).
    pub fn new(ordinals: &[usize], problem: ApProblem, cfg: ApConfig) -> Result<Self, DistError> {
        let a = &problem.addr;
        let n = a.frame_shape()[0] * a.frame_shape()[1];
        let [nyo, nxo] = a.obj_shape();
        for (what, got, want) in [
            ("counts", problem.counts.len(), a.nframe() * n),
            ("object", problem.object.len(), a.nobj() * nyo * nxo),
            ("probe", problem.probe.len(), a.nprobe() * n),
        ] {
            if got != want {
                return Err(DistError::Partition(format!(
                    "{what} holds {got}, needs {want}"
                )));
            }
        }
        let plan = Arc::new(Plan::new(a, &cfg, problem.quad_phase, ordinals.len())?);
        let data = Arc::new(Data {
            counts: problem.counts,
            object: problem.object,
            probe: problem.probe,
        });
        let p = plan.clone();
        let workers = Workers::spawn(ordinals, move |gpu| State::new(gpu, &p, &data))?;
        let ptrs = workers.run(|_, st| Ok(st.ptrs()))?;
        let links: Vec<_> = (0..ordinals.len())
            .map(|r| {
                let leaf = std::array::from_fn(|l| {
                    let o = l / plan.k;
                    ptrs[o].leaf.get(l - o * plan.k).copied().unwrap_or((0, 0))
                });
                (plan.gather(r, &ptrs), plan.halo_pulls(r, &ptrs), leaf)
            })
            .collect();
        let max_srcs: Vec<u64> = ptrs.iter().map(|p| p.max_bits).collect();
        let links = Arc::new(links);
        workers.run(move |gpu, st| {
            let (gather, halo, leaf) = &links[gpu.region()];
            let s = gpu.stream();
            st.links = Some(Links {
                gather: Runs::new(s, gather)?,
                halo: halo.clone(),
                max_srcs: DeviceBuffer::from_host(s, &max_srcs)?,
                leaf: *leaf,
            });
            Ok(())
        })?;
        Ok(Self {
            workers,
            plan,
            cycle: 0,
            history: Vec::new(),
        })
    }

    pub fn ngpu(&self) -> usize {
        self.workers.len()
    }

    /// Per region, in region order.
    pub fn loads(&self) -> Vec<RegionLoad> {
        self.plan
            .regions
            .iter()
            .map(|rp| {
                let computed = rp.own.iter().map(|&s| rp.stacks[s].frames.len()).sum();
                RegionLoad {
                    computed,
                    copied: rp.global.len() - computed,
                    halo_tiles: rp.halo_read.len(),
                }
            })
            .collect()
    }

    /// Runs `ncycle` cycles and returns once they are complete.
    pub fn run(&mut self, ncycle: usize) -> Result<(), DistError> {
        let plan = self.plan.clone();
        let calc = plan.calc_llk;
        let due = move |ic: usize| calc > 0 && (ic.is_multiple_of(calc) || ic + 1 == ncycle);
        let out = self.workers.run(move |gpu, st| {
            let mut out = Vec::new();
            for ic in 0..ncycle {
                let p = st.cycle(gpu, &plan, due(ic))?;
                if due(ic) {
                    out.push(p);
                }
            }
            gpu.stream().synchronize()?;
            Ok(out)
        })?;
        let nframe = self.plan.run.nframe();
        for (d, ic) in (0..ncycle).filter(|&ic| due(ic)).enumerate() {
            let mut partials = vec![None; nframe];
            for (g, p) in out.iter().flat_map(|o| &o[d]) {
                partials[*g as usize] = Some(*p);
            }
            let partials: Vec<LlkPartial> = partials
                .into_iter()
                .map(|p| p.expect("every frame has one owner"))
                .collect();
            self.history
                .push((self.cycle + ic, LlkSums::from_partials(&partials)));
        }
        self.cycle += ncycle;
        Ok(())
    }

    /// `(cycle, totals)` of each recorded LLK, unnormalised.
    pub fn history(&self) -> &[(usize, LlkSums)] {
        &self.history
    }

    /// The object `[nobj, nyo, nxo]`, each tile from its owner.
    pub fn object(&self) -> Result<Vec<C32>, DistError> {
        let objs = self
            .workers
            .run(|gpu, st| Ok(st.obj.to_host_vec(gpu.stream())?))?;
        let grid = self.plan.run.grid();
        let [nyo, nxo] = grid.obj();
        let tile = grid.tile();
        let mut out = objs[0].clone();
        for (t, &o) in self.plan.run.owner().iter().enumerate() {
            let [ty, tx] = grid.coords(t);
            let (x0, x1) = (tx * tile, ((tx + 1) * tile).min(nxo));
            for m in 0..self.plan.dims().nobj as usize {
                for y in ty * tile..((ty + 1) * tile).min(nyo) {
                    let row = (m * nyo + y) * nxo;
                    out[row + x0..row + x1].copy_from_slice(&objs[o as usize][row + x0..row + x1]);
                }
            }
        }
        Ok(out)
    }

    /// The probe `[nprobe, ny, nx]`; every region holds the same.
    pub fn probe(&self) -> Result<Vec<C32>, DistError> {
        let mut p = self
            .workers
            .run(|gpu, st| Ok(st.probe.to_host_vec(gpu.stream())?))?;
        Ok(p.swap_remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::tests::gpus;
    use ptyrs_core::{C64, Geometry, Object, Positions, Scan, Stack, required_obj_shape, sim};
    use ptyrs_engine::{ApParams, Ptycho};

    const PHOTONS: f64 = 1e5;

    /// A simulated 32x32 scan at `positions(pixel)` and its PyNX start: flat object and the
    /// truth probe, scaled to the observed photons.
    fn case(nmode: usize, positions: impl FnOnce(f64) -> Positions) -> (Scan, Ptycho) {
        let g = Geometry::new(1.5e-10, 1.0, 75e-6, [32, 32]).unwrap();
        let pixel = g.pixel_size_object();
        let pos = positions(pixel[1]);
        let shape = required_obj_shape(&pos.pynx_preprocess(pixel), pixel, g.frame, 4);
        let object = sim::object(shape);
        let probe = sim::probe(g.frame, nmode).unwrap();
        let scan = ptyrs_engine::sim::simulate(g, pos, &object, &probe, PHOTONS, None, 7).unwrap();
        let flat = Object::new(Stack::filled(object.dims(), C64::new(1.0, 0.0)));
        let mut p = Ptycho::new(&scan, flat, probe).unwrap();
        p.scale_obj_probe();
        (scan, p)
    }

    fn c32(v: &[C64]) -> Vec<C32> {
        v.iter()
            .map(|c| C32 {
                re: c.re as f32,
                im: c.im as f32,
            })
            .collect()
    }

    fn problem(scan: &Scan, p: &Ptycho) -> ApProblem {
        let g = scan.geometry;
        let px = g.pixel_size_object()[1];
        ApProblem {
            addr: p.addr().clone(),
            counts: scan.counts.as_slice().to_vec(),
            object: c32(p.object().as_slice()),
            probe: c32(p.probe().as_slice()),
            quad_phase: (std::f64::consts::PI / (g.wavelength * g.distance) * px * px) as f32,
        }
    }

    fn bits(v: &[C32]) -> Vec<[u32; 2]> {
        v.iter().map(|c| [c.re.to_bits(), c.im.to_bits()]).collect()
    }

    fn llk_bits(h: &[(usize, LlkSums)]) -> Vec<(usize, [u64; 4])> {
        h.iter()
            .map(|(c, s)| {
                let b = [s.poisson, s.gaussian, s.euclidean, s.icalc].map(f64::to_bits);
                (*c, b)
            })
            .collect()
    }

    /// Object, probe and LLK history bits after `ncycle` cycles on `ngpu` GPUs.
    type Bits = (Vec<[u32; 2]>, Vec<[u32; 2]>, Vec<(usize, [u64; 4])>);

    fn run(ngpu: usize, problem: ApProblem, cfg: ApConfig, ncycle: usize) -> Bits {
        let mut ap = DistAp::new(&gpus(ngpu), problem, cfg).unwrap();
        assert_eq!(ap.ngpu(), ngpu);
        ap.run(ncycle).unwrap();
        (
            bits(&ap.object().unwrap()),
            bits(&ap.probe().unwrap()),
            llk_bits(ap.history()),
        )
    }

    /// Object, probe and every LLK are the same bits on 1, 2 and 4 GPUs, for a spiral cut in
    /// sectors and a raster cut in strips, with partial last stacks and two probe modes.
    #[test]
    fn ap_is_bit_identical_on_1_2_and_4_gpus() {
        let fermat = case(2, |px| sim::fermat(200, 2.5 * px));
        let raster = case(2, |px| sim::raster(8, 40, 3.0 * px));
        for (name, scheme, (scan, p)) in [
            ("fermat", Scheme::Sectors { centre: None }, fermat),
            ("raster", Scheme::Strips, raster),
        ] {
            let cfg = ApConfig {
                scheme,
                stack: 7,
                obj_inertia: Some(0.01),
                probe_inertia: Some(0.001),
                calc_llk: 1,
            };
            let one = run(1, problem(&scan, &p), cfg, 5);
            assert_eq!(one.2.len(), 5, "{name}");
            assert_ne!(
                one.0,
                bits(&problem(&scan, &p).object),
                "{name}: object unchanged"
            );
            assert_ne!(
                one.1,
                bits(&problem(&scan, &p).probe),
                "{name}: probe unchanged"
            );
            for ngpu in [2, 4] {
                let got = run(ngpu, problem(&scan, &p), cfg, 5);
                assert!(got.0 == one.0, "{name}: object differs on {ngpu} GPUs");
                assert!(got.1 == one.1, "{name}: probe differs on {ngpu} GPUs");
                assert_eq!(got.2, one.2, "{name}: LLK differs on {ngpu} GPUs");
            }
        }
    }

    fn rel_l2(a: &[C32], b: &[C64]) -> f64 {
        let (mut d, mut n) = (0.0, 0.0);
        for (a, b) in a.iter().zip(b) {
            d += (C64::new(a.re as f64, a.im as f64) - b).norm_sqr();
            n += b.norm_sqr();
        }
        (d / n).sqrt()
    }

    /// One GPU follows the CPU f64 AP (object and probe updates, LLK of every cycle) to f32
    /// rounding; the run is split in two calls to check the history's cycle numbers.
    #[test]
    fn ap_follows_the_cpu_reference() {
        let (scan, mut cpu) = case(2, |px| sim::fermat(100, 2.5 * px));
        let cfg = ApConfig {
            scheme: Scheme::Sectors { centre: None },
            stack: 8,
            obj_inertia: Some(0.01),
            probe_inertia: Some(0.001),
            calc_llk: 3,
        };
        let mut gpu = DistAp::new(&gpus(1), problem(&scan, &cpu), cfg).unwrap();
        gpu.run(4).unwrap();
        gpu.run(6).unwrap();
        let params = ApParams {
            obj_inertia: Some(0.01),
            probe_inertia: Some(0.001),
            calc_llk: 3,
            zero_phase_ramp: false,
        };
        cpu.ap(&params, 4);
        cpu.ap(&params, 6);
        let nb_obs = cpu.observed().nb_obs() as f64;
        let h = cpu.history();
        assert_eq!(
            gpu.history().iter().map(|r| r.0).collect::<Vec<_>>(),
            h.iter().map(|r| r.cycle).collect::<Vec<_>>()
        );
        for ((c, g), r) in gpu.history().iter().zip(h) {
            let e = (g.poisson / nb_obs - r.llk.poisson).abs() / r.llk.poisson;
            let i = (g.icalc - r.llk.nb_photons_calc).abs() / r.llk.nb_photons_calc;
            eprintln!("cycle {c}: poisson rel {e:.2e}, icalc rel {i:.2e}");
            assert!(e < 1e-4 && i < 1e-4, "cycle {c}");
        }
        let o = rel_l2(&gpu.object().unwrap(), cpu.object().as_slice());
        let p = rel_l2(&gpu.probe().unwrap(), cpu.probe().as_slice());
        eprintln!("object rel L2 {o:.2e}, probe rel L2 {p:.2e}");
        assert!(o < 1e-4 && p < 1e-4);
    }
}
