//! PyNX AP on 1, 2 or 4 GPUs, bit-identical whatever the GPU count (DESIGN §7 M4).
//!
//! The scan is cut once into [`NLEAF`] leaf regions and a run on `g` GPUs merges them
//! ([`Partition::merge`]), so every order fixed per leaf is the same on 1, 2 and 4 GPUs:
//! a region computes the exit waves of its own leaves' frames and recomputes those of the other
//! leaves' frames that touch its tiles, in stacks of `stack` frames per class and leaf, the
//! interior frames (reading only their own leaf's tiles, so no halo on any GPU count) before
//! the border frames; each object pixel adds its views in (class, leaf, frame) order; the probe
//! sums are one per leaf, added through a fixed pair tree; the regulariser's `max(den)` is an
//! exact max over regions; the LLK is [`LlkSums::from_partials`] over frames by global index.
//! A recomputed exit wave has its owner's bits: the halo holds the owner's object under the
//! whole footprint, the probe and counts are the same, and [`Kernels::fused_project`]
//! transforms each frame in its own blocks, whatever else its stack holds.
//!
//! Per cycle, with [`Gpu::fence`] between the steps: A per stack one fused pass (exit waves,
//! LLK, projection, an own stack's leaf probe sums, object sums), the border stacks only once
//! the halo has landed, then the probe subtree and the object sums' max; B the other regions'
//! max words pulled by copy engine, the object update, then each owner packs the updated rows
//! its neighbours read, while the side stream pulls the other regions' subtree roots and adds
//! the pair tree above them, then the probe update; C each receiver pulls the packed rows by
//! copy engine and unpacks its halo on its side stream, under the next cycle's interior stacks.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Context, DeviceBuffer, Stream};
use ptyrs_core::{AddrTable, View};
use ptyrs_gpu::{
    C32, Exit, FarSums, FusedBufs, FusedInputs, FusedSums, Kernels, LlkPartial, LlkSums, Model,
    StackTable, UpdateSums, ViewAddr, ViewDims, Views, view_tiles,
};

use crate::exchange::{self, Pull, Run, Runs};
use crate::{DEFAULT_TILE, DistError, Exchange, Gpu, Partition, Scheme, TileGrid, Workers};

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
    /// Frames per stack, the fused passes' batch.
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

/// The part of the object a region's GPU holds: the bounding box of its own tiles and the
/// halo tiles it reads, in whole tiles, so its kernel tiles are the object's shifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    /// First object pixel `[y, x]`.
    at: [usize; 2],
    /// `[ny, nx]` pixels per object mode.
    shape: [usize; 2],
}

impl Window {
    fn new(grid: &TileGrid, tiles: impl IntoIterator<Item = usize>) -> Self {
        let (mut lo, mut hi) = ([usize::MAX; 2], [0; 2]);
        for t in tiles {
            let c = grid.coords(t);
            for a in 0..2 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a] + 1);
            }
        }
        let (obj, tile) = (grid.obj(), grid.tile());
        let at = lo.map(|l| l * tile);
        Self {
            at,
            shape: [0, 1].map(|a| (hi[a] * tile).min(obj[a]) - at[a]),
        }
    }

    fn len(&self, nobj: usize) -> usize {
        nobj * self.shape[0] * self.shape[1]
    }

    /// Index in the window's `[nobj, ny, nx]` of object pixel `(y, x)` of mode `m`.
    fn index(&self, m: usize, y: usize, x: usize) -> usize {
        let [dy, dx] = [(y, 0), (x, 1)].map(|(p, a)| {
            p.checked_sub(self.at[a])
                .filter(|&d| d < self.shape[a])
                .expect("pixel inside the window")
        });
        (m * self.shape[0] + dy) * self.shape[1] + dx
    }

    /// The window of the `[nobj, nyo, nxo]` object `obj`.
    fn crop(&self, obj: &[C32], [nyo, nxo]: [usize; 2]) -> Vec<C32> {
        let ([y0, x0], [ny, nx]) = (self.at, self.shape);
        let mut out = Vec::with_capacity(self.len(obj.len() / (nyo * nxo)));
        for plane in obj.chunks_exact(nyo * nxo) {
            for y in y0..y0 + ny {
                out.extend_from_slice(&plane[y * nxo + x0..][..nx]);
            }
        }
        out
    }

    /// `v` with offsets into the window.
    fn view(&self, v: &View) -> ViewAddr {
        let shift = |o: u32, a: usize| o - self.at[a] as u32;
        ViewAddr {
            oy: shift(v.oy, 0),
            ox: shift(v.ox, 1),
            ..view_addr(v)
        }
    }
}

/// One region's share, fixed at setup.
struct RegionPlan {
    window: Window,
    /// Over the window.
    dims: ViewDims,
    /// Offsets into the window.
    views: Vec<ViewAddr>,
    /// Global frame of each local frame, in (class, leaf, frame) order.
    global: Vec<u32>,
    stacks: Vec<Stack>,
    /// Leading stacks of interior frames, which read no halo.
    interior: usize,
    /// Stacks of the own leaves, ascending; the others' frames are recomputed.
    own: Vec<usize>,
    /// Halo tiles some local view reads.
    halo_read: Vec<u32>,
}

/// `len` object pixels of mode `m` from `(y, x)` on.
#[derive(Clone)]
struct HaloRow {
    m: usize,
    y: usize,
    x: usize,
    len: usize,
}

/// The object rows region `recv` reads from region `owner`'s tiles. `owner` packs them from
/// word `send_at` of its send buffer on; `recv` pulls them to word `recv_at` of its receive
/// buffer and unpacks them into its window.
struct HaloSeg {
    recv: usize,
    owner: usize,
    rows: Vec<HaloRow>,
    words: u64,
    send_at: u64,
    recv_at: u64,
}

struct Plan {
    run: Partition,
    /// Leaves per region.
    k: usize,
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
        let leaf_of = leaves.owner();
        let mut interior = vec![false; addr.nframe()];
        for (l, frames) in leaf_frames.iter().enumerate() {
            for &f in frames {
                let f = f as usize;
                interior[f] = tiles[f].iter().all(|&t| leaf_of[t] as usize == l);
            }
        }
        let owner = run.owner();
        let mut regions = Vec::with_capacity(ngpu);
        for (r, region) in run.regions().iter().enumerate() {
            let mut zone = vec![0u8; grid.len()];
            region.own.iter().for_each(|&t| zone[t as usize] = 1);
            region.halo.iter().for_each(|&t| zone[t as usize] = 2);
            let mut global = Vec::new();
            let mut stacks = Vec::new();
            let mut own = Vec::new();
            let mut ninterior = 0;
            // Interior frames first; another leaf's interior frames read none of this region's
            // tiles.
            for class in [true, false] {
                for (l, all) in leaf_frames.iter().enumerate() {
                    let mine = l / k == r;
                    let frames: Vec<u32> = all
                        .iter()
                        .copied()
                        .filter(|&f| interior[f as usize] == class)
                        .filter(|&f| {
                            mine || tiles[f as usize].iter().any(|&t| owner[t] as usize == r)
                        })
                        .collect();
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
                if class {
                    ninterior = stacks.len();
                }
            }
            if global.is_empty() {
                return bad(format!("region {r} of {ngpu} has no frames"));
            }
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
            let halo_read: Vec<u32> = (0..grid.len())
                .filter(|&t| read[t])
                .map(|t| t as u32)
                .collect();
            let window = Window::new(
                grid,
                region.own.iter().chain(&halo_read).map(|&t| t as usize),
            );
            regions.push(RegionPlan {
                window,
                dims: ViewDims {
                    nframe: u(global.len())?,
                    obj: [u(window.shape[0])?, u(window.shape[1])?],
                    ..dims
                },
                views: global
                    .iter()
                    .flat_map(|&f| addr.frame_views(f as usize).iter().map(|v| window.view(v)))
                    .collect(),
                halo_read,
                global,
                stacks,
                interior: ninterior,
                own,
            });
        }
        let halo = halo_segs(&run, &regions, dims.nobj as usize);
        Ok(Self {
            run,
            k,
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

    /// Words of region `r`'s halo send and receive buffers.
    fn halo_words(&self, r: usize) -> [u64; 2] {
        let sum = |f: &dyn Fn(&HaloSeg) -> bool| -> u64 {
            self.halo.iter().filter(|s| f(s)).map(|s| s.words).sum()
        };
        [sum(&|s| s.owner == r), sum(&|s| s.recv == r)]
    }

    /// `[byte offset in region r's window, buffer word, words]` of every row of the segments
    /// `pick` selects, laid out in the buffer from word `at(segment)` on.
    fn halo_rows(
        &self,
        r: usize,
        pick: impl Fn(&HaloSeg) -> bool,
        at: fn(&HaloSeg) -> u64,
    ) -> impl Iterator<Item = [u64; 3]> {
        let win = self.regions[r].window;
        self.halo
            .iter()
            .filter(move |s| pick(s))
            .flat_map(move |s| {
                let mut w = at(s);
                s.rows.iter().map(move |row| {
                    let (off, len) = (win.index(row.m, row.y, row.x), row.len as u64);
                    w += len;
                    [(off * size_of::<C32>()) as u64, w - len, len]
                })
            })
    }

    /// Runs copying region `r`'s object rows that others read into its send buffer.
    fn halo_pack(&self, r: usize, obj: u64, send: u64) -> Vec<Run> {
        self.halo_rows(r, |s| s.owner == r, |s| s.send_at)
            .map(|[off, w, len]| Run {
                dst: send + w * 8,
                src: obj + off,
                len,
            })
            .collect()
    }

    /// Runs copying region `r`'s receive buffer into the halo rows of its object.
    fn halo_unpack(&self, r: usize, obj: u64, recv: u64) -> Vec<Run> {
        self.halo_rows(r, |s| s.recv == r, |s| s.recv_at)
            .map(|[off, w, len]| Run {
                dst: obj + off,
                src: recv + w * 8,
                len,
            })
            .collect()
    }

    /// Copy-engine pulls of region `r`'s receive buffer from the owners' send buffers, in
    /// [`peers`] order.
    fn halo_pulls(&self, r: usize, ptrs: &[Ptrs]) -> Vec<Pull> {
        peers(r, ptrs.len())
            .flat_map(|o| {
                self.halo
                    .iter()
                    .filter(move |s| s.recv == r && s.owner == o)
            })
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

/// The regions other than `r` of `n`, from `r + 1` on, wrapping: at each step of pull lists
/// in this order every region reads a different source, where in index order all would read
/// region 0 first and queue on its link.
fn peers(r: usize, n: usize) -> impl Iterator<Item = usize> {
    (1..n).map(move |i| (r + i) % n)
}

/// A view as the kernels take it; integer offsets only.
fn view_addr(v: &View) -> ViewAddr {
    ViewAddr {
        oy: v.oy,
        ox: v.ox,
        fy: 0.0,
        fx: 0.0,
        probe: v.probe,
        obj: v.obj,
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
                let (x, x1) = (tx0 * tile, ((tx1 + 1) * tile).min(nxo));
                for y in ty * tile..((ty + 1) * tile).min(nyo) {
                    rows[o].push(HaloRow {
                        m,
                        y,
                        x,
                        len: x1 - x,
                    });
                }
            }
        }
        let mut recv_at = 0;
        for (o, rows) in rows.into_iter().enumerate() {
            let words: u64 = rows.iter().map(|w| w.len as u64).sum();
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
    halo_send: u64,
    halo_recv: u64,
    max_bits: u64,
    /// [`Tree::top`], when the probe updates.
    tree_top: Option<Node>,
}

/// What a region's exchange reads, built once all regions exist.
struct Links {
    halo: Vec<Pull>,
    /// The other regions' probe subtree roots into [`Tree::pulled`].
    tree: Vec<Pull>,
    /// The other regions' `max_bits` words into [`State::maxes`].
    max: Vec<Pull>,
    /// Every region's `max_bits` word, all local.
    max_srcs: DeviceBuffer<u64>,
}

/// `(num, den)` addresses of one node's probe sums.
type Node = (u64, u64);

/// `dst = a + b` over the probe sums of tree nodes.
struct Add {
    dst: Node,
    a: Node,
    b: Node,
}

impl Add {
    /// # Safety
    /// The nodes must hold `np` probe elements on the stream's device, `a` and `b` written
    /// before this launch.
    unsafe fn run(&self, ex: &Exchange, stream: &Stream, np: u32) -> Result<(), DistError> {
        // SAFETY: as the caller guarantees; `num` holds two floats per element.
        unsafe {
            ex.add_f32(stream, self.dst.0, self.a.0, self.b.0, 2 * np)?;
            ex.add_f32(stream, self.dst.1, self.a.1, self.b.1, np)
        }
    }
}

/// The probe pair tree over the leaf sums, level by level: node `(h, i)` adds nodes
/// `(h - 1, 2i)` and `(h - 1, 2i + 1)` up to the root. A region adds the subtree of its own
/// leaves, pulls the other regions' subtree roots and adds the levels above them, so every
/// region adds the same nodes in the same order on 1, 2 and 4 GPUs.
struct Tree {
    /// Inner nodes and pulled subtree roots.
    _nodes: Vec<UpdateSums>,
    /// Adds of the own subtree, after the own leaf sums.
    own: Vec<Add>,
    /// Adds above the subtree roots, after the pulls.
    upper: Vec<Add>,
    /// The own subtree root, which the other regions pull.
    top: Node,
    /// Where each other region's subtree root is pulled to, by region.
    pulled: Vec<Option<Node>>,
    root: UpdateSums,
}

impl Tree {
    /// Tree of region `r` of `NLEAF / k`, whose own leaf sums are `leaves`.
    fn new(
        stream: &Stream,
        r: usize,
        k: usize,
        np: usize,
        leaves: &[UpdateSums],
    ) -> Result<Self, DistError> {
        let node = |s: &UpdateSums| (s.num.cu_deviceptr(), s.den.cu_deviceptr());
        // Levels inside a region's subtree, and to the root.
        let (h, height) = (k.trailing_zeros(), NLEAF.trailing_zeros());
        let root = UpdateSums::zeroed(stream, np)?;
        let mut nodes = Vec::new();
        let mut alloc = || -> Result<Node, DistError> {
            nodes.push(UpdateSums::zeroed(stream, np)?);
            Ok(node(nodes.last().expect("just pushed")))
        };
        let (mut own, mut upper) = (Vec::new(), Vec::new());
        let (mut top, mut pulled) = (None, vec![None; NLEAF / k]);
        // Nodes of the current level; `None` inside another region's subtree.
        let mut cur: Vec<Option<Node>> = (0..NLEAF)
            .map(|l| (l / k == r).then(|| node(&leaves[l - r * k])))
            .collect();
        for lvl in 0..=height {
            if lvl > 0 {
                let mut next = Vec::with_capacity(cur.len() / 2);
                for pair in cur.chunks(2) {
                    let (Some(a), Some(b)) = (pair[0], pair[1]) else {
                        next.push(None);
                        continue;
                    };
                    let dst = if lvl == height { node(&root) } else { alloc()? };
                    let add = Add { dst, a, b };
                    if lvl <= h { &mut own } else { &mut upper }.push(add);
                    next.push(Some(dst));
                }
                cur = next;
            }
            if lvl == h {
                top = cur[r];
                for (o, c) in cur.iter_mut().enumerate() {
                    if o != r {
                        let p = alloc()?;
                        (*c, pulled[o]) = (Some(p), Some(p));
                    }
                }
            }
        }
        Ok(Self {
            _nodes: nodes,
            own,
            upper,
            top: top.expect("a region holds its subtree root"),
            pulled,
            root,
        })
    }
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
    views: Views,
    table: StackTable,
    counts: DeviceBuffer<u16>,
    obj: DeviceBuffer<C32>,
    halo: Halo,
    probe: DeviceBuffer<C32>,
    bufs: FusedBufs,
    /// Per own stack.
    llk: Vec<DeviceBuffer<LlkPartial>>,
    obj_sums: Option<UpdateSums>,
    /// Per own leaf, when the probe updates.
    leaf_sums: Vec<UpdateSums>,
    tree: Option<Tree>,
    /// Every region's `max_bits` word, the others' pulled.
    maxes: DeviceBuffer<u32>,
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
        let table = StackTable::new(stream, &rp.dims, &ranges)?;
        let np = plan.np();
        let leaf_sums: Vec<UpdateSums> = (0..if plan.probe_inertia.is_some() {
            plan.k
        } else {
            0
        })
            .map(|_| UpdateSums::zeroed(stream, np))
            .collect::<Result<_, _>>()?;
        let tree = plan
            .probe_inertia
            .map(|_| Tree::new(stream, gpu.region(), plan.k, np, &leaf_sums))
            .transpose()?;
        let obj = rp.window.crop(&data.object, plan.run.grid().obj());
        let obj = DeviceBuffer::from_host(stream, &obj)?;
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
            views: Views::new(stream, rp.dims, &rp.views)?,
            bufs: FusedBufs::new(stream, &rp.dims, table.nf_max())?,
            table,
            counts: DeviceBuffer::from_host(stream, &counts)?,
            obj,
            halo,
            probe: DeviceBuffer::from_host(stream, &data.probe)?,
            llk: (0..rp.own.len())
                .map(|_| DeviceBuffer::zeroed(stream, plan.stack))
                .collect::<Result<_, _>>()?,
            obj_sums: plan
                .obj_inertia
                .map(|_| UpdateSums::zeroed(stream, rp.window.len(rp.dims.nobj as usize)))
                .transpose()?,
            leaf_sums,
            tree,
            maxes: DeviceBuffer::zeroed(stream, gpu.nregion())?,
            gmax: DeviceBuffer::zeroed(stream, 1)?,
            links: None,
        })
    }

    fn ptrs(&self) -> Ptrs {
        Ptrs {
            ctx: self.obj.context().clone(),
            halo_send: self.halo.send.cu_deviceptr(),
            halo_recv: self.halo.recv.cu_deviceptr(),
            max_bits: self
                .obj_sums
                .as_ref()
                .map_or(0, |s| s.max_bits.cu_deviceptr()),
            tree_top: self.tree.as_ref().map(|t| t.top),
        }
    }

    /// Region `r`'s exchange with the regions exposing `ptrs`.
    fn links(&self, gpu: &Gpu, plan: &Plan, ptrs: &[Ptrs]) -> Result<Links, DistError> {
        let (r, s) = (gpu.region(), gpu.stream());
        let others = || peers(r, ptrs.len()).map(|o| (o, &ptrs[o]));
        let np = plan.np();
        let tree = match &self.tree {
            Some(t) => others()
                .flat_map(|(o, p)| {
                    let (dst, src) = (t.pulled[o].expect("pulled"), p.tree_top.expect("top"));
                    [
                        (dst.0, src.0, np * size_of::<C32>()),
                        (dst.1, src.1, np * 4),
                    ]
                    .map(|(dst, src, bytes)| Pull {
                        dst,
                        src,
                        bytes,
                        from: p.ctx.clone(),
                    })
                })
                .collect(),
            None => Vec::new(),
        };
        let word = |o: usize| self.maxes.cu_deviceptr() + 4 * o as u64;
        let max = others()
            .map(|(o, p)| Pull {
                dst: word(o),
                src: p.max_bits,
                bytes: 4,
                from: p.ctx.clone(),
            })
            .collect();
        let max_srcs: Vec<u64> = (0..ptrs.len())
            .map(|o| if o == r { ptrs[r].max_bits } else { word(o) })
            .collect();
        Ok(Links {
            halo: plan.halo_pulls(r, ptrs),
            tree,
            max: if self.obj_sums.is_some() {
                max
            } else {
                Vec::new()
            },
            max_srcs: DeviceBuffer::from_host(s, &max_srcs)?,
        })
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
            views,
            table,
            counts,
            obj,
            halo,
            probe,
            bufs,
            llk,
            obj_sums,
            leaf_sums,
            tree,
            maxes: _,
            gmax,
            links,
        } = self;
        let links = links.as_ref().expect("DistAp::new installs the links");
        let np = plan.np() as u32;

        // A: per stack the fused pass, which adds an own stack's probe sums too; the border
        // stacks read the halo, so they wait for the side stream's unpack.
        for s in leaf_sums.iter_mut() {
            ptyrs_gpu::zero(&ctx, &mut s.num)?;
            ptyrs_gpu::zero(&ctx, &mut s.den)?;
        }
        let inputs = FusedInputs {
            views,
            counts,
            obj,
            probe,
            model: Model::default(),
        };
        for (s, st) in rp.stacks.iter().enumerate() {
            if s == rp.interior {
                gpu.join()?;
            }
            let mine = rp.own.binary_search(&s).ok();
            let at = table.at(s);
            let llk = match (due, mine) {
                (true, Some(i)) => Some(&mut llk[i]),
                _ => None,
            };
            let sums = FusedSums {
                obj: obj_sums.as_mut(),
                probe: mine.and_then(|_| leaf_sums.get_mut(st.leaf - r * plan.k)),
                ..Default::default()
            };
            let far = FarSums {
                llk,
                ..Default::default()
            };
            k.fused_project(&ctx, &inputs, at, Exit::Ap { far }, bufs, sums)?;
        }
        if let Some(t) = tree.as_ref() {
            for a in &t.own {
                // SAFETY: the own subtree's nodes are local, its leaves written above.
                unsafe { a.run(ex, &stream, np)? };
            }
        }
        if let Some(sums) = obj_sums.as_mut() {
            k.obj_den_max(&ctx, views, sums)?;
        }
        gpu.fence()?;

        // B: the object update, with the probe pair tree above the subtree roots on the side
        // stream under it; then the probe update.
        let obj_update = obj_sums.as_mut().zip(plan.obj_inertia);
        if obj_update.is_some() {
            // SAFETY: each pull reads a region's `max_bits` word, written before the fence and
            // rewritten only after the next; every source of the max is then local.
            unsafe {
                exchange::pull(&stream, &links.max)?;
                ex.max_u32(&stream, gmax.cu_deviceptr(), &links.max_srcs)?;
            }
        }
        if let Some(t) = tree.as_ref() {
            // Forked after the max words: copies share the copy engines in order, and the
            // tree's 1.5 MB pulls ahead of them would hold back the object update.
            // SAFETY: each pull reads an owner's subtree root, added before the fence and
            // rewritten only after the next; the pulled copies and the upper nodes are local,
            // last read by the previous cycle's adds on this same stream.
            gpu.side(|s| unsafe {
                exchange::pull(s, &links.tree)?;
                t.upper.iter().try_for_each(|a| a.run(ex, s, np))
            })?;
        }
        if let Some((sums, inertia)) = obj_update {
            k.normalize_obj(&ctx, views, obj, sums, gmax, inertia)?;
            // SAFETY: the runs address this region's object and send buffer, which its peers
            // last pulled on their side streams, joined before the fence.
            unsafe { ex.copy_runs(&stream, &halo.pack)? };
        }
        if let (Some(t), Some(inertia)) = (tree.as_mut(), plan.probe_inertia) {
            gpu.join()?;
            k.update_from_sums(&ctx, probe, &mut t.root, np, inertia)?;
        }
        gpu.fence()?;

        // C: halo pull from the owners' send buffers; the unpack overwrites what
        // `normalize_obj` left in the halo tiles it read.
        if obj_sums.is_some() {
            // SAFETY: each pull reads the part of an owner's send buffer packed for this
            // region before the fence; the next cycle joins it before its first fence, after
            // which the owner repacks. The receive buffer and the halo rows of the object are
            // touched by nothing else until that join: the interior stacks read own tiles only.
            gpu.side(|s| unsafe {
                exchange::pull(s, &links.halo)?;
                ex.copy_runs(s, &halo.unpack)
            })?;
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
    /// Frames of its own leaves: exit waves, LLK and probe sums.
    pub computed: usize,
    /// Of those, the interior frames, computed while the halo is in flight.
    pub interior: usize,
    /// Other regions' frames touching its tiles, whose exit waves it recomputes for its object
    /// sums.
    pub recomputed: usize,
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
        let ptrs = Arc::new(workers.run(|_, st| Ok(st.ptrs()))?);
        let p = plan.clone();
        workers.run(move |gpu, st| {
            st.links = Some(st.links(gpu, &p, &ptrs)?);
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
                    interior: rp.stacks[..rp.interior]
                        .iter()
                        .map(|s| s.frames.len())
                        .sum(),
                    recomputed: rp.global.len() - computed,
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
            gpu.synchronize()?;
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
        let nobj = self.plan.dims().nobj as usize;
        let mut out = vec![C32::default(); nobj * nyo * nxo];
        for (t, &o) in self.plan.run.owner().iter().enumerate() {
            let [ty, tx] = grid.coords(t);
            let (x0, x1) = (tx * tile, ((tx + 1) * tile).min(nxo));
            let (win, src) = (&self.plan.regions[o as usize].window, &objs[o as usize]);
            for m in 0..nobj {
                for y in ty * tile..((ty + 1) * tile).min(nyo) {
                    let row = (m * nyo + y) * nxo;
                    let at = win.index(m, y, x0);
                    out[row + x0..row + x1].copy_from_slice(&src[at..at + x1 - x0]);
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

    /// A simulated `frame`x`frame` scan at `positions(pixel)` and its start: unit-modulus object with a uniform random phase in
    /// [-0.5, 0.5) rad and the truth probe, scaled to the observed photons. From a flat object
    /// the far field of the Gaussian probe is f32 rounding noise in pixels with counts, where
    /// the Poisson LLK is infinite; the random phase scatters into every pixel.
    fn case(
        frame: usize,
        nmode: usize,
        positions: impl FnOnce(f64) -> Positions,
    ) -> (Scan, Ptycho) {
        let g = Geometry::new(1.5e-10, 1.0, 75e-6, [frame, frame]).unwrap();
        let pixel = g.pixel_size_object();
        let pos = positions(pixel[1]);
        let shape = required_obj_shape(&pos.pynx_preprocess(pixel), pixel, g.frame, 4);
        let object = sim::object(shape);
        let probe = sim::probe(g.frame, nmode).unwrap();
        let scan = ptyrs_engine::sim::simulate(g, pos, &object, &probe, PHOTONS, None, 7).unwrap();
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        let phase: Vec<C64> = (0..object.as_slice().len())
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                C64::from_polar(1.0, (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
            })
            .collect();
        let start = Object::new(Stack::from_vec(object.dims(), phase).unwrap());
        let mut p = Ptycho::new(&scan, start, probe).unwrap();
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

    #[test]
    fn peers_read_a_different_source_at_each_step() {
        for n in [1, 2, 4] {
            for r in 0..n {
                let mut all: Vec<usize> = peers(r, n).collect();
                all.sort();
                assert_eq!(all, (0..n).filter(|&o| o != r).collect::<Vec<_>>());
            }
            for step in 0..n - 1 {
                let mut src: Vec<usize> = (0..n).map(|r| peers(r, n).nth(step).unwrap()).collect();
                src.sort();
                assert_eq!(src, (0..n).collect::<Vec<_>>(), "n {n} step {step}");
            }
        }
    }

    /// Object, probe and every LLK are the same bits on 1, 2 and 4 GPUs, for a spiral cut in
    /// sectors and a raster cut in strips, with partial last stacks and two probe modes.
    #[test]
    fn ap_is_bit_identical_on_1_2_and_4_gpus() {
        let fermat = case(128, 2, |px| sim::fermat(200, 10.0 * px));
        let raster = case(128, 2, |px| sim::raster(8, 40, 12.0 * px));
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

    /// On the cases above at 2 and 4 GPUs every region's window holds its own tiles and the
    /// halo tiles it reads and is smaller than the object, and some start inside it, so the
    /// bit-identity test runs windows with shifted offsets and their own row strides.
    #[test]
    fn windows_hold_own_and_read_tiles_only() {
        let fermat = case(128, 2, |px| sim::fermat(200, 10.0 * px));
        let raster = case(128, 2, |px| sim::raster(8, 40, 12.0 * px));
        for (scheme, (_, p)) in [
            (Scheme::Sectors { centre: None }, fermat),
            (Scheme::Strips, raster),
        ] {
            let cfg = ApConfig {
                scheme,
                stack: 7,
                obj_inertia: Some(0.01),
                probe_inertia: Some(0.001),
                calc_llk: 1,
            };
            for ngpu in [2, 4] {
                let plan = Plan::new(p.addr(), &cfg, 0.0, ngpu).unwrap();
                let grid = plan.run.grid();
                let (obj, tile) = (grid.obj(), grid.tile());
                let mut shifted = false;
                for (rp, region) in plan.regions.iter().zip(plan.run.regions()) {
                    let w = rp.window;
                    assert!(
                        w.shape[0] * w.shape[1] < obj[0] * obj[1],
                        "{scheme:?} on {ngpu}: {w:?} of {obj:?}"
                    );
                    for &t in region.own.iter().chain(&rp.halo_read) {
                        let [ty, tx] = grid.coords(t as usize);
                        let end = |i: usize, a: usize| ((i + 1) * tile).min(obj[a]) - 1;
                        w.index(0, ty * tile, tx * tile);
                        w.index(0, end(ty, 0), end(tx, 1));
                    }
                    shifted |= w.at != [0, 0];
                }
                assert!(shifted, "{scheme:?} on {ngpu}: every window starts at 0");
            }
        }
    }

    /// A region recomputes its neighbours' border frames in stacks of its own, so a frame's
    /// exit wave, LLK partial and object-sum terms are the owner's only if the fused pass
    /// gives a frame the same bits whatever else its stack holds and wherever in it the frame
    /// sits. Some frames of a scan, in two stacks of their own, against each of them alone
    /// in a stack over the whole scan's views; the object sums add the same frames in the same
    /// order either way. At both frame sides the fused passes take.
    #[test]
    fn fused_bits_do_not_depend_on_the_stack() {
        let ctx = cuda_core::Context::new(gpus(1)[0]).unwrap();
        let stream = ctx.new_stream().unwrap();
        let k = Kernels::load(&ctx).unwrap();
        for frame in [128, 256] {
            fused_stack_case(&stream, &k, frame);
        }
    }

    fn fused_stack_case(stream: &Arc<Stream>, k: &Kernels, frame: usize) {
        let step = frame as f64 / 12.8;
        let (scan, p) = case(frame, 2, |px| sim::fermat(24, step * px));
        let pick: [u32; 8] = [1, 4, 6, 7, 11, 15, 16, 22];
        let addr = p.addr();
        let [ny, nx] = addr.frame_shape();
        let [nyo, nxo] = addr.obj_shape();
        let n = ny * nx;
        let nmode = addr.nmode();
        let obj = c32(sim::object([nyo, nxo]).as_slice());
        let obj = DeviceBuffer::from_host(stream, &obj).unwrap();
        let probe = DeviceBuffer::from_host(stream, &c32(p.probe().as_slice())).unwrap();
        // Per picked frame its exit waves and LLK partial, then the object sums.
        let run = |frames: &[u32], stacks: &[Range<u32>], at: &dyn Fn(u32) -> u32| {
            let dims = ViewDims {
                nframe: frames.len() as u32,
                nmode: nmode as u32,
                nprobe: addr.nprobe() as u32,
                nobj: addr.nobj() as u32,
                frame: [ny as u32, nx as u32],
                obj: [nyo as u32, nxo as u32],
                quad_phase: problem(&scan, &p).quad_phase,
            };
            let va: Vec<ViewAddr> = frames
                .iter()
                .flat_map(|&f| addr.frame_views(f as usize).iter().map(view_addr))
                .collect();
            let counts: Vec<u16> = frames
                .iter()
                .flat_map(|&f| &scan.counts.as_slice()[f as usize * n..][..n])
                .copied()
                .collect();
            let views = Views::new(stream, dims, &va).unwrap();
            let table = StackTable::new(stream, &dims, stacks).unwrap();
            let counts = DeviceBuffer::from_host(stream, &counts).unwrap();
            let mut bufs = FusedBufs::new(stream, &dims, table.nf_max()).unwrap();
            let mut llk = DeviceBuffer::zeroed(stream, table.nf_max() as usize).unwrap();
            let mut sums = UpdateSums::zeroed(stream, obj.len()).unwrap();
            let inputs = FusedInputs {
                views: &views,
                counts: &counts,
                obj: &obj,
                probe: &probe,
                model: Model::default(),
            };
            let mut out = vec![None; pick.len()];
            for (s, r) in stacks.iter().enumerate() {
                let ec = ExecutionContext::new(stream.clone());
                let exit = Exit::Ap {
                    far: FarSums {
                        llk: Some(&mut llk),
                        ..Default::default()
                    },
                };
                let sums = FusedSums {
                    obj: Some(&mut sums),
                    ..Default::default()
                };
                k.fused_project(&ec, &inputs, table.at(s), exit, &mut bufs, sums)
                    .unwrap();
                drop(ec);
                let work = bufs.work().to_host_vec(stream).unwrap();
                let llk = llk.to_host_vec(stream).unwrap();
                for (i, f) in r.clone().enumerate() {
                    let psi = bits(&work[i * nmode * n..][..nmode * n]);
                    let l = [
                        llk[i].poisson,
                        llk[i].gaussian,
                        llk[i].euclidean,
                        llk[i].icalc,
                    ];
                    out[at(f) as usize] = Some((psi, l.map(f32::to_bits)));
                }
            }
            let num = bits(&sums.num.to_host_vec(stream).unwrap());
            let den: Vec<u32> = sums
                .den
                .to_host_vec(stream)
                .unwrap()
                .iter()
                .map(|d| d.to_bits())
                .collect();
            (out, num, den)
        };
        let alone: Vec<Range<u32>> = pick.iter().map(|&f| f..f + 1).collect();
        let pos = |f: u32| pick.iter().position(|&g| g == f).unwrap() as u32;
        let want = run(&(0..24).collect::<Vec<_>>(), &alone, &pos);
        let got = run(&pick, &[0..3, 3..8], &|i| i);
        assert!(want.0.iter().all(Option::is_some));
        assert!(want.2.iter().any(|&d| d != 0), "{frame}: no object sums");
        for (i, (g, w)) in got.0.iter().zip(&want.0).enumerate() {
            assert!(
                g == w,
                "{frame}: frame {}: exit waves or LLK differ",
                pick[i]
            );
        }
        assert!(got.1 == want.1, "{frame}: object sums num differ");
        assert!(got.2 == want.2, "{frame}: object sums den differ");
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
        let (scan, mut cpu) = case(128, 2, |px| sim::fermat(100, 10.0 * px));
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
