//! Live AP over GPUs (DESIGN B2, B3, B6): [`LiveAp`](crate::live::recon::LiveAp) with the
//! object cut by a [`LivePlan`], one region per GPU, each iterating its share of the frames
//! from graphs captured before the first frame lands. Each GPU keeps the counts of its own
//! frames in a [`DeviceStore`] the uploader fills from the ring, routed by the plan
//! ([`DistLiveAp::stores`], [`DistLiveAp::route`]): a frame's counts are on a GPU before
//! its views are pushed, and never on a GPU that does not iterate it.
//!
//! An epoch on each GPU packs the halo rows of its own tiles, pulls and unpacks the owners'
//! rows into its halo after a fence, bins and sweeps its landed frames as one GPU does, takes
//! `max(den)` over its own tiles, and after a second fence the max of every region's word
//! and the update of its own tiles. A region's own tiles bin the frames one GPU bins there,
//! in the same order, and its frames read the owners' pixels of the last update, so its own
//! tiles hold one GPU's object bit for bit after every epoch.
//!
//! With LLK recording each GPU's sweep also writes its frames' LLK partials into its own
//! [`LlkRing`], and [`DistLiveAp::llk`] sums them on the host: each frame once, from the
//! lowest region holding it, in the one-GPU scan's frame order, so the sum is one GPU's.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Context, DeviceBuffer, Event};

use crate::gpu::{
    C32, Ended, FrameLlk, FusedBufs, Kernels, LlkPartial, LlkRing, LlkTotals, ViewAddr,
};
use crate::live::fill::Fill;
use crate::live::graph::LiveGraph;
use crate::live::grow::GrowError;
use crate::live::object::{Snapshot, timed_sleeping_event};
use crate::live::recon::{ApConfig, LLK_SLOTS, ViewPlacement, capture_sweep};
use crate::live::ring::{DeviceStore, STORE_LEAD};
use crate::live::scan::{LiveScan, Refusal, ScanBacker, one_object_mode};
use crate::live::stats::wall_ns;

use crate::dist::exchange::{self, Exchange, Pull};
use crate::dist::halo::{HaloBacker, HaloBufs, peers};
use crate::dist::{DistError, Gpu, LivePlan, LiveRegion, LiveRoute, Scheme, Share, Workers};

/// One region's GPU.
struct State {
    // The graphs drop before the buffers and kernels they replay on.
    pack: LiveGraph,
    unpack: LiveGraph,
    ingest: LiveGraph,
    sweep: LiveGraph,
    den: LiveGraph,
    update: LiveGraph,
    llk: Option<RegionLlk>,
    halo: HaloBufs,
    halo_backer: HaloBacker,
    scan: LiveScan,
    scan_backer: ScanBacker,
    /// How long backing the region's planned views took.
    plan_time: Duration,
    /// The counts of the region's frames in landing order, which the uploader fills.
    store: Arc<DeviceStore>,
    counts_view: DeviceBuffer<u16>,
    _kernels: Kernels,
    ex: Exchange,
    _probe: DeviceBuffer<C32>,
    _fused: FusedBufs,
    /// The other regions' `max(den)` words, pulled.
    maxes: DeviceBuffer<u32>,
    /// The max over every region's word, which the update graph reads.
    gmax: DeviceBuffer<u32>,
    links: Option<Links>,
    /// [`DistLiveAp::object`]'s copy.
    snap: Snapshot,
    /// The copies of the [`DistSnapshot`]s, by id.
    snaps: Vec<Snapshot>,
    /// Recorded after the epochs of each parity; `anchor` at wall ns `t_anchor`.
    fence: [Event; 2],
    anchor: Event,
    t_anchor: i64,
    /// Frames the ingest graph has binned.
    binned: u32,
    /// Stacks per sweep replay.
    stacks: u32,
}

/// A region's record of each epoch's LLK over the frames it holds.
struct RegionLlk {
    ring: LlkRing<u64>,
    _partials: DeviceBuffer<LlkPartial>,
    /// The update graph ends a record each replay.
    ended: Ended,
}

/// Which region's partials count for each frame of the scan, and the records not yet read.
#[derive(Default)]
struct Counted {
    /// Per region, `(its frame, the scan's frame)` of the frames it is the lowest region
    /// to hold, ascending: every frame taken is in exactly one list.
    frames: Vec<Vec<(u32, u32)>>,
    /// `(epoch, frames taken when it was queued)` of the epochs queued since the last
    /// [`DistLiveAp::llk`].
    queued: Vec<(u64, u32)>,
}

/// What a region's GPU exposes to the others' exchange.
struct Ptrs {
    ctx: Arc<Context>,
    /// Its halo send and receive buffers.
    halo: [u64; 2],
    max_bits: u64,
}

/// What a region's exchange reads, fixed for the whole scan.
struct Links {
    halo: Vec<Pull>,
    /// The other regions' `max_bits` words into [`State::maxes`].
    max: Vec<Pull>,
    /// Every region's `max(den)` word, all local.
    max_srcs: DeviceBuffer<u64>,
}

impl State {
    fn new(
        gpu: &Gpu,
        region: &LiveRegion,
        cfg: &ApConfig,
        probe: &[C32],
        store: Option<Arc<DeviceStore>>,
        llk: bool,
    ) -> Result<Self, DistError> {
        let stream = gpu.stream().clone();
        let kernels = Kernels::load(gpu.ctx())?;
        let ex = Exchange::load(gpu.ctx())?;
        let (nf_max, chunk) = (cfg.nf_max, cfg.chunk);
        let mut scan = LiveScan::owned(
            &stream,
            &region.shape,
            nf_max,
            cfg.cap,
            cfg.fill,
            chunk,
            &region.own,
        )?;
        let d = *scan.dims();
        let n = (d.frame[0] * d.frame[1]) as usize;
        let mut scan_backer = scan.backer();
        let t = Instant::now();
        scan_backer.plan(&region.planned)?;
        let plan_time = t.elapsed();
        let store = match store {
            Some(s)
                if s.frame_elems() == n
                    && s.max_frames() >= u64::from(d.nframe)
                    && s.ctx().ordinal() == gpu.ctx().ordinal() =>
            {
                s
            }
            Some(s) => {
                return Err(DistError::Partition(format!(
                    "a store of {} {}-pixel frames on GPU {} for region {}: {} {n}-pixel \
                     frames on GPU {}",
                    s.max_frames(),
                    s.frame_elems(),
                    s.ctx().ordinal(),
                    gpu.region(),
                    d.nframe,
                    gpu.ctx().ordinal()
                )));
            }
            None => Arc::new(DeviceStore::growing(
                gpu.ctx(),
                d.nframe as usize,
                n,
                region.frames.len(),
                STORE_LEAD,
            )?),
        };
        // SAFETY: the fused pass reads the counts of landed frames only, each in the store
        // before its views are pushed ([`DistLiveAp::push`]).
        let counts_view = unsafe { store.frames_view()? };
        let halo = region.halo_bufs(&stream, scan.obj().ptr(), chunk)?;
        let stacks = cfg.stacks.get().min(d.nframe.div_ceil(nf_max.get()));
        let probe = DeviceBuffer::from_host(&stream, probe)?;
        let mut fused = FusedBufs::new(&stream, &d, nf_max.get())?;
        let gmax = DeviceBuffer::zeroed(&stream, 1)?;
        let (k, ex_ref) = (&kernels, &ex);

        let (pack, unpack) = {
            let (win, halo) = (scan.obj().pixels(), &halo);
            // SAFETY: the window is the scan's object, at the address the halo took, live as
            // long as the graphs and leased in them. Each epoch packs after the last update
            // and before the fence its peers pull after, and unpacks after its own pulls,
            // on this stream.
            let pack = LiveGraph::capture(&stream, |c| unsafe {
                crate::gpu::lease(c, &[win.storage()], &[])?;
                halo.pack(ex_ref, c)
            })?;
            let unpack = LiveGraph::capture(&stream, |c| unsafe {
                crate::gpu::lease(c, &[], &[win.storage()])?;
                halo.unpack(ex_ref, c)
            })?;
            (pack, unpack)
        };
        let ingest = {
            let s = &mut scan;
            LiveGraph::capture(&stream, move |c| s.advance(k, c))?
        };
        let mut bufs = if llk {
            Some((
                LlkRing::new(&kernels, &stream, &d, LLK_SLOTS)?,
                DeviceBuffer::zeroed(&stream, nf_max.get() as usize)?,
            ))
        } else {
            None
        };
        // One writer for every record: the sweep graph pushes through it and the update
        // graph ends it.
        let mut rec = bufs
            .as_mut()
            .map(|(ring, partials)| (ring.device().record(), partials));
        let sweep = {
            let llk = rec.as_mut().map(|(w, partials)| (w, &mut **partials));
            capture_sweep(&mut scan, k, stacks, &probe, &counts_view, &mut fused, llk)?
        };
        let den = {
            let s = &mut scan;
            LiveGraph::capture(&stream, move |c| {
                let p = s.parts_mut();
                k.obj_den_max(c, p.views, p.sums)
            })?
        };
        let mut ended = None;
        let update = {
            let (s, gmax, inertia, ended) = (&mut scan, &gmax, cfg.inertia, &mut ended);
            let w = rec.map(|(w, _)| w);
            LiveGraph::capture(&stream, move |c| {
                let p = s.parts_mut();
                k.normalize_obj(c, p.views, p.pixels, p.sums, gmax, inertia)?;
                p.frames.rewind(c)?;
                if let Some(w) = w {
                    *ended = Some(w.end(c)?);
                }
                Ok(())
            })?
        };
        let llk = bufs.map(|(ring, partials)| RegionLlk {
            ring,
            _partials: partials,
            ended: ended.expect("the update capture ends the record"),
        });
        let snap = Snapshot::new(scan.obj())?;
        Ok(Self {
            pack,
            unpack,
            ingest,
            sweep,
            den,
            update,
            llk,
            halo_backer: halo.backer(),
            halo,
            scan,
            scan_backer,
            plan_time,
            store,
            counts_view,
            _kernels: kernels,
            ex,
            _probe: probe,
            _fused: fused,
            maxes: DeviceBuffer::zeroed(&stream, gpu.nregion())?,
            gmax,
            links: None,
            snap,
            snaps: Vec::new(),
            fence: [
                timed_sleeping_event(gpu.ctx())?,
                timed_sleeping_event(gpu.ctx())?,
            ],
            anchor: gpu.ctx().new_event()?,
            t_anchor: 0,
            binned: 0,
            stacks,
        })
    }

    fn ptrs(&self, gpu: &Gpu) -> Ptrs {
        Ptrs {
            ctx: gpu.ctx().clone(),
            halo: self.halo.ptrs(),
            max_bits: self.scan.obj().sums().max_bits.cu_deviceptr(),
        }
    }

    /// Region `r`'s links, its halo pulls `halo` from the plan.
    fn links(&self, gpu: &Gpu, ptrs: &[Ptrs], halo: Vec<Pull>) -> Result<Links, DistError> {
        let r = gpu.region();
        let word = |o: usize| self.maxes.cu_deviceptr() + 4 * o as u64;
        let max = peers(r, ptrs.len())
            .map(|o| Pull {
                dst: word(o),
                src: ptrs[o].max_bits,
                bytes: 4,
                from: ptrs[o].ctx.clone(),
            })
            .collect();
        let max_srcs: Vec<u64> = (0..ptrs.len())
            .map(|o| if o == r { ptrs[r].max_bits } else { word(o) })
            .collect();
        Ok(Links {
            halo,
            max,
            max_srcs: DeviceBuffer::from_host(gpu.stream(), &max_srcs)?,
        })
    }

    /// Lands `share`, the region's part of a batch; returns the frames landed so far and
    /// the halo rows listed.
    fn push(&mut self, share: &Share) -> Result<(u32, [usize; 2]), DistError> {
        self.halo_backer.back(&share.halo)?;
        self.halo.push(&share.halo)?;
        self.scan_backer.back(&share.views)?;
        let landed = self.scan.push(&share.views)?;
        if landed as usize != share.frames.len() {
            return Err(DistError::Partition(format!(
                "the region's scan landed {landed} of the {} frames the plan gave it",
                share.frames.len()
            )));
        }
        Ok((self.scan.landed(), self.halo.rows()))
    }

    /// Stamps the anchor the epoch ends are timed from on the idle stream; returns how
    /// long stamping it took, ns.
    fn anchor(&mut self, gpu: &Gpu) -> Result<i64, DistError> {
        gpu.synchronize()?;
        let t0 = wall_ns();
        self.anchor.record(gpu.stream())?;
        self.anchor.synchronize()?;
        self.t_anchor = wall_ns();
        Ok(self.t_anchor - t0)
    }

    /// Wall ns this GPU finished the epoch queued last with `parity`; waits for it.
    fn end_of(&self, parity: usize) -> Result<i64, DistError> {
        let f = &self.fence[parity];
        f.synchronize()?;
        Ok(self.t_anchor + (f64::from(self.anchor.elapsed_time(f)?) * 1e6) as i64)
    }

    /// Queues epoch `epoch` over the frames landed so far, two fences, and after it the
    /// event of its parity.
    fn epoch(&mut self, gpu: &mut Gpu, epoch: u64) -> Result<(), DistError> {
        let links = self
            .links
            .as_ref()
            .expect("DistLiveAp::new installs the links");
        let stream = gpu.stream().clone();
        self.pack.replay()?;
        gpu.fence()?;
        // SAFETY: each pull reads the part of an owner's send buffer it packed before the
        // fence, packed again only after the next one; the receive buffer is this GPU's own.
        unsafe { exchange::pull(&stream, &links.halo)? };
        self.unpack.replay()?;
        let (landed, nf) = (self.scan.landed(), self.scan.frames().nf_max());
        let binned = self.binned;
        while self.binned < landed {
            self.ingest.replay()?;
            self.binned = (self.binned + nf).min(landed);
        }
        let ctx = ExecutionContext::new(self.scan.stream().clone());
        if let Some(l) = &mut self.llk {
            l.ring
                .device()
                .count(&ctx, binned..landed, &self.counts_view)?;
        }
        for _ in 0..landed.div_ceil(nf).div_ceil(self.stacks) {
            self.sweep.replay()?;
        }
        self.den.replay()?;
        gpu.fence()?;
        // SAFETY: each pull reads a region's `max_bits` word, written before the fence and
        // rewritten only after the next epoch's first fence; every source of the max is then
        // local, and the update graph reads `gmax` after it on this stream.
        unsafe {
            exchange::pull(&stream, &links.max)?;
            self.ex
                .max_u32(&stream, self.gmax.cu_deviceptr(), &links.max_srcs)?;
        }
        self.update.replay()?;
        if let Some(l) = &mut self.llk {
            l.ring
                .enqueued(&ctx, &l.ended, epoch, landed)
                .map_err(GrowError::from)?;
        }
        Ok(self.fence[(epoch % 2) as usize].record(&stream)?)
    }

    /// Every record counted since the last call, each as the region's frames; waits for
    /// them.
    fn llk(&mut self) -> Result<Vec<(u64, Vec<FrameLlk>)>, DistError> {
        let l = self.llk.as_mut().expect("the run records LLK on every GPU");
        let ctx = ExecutionContext::new(self.scan.stream().clone());
        Ok(l.ring.flush_frames(&ctx).map_err(GrowError::from)?)
    }
}

/// A host copy of a [`DistLiveAp`]'s object, queued by [`DistLiveAp::snapshot`] and read
/// through [`DistLiveAp::wait`].
pub struct DistSnapshot {
    id: usize,
    pixels: Vec<[f32; 2]>,
}

/// A live AP run over one GPU per region of a [`LivePlan`].
pub struct DistLiveAp {
    workers: Workers<State>,
    plan: LivePlan,
    /// Per region: the counts of its frames.
    stores: Vec<Arc<DeviceStore>>,
    /// Per region: frames landed.
    landed: Vec<u32>,
    /// Per region: halo rows listed, `[pack, unpack]`.
    halo_rows: Vec<[usize; 2]>,
    lost: u64,
    taken: u64,
    epochs: u64,
    clock_ns: i64,
    /// With LLK recording.
    llk: Option<Counted>,
    /// A push or epoch failed part-way: the GPUs may disagree, so every later call fails.
    failed: bool,
}

impl DistLiveAp {
    /// An empty scan of `cfg.shape` split by `scheme` over `ordinals` (1, 2 or 4 GPUs, region
    /// `r` on `ordinals[r]`) from its first frames' views `planned`, with the fixed `probe`
    /// (`[nprobe, ny, nx]`); backs the plan and captures every graph, which record each
    /// epoch's LLK with `llk` ([`Self::llk`]). Each GPU's store backs its planned frames and
    /// [`STORE_LEAD`] more before the first frame lands, the rest as the frames come.
    pub fn new(
        ordinals: &[usize],
        cfg: &ApConfig,
        planned: &[ViewAddr],
        scheme: Scheme,
        probe: &[C32],
        llk: bool,
    ) -> Result<Self, DistError> {
        Self::build(ordinals, cfg, planned, scheme, probe, None, llk)
    }

    /// [`Self::new`] on the GPUs of `stores`, region `r` reading its frames' counts from
    /// `stores[r]`: the stores of an earlier run of the same plan, whose frames the caller
    /// pushes again in the order they landed.
    pub fn on_stores(
        stores: &[Arc<DeviceStore>],
        cfg: &ApConfig,
        planned: &[ViewAddr],
        scheme: Scheme,
        probe: &[C32],
        llk: bool,
    ) -> Result<Self, DistError> {
        let ordinals: Vec<usize> = stores.iter().map(|s| s.ctx().ordinal()).collect();
        let stores = Some(Arc::new(stores.to_vec()));
        Self::build(&ordinals, cfg, planned, scheme, probe, stores, llk)
    }

    fn build(
        ordinals: &[usize],
        cfg: &ApConfig,
        planned: &[ViewAddr],
        scheme: Scheme,
        probe: &[C32],
        stores: Option<Arc<Vec<Arc<DeviceStore>>>>,
        llk: bool,
    ) -> Result<Self, DistError> {
        let plan = LivePlan::new(&cfg.shape, planned, scheme, ordinals.len(), cfg.cap)?;
        let regions: Arc<Vec<LiveRegion>> = Arc::new(plan.regions().to_vec());
        let (cfg, probe) = (*cfg, Arc::new(probe.to_vec()));
        let workers = Workers::spawn(ordinals, move |gpu| {
            let store = stores.as_ref().map(|s| s[gpu.region()].clone());
            State::new(gpu, &regions[gpu.region()], &cfg, &probe, store, llk)
        })?;
        let ptrs = workers.run(|gpu, st| Ok(st.ptrs(gpu)))?;
        let halo: Vec<Vec<Pull>> = (0..ptrs.len())
            .map(|r| {
                plan.halo_pulls(r, ptrs[r].halo[1], |o| {
                    (ptrs[o].ctx.clone(), ptrs[o].halo[0])
                })
            })
            .collect();
        let (ptrs, halo) = (Arc::new(ptrs), Arc::new(halo));
        workers.run(move |gpu, st| {
            st.links = Some(st.links(gpu, &ptrs, halo[gpu.region()].clone())?);
            Ok(())
        })?;
        let stores = workers.run(|_, st| Ok(st.store.clone()))?;
        let clock_ns = workers.run(|gpu, st| st.anchor(gpu))?.into_iter().max();
        let nregion = plan.nregion();
        Ok(Self {
            workers,
            plan,
            stores,
            landed: vec![0; nregion],
            halo_rows: vec![[0; 2]; nregion],
            lost: 0,
            taken: 0,
            epochs: 0,
            clock_ns: clock_ns.unwrap_or(0),
            llk: llk.then(|| Counted {
                frames: vec![Vec::new(); nregion],
                queued: Vec::new(),
            }),
            failed: false,
        })
    }

    fn check(&self) -> Result<(), DistError> {
        if self.failed {
            return Err(DistError::Aborted);
        }
        Ok(())
    }

    /// Per region, the store of its frames' counts, for the uploader.
    pub fn stores(&self) -> Vec<Arc<DeviceStore>> {
        self.stores.clone()
    }

    /// The regions the uploader lands a frame on, positions placed by `place` as the views
    /// pushed are ([`LivePlan::route`]).
    pub fn route(&self, place: ViewPlacement) -> LiveRoute {
        self.plan.route(place)
    }

    /// Lands the frames whose views are `views`, `nmode` per frame, on every GPU that owns a
    /// tile they touch, up to the first one the one-GPU scan refuses; returns how many it
    /// took ([`LivePlan::split`]). The uploader, routing by [`Self::route`], must have
    /// landed their counts in [`Self::stores`], in this order and with no frame left out:
    /// they are the frames of the batches it sent.
    pub fn push(&mut self, views: &[ViewAddr]) -> Result<u32, DistError> {
        self.check()?;
        let nmode = self.plan.regions()[0].shape.nmode as usize;
        let mut shares = vec![Share::default(); self.plan.nregion()];
        let took = self.plan.split(views, &mut shares)?;
        self.lost += (views.len() / nmode) as u64 - u64::from(took);
        if let Some(c) = &mut self.llk {
            let mut counted = vec![false; took as usize];
            for (r, share) in shares.iter().enumerate() {
                for (at, &f) in (self.landed[r]..).zip(&share.frames) {
                    if !std::mem::replace(&mut counted[f as usize], true) {
                        c.frames[r].push((at, self.taken as u32 + f));
                    }
                }
            }
        }
        self.taken += u64::from(took);
        self.failed = true;
        let got = self.workers.run(|gpu, st| st.push(&shares[gpu.region()]))?;
        self.failed = false;
        for (r, (landed, rows)) in got.into_iter().enumerate() {
            self.landed[r] = landed;
            self.halo_rows[r] = rows;
        }
        Ok(took)
    }

    /// Queues `fill` onto every GPU's window behind the epochs queued so far
    /// ([`LiveScan::fill`]), halo tiles included, so the next epoch starts from it on every
    /// GPU; returns the fill's pixels their owner's GPU had not ready, each once: 0 for the
    /// fill of a batch [`Self::push`] has landed whole, as every tile a frame touches is
    /// readied on its owner. Each GPU queues an epoch whole within one [`Self::epoch`], on
    /// the stream its fills are queued on, so a fill never lands inside one.
    pub fn fill(&mut self, fill: Fill) -> Result<usize, DistError> {
        self.check()?;
        let regions = self.plan.regions();
        one_object_mode(&regions[0].shape.dims())?;
        self.failed = true;
        let skipped = self.workers.run(|gpu, st| {
            let origin = regions[gpu.region()].window.at;
            Ok(st.scan.fill(fill.clone(), origin)?)
        })?;
        self.failed = false;
        Ok(skipped.into_iter().sum())
    }

    /// Queues one epoch over every frame landed so far on every GPU, without waiting for it.
    pub fn epoch(&mut self) -> Result<(), DistError> {
        self.check()?;
        self.failed = true;
        let epoch = self.epochs;
        self.workers.run(|gpu, st| st.epoch(gpu, epoch))?;
        self.failed = false;
        if let Some(c) = &mut self.llk {
            c.queued.push((epoch, self.taken as u32));
        }
        self.epochs += 1;
        Ok(())
    }

    /// `(e, llk)` for each epoch queued since the last call, oldest first, as
    /// [`LiveAp::llk`](crate::live::recon::LiveAp::llk) gives them on one GPU, bit for
    /// bit. Waits for those epochs. `None` unless the run was asked to record it.
    pub fn llk(&mut self) -> Result<Option<Vec<(u64, f64)>>, DistError> {
        self.check()?;
        if self.llk.is_none() {
            return Ok(None);
        }
        let recs = self.workers.run(|_, st| st.llk())?;
        let c = self.llk.as_mut().expect("checked above");
        let none = FrameLlk {
            llk: LlkPartial::default(),
            nobs: 0,
        };
        let mut out = Vec::with_capacity(c.queued.len());
        for (k, (e, taken)) in c.queued.drain(..).enumerate() {
            let mut all = vec![none; taken as usize];
            for (own, recs) in c.frames.iter().zip(&recs) {
                let (tag, held) = &recs[k];
                assert_eq!(*tag, e, "every GPU records every epoch");
                for &(at, f) in own.iter().take_while(|&&(_, f)| f < taken) {
                    all[f as usize] = held[at as usize];
                }
            }
            let t = LlkTotals::of(&all);
            out.push((e, t.sums.poisson / t.nb_obs as f64));
        }
        Ok(Some(out))
    }

    /// Wall ns the slowest GPU finished `epoch`, one of the last two queued; waits for it.
    pub fn end_of(&self, epoch: u64) -> Result<i64, DistError> {
        assert!(epoch < self.epochs && epoch + 2 >= self.epochs);
        let parity = (epoch % 2) as usize;
        let ends = self.workers.run(|_, st| st.end_of(parity))?;
        Ok(ends.into_iter().max().expect("a run has a GPU"))
    }

    /// The longest any GPU took to stamp the clock [`Self::end_of`] reads: its error bound.
    pub fn clock_ns(&self) -> i64 {
        self.clock_ns
    }

    /// Waits for every epoch queued so far.
    pub fn synchronize(&self) -> Result<(), DistError> {
        self.workers.run(|gpu, _| gpu.synchronize())?;
        Ok(())
    }

    /// The object `[nobj, nyo, nxo]` of the one-GPU scan after the epochs queued so far, each
    /// tile from its owner; waits for them.
    pub fn object(&self) -> Result<Vec<[f32; 2]>, DistError> {
        self.check()?;
        let wins = self.workers.run(|_, st| {
            st.scan.obj().snapshot(&mut st.snap)?;
            Ok(st.snap.wait()?.to_vec())
        })?;
        let mut out = vec![[0.0; 2]; self.pixels()];
        self.stitch(&wins, &mut out);
        Ok(out)
    }

    /// A host copy to queue with [`Self::snapshot`].
    pub fn new_snapshot(&mut self) -> Result<DistSnapshot, DistError> {
        let ids = self.workers.run(|_, st| {
            st.snaps.push(Snapshot::new(st.scan.obj())?);
            Ok(st.snaps.len() - 1)
        })?;
        Ok(DistSnapshot {
            id: ids[0],
            pixels: vec![[0.0; 2]; self.pixels()],
        })
    }

    /// Queues a copy of the object after the epochs queued so far into `snap`, without
    /// waiting for it.
    pub fn snapshot(&self, snap: &mut DistSnapshot) -> Result<(), DistError> {
        self.check()?;
        let id = snap.id;
        self.workers
            .run(|_, st| Ok(st.scan.obj().snapshot(&mut st.snaps[id])?))?;
        Ok(())
    }

    /// Waits for the copy queued last into `snap`: the object `[nobj, nyo, nxo]` of the
    /// one-GPU scan, each tile from its owner.
    pub fn wait<'a>(&self, snap: &'a mut DistSnapshot) -> Result<&'a [[f32; 2]], DistError> {
        let id = snap.id;
        let wins = self
            .workers
            .run(|_, st| Ok(st.snaps[id].wait()?.to_vec()))?;
        self.stitch(&wins, &mut snap.pixels);
        Ok(&snap.pixels)
    }

    /// Pixels of the object, `nobj * nyo * nxo`.
    fn pixels(&self) -> usize {
        let [nyo, nxo] = self.shape();
        self.plan.regions()[0].shape.nobj as usize * nyo * nxo
    }

    /// Copies each tile from its owner's window in `wins`, by region, into `out`.
    fn stitch(&self, wins: &[Vec<[f32; 2]>], out: &mut [[f32; 2]]) {
        let grid = self.plan.partition().grid();
        let ([nyo, nxo], tile) = (grid.obj(), grid.tile());
        let nobj = self.plan.regions()[0].shape.nobj as usize;
        for (t, &o) in self.plan.partition().owner().iter().enumerate() {
            let [ty, tx] = grid.coords(t);
            let (x0, x1) = (tx * tile, ((tx + 1) * tile).min(nxo));
            let (win, src) = (&self.plan.regions()[o as usize].window, &wins[o as usize]);
            for m in 0..nobj {
                for y in ty * tile..((ty + 1) * tile).min(nyo) {
                    let row = (m * nyo + y) * nxo;
                    let at = win.index(m, y, x0);
                    out[row + x0..row + x1].copy_from_slice(&src[at..at + x1 - x0]);
                }
            }
        }
    }

    /// `[nyo, nxo]` of the object.
    pub fn shape(&self) -> [usize; 2] {
        self.plan.partition().grid().obj()
    }

    /// Object bytes mapped over every GPU, halos included.
    pub fn mapped_bytes(&self) -> Result<usize, DistError> {
        let b = self.workers.run(|_, st| Ok(st.scan.obj().mapped_bytes()))?;
        Ok(b.into_iter().sum())
    }

    /// How long backing the planned views took, summed over the GPUs, as
    /// [`Self::mapped_bytes`] sums their bytes; they back theirs at once.
    pub fn plan_time(&self) -> Result<Duration, DistError> {
        let t = self.workers.run(|_, st| Ok(st.plan_time))?;
        Ok(t.into_iter().sum())
    }

    pub fn ngpu(&self) -> usize {
        self.workers.len()
    }

    pub fn plan(&self) -> &LivePlan {
        &self.plan
    }

    /// Per region, the frames landed on it.
    pub fn landed(&self) -> &[u32] {
        &self.landed
    }

    /// Per region, the halo rows it packs and unpacks, `[pack, unpack]`.
    pub fn halo_rows(&self) -> &[[usize; 2]] {
        &self.halo_rows
    }

    /// Frames the scan took.
    pub fn taken(&self) -> u64 {
        self.taken
    }

    /// Frames refused or pushed after the plan closed.
    pub fn lost(&self) -> u64 {
        self.lost
    }

    /// The frame that closed the plan, as it closes the one-GPU scan.
    pub fn closed(&self) -> Option<Refusal> {
        self.plan.closed()
    }

    /// Epochs queued so far.
    pub fn epochs(&self) -> u64 {
        self.epochs
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use crossbeam_channel::Receiver;
    use cuda_core::Context;

    use super::*;
    use crate::dist::worker::tests::gpus;
    use crate::live::farfield::{Ptycho, PtychoSpec, ScanPath};
    use crate::live::fill::standin::{Wrong, batch_fill};
    use crate::live::fill::{Fill, PatchFill};
    use crate::live::graph::WarmUp;
    use crate::live::grow::GrowError;
    use crate::live::pos::{Pos, PosTable};
    use crate::live::recon::{LiveAp, ViewPlacement};
    use crate::live::ring::{
        FrameBatch, Landing, Ring, UploadReport, UploaderConfig, run_region_uploader,
    };
    use crate::live::scan::ScanShape;
    use crate::live::sent::sent;

    /// 64 px frames at `points` points 8 px apart.
    fn sim(path: ScanPath, points: usize) -> Ptycho {
        Ptycho::new(PtychoSpec {
            n: 64,
            path,
            points,
            step: 8.0,
            photons: 1e6,
            seed: 5,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        })
    }

    /// A scan landed in the order: the first half of the simulated frames, which is the
    /// plan, copies of every simulated frame moved by `[5, -9]` and `[-13, 11]` px whose views
    /// stay on the object, each with its source's counts, then the other half.
    struct Case {
        cfg: ApConfig,
        place: ViewPlacement,
        /// `(x, y)` m of each frame.
        pos: Vec<(f64, f64)>,
        views: Vec<ViewAddr>,
        counts: Vec<u16>,
        probe: Vec<C32>,
        planned: usize,
        moved: usize,
    }

    fn case(p: &Ptycho) -> Case {
        let pos: Vec<(f64, f64)> = (0..p.points()).map(|k| p.position(k)).collect();
        let dx = p.spec().dx();
        let (place, reach) = ViewPlacement::plan([dx, dx], &pos);
        let views_at = |q: (f64, f64)| {
            let mut v = Vec::new();
            place.views(q, 2, 1, &mut v);
            v
        };
        let mut shape = ScanShape {
            nframe: 0,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [p.spec().n as u32; 2],
            reach,
            quad_phase: 0.0,
        };
        let dims = shape.dims();
        let half = p.points() / 2;
        // Each frame's position and the simulated frame its counts are.
        let mut order: Vec<((f64, f64), usize)> = (0..half).map(|k| (pos[k], k)).collect();
        for d in [[5.0, -9.0], [-13.0, 11.0]] {
            order.extend((0..p.points()).filter_map(|k| {
                let q = (pos[k].0 + d[1] * dx, pos[k].1 + d[0] * dx);
                let on = views_at(q).iter().all(|v| dims.check_view(v).is_ok());
                on.then_some((q, k))
            }));
        }
        let moved = order.len() - half;
        order.extend((half..p.points()).map(|k| (pos[k], k)));
        let pos: Vec<(f64, f64)> = order.iter().map(|&(q, _)| q).collect();
        let views: Vec<ViewAddr> = pos.iter().flat_map(|&q| views_at(q)).collect();
        let counts = order.iter().flat_map(|&(_, k)| p.frame(k)).collect();
        shape.nframe = order.len() as u32;
        let cfg = ApConfig {
            shape,
            nf_max: NonZeroU32::new(8).unwrap(),
            cap: shape.tile_cap(&views),
            fill: [1.0, 0.0],
            chunk: 1,
            inertia: 0.01,
            stacks: NonZeroU32::new(2).unwrap(),
        };
        let probe = p
            .probe()
            .iter()
            .map(|c| C32 { re: c.re, im: c.im })
            .collect();
        Case {
            cfg,
            place,
            pos,
            views,
            counts,
            probe,
            planned: half,
            moved,
        }
    }

    /// Frames, in landing order, of each batch: the plan in uneven batches, then 17 at a time.
    fn batches(c: &Case) -> Vec<Range<usize>> {
        let mut out = Vec::new();
        let mut at = 0;
        for len in [13, 1, 30, 0, c.planned - 44] {
            out.push(at..at + len);
            at += len;
        }
        let nframe = c.cfg.shape.nframe as usize;
        while at < nframe {
            out.push(at..(at + 17).min(nframe));
            at += 17;
        }
        out
    }

    /// The stream of a case's frames into a run's stores: a ring and the uploader routing
    /// by the run's plan.
    struct Feed {
        ring: Arc<Ring>,
        table: Arc<PosTable>,
        stop: Arc<AtomicBool>,
        uploader: JoinHandle<Result<UploadReport, GrowError>>,
        batches: Receiver<FrameBatch>,
    }

    impl Feed {
        fn new(ctx: &Arc<Context>, c: &Case, ap: &DistLiveAp) -> Self {
            let d = c.cfg.shape.dims();
            let n = (d.frame[0] * d.frame[1]) as usize;
            let ring = Arc::new(Ring::new(ctx, 32, n).unwrap());
            let table = Arc::new(PosTable::new(1 << 16));
            let stop = Arc::new(AtomicBool::new(false));
            let (tx, batches) = crossbeam_channel::unbounded();
            let up = UploaderConfig {
                batch: NonZeroUsize::new(8).unwrap(),
                flush_ns: 100_000,
                idle: Duration::from_micros(20),
                events: 4,
                pos_wait_ns: 1_000_000_000,
            };
            let uploader = {
                let (ring, table, stop) = (ring.clone(), table.clone(), stop.clone());
                let landing = Landing::new(ap.stores(), ap.route(c.place));
                std::thread::spawn(move || {
                    run_region_uploader(ring, landing, table, up, stop, Some(tx))
                })
            };
            Self {
                ring,
                table,
                stop,
                uploader,
                batches,
            }
        }

        /// Streams `frames` and returns once the uploader has landed them.
        fn land(mut self, c: &Case, frames: &Range<usize>) -> Self {
            let n = self.ring.slot_elems();
            for k in frames.clone() {
                let (x, y) = c.pos[k];
                self.table.insert(k as i32, Pos::new(x, y, 0));
                let mut s = self.ring.reserve();
                s.frame().copy_from_slice(&c.counts[k * n..(k + 1) * n]);
                s.commit(k as i32, 0, 0);
            }
            let mut at = frames.start;
            while at < frames.end {
                let (b, uploader) = sent(&self.batches, self.uploader);
                self.uploader = uploader;
                assert_eq!(b.f0, at as u64);
                assert_eq!(b.positions, c.pos[at..at + b.n as usize]);
                at += b.n as usize;
            }
            assert_eq!(at, frames.end);
            self
        }

        fn finish(self) -> UploadReport {
            self.stop.store(true, Ordering::Release);
            self.uploader.join().unwrap().unwrap()
        }
    }

    fn bits(o: &[[f32; 2]]) -> Vec<[u32; 2]> {
        o.iter().map(|p| p.map(f32::to_bits)).collect()
    }

    /// A raster in strips and a Fermat spiral in sectors, streamed in batches through the
    /// ring and the uploader into each GPU's store with two epochs after each, from graphs
    /// captured before the first frame: on 1, 2 and 4 GPUs the object is one GPU's bit for
    /// bit after every batch, with no recapture, including after frames moved off the plan
    /// read halo tiles no planned frame read.
    #[test]
    fn gpus_iterate_a_live_scan_as_one_gpu() {
        let ctx = Context::new(0).expect("GPU 0");
        for (name, path, points, scheme) in [
            ("raster", ScanPath::Raster { cols: 12 }, 240, Scheme::Strips),
            (
                "fermat",
                ScanPath::Fermat,
                200,
                Scheme::Sectors { centre: None },
            ),
        ] {
            let c = case(&sim(path, points));
            let batches = batches(&c);
            let nmode = c.cfg.shape.nmode as usize;
            let views = |b: &Range<usize>| &c.views[b.start * nmode..b.end * nmode];
            let st = ctx.new_stream().unwrap();
            let dev = DeviceBuffer::from_host(&st, &c.counts).unwrap();
            let mut one = LiveAp::new(&st, &c.cfg, &c.probe, dev, true).unwrap();
            let (mut snap, mut backer) = (one.new_snapshot().unwrap(), one.backer());
            let mut want = Vec::new();
            for b in &batches {
                let v = views(b);
                backer.back(v).unwrap();
                assert_eq!(one.push(v).unwrap() as usize, b.len(), "{name}");
                for _ in 0..2 {
                    one.epoch().unwrap();
                }
                one.snapshot(&mut snap).unwrap();
                want.push(bits(snap.wait().unwrap()));
            }
            assert_eq!(one.scan().lost(), 0, "{name}");
            let llk_bits = |l: Vec<(u64, f64)>| -> Vec<(u64, u64)> {
                l.into_iter().map(|(e, l)| (e, l.to_bits())).collect()
            };
            let want_llk = llk_bits(one.llk().unwrap().unwrap());
            assert_eq!(want_llk.len(), 2 * batches.len(), "{name}");

            for ngpu in [1, 2, 4] {
                let what = format!("{name} on {ngpu}");
                let planned = &c.views[..c.planned * nmode];
                let mut ap =
                    DistLiveAp::new(&gpus(ngpu), &c.cfg, planned, scheme, &c.probe, true).unwrap();
                let warm = WarmUp::end();
                let mut feed = Feed::new(&ctx, &c, &ap);
                let mut grew = 0;
                for (k, (b, want)) in batches.iter().zip(&want).enumerate() {
                    let v = views(b);
                    let before: Vec<usize> = ap.halo_rows().iter().map(|r| r[1]).collect();
                    feed = feed.land(&c, b);
                    assert_eq!(ap.push(v).unwrap() as usize, b.len(), "{what}");
                    let moved = c.planned..c.planned + c.moved;
                    if b.start >= moved.start && b.end <= moved.end {
                        grew += ap.halo_rows().iter().zip(&before).any(|(r, &b)| r[1] > b) as u32;
                    }
                    for _ in 0..2 {
                        ap.epoch().unwrap();
                    }
                    let got = bits(&ap.object().unwrap());
                    let diff = got.iter().zip(want).filter(|(g, w)| g != w).count();
                    assert_eq!(diff, 0, "{what}: pixels unlike one GPU's after batch {k}");
                }
                let up = feed.finish();
                assert_eq!(
                    (up.frames, up.pos_missing),
                    (u64::from(c.cfg.shape.nframe), 0)
                );
                let placed: Vec<u64> = up.regions.iter().map(|r| r.frames).collect();
                let landed: Vec<u64> = ap.landed().iter().map(|&l| u64::from(l)).collect();
                assert_eq!(placed, landed, "{what}: frames stored and iterated per GPU");
                assert_eq!(warm.recaptures(), 0, "{what}");
                assert_eq!((ap.lost(), ap.closed()), (0, None), "{what}");
                assert_eq!(ap.epochs(), 2 * batches.len() as u64, "{what}");
                let llk = llk_bits(ap.llk().unwrap().unwrap());
                assert_eq!(llk, want_llk, "{what}: LLK unlike one GPU's");
                if ngpu > 1 {
                    assert!(grew > 0, "{what}: no moved frame read a new halo tile");
                }
            }
        }
    }

    /// A raster in strips with a fill after each batch's push and two epochs after it: on
    /// `ngpu` GPUs the fill skips no pixel, and the object is one GPU's bit for bit both
    /// before the batch's first epoch and after its epochs, with fills that lie over tiles
    /// of two GPUs.
    fn fills_give_the_one_gpu_object(ngpu: usize) {
        let ctx = Context::new(0).expect("GPU 0");
        let c = case(&sim(ScanPath::Raster { cols: 12 }, 240));
        let batches = batches(&c);
        let (nmode, n) = (c.cfg.shape.nmode as usize, 64 * 64);
        let views = |b: &Range<usize>| &c.views[b.start * nmode..b.end * nmode];
        let obj = c.cfg.shape.dims().obj.map(|n| n as usize);
        let most = batches.iter().map(Range::len).max().unwrap();
        // Every fill stays out for both runs.
        let mut pf = PatchFill::new(&ctx, obj, [32, 32], most, batches.len()).unwrap();
        let mut wrong = Wrong {
            patch: [32, 32],
            issued: 0,
        };
        let fills: Vec<Fill> = batches
            .iter()
            .map(|b| {
                let counts = &c.counts[b.start * n..b.end * n];
                batch_fill(&mut pf, &mut wrong, counts, views(b), nmode, [64, 64])
            })
            .collect();

        let st = ctx.new_stream().unwrap();
        let dev = DeviceBuffer::from_host(&st, &c.counts).unwrap();
        let mut one = LiveAp::new(&st, &c.cfg, &c.probe, dev, false).unwrap();
        let (mut snap, mut backer) = (one.new_snapshot().unwrap(), one.backer());
        let mut want = Vec::new();
        for (b, fill) in batches.iter().zip(&fills) {
            let v = views(b);
            backer.back(v).unwrap();
            assert_eq!(one.push(v).unwrap() as usize, b.len());
            assert_eq!(one.fill(fill.clone()).unwrap(), 0);
            one.snapshot(&mut snap).unwrap();
            let filled = bits(snap.wait().unwrap());
            for _ in 0..2 {
                one.epoch().unwrap();
            }
            one.snapshot(&mut snap).unwrap();
            want.push([filled, bits(snap.wait().unwrap())]);
        }

        let planned = &c.views[..c.planned * nmode];
        let mut ap = DistLiveAp::new(
            &gpus(ngpu),
            &c.cfg,
            planned,
            Scheme::Strips,
            &c.probe,
            false,
        )
        .unwrap();
        let grid = ap.plan().partition().grid();
        let (tile, ntx) = (grid.tile(), grid.obj()[1].div_ceil(grid.tile()));
        let owner = ap.plan().partition().owner().to_vec();
        // Fills with pixels on tiles of more than one GPU.
        let across = fills
            .iter()
            .filter(|f| {
                let owners = |r: &crate::live::fill::Rect| {
                    let (y, x) = (r.y as usize, r.x as usize);
                    let corners = [[y, x], [y + r.h as usize - 1, x + r.w as usize - 1]];
                    corners.map(|[y, x]| owner[y / tile * ntx + x / tile])
                };
                let mut seen = f.rects().iter().flat_map(owners);
                seen.next().is_some_and(|o| seen.any(|p| p != o))
            })
            .count();
        assert!(across > 0, "no fill lies over two GPUs' tiles");
        let mut feed = Feed::new(&ctx, &c, &ap);
        for (k, ((b, fill), want)) in batches.iter().zip(fills).zip(&want).enumerate() {
            feed = feed.land(&c, b);
            assert_eq!(ap.push(views(b)).unwrap() as usize, b.len());
            assert_eq!(ap.fill(fill).unwrap(), 0, "batch {k}");
            let unlike = |ap: &DistLiveAp, want: &[[u32; 2]]| {
                let got = bits(&ap.object().unwrap());
                got.iter().zip(want).filter(|(g, w)| g != w).count()
            };
            assert_eq!(
                unlike(&ap, &want[0]),
                0,
                "batch {k}: unlike one GPU's when filled"
            );
            for _ in 0..2 {
                ap.epoch().unwrap();
            }
            assert_eq!(
                unlike(&ap, &want[1]),
                0,
                "batch {k}: unlike one GPU's after epochs"
            );
        }
        feed.finish();
        assert_eq!((ap.lost(), ap.closed()), (0, None));
    }

    /// The raster of [`gpus_iterate_a_live_scan_as_one_gpu`] on stores that back their
    /// planned frames and one more before the first frame lands, then a chunk at a time as
    /// the uploader reaches the edge: on 1, 2 and 4 GPUs the stores grow while the scan
    /// lands, and the object is one GPU's on counts uploaded whole after every batch, bit for
    /// bit.
    #[test]
    fn stores_that_grow_mid_scan_give_the_one_gpu_object() {
        let ctx = Context::new(0).expect("GPU 0");
        let c = case(&sim(ScanPath::Raster { cols: 12 }, 240));
        let batches = batches(&c);
        let (nmode, n) = (c.cfg.shape.nmode as usize, 64 * 64);
        let views = |b: &Range<usize>| &c.views[b.start * nmode..b.end * nmode];
        let st = ctx.new_stream().unwrap();
        let dev = DeviceBuffer::from_host(&st, &c.counts).unwrap();
        let mut one = LiveAp::new(&st, &c.cfg, &c.probe, dev, false).unwrap();
        let (mut snap, mut backer) = (one.new_snapshot().unwrap(), one.backer());
        let mut want = Vec::new();
        for b in &batches {
            let v = views(b);
            backer.back(v).unwrap();
            assert_eq!(one.push(v).unwrap() as usize, b.len());
            one.epoch().unwrap();
            one.snapshot(&mut snap).unwrap();
            want.push(bits(snap.wait().unwrap()));
        }

        let (nframe, planned) = (c.cfg.shape.nframe as usize, &c.views[..c.planned * nmode]);
        for ngpu in [1, 2, 4] {
            let plan =
                LivePlan::new(&c.cfg.shape, planned, Scheme::Strips, ngpu, c.cfg.cap).unwrap();
            let stores: Vec<Arc<DeviceStore>> = gpus(ngpu)
                .into_iter()
                .zip(plan.regions())
                .map(|(o, r)| {
                    let ctx = Context::new(o).unwrap();
                    Arc::new(DeviceStore::growing(&ctx, nframe, n, r.frames.len(), 1).unwrap())
                })
                .collect();
            let first: Vec<u64> = stores.iter().map(|s| s.edge()).collect();
            let mut ap =
                DistLiveAp::on_stores(&stores, &c.cfg, planned, Scheme::Strips, &c.probe, false)
                    .unwrap();
            let mut feed = Feed::new(&ctx, &c, &ap);
            for (k, (b, want)) in batches.iter().zip(&want).enumerate() {
                feed = feed.land(&c, b);
                assert_eq!(ap.push(views(b)).unwrap() as usize, b.len());
                ap.epoch().unwrap();
                let got = bits(&ap.object().unwrap());
                let diff = got.iter().zip(want).filter(|(g, w)| g != w).count();
                assert_eq!(
                    diff, 0,
                    "{ngpu} GPUs: pixels unlike one GPU's after batch {k}"
                );
            }
            let up = feed.finish();
            let placed: Vec<u64> = up.regions.iter().map(|r| r.frames).collect();
            let edges: Vec<u64> = stores.iter().map(|s| s.edge()).collect();
            for (r, ((&p, &f), &e)) in placed.iter().zip(&first).zip(&edges).enumerate() {
                assert!(
                    p > f,
                    "{ngpu} GPUs: region {r} took {p} frames, inside its first {f}"
                );
                assert!(
                    e >= p,
                    "{ngpu} GPUs: region {r} backs {e} places for {p} frames"
                );
            }
            assert_eq!((ap.lost(), ap.closed()), (0, None));
        }
    }

    #[test]
    fn fills_on_2_gpus_give_the_one_gpu_object() {
        fills_give_the_one_gpu_object(2);
    }

    #[test]
    fn fills_on_4_gpus_give_the_one_gpu_object() {
        fills_give_the_one_gpu_object(4);
    }
}
