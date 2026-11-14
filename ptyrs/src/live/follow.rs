//! The live AP consumer (DESIGN.md §3 B6, §7 M5): lands each frame batch
//! the uploader sends into a [`LiveAp`] and runs epochs over the landed
//! frames between batches, from graphs captured before the first frame.
//! [`run_dist_consumer`] runs the same loop over a [`DistLiveAp`], one GPU
//! per region of the object, each fed its own frames by the uploader.
//! The object the scan's plan reaches is backed before the first frame,
//! and a mapper thread backs what each batch needs beyond it, its object
//! tiles and view records, before handing it on: a scan inside its plan
//! makes no growth call, which would hold the driver lock against the
//! epochs' own calls, and growing past it never stalls them on the call.
//!
//! Once every frame of the scan has landed, or the uploader stopped with
//! some missing, it goes on iterating and keeps a host copy of the object
//! every `snap_every` epochs, stamped with the wall time the host had it,
//! for [`Clock::check`]; a copy from before then lacks frames the offline
//! run has. At most two epochs are queued ahead of the host, so frames that
//! arrive land behind at most one epoch already running. With a
//! [`Preview`], it also hands a viewer a copy of the object from the first
//! epoch on, read on the host only once a later epoch's fence shows it done,
//! and each epoch's LLK, read once a period.
//! [`offline_ap`] runs the same AP on the same stored frames until it
//! settles, the reference the clock compares against.

use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::base::{C64, CoreError};
use crate::gpu::{C32, ViewAddr};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use cuda_core::{Context, Event, Stream};
use serde::Deserialize;

use crate::dist::{DistError, DistLiveAp, DistSnapshot, Scheme};
use crate::live::args::Args;
use crate::live::converge::{Clock, ClockError};
use crate::live::farfield::Ptycho;
use crate::live::grow::GrowError;
use crate::live::object::{Snapshot, timed_sleeping_event};
use crate::live::pv_out::PvCfg;
use crate::live::recon::{ApConfig, LiveAp, ViewPlacement};
use crate::live::ring::{DeviceStore, FrameBatch, Route, STORE_LEAD, Whole};
use crate::live::scan::{ScanBacker, ScanShape};
use crate::live::stats::wall_ns;

/// The reconstruction run on the received frames: `pty-recv --recon ap`
/// and its `--recon-*` flags, or a `[recon]` table with the same keys.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ReconCfg {
    pub algo: Algo,
    /// Frames per stack; at most the scan's.
    pub nf_max: NonZeroU32,
    /// Stacks per sweep replay ([`ApConfig::stacks`]).
    pub stacks: NonZeroU32,
    /// PyNX `ObjNorm` inertia.
    pub inertia: f32,
    /// The object is backed this many MiB at a time.
    pub chunk_mib: usize,
    /// Epochs between the live run's kept snapshots once the scan landed.
    pub snap_every: NonZeroU32,
    /// Snapshots kept; the live run stops after the last.
    pub snaps: usize,
    /// The offline reference's epoch cap; see [`settle`].
    pub settle_max: u64,
    /// Publish the live results as PVs; off when unset.
    pub pv: Option<PvCfg>,
}

impl Default for ReconCfg {
    fn default() -> Self {
        Self {
            algo: Algo::Ap,
            nf_max: NonZeroU32::new(32).unwrap(),
            stacks: NonZeroU32::new(16).unwrap(),
            inertia: 0.01,
            chunk_mib: 2,
            snap_every: NonZeroU32::new(5).unwrap(),
            snaps: 200,
            settle_max: 8192,
            pv: None,
        }
    }
}

impl ReconCfg {
    /// `--recon ALGO --recon-nf-max --recon-stacks --recon-inertia --recon-chunk-mib
    /// --recon-snap-every --recon-snaps --recon-settle-max`, and
    /// `--recon-pv PREFIX --recon-pv-bin --recon-pv-period-ms
    /// --recon-pv-llk-period-ms`; `None` without `--recon`.
    pub fn from_args(a: &Args) -> Option<Self> {
        let algo = a.opt("recon")?;
        let d = Self::default();
        Some(Self {
            algo,
            nf_max: a.get("recon-nf-max", d.nf_max),
            stacks: a.get("recon-stacks", d.stacks),
            inertia: a.get("recon-inertia", d.inertia),
            chunk_mib: a.get("recon-chunk-mib", d.chunk_mib),
            snap_every: a.get("recon-snap-every", d.snap_every),
            snaps: a.get("recon-snaps", d.snaps),
            settle_max: a.get("recon-settle-max", d.settle_max),
            pv: a.opt("recon-pv").map(|prefix| {
                let p = PvCfg::new(prefix);
                PvCfg {
                    bin: a.get("recon-pv-bin", p.bin),
                    period_ms: a.get("recon-pv-period-ms", p.period_ms),
                    llk_period_ms: a.get("recon-pv-llk-period-ms", p.llk_period_ms),
                    ..p
                }
            }),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Algo {
    Ap,
}

impl std::str::FromStr for Algo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "ap" => Ok(Algo::Ap),
            _ => Err("expected ap".into()),
        }
    }
}

/// What the live run reconstructs with, fixed before the first frame.
#[derive(Clone, Debug)]
pub struct FollowCfg {
    /// `shape.nframe` is the scan's frame count.
    pub ap: ApConfig,
    /// `[nprobe, n, n]`, natural order.
    pub probe: Vec<C32>,
    pub place: ViewPlacement,
    /// The scan's planned positions, frame by frame from the first; the
    /// object they reach is backed before the first frame. Empty when the
    /// scan has no plan.
    pub plan: Vec<(f64, f64)>,
    /// Epochs between kept snapshots.
    pub snap_every: NonZeroU32,
    /// Snapshots kept; the run stops after the last.
    pub snaps: usize,
}

impl FollowCfg {
    /// `rc` on the first `nframe` triggers of `p`'s scan, trigger `u` at
    /// its point `u % p.points()`: one object, `p`'s probe modes held fixed,
    /// the object placed, its tile slots sized and its plan set from those
    /// positions.
    pub fn plan(p: &Ptycho, nframe: NonZeroU32, rc: &ReconCfg) -> Self {
        let n = p.spec().n;
        let dx = p.spec().dx();
        let pos: Vec<(f64, f64)> = (0..nframe.get() as usize)
            .map(|u| p.position(u % p.points()))
            .collect();
        let (place, reach) = ViewPlacement::plan([dx, dx], &pos);
        let nprobe = (p.probe().len() / (n * n)) as u32;
        let shape = ScanShape {
            nframe: nframe.get(),
            nmode: nprobe,
            nprobe,
            nobj: 1,
            frame: [n as u32; 2],
            reach,
            quad_phase: 0.0,
        };
        let mut ap = ApConfig {
            shape,
            nf_max: rc.nf_max.min(nframe),
            cap: 0,
            fill: [1.0, 0.0],
            chunk: rc.chunk_mib << 20,
            inertia: rc.inertia,
            stacks: rc.stacks,
        };
        ap.cap = shape.tile_cap(&views_of(&ap, &place, &pos));
        Self {
            ap,
            probe: p
                .probe()
                .iter()
                .map(|c| C32 { re: c.re, im: c.im })
                .collect(),
            place,
            plan: pos,
            snap_every: rc.snap_every,
            snaps: rc.snaps,
        }
    }
}

/// Object 0 of the live run some epochs after its scan landed.
pub struct Kept {
    /// Wall ns once the host had it.
    pub taken: i64,
    /// Epochs it follows.
    pub epochs: u64,
    /// `[ny, nx]` row-major.
    pub pixels: Vec<[f32; 2]>,
}

/// Object 0 of the live run after some epochs, for a viewer.
pub struct View {
    pub epochs: u64,
    /// `[ny, nx]`.
    pub shape: [usize; 2],
    /// `[ny, nx]` row-major.
    pub pixels: Vec<[f32; 2]>,
}

/// Where the live run sends [`View`]s: one copy is queued after an epoch
/// once `period` has passed since the last and `tx` holds none, and one
/// after the last epoch. With it the run records each epoch's per-pixel
/// LLK ([`LiveAp::llk`]) and sends on `llk`, in epoch order from the first,
/// those of the epochs queued so far once `llk_period` has passed since the
/// last send, and the rest after the last epoch. Each such read waits for
/// the epoch queued last, so the next starts a host turn late.
pub struct Preview {
    pub period: Duration,
    pub tx: Sender<View>,
    pub llk_period: Duration,
    pub llk: Sender<Vec<f64>>,
}

pub struct FollowReport {
    /// Frames the uploader sent.
    pub stored: u64,
    /// Frames the live scan took, the first `landed` stored ones.
    pub landed: u64,
    pub epochs: u64,
    /// Epochs queued before the scan had landed; `None` if no frame came.
    pub scan_epochs: Option<u64>,
    /// `[ny, nx]` of the object.
    pub shape: [usize; 2],
    /// `(x, y)` m of each stored frame, by sequence number.
    pub positions: Vec<(f64, f64)>,
    pub kept: Vec<Kept>,
    pub obj_mapped_bytes: usize,
    /// Time backing the plan took, before the first frame.
    pub plan_time: Duration,
    /// Object bytes mapped once the plan was backed; the scan grew it to
    /// `obj_mapped_bytes`.
    pub plan_mapped_bytes: usize,
    /// Per batch, the epochs queued before it landed and the frames stored
    /// with it: the schedule that, replayed, gives the same object.
    pub landings: Vec<(u64, u64)>,
    /// Wall time, ns, each epoch finished on the GPU, by epoch.
    pub epoch_done: Vec<i64>,
    /// How long stamping the clock of `epoch_done` took: its error bound.
    pub epoch_clock_ns: i64,
}

impl FollowReport {
    /// GPU time, ns, of each epoch queued once the scan had landed, from
    /// [`Self::epoch_done`]: the only source of epoch time. The first epoch
    /// has no end before it to time from.
    pub fn epochs_after_scan(&self) -> impl Iterator<Item = i64> + '_ {
        let from = self
            .scan_epochs
            .map_or(self.epoch_done.len(), |s| s.max(1) as usize - 1);
        self.epoch_done[from..].windows(2).map(|w| w[1] - w[0])
    }

    /// Mean of [`Self::epochs_after_scan`], ms.
    pub fn epoch_ms(&self) -> Option<f64> {
        let (n, ns) = self
            .epochs_after_scan()
            .fold((0u64, 0i64), |(n, ns), t| (n + 1, ns + t));
        (n > 0).then(|| ns as f64 / 1e6 / n as f64)
    }
}

/// The views of the frames at `positions`.
fn views_of(cfg: &ApConfig, place: &ViewPlacement, positions: &[(f64, f64)]) -> Vec<ViewAddr> {
    let (nprobe, nobj) = (cfg.shape.nprobe, cfg.shape.nobj);
    let mut out = Vec::with_capacity(positions.len() * (nprobe * nobj) as usize);
    for &p in positions {
        place.views(p, nprobe, nobj, &mut out);
    }
    out
}

/// Backs each batch's frames through `backer`, if the run has one, then
/// hands the batch on with its views, until `batches` closes or `ahead`'s
/// receiver goes.
fn back_ahead(
    mut backer: Option<ScanBacker>,
    ap: ApConfig,
    place: ViewPlacement,
    batches: Receiver<FrameBatch>,
    ahead: Sender<(FrameBatch, Vec<ViewAddr>)>,
) -> Result<(), GrowError> {
    // Dropped once the scan's last frame is backed: frames past it never
    // land, and the consumer may finish before `batches` closes.
    for b in batches {
        let views = views_of(&ap, &place, &b.positions);
        if let Some(k) = &mut backer {
            k.back(&views)?;
            if b.f0 + b.n >= u64::from(ap.shape.nframe) {
                backer = None;
            }
        }
        if ahead.send((b, views)).is_err() {
            break;
        }
    }
    Ok(())
}

/// The mapper's outcome, once it stopped handing batches on.
fn joined(mapper: &mut Option<JoinHandle<Result<(), GrowError>>>) -> Result<(), GrowError> {
    match mapper.take().map(JoinHandle::join) {
        Some(Ok(r)) => r,
        Some(Err(panic)) => std::panic::resume_unwind(panic),
        None => Ok(()),
    }
}

/// A live AP run the follow loop drives: a [`LiveAp`] on one GPU or a
/// [`DistLiveAp`] on several.
pub trait Solve {
    type Error: From<GrowError>;
    /// A host copy of the object.
    type Snap;

    /// Queues an epoch over the frames landed, without waiting for it.
    fn epoch(&mut self) -> Result<(), Self::Error>;
    /// Epochs queued so far.
    fn epochs(&self) -> u64;
    fn new_snapshot(&mut self) -> Result<Self::Snap, Self::Error>;
    /// Queues a copy of the object after the epochs queued so far.
    fn snapshot(&mut self, snap: &mut Self::Snap) -> Result<(), Self::Error>;
    /// Waits for the copy queued last into `snap`: `[nobj, ny, nx]`.
    fn pixels<'a>(&mut self, snap: &'a mut Self::Snap) -> Result<&'a [[f32; 2]], Self::Error>;
    /// `[ny, nx]` of the object.
    fn shape(&self) -> [usize; 2];
}

trait Engine: Solve {
    /// Lands the frames whose views are `views`.
    fn push(&mut self, views: &[ViewAddr]) -> Result<(), Self::Error>;
    /// Wall ns the GPUs finished `epoch`, one of the last two queued; waits
    /// for it.
    fn end_of(&mut self, epoch: u64) -> Result<i64, Self::Error>;
    /// How long stamping the clock [`Self::end_of`] reads took, ns.
    fn clock_ns(&self) -> i64;
    /// Each epoch's LLK since the last read, waiting for them; `None` if
    /// the run records none.
    fn llk(&mut self) -> Result<Option<Vec<f64>>, Self::Error>;
    /// Frames the scan took.
    fn landed(&self) -> u64;
    /// Object bytes mapped.
    fn mapped_bytes(&mut self) -> Result<usize, Self::Error>;
    /// Waits for everything queued.
    fn synchronize(&mut self) -> Result<(), Self::Error>;
}

/// A [`LiveAp`] and the fences that time its epochs.
struct One {
    live: LiveAp,
    stream: Arc<Stream>,
    /// Recorded after the epochs of each parity.
    fence: [Event; 2],
    /// Epoch ends are timed from `anchor`, stamped at wall ns `t_anchor`.
    anchor: Event,
    t_anchor: i64,
    clock_ns: i64,
}

impl Solve for LiveAp {
    type Error = GrowError;
    type Snap = Snapshot;

    fn epoch(&mut self) -> Result<(), GrowError> {
        LiveAp::epoch(self)
    }

    fn epochs(&self) -> u64 {
        LiveAp::epochs(self)
    }

    fn new_snapshot(&mut self) -> Result<Snapshot, GrowError> {
        LiveAp::new_snapshot(self)
    }

    fn snapshot(&mut self, snap: &mut Snapshot) -> Result<(), GrowError> {
        LiveAp::snapshot(self, snap)
    }

    fn pixels<'a>(&mut self, snap: &'a mut Snapshot) -> Result<&'a [[f32; 2]], GrowError> {
        Ok(snap.wait()?)
    }

    fn shape(&self) -> [usize; 2] {
        self.scan().obj().shape()
    }
}

impl Solve for One {
    type Error = GrowError;
    type Snap = Snapshot;

    fn epoch(&mut self) -> Result<(), GrowError> {
        let f = &self.fence[(self.live.epochs() % 2) as usize];
        self.live.epoch()?;
        Ok(f.record(&self.stream)?)
    }

    fn epochs(&self) -> u64 {
        self.live.epochs()
    }

    fn new_snapshot(&mut self) -> Result<Snapshot, GrowError> {
        self.live.new_snapshot()
    }

    fn snapshot(&mut self, snap: &mut Snapshot) -> Result<(), GrowError> {
        self.live.snapshot(snap)
    }

    fn pixels<'a>(&mut self, snap: &'a mut Snapshot) -> Result<&'a [[f32; 2]], GrowError> {
        self.live.pixels(snap)
    }

    fn shape(&self) -> [usize; 2] {
        Solve::shape(&self.live)
    }
}

impl Engine for One {
    fn push(&mut self, views: &[ViewAddr]) -> Result<(), GrowError> {
        self.live.push(views).map(drop)
    }

    fn end_of(&mut self, epoch: u64) -> Result<i64, GrowError> {
        let f = &self.fence[(epoch % 2) as usize];
        f.synchronize()?;
        Ok(self.t_anchor + (f64::from(self.anchor.elapsed_time(f)?) * 1e6) as i64)
    }

    fn clock_ns(&self) -> i64 {
        self.clock_ns
    }

    fn llk(&mut self) -> Result<Option<Vec<f64>>, GrowError> {
        let llk = self.live.llk()?;
        Ok(llk.map(|l| l.into_iter().map(|(_, l)| l).collect()))
    }

    fn landed(&self) -> u64 {
        u64::from(self.live.scan().landed())
    }

    fn mapped_bytes(&mut self) -> Result<usize, GrowError> {
        Ok(self.live.scan().obj().mapped_bytes())
    }

    fn synchronize(&mut self) -> Result<(), GrowError> {
        Ok(self.stream.synchronize()?)
    }
}

impl Solve for DistLiveAp {
    type Error = DistError;
    type Snap = DistSnapshot;

    fn epoch(&mut self) -> Result<(), DistError> {
        DistLiveAp::epoch(self)
    }

    fn epochs(&self) -> u64 {
        DistLiveAp::epochs(self)
    }

    fn new_snapshot(&mut self) -> Result<DistSnapshot, DistError> {
        DistLiveAp::new_snapshot(self)
    }

    fn snapshot(&mut self, snap: &mut DistSnapshot) -> Result<(), DistError> {
        DistLiveAp::snapshot(self, snap)
    }

    fn pixels<'a>(&mut self, snap: &'a mut DistSnapshot) -> Result<&'a [[f32; 2]], DistError> {
        self.wait(snap)
    }

    fn shape(&self) -> [usize; 2] {
        DistLiveAp::shape(self)
    }
}

impl Engine for DistLiveAp {
    fn push(&mut self, views: &[ViewAddr]) -> Result<(), DistError> {
        DistLiveAp::push(self, views).map(drop)
    }

    fn end_of(&mut self, epoch: u64) -> Result<i64, DistError> {
        DistLiveAp::end_of(self, epoch)
    }

    fn clock_ns(&self) -> i64 {
        DistLiveAp::clock_ns(self)
    }

    fn llk(&mut self) -> Result<Option<Vec<f64>>, DistError> {
        let llk = DistLiveAp::llk(self)?;
        Ok(llk.map(|l| l.into_iter().map(|(_, l)| l).collect()))
    }

    fn landed(&self) -> u64 {
        self.taken()
    }

    fn mapped_bytes(&mut self) -> Result<usize, DistError> {
        DistLiveAp::mapped_bytes(self)
    }

    fn synchronize(&mut self) -> Result<(), DistError> {
        DistLiveAp::synchronize(self)
    }
}

/// Live AP on `store`'s frames as `batches` say they landed, until the
/// uploader drops its sender and `cfg.snaps` snapshots are kept, with
/// copies to `preview` if given. Sends on `captured` once its graphs are
/// captured, and drops it unsent if that fails.
pub fn run_ap_consumer(
    ctx: Arc<Context>,
    store: Arc<DeviceStore>,
    cfg: FollowCfg,
    batches: Receiver<FrameBatch>,
    captured: Sender<()>,
    preview: Option<Preview>,
) -> Result<FollowReport, GrowError> {
    follow_one(ctx, store, cfg, batches, preview, || {
        let _ = captured.send(());
    })
}

/// [`run_ap_consumer`], calling `ready` once the graphs are captured.
fn follow_one(
    ctx: Arc<Context>,
    store: Arc<DeviceStore>,
    cfg: FollowCfg,
    batches: Receiver<FrameBatch>,
    preview: Option<Preview>,
    ready: impl FnOnce(),
) -> Result<FollowReport, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let shape = cfg.ap.shape;
    let elems = shape.frame[0] as usize * shape.frame[1] as usize;
    if store.frame_elems() != elems || store.max_frames() < u64::from(shape.nframe) {
        return Err(crate::gpu::Error::Shape(format!(
            "a store of {} frames of {} pixels for a scan of {} frames of {:?}",
            store.max_frames(),
            store.frame_elems(),
            shape.nframe,
            shape.frame
        ))
        .into());
    }
    // SAFETY: an epoch reads only landed frames, whose copies into the
    // store completed before the uploader sent their batch.
    let counts = unsafe { store.frames_view()? };
    let live = LiveAp::new(&stream, &cfg.ap, &cfg.probe, counts, preview.is_some())?;
    // The plan is backed before `captured` lets the first frame in; from
    // here on only the mapper grows device memory, past the plan.
    let mut backer = live.backer();
    let t = Instant::now();
    backer.plan(&views_of(&cfg.ap, &cfg.place, &cfg.plan))?;
    let plan_time = t.elapsed();
    let fence = [timed_sleeping_event(&ctx)?, timed_sleeping_event(&ctx)?];
    // Stamped on the idle stream.
    let anchor = ctx.new_event()?;
    stream.synchronize()?;
    let t0 = wall_ns();
    anchor.record(&stream)?;
    anchor.synchronize()?;
    let t_anchor = wall_ns();
    let run = One {
        live,
        stream,
        fence,
        anchor,
        t_anchor,
        clock_ns: t_anchor - t0,
    };
    follow(run, Some(backer), plan_time, &cfg, batches, preview, |_| {
        ready()
    })
}

/// Each region's store and the route the uploader lands the frames by
/// ([`run_region_uploader`](crate::live::ring::run_region_uploader)).
pub type Landing = (Vec<Arc<DeviceStore>>, Box<dyn Route>);

/// Live AP on the GPUs `ordinals`, on stores of its own, sent on `ready`
/// once its graphs are captured and dropped unsent if that fails. One GPU
/// runs as [`run_ap_consumer`] does. More split the object by `scheme`,
/// region `r` on `ordinals[r]`, and [`FollowReport::plan_time`] is the
/// whole setup's.
pub fn run_dist_consumer(
    ordinals: &[usize],
    scheme: Scheme,
    cfg: FollowCfg,
    batches: Receiver<FrameBatch>,
    ready: Sender<Landing>,
    preview: Option<Preview>,
) -> Result<FollowReport, DistError> {
    if let [one] = *ordinals {
        let ctx = Context::new(one)?;
        let shape = cfg.ap.shape;
        let elems = shape.frame[0] as usize * shape.frame[1] as usize;
        let store = Arc::new(DeviceStore::growing(
            &ctx,
            shape.nframe as usize,
            elems,
            cfg.plan.len(),
            STORE_LEAD,
        )?);
        let landing: Landing = (vec![store.clone()], Box::new(Whole));
        return Ok(follow_one(ctx, store, cfg, batches, preview, || {
            let _ = ready.send(landing);
        })?);
    }
    let t = Instant::now();
    let planned = views_of(&cfg.ap, &cfg.place, &cfg.plan);
    let llk = preview.is_some();
    let run = DistLiveAp::new(ordinals, &cfg.ap, &planned, scheme, &cfg.probe, llk)?;
    let plan_time = t.elapsed();
    let place = cfg.place;
    follow(run, None, plan_time, &cfg, batches, preview, |run| {
        let _ = ready.send((run.stores(), Box::new(run.route(place))));
    })
}

/// Lands the batches into `live` and iterates it, as [`run_ap_consumer`]
/// says; `backer` backs each batch ahead of its landing if the run does not
/// itself, and `ready` is called once the run can take frames.
fn follow<R: Engine>(
    mut live: R,
    backer: Option<ScanBacker>,
    plan_time: Duration,
    cfg: &FollowCfg,
    batches: Receiver<FrameBatch>,
    preview: Option<Preview>,
    ready: impl FnOnce(&R),
) -> Result<FollowReport, R::Error> {
    let shape = cfg.ap.shape;
    let plan_mapped_bytes = live.mapped_bytes()?;
    let (fwd, ahead) = crossbeam_channel::unbounded();
    let mut mapper = {
        let (ap, place) = (cfg.ap, cfg.place);
        Some(
            std::thread::Builder::new()
                .name("backer".into())
                .spawn(move || back_ahead(backer, ap, place, batches, fwd))
                .expect("spawn the backer"),
        )
    };
    let mut snaps = [live.new_snapshot()?, live.new_snapshot()?];
    let mut view = match preview {
        Some(p) => Some((p, live.new_snapshot()?)),
        None => None,
    };
    // The epochs the view copy queued last follows, until it is sent, and
    // when it was queued.
    let mut viewing: Option<u64> = None;
    let mut viewed: Option<Instant> = None;
    let mut llk_sent: Option<Instant> = None;
    let mut epoch_done = Vec::new();
    ready(&live);

    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(shape.nframe as usize);
    let mut kept = Vec::with_capacity(cfg.snaps);
    let (mut open, mut scan_epochs) = (true, None);
    // The snapshot buffer queued last and the epochs it follows.
    let mut pending: Option<(usize, u64)> = None;
    let mut next_snap = 0usize;
    let mut landings = Vec::new();
    let fed = (|| -> Result<(), R::Error> {
        let mut got = Vec::new();
        while kept.len() + usize::from(pending.is_some()) < cfg.snaps {
            // Block for frames only while none has come.
            if open && positions.is_empty() {
                match ahead.recv() {
                    Ok(b) => got.push(b),
                    Err(_) => {
                        open = false;
                        joined(&mut mapper)?;
                    }
                }
            }
            while open {
                match ahead.try_recv() {
                    Ok(b) => got.push(b),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        open = false;
                        joined(&mut mapper)?;
                    }
                }
            }
            for (b, views) in got.drain(..) {
                assert_eq!(
                    b.f0,
                    positions.len() as u64,
                    "batches arrive in order and without gaps"
                );
                live.push(&views)?;
                positions.extend(b.positions);
                landings.push((live.epochs(), positions.len() as u64));
            }
            if positions.is_empty() {
                return Ok(());
            }
            let done = positions.len() as u64 == u64::from(shape.nframe) || !open;
            if done && scan_epochs.is_none() {
                scan_epochs = Some(live.epochs());
            }
            // The epoch two before this one has finished.
            if live.epochs() >= 2 {
                epoch_done.push(live.end_of(live.epochs() - 2)?);
            }
            // So has a copy queued after the epoch before that one, behind
            // its fence: it is sent without waiting.
            if let Some(e) = viewing
                && live.epochs() >= e + 2
                && let Some((p, snap)) = &mut view
            {
                viewing = None;
                // Only this thread sends, so only a viewer gone refuses it.
                if p.tx.try_send(view_of(&mut live, snap, e)?).is_err() {
                    view = None;
                }
            }
            if let Some((p, _)) = &view
                && live.epochs() > 0
                && llk_sent.is_none_or(|t| t.elapsed() >= p.llk_period)
            {
                llk_sent = Some(Instant::now());
                if !send_llk(&mut live, &p.llk)? {
                    view = None;
                }
            }
            live.epoch()?;
            if let Some((p, snap)) = &mut view
                && viewing.is_none()
                && p.tx.is_empty()
                && viewed.is_none_or(|t| t.elapsed() >= p.period)
            {
                live.snapshot(snap)?;
                viewing = Some(live.epochs());
                viewed = Some(Instant::now());
            }
            if let Some(k) = scan_epochs.map(|s| live.epochs() - s)
                && k.is_multiple_of(u64::from(cfg.snap_every.get()))
            {
                let i = next_snap;
                next_snap ^= 1;
                live.snapshot(&mut snaps[i])?;
                if let Some((j, epochs)) = pending.replace((i, live.epochs())) {
                    kept.push(keep(&mut live, &mut snaps[j], epochs)?);
                }
            }
        }
        if let Some((i, epochs)) = pending.take() {
            kept.push(keep(&mut live, &mut snaps[i], epochs)?);
        }
        Ok(())
    })();
    // Nothing may still read the stores once this returns.
    live.synchronize()?;
    fed?;
    // Each fence follows one of the last two epochs.
    for e in epoch_done.len() as u64..live.epochs() {
        epoch_done.push(live.end_of(e)?);
    }
    // The viewer's last LLK and copy are those of the last epoch.
    if let Some((p, mut snap)) = view
        && live.epochs() > 0
        && send_llk(&mut live, &p.llk)?
    {
        if viewing != Some(live.epochs()) {
            live.snapshot(&mut snap)?;
        }
        let epochs = live.epochs();
        let _ = p.tx.send(view_of(&mut live, &mut snap, epochs)?);
    }
    Ok(FollowReport {
        stored: positions.len() as u64,
        landed: live.landed(),
        epochs: live.epochs(),
        scan_epochs,
        shape: live.shape(),
        positions,
        kept,
        obj_mapped_bytes: live.mapped_bytes()?,
        plan_time,
        plan_mapped_bytes,
        landings,
        epoch_done,
        epoch_clock_ns: live.clock_ns(),
    })
}

/// Sends `live`'s LLK of the epochs queued since the last read, if it
/// records any, waiting for them; false once the receiver is gone.
fn send_llk<R: Engine>(live: &mut R, tx: &Sender<Vec<f64>>) -> Result<bool, R::Error> {
    Ok(live.llk()?.is_none_or(|llk| tx.send(llk).is_ok()))
}

/// Waits for `snap`, which follows `epochs`, and copies its object 0.
fn view_of<R: Engine>(live: &mut R, snap: &mut R::Snap, epochs: u64) -> Result<View, R::Error> {
    let shape = live.shape();
    let px = live.pixels(snap)?;
    Ok(View {
        epochs,
        shape,
        pixels: px[..shape[0] * shape[1]].to_vec(),
    })
}

/// Waits for `snap` and copies its object 0, stamped now.
fn keep<R: Engine>(live: &mut R, snap: &mut R::Snap, epochs: u64) -> Result<Kept, R::Error> {
    let [ny, nx] = live.shape();
    let px = live.pixels(snap)?;
    let taken = wall_ns();
    Ok(Kept {
        taken,
        epochs,
        pixels: px[..ny * nx].to_vec(),
    })
}

pub struct Offline {
    pub epochs: u64,
    /// The clock against the run's last object, or why it would not start.
    pub clock: Result<Clock, ClockError>,
}

/// AP as the live run does it on `store`'s frames `0..positions.len()` at
/// `positions`, landed at once, run until it [`settle`]s.
pub fn offline_ap(
    ctx: &Arc<Context>,
    store: &DeviceStore,
    cfg: &FollowCfg,
    positions: &[(f64, f64)],
    region: [Range<usize>; 2],
    scan_end: i64,
    max_epochs: u64,
) -> Result<Offline, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let mut ap = cfg.ap;
    ap.shape.nframe = u32::try_from(positions.len())
        .ok()
        .filter(|&n| n <= cfg.ap.shape.nframe)
        .ok_or_else(|| {
            crate::gpu::Error::Shape(format!(
                "{} frames for a scan of {}",
                positions.len(),
                cfg.ap.shape.nframe
            ))
        })?;
    // SAFETY: every frame read is stored, and nothing writes the store now.
    let counts = unsafe { store.frames_view()? };
    let mut run = LiveAp::new(&stream, &ap, &cfg.probe, counts, false)?;
    let views = views_of(&ap, &cfg.place, positions);
    run.backer().back(&views)?;
    run.push(&views)?;
    settle(&mut run, region, scan_end, max_epochs)
}

/// [`offline_ap`] over the regions of `scheme`, region `r` on the GPU of
/// `stores[r]` and on its frames as the live run's uploader left them: the
/// frames at `positions` routed as [`run_dist_consumer`] routes them. One
/// store is [`offline_ap`] itself.
pub fn offline_dist(
    stores: &[Arc<DeviceStore>],
    scheme: Scheme,
    cfg: &FollowCfg,
    positions: &[(f64, f64)],
    region: [Range<usize>; 2],
    scan_end: i64,
    max_epochs: u64,
) -> Result<Offline, DistError> {
    if let [store] = stores {
        let off = offline_ap(
            store.ctx(),
            store,
            cfg,
            positions,
            region,
            scan_end,
            max_epochs,
        );
        return Ok(off?);
    }
    let planned = views_of(&cfg.ap, &cfg.place, &cfg.plan);
    let mut run = DistLiveAp::on_stores(stores, &cfg.ap, &planned, scheme, &cfg.probe, false)?;
    let views = views_of(&cfg.ap, &cfg.place, positions);
    let took = run.push(&views)?;
    if took as usize != positions.len() {
        return Err(DistError::Partition(format!(
            "the offline run took {took} of {} stored frames",
            positions.len()
        )));
    }
    settle(&mut run, region, scan_end, max_epochs)
}

/// Iterates `run`, every frame landed, to epochs `E` = `E0`, `2 E0`, … from
/// its count `E0` now (1 when fresh) until its object at `E` matches the
/// one at `E/2` over `region` to
/// [`SETTLED`](crate::live::converge::SETTLED), the clock's reference; refused
/// as still moving once the next doubling would pass `max_epochs`, after
/// at least the check at `E` = 2.
pub fn settle<R: Solve>(
    run: &mut R,
    region: [Range<usize>; 2],
    scan_end: i64,
    max_epochs: u64,
) -> Result<Offline, R::Error> {
    let mut snap = run.new_snapshot()?;
    let mut half: Option<Vec<C64>> = None;
    let mut e = run.epochs().max(1);
    loop {
        while run.epochs() < e {
            run.epoch()?;
        }
        run.snapshot(&mut snap)?;
        let shape = run.shape();
        let now = plane(run.pixels(&mut snap)?, shape);
        if let Some(h) = &half {
            match Clock::new(&now, h, shape, region.clone(), scan_end) {
                Err(ClockError::Moving { .. }) if 2 * e <= max_epochs => {}
                clock => return Ok(Offline { epochs: e, clock }),
            }
        }
        half = Some(now);
        e *= 2;
    }
}

/// Object 0 of a snapshot's pixels, `[ny, nx]` = `shape`.
pub fn plane(px: &[[f32; 2]], shape: [usize; 2]) -> Vec<C64> {
    px[..shape[0] * shape[1]]
        .iter()
        .map(|&[re, im]| C64::new(re.into(), im.into()))
        .collect()
}

/// The first of `kept`, in order, that `clock` finds converged, with its
/// time from the scan's end to [`Kept::taken`], both host clock; checked on
/// `threads` threads.
pub fn first_converged(
    clock: &Clock,
    kept: &[Kept],
    shape: [usize; 2],
    threads: usize,
) -> Result<Option<(usize, Duration)>, CoreError> {
    let per = kept.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = kept
            .chunks(per)
            .enumerate()
            .map(|(c, chunk)| {
                sc.spawn(move || -> Result<Option<(usize, Duration)>, CoreError> {
                    for (i, k) in chunk.iter().enumerate() {
                        if let Some(d) = clock.check(&plane(&k.pixels, shape), k.taken)? {
                            return Ok(Some((c * per + i, d)));
                        }
                    }
                    Ok(None)
                })
            })
            .collect();
        let mut first = None;
        for h in hs {
            if let (None, Some(hit)) = (first, h.join().unwrap()?) {
                first = Some(hit);
            }
        }
        Ok(first)
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::live::farfield::{Ptycho, PtychoSpec, ScanPath};
    use crate::live::graph::WarmUp;
    use crate::live::pos::{Pos, PosTable};
    use crate::live::ring::{Ring, UploaderConfig, run_region_uploader, run_uploader};
    use crate::live::sent::sent;

    struct Streamed {
        ctx: Arc<Context>,
        store: Arc<DeviceStore>,
        cfg: FollowCfg,
        rep: FollowReport,
        /// Wall ns just before the last frame was committed.
        scan_end: i64,
        recaptures: u64,
        views: Vec<View>,
        /// The LLK history the viewer was sent.
        llk: Vec<f64>,
    }

    /// A 6x6 raster of 64x64 frames, two probe modes, streamed through the
    /// ring and the uploader in bursts 20 ms apart into the consumer, which
    /// hands a viewer copies at least `preview` apart if given.
    fn streamed(preview: Option<Duration>) -> Streamed {
        let p = Ptycho::new(PtychoSpec {
            n: 64,
            path: ScanPath::Raster { cols: 6 },
            points: 36,
            step: 12.0,
            photons: 1e6,
            seed: 5,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        });
        let (n, total) = (64usize, p.points());
        let pos: Vec<(f64, f64)> = (0..total).map(|k| p.position(k)).collect();
        let rc = ReconCfg {
            nf_max: NonZeroU32::new(16).unwrap(),
            snap_every: NonZeroU32::new(25).unwrap(),
            snaps: 80,
            ..ReconCfg::default()
        };
        let cfg = FollowCfg::plan(&p, NonZeroU32::new(total as u32).unwrap(), &rc);
        assert_eq!((cfg.ap.shape.nmode, cfg.ap.shape.nprobe), (2, 2));

        let ctx = Context::new(0).expect("GPU 0");
        let ring = Arc::new(Ring::new(&ctx, 16, n * n).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, total, n * n).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let (cap_tx, cap_rx) = crossbeam_channel::bounded(1);
        let (view_tx, view_rx) = crossbeam_channel::bounded(1);
        let viewer = std::thread::spawn(move || view_rx.iter().collect::<Vec<View>>());
        let (llk_tx, llk_rx) = crossbeam_channel::unbounded();
        let llk_reader = std::thread::spawn(move || llk_rx.iter().flatten().collect());
        let consumer = {
            let (ctx, store, cfg) = (ctx.clone(), store.clone(), cfg.clone());
            let preview = preview.map(|period| Preview {
                period,
                tx: view_tx,
                llk_period: period,
                llk: llk_tx,
            });
            std::thread::spawn(move || run_ap_consumer(ctx, store, cfg, rx, cap_tx, preview))
        };
        let ((), consumer) = sent(&cap_rx, consumer);
        let warm = WarmUp::end();
        let table = Arc::new(PosTable::new(1 << 16));
        let uploader = {
            let (ring, stop, store) = (ring.clone(), stop.clone(), store.clone());
            let table = table.clone();
            let up = UploaderConfig {
                batch: NonZeroUsize::new(4).unwrap(),
                flush_ns: 100_000,
                idle: Duration::from_micros(20),
                events: 4,
                pos_wait_ns: 1_000_000_000,
            };
            std::thread::spawn(move || run_uploader(ring, store, table, up, stop, Some(tx)))
        };
        let (mut k, mut scan_end) = (0, 0);
        for burst in [5, 1, 13, 0, 17] {
            for _ in 0..burst {
                if k == total - 1 {
                    scan_end = wall_ns();
                }
                let mut s = ring.reserve();
                s.frame().copy_from_slice(&p.frame(k));
                let (x, y) = pos[k];
                table.insert(k as i32, Pos { x, y, t_rx: 0 });
                s.commit(k as i32, 0, 0);
                k += 1;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(k, total);
        stop.store(true, Ordering::Release);
        assert_eq!(uploader.join().unwrap().unwrap().frames, total as u64);
        let rep = consumer.join().unwrap().unwrap();
        assert_eq!(rep.positions, pos);
        Streamed {
            ctx,
            store,
            cfg,
            rep,
            scan_end,
            recaptures: warm.recaptures(),
            views: viewer.join().unwrap(),
            llk: llk_reader.join().unwrap(),
        }
    }

    /// The consumer lands every streamed frame at its streamed position from
    /// graphs captured before the first, with epochs between landings, and
    /// keeps snapshots every 25 epochs once the scan has landed, each
    /// stamped after the scan's end. Its last one is the object of its
    /// landings replayed on an idle stream, each landing and epoch waited
    /// for: no landing raced an epoch in flight.
    ///
    /// The viewer's copies, the first after the first epoch and so during
    /// the scan, the last after the last epoch, are each that replay at
    /// their epochs, before the landings that followed them; the LLK it was
    /// sent is the replay's of every epoch.
    ///
    /// The plan, every streamed position, was backed before the first frame:
    /// landing the scan mapped no more of the object.
    ///
    /// The live object is not held to the offline one here: these frames,
    /// 12 px apart, overlap the probe's 16 px too little for AP's settled
    /// object to be set by the frames, so how near the live run gets to the
    /// offline one depends on rounding and on how many epochs ran over the
    /// partial scan, which wall time sets.
    #[test]
    fn the_live_object_is_its_landings_replayed_one_at_a_time() {
        let Streamed {
            ctx,
            store,
            cfg,
            rep,
            scan_end,
            recaptures,
            views,
            llk,
        } = streamed(Some(Duration::from_millis(10)));
        assert_eq!(recaptures, 0);
        let total = rep.positions.len() as u64;
        assert_eq!((rep.stored, rep.landed), (total, total));
        assert!(rep.plan_mapped_bytes > 0);
        assert_eq!(rep.obj_mapped_bytes, rep.plan_mapped_bytes);
        let scan_epochs = rep.scan_epochs.unwrap();
        assert!(scan_epochs > 0, "no epoch ran while the scan landed");
        assert_eq!(rep.landings.last().unwrap(), &(scan_epochs, total));
        assert_eq!(rep.kept.len(), cfg.snaps);
        for (i, kept) in rep.kept.iter().enumerate() {
            assert_eq!(kept.epochs, scan_epochs + 25 * (i as u64 + 1));
            assert!(kept.taken > scan_end);
        }
        let last = rep.kept.last().unwrap();
        assert_eq!(rep.epochs, last.epochs);

        ctx.bind_to_thread().unwrap();
        let st = ctx.new_stream().unwrap();
        // SAFETY: the consumer has returned; nothing writes the store.
        let counts = unsafe { store.frames_view().unwrap() };
        let mut run = LiveAp::new(&st, &cfg.ap, &cfg.probe, counts, true).unwrap();
        let mut backer = run.backer();
        assert!(views.len() > 2, "{} views", views.len());
        assert!(views.windows(2).all(|w| w[0].epochs < w[1].epochs));
        assert_eq!(views[0].epochs, 1);
        assert!(views[0].epochs <= scan_epochs);
        let end = views.last().unwrap();
        assert_eq!(end.epochs, last.epochs);
        assert_eq!((end.shape, &end.pixels[..]), (rep.shape, &last.pixels[..]));
        let mut snap = run.new_snapshot().unwrap();
        let mut unseen = views.iter().peekable();
        let mut step = |run: &mut LiveAp| {
            run.epoch().unwrap();
            if let Some(v) = unseen.next_if(|v| v.epochs == run.epochs()) {
                run.snapshot(&mut snap).unwrap();
                let got = snap.wait().unwrap();
                assert!(
                    got[..v.pixels.len()] == v.pixels[..],
                    "view at {}",
                    v.epochs
                );
            }
        };
        let mut at = 0;
        for &(epochs, upto) in &rep.landings {
            while run.epochs() < epochs {
                step(&mut run);
                st.synchronize().unwrap();
            }
            let views = views_of(
                &cfg.ap,
                &cfg.place,
                &rep.positions[at as usize..upto as usize],
            );
            backer.back(&views).unwrap();
            run.push(&views).unwrap();
            st.synchronize().unwrap();
            at = upto;
        }
        while run.epochs() < last.epochs {
            step(&mut run);
        }
        drop(step);
        assert_eq!(unseen.count(), 0, "views off the replay's epochs");
        run.snapshot(&mut snap).unwrap();
        let got = snap.wait().unwrap();
        assert!(got[..last.pixels.len()] == last.pixels[..]);
        let replayed = run.llk().unwrap().unwrap();
        assert!(replayed.iter().map(|&(e, _)| e).eq(0..last.epochs));
        let bits = |l: &f64| l.to_bits();
        assert!(
            replayed
                .iter()
                .map(|(_, l)| bits(l))
                .eq(llk.iter().map(bits)),
            "{} sent, {} replayed",
            llk.len(),
            replayed.len()
        );
        // +inf where the flat start object predicts no photons where some
        // landed, as PyNX gives it: an epoch-0 value whenever epoch 0
        // covers such a frame, which depends on how many it covers.
        assert!(llk.iter().all(|l| *l > 0.0), "{llk:?}");
    }

    /// The offline run settles at a doubling of its epochs within the cap,
    /// and a cap under that refuses it as still moving at the doubling
    /// before. The live object need not match it, so the search runs against
    /// a clock on the live run's own last snapshot: on one thread per
    /// snapshot it finds the same first converged one as an in-order search,
    /// timed from the scan's end.
    #[test]
    fn the_offline_run_settles_and_the_search_finds_the_first_match() {
        let s = streamed(None);
        let rep = &s.rep;
        let region = s.cfg.place.scanned(&rep.positions, s.cfg.ap.shape.frame);
        let offline = |max| {
            let r = region.clone();
            offline_ap(&s.ctx, &s.store, &s.cfg, &rep.positions, r, s.scan_end, max).unwrap()
        };
        let settled = offline(1 << 14);
        let e = settled.epochs;
        settled.clock.unwrap();
        assert!(e.is_power_of_two() && e >= 4, "settled at {e}");
        let capped = offline(e - 1);
        assert_eq!(capped.epochs, e / 2);
        assert!(matches!(capped.clock, Err(ClockError::Moving { .. })));

        let n = rep.kept.len();
        let obj = |i: usize| plane(&rep.kept[i].pixels, rep.shape);
        let last = obj(n - 1);
        let clock = Clock::new(&last, &last, rep.shape, region, s.scan_end).unwrap();
        let first = (0..n)
            .find(|&i| clock.check(&obj(i), rep.kept[i].taken).unwrap().is_some())
            .unwrap();
        let (i, after) = first_converged(&clock, &rep.kept, rep.shape, n)
            .unwrap()
            .unwrap();
        assert_eq!(i, first);
        assert_eq!(after.as_nanos() as i64, rep.kept[i].taken - s.scan_end);
    }

    /// The reported epoch time is the GPU fences' over the epochs after the
    /// scan landed. Every epoch has an end, in order, each before the host
    /// held a snapshot that follows it.
    #[test]
    fn the_epoch_time_is_the_gpu_fences_after_the_scan() {
        let rep = streamed(None).rep;
        let (e, s) = (rep.epochs as usize, rep.scan_epochs.unwrap() as usize);
        assert!(s > 0, "no epoch ran while the scan landed");
        let done = &rep.epoch_done;
        assert_eq!(done.len(), e);
        assert!(done.windows(2).all(|w| w[0] < w[1]));
        // CUDA times events to about 0.5 µs.
        for k in &rep.kept {
            assert!(done[k.epochs as usize - 1] <= k.taken + rep.epoch_clock_ns + 1_000);
        }
        let gpu = (done[e - 1] - done[s - 1]) as f64 / 1e6 / (e - s) as f64;
        assert_eq!(rep.epoch_ms(), Some(gpu));
    }

    /// A 12-column raster streamed through the ring into the consumer on 1,
    /// 2 and 4 GPUs, each GPU's store filled by the uploader with the frames
    /// the plan gives it: every frame lands with no recapture, epochs run
    /// while the scan lands, and the last kept object and the viewer's last
    /// copy are one GPU's replay of the landings, bit for bit. The offline
    /// run on those stores settles alike on 1, 2 and 4 GPUs: at the same
    /// epochs, with the same verdict.
    #[test]
    fn gpus_follow_a_stream_as_one_gpu_replays_its_landings() {
        let p = Ptycho::new(PtychoSpec {
            n: 64,
            path: ScanPath::Raster { cols: 12 },
            points: 240,
            step: 8.0,
            photons: 1e6,
            seed: 5,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        });
        let (n, total) = (64usize, p.points());
        let pos: Vec<(f64, f64)> = (0..total).map(|k| p.position(k)).collect();
        let counts: Vec<u16> = (0..total).flat_map(|k| p.frame(k)).collect();
        let rc = ReconCfg {
            nf_max: NonZeroU32::new(8).unwrap(),
            snap_every: NonZeroU32::new(5).unwrap(),
            snaps: 3,
            ..ReconCfg::default()
        };
        let cfg = FollowCfg::plan(&p, NonZeroU32::new(total as u32).unwrap(), &rc);
        let ctx = Context::new(0).expect("GPU 0");
        // The one-GPU run's last object and what its offline run found.
        let (mut probe_obj, mut one) = (None, None);
        for ngpu in [1, 2, 4] {
            let ordinals: Vec<usize> = (0..ngpu).collect();
            let ring = Arc::new(Ring::new(&ctx, 16, n * n).unwrap());
            let stop = Arc::new(AtomicBool::new(false));
            let (tx, rx) = crossbeam_channel::unbounded();
            let (ready_tx, ready_rx) = crossbeam_channel::bounded(1);
            let (view_tx, view_rx) = crossbeam_channel::bounded(1);
            let viewer = std::thread::spawn(move || view_rx.iter().collect::<Vec<View>>());
            let (llk_tx, llk_rx) = crossbeam_channel::unbounded::<Vec<f64>>();
            let consumer = {
                let cfg = cfg.clone();
                let preview = Some(Preview {
                    period: Duration::from_millis(10),
                    tx: view_tx,
                    llk_period: Duration::from_millis(10),
                    llk: llk_tx,
                });
                std::thread::spawn(move || {
                    run_dist_consumer(&ordinals, Scheme::Strips, cfg, rx, ready_tx, preview)
                })
            };
            let ((stores, route), consumer) = sent(&ready_rx, consumer);
            assert_eq!(stores.len(), ngpu);
            let filled = stores.clone();
            let warm = WarmUp::end();
            let table = Arc::new(PosTable::new(1 << 16));
            let uploader = {
                let (ring, stop, table) = (ring.clone(), stop.clone(), table.clone());
                let up = UploaderConfig {
                    batch: NonZeroUsize::new(4).unwrap(),
                    flush_ns: 100_000,
                    idle: Duration::from_micros(20),
                    events: 4,
                    pos_wait_ns: 1_000_000_000,
                };
                std::thread::spawn(move || {
                    run_region_uploader(ring, stores, route, table, up, stop, Some(tx))
                })
            };
            let mut k = 0;
            for burst in [50, 1, 90, 0, 99] {
                for _ in 0..burst {
                    let mut s = ring.reserve();
                    s.frame()
                        .copy_from_slice(&counts[k * n * n..(k + 1) * n * n]);
                    let (x, y) = pos[k];
                    table.insert(k as i32, Pos { x, y, t_rx: 0 });
                    s.commit(k as i32, 0, 0);
                    k += 1;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            assert_eq!(k, total);
            stop.store(true, Ordering::Release);
            let up = uploader.join().unwrap().unwrap();
            assert_eq!((up.frames, up.pos_missing), (total as u64, 0));
            let rep = consumer.join().unwrap().unwrap();
            let views = viewer.join().unwrap();
            let llk: Vec<u64> = llk_rx.iter().flatten().map(f64::to_bits).collect();
            assert_eq!(warm.recaptures(), 0, "{ngpu} GPUs");
            assert_eq!(rep.positions, pos);
            assert_eq!((rep.stored, rep.landed), (total as u64, total as u64));
            let scan_epochs = rep.scan_epochs.unwrap();
            assert!(scan_epochs > 0, "no epoch ran while the scan landed");
            assert_eq!(rep.kept.len(), 3);
            let last = rep.kept.last().unwrap();
            assert_eq!((rep.epochs, last.epochs), (scan_epochs + 15, rep.epochs));
            assert_eq!(rep.epoch_done.len() as u64, rep.epochs);
            assert!(rep.epoch_done.windows(2).all(|w| w[0] < w[1]));
            let end = views.last().unwrap();
            assert_eq!(end.epochs, last.epochs);
            assert_eq!((end.shape, &end.pixels[..]), (rep.shape, &last.pixels[..]));

            ctx.bind_to_thread().unwrap();
            let st = ctx.new_stream().unwrap();
            let dev = cuda_core::DeviceBuffer::from_host(&st, &counts).unwrap();
            let mut run = LiveAp::new(&st, &cfg.ap, &cfg.probe, dev, true).unwrap();
            let mut backer = run.backer();
            let mut at = 0;
            for &(epochs, upto) in &rep.landings {
                while run.epochs() < epochs {
                    run.epoch().unwrap();
                }
                let v = views_of(&cfg.ap, &cfg.place, &pos[at as usize..upto as usize]);
                backer.back(&v).unwrap();
                run.push(&v).unwrap();
                at = upto;
            }
            while run.epochs() < last.epochs {
                run.epoch().unwrap();
            }
            let mut snap = run.new_snapshot().unwrap();
            run.snapshot(&mut snap).unwrap();
            assert_eq!(rep.shape, snap.shape());
            let got = snap.wait().unwrap();
            let diff = got[..last.pixels.len()]
                .iter()
                .zip(&last.pixels)
                .filter(|(a, b)| a.map(f32::to_bits) != b.map(f32::to_bits))
                .count();
            assert_eq!(diff, 0, "{ngpu} GPUs: pixels unlike the replay's");
            let replayed = run.llk().unwrap().unwrap();
            assert!(
                replayed
                    .iter()
                    .map(|(_, l)| l.to_bits())
                    .eq(llk.iter().copied()),
                "{ngpu} GPUs: {} LLKs sent, {} replayed, or unlike",
                llk.len(),
                replayed.len()
            );

            // Capped, so the run may stop while still moving: what it found then
            // names the ring and its FRC, which depend on the object as well.
            let region = cfg.place.scanned(&pos, cfg.ap.shape.frame);
            let off = offline_dist(&filled, Scheme::Strips, &cfg, &pos, region, 0, 16).unwrap();
            let obj = &*probe_obj.get_or_insert_with(|| plane(&last.pixels, rep.shape));
            let found = match off.clock {
                Ok(c) => format!(
                    "settled {:x}, scores {:x} {:x}",
                    c.settled().to_bits(),
                    c.region_frc(obj).unwrap().to_bits(),
                    c.whole(obj).unwrap().to_bits()
                ),
                Err(e) => e.to_string(),
            };
            let found = (off.epochs, found);
            assert_eq!(
                &found,
                one.get_or_insert_with(|| found.clone()),
                "{ngpu} GPUs: the offline run"
            );
        }
    }
}
