//! One OS thread per GPU, each with its device's context, its own streams and its own state.
//!
//! Cross-GPU ordering has one owner, [`Gpu::fence`]: work any GPU enqueued before a fence, on
//! either of its streams, is complete before work any GPU enqueues after it. Peers only read
//! each other's memory (invariant 5), so a fence between a producer step and its readers, and
//! another before the producer overwrites, is all the exchange needs.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use cuda_core::{Context, Event, IntoResult, Stream, sys};

use crate::dist::DistError;
use crate::dist::topo::enable_all_pairs;

/// What a job sees: its region, its device, its main and side streams and the shared fence.
pub struct Gpu {
    region: usize,
    ctx: Arc<Context>,
    stream: Arc<Stream>,
    side: Side,
    shared: Arc<Shared>,
    fences: usize,
}

/// A second stream, reachable only through [`Gpu::side`], which forks it from the main stream
/// first: side work is ordered after everything enqueued on the main stream before it, the
/// fences' waits included, and overlaps what the main stream runs next until [`Gpu::join`].
struct Side {
    stream: Arc<Stream>,
    fork: Event,
    join: Event,
}

impl Gpu {
    pub fn region(&self) -> usize {
        self.region
    }

    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    pub fn nregion(&self) -> usize {
        self.shared.rendezvous.n
    }

    pub fn ctx(&self) -> &Arc<Context> {
        &self.ctx
    }

    /// The main stream.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Runs `f` on the side stream, forked from the main stream first. The side stream has the
    /// device's highest priority, so its kernels take free SMs ahead of the main stream's.
    pub fn side<R>(
        &self,
        f: impl FnOnce(&Arc<Stream>) -> Result<R, DistError>,
    ) -> Result<R, DistError> {
        let s = &self.side;
        s.fork.record(&self.stream)?;
        s.stream.wait(&s.fork)?;
        f(&s.stream)
    }

    /// Makes the main stream wait for everything enqueued on the side stream so far.
    pub fn join(&self) -> Result<(), DistError> {
        let s = &self.side;
        s.join.record(&s.stream)?;
        Ok(self.stream.wait(&s.join)?)
    }

    /// Blocks until both streams are idle.
    pub fn synchronize(&self) -> Result<(), DistError> {
        self.join()?;
        Ok(self.stream.synchronize()?)
    }

    /// Orders every GPU's prior work, on both streams, before every GPU's later work, without
    /// a host sync: join the side stream, record this GPU's event, meet the other workers on
    /// the host so every record is enqueued, then make the main stream wait on every peer's
    /// event. All workers of a run must call it the same number of times.
    pub fn fence(&mut self) -> Result<(), DistError> {
        let events = &self.shared.events;
        // Alternate two events: a peer can only re-record parity p after the next fence's
        // rendezvous, which this worker reaches only after enqueuing its waits on parity p.
        let p = self.fences % 2;
        self.fences += 1;
        self.join()?;
        events[self.region][p].record(&self.stream)?;
        self.shared.rendezvous.wait()?;
        for (r, e) in events.iter().enumerate() {
            if r != self.region {
                wait_event(&self.stream, &e[p])?;
            }
        }
        Ok(())
    }
}

/// `stream` waits for `event`, which may belong to another device's context.
fn wait_event(stream: &Stream, event: &Event) -> Result<(), DistError> {
    stream.context().bind_to_thread()?;
    // SAFETY: both handles are live; cross-context waits are valid in the driver API.
    unsafe { sys::cuStreamWaitEvent(stream.cu_stream(), event.cu_event(), 0).result()? };
    Ok(())
}

struct Shared {
    rendezvous: Rendezvous,
    /// Each region's two fence events, made before any worker starts.
    events: Vec<[Event; 2]>,
}

/// Host barrier of the `n` workers. A worker that fails, panics or finishes while others
/// wait aborts it, so no worker is left blocked.
struct Rendezvous {
    n: usize,
    state: Mutex<RvState>,
    cv: Condvar,
}

#[derive(Default)]
struct RvState {
    arrived: usize,
    departed: usize,
    generation: u64,
    stop: Option<Stop>,
}

#[derive(Clone, Copy)]
enum Stop {
    /// A worker failed.
    Failed,
    /// A worker finished while others still had fences to meet.
    Mismatch,
}

impl Stop {
    fn error(self) -> DistError {
        match self {
            Stop::Failed => DistError::Aborted,
            Stop::Mismatch => DistError::FenceMismatch,
        }
    }
}

impl Rendezvous {
    fn lock(&self) -> std::sync::MutexGuard<'_, RvState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn stop(&self, s: &mut RvState, why: Stop) -> DistError {
        let why = *s.stop.get_or_insert(why);
        self.cv.notify_all();
        why.error()
    }

    fn wait(&self) -> Result<(), DistError> {
        let mut s = self.lock();
        if let Some(why) = s.stop {
            return Err(why.error());
        }
        if s.departed > 0 {
            return Err(self.stop(&mut s, Stop::Mismatch));
        }
        let generation = s.generation;
        s.arrived += 1;
        if s.arrived == self.n {
            s.arrived = 0;
            s.generation += 1;
            self.cv.notify_all();
            return Ok(());
        }
        while s.generation == generation && s.stop.is_none() {
            s = self.cv.wait(s).unwrap_or_else(|p| p.into_inner());
        }
        match s.stop {
            Some(why) if s.generation == generation => Err(why.error()),
            _ => Ok(()),
        }
    }

    /// A worker's job ended; `ok` is false when it failed. Workers still waiting are released.
    fn depart(&self, ok: bool) {
        let mut s = self.lock();
        s.departed += 1;
        if !ok {
            self.stop(&mut s, Stop::Failed);
        } else if s.arrived > 0 {
            self.stop(&mut s, Stop::Mismatch);
        }
    }

    fn reset(&self) {
        *self.lock() = RvState::default();
    }
}

type Job<S> = Box<dyn FnOnce(&mut Gpu, &mut S) + Send>;

/// Worker threads, region `r` on `ordinals[r]`. Regions may share a GPU, each with its own
/// thread and stream; peer access is enabled for every pair of distinct GPUs.
pub struct Workers<S: Send + 'static> {
    shared: Arc<Shared>,
    senders: Vec<Sender<Job<S>>>,
    handles: Vec<JoinHandle<()>>,
}

impl<S: Send + 'static> Workers<S> {
    /// Starts one thread per ordinal; each builds its state with `init` on its own thread.
    /// `init` sees its [`Gpu`] shared, so it cannot [`Gpu::fence`]: the rendezvous counts
    /// only jobs' fences.
    pub fn spawn<I>(ordinals: &[usize], init: I) -> Result<Self, DistError>
    where
        I: Fn(&Gpu) -> Result<S, DistError> + Send + Sync + 'static,
    {
        if ordinals.is_empty() {
            return Err(DistError::Partition("no GPUs".into()));
        }
        let ctxs = ordinals
            .iter()
            .map(|&o| Context::new(o))
            .collect::<Result<Vec<_>, _>>()?;
        let distinct: Vec<_> = ctxs
            .iter()
            .enumerate()
            .filter(|(i, c)| !ordinals[..*i].contains(&c.ordinal()))
            .map(|(_, c)| c.clone())
            .collect();
        enable_all_pairs(&distinct)?;
        let events = ctxs
            .iter()
            .map(|c| Ok([c.new_event_with_flags(None)?, c.new_event_with_flags(None)?]))
            .collect::<Result<_, DistError>>()?;
        let shared = Arc::new(Shared {
            rendezvous: Rendezvous {
                n: ordinals.len(),
                state: Mutex::default(),
                cv: Condvar::new(),
            },
            events,
        });
        let init = Arc::new(init);
        let (ready_tx, ready_rx) = channel();
        let mut workers = Self {
            shared: shared.clone(),
            senders: Vec::new(),
            handles: Vec::new(),
        };
        for (region, ctx) in ctxs.into_iter().enumerate() {
            let (tx, rx) = channel::<Job<S>>();
            let (init, shared, ready) = (init.clone(), shared.clone(), ready_tx.clone());
            let handle = std::thread::Builder::new()
                .name(format!("ptyrs-r{region}-gpu{}", ctx.ordinal()))
                .spawn(move || worker(region, ctx, shared, init, ready, rx))
                .map_err(|e| DistError::Topology(format!("spawn worker: {e}")))?;
            workers.senders.push(tx);
            workers.handles.push(handle);
        }
        drop(ready_tx);
        let mut reported = vec![false; ordinals.len()];
        let mut first_err = None;
        for (r, res) in ready_rx {
            reported[r] = true;
            if let Err(e) = res {
                first_err = first_err.or(Some(e));
            }
        }
        let lost = reported.iter().position(|&r| !r).map(thread_ended);
        match first_err.or(lost) {
            Some(e) => Err(e),
            None => Ok(workers),
        }
    }

    pub fn len(&self) -> usize {
        self.senders.len()
    }

    pub fn is_empty(&self) -> bool {
        self.senders.is_empty()
    }

    /// Runs `f` on every worker at once and returns the results in region order. On failure
    /// the error of the lowest failing region is returned, preferring a root cause over
    /// [`DistError::Aborted`] from the workers it released. `f` may borrow from the caller:
    /// `run` returns only after every worker is done with it.
    pub fn run<R, F>(&self, f: F) -> Result<Vec<R>, DistError>
    where
        R: Send + 'static,
        F: Fn(&mut Gpu, &mut S) -> Result<R, DistError> + Sync,
    {
        self.shared.rendezvous.reset();
        let f: &Body<'_, S, R> = &f;
        // SAFETY: only the jobs below hold `f`, and `results` does not let this frame end,
        // unwinding included, before every job has run or been dropped.
        let f: &'static Body<'static, S, R> = unsafe { std::mem::transmute(f) };
        let mut results = Results::new();
        // Regions up to this one were handed the job or found their worker gone; the rest
        // never run.
        let mut reached = self.len();
        for (region, s) in self.senders.iter().enumerate() {
            let tx = results.sender();
            let job: Job<S> = Box::new(move |gpu, state| {
                let out = catch_unwind(AssertUnwindSafe(|| {
                    gpu.ctx.bind_to_thread()?;
                    f(gpu, state)
                }))
                .unwrap_or_else(|p| {
                    Err(DistError::Panic {
                        region,
                        msg: panic_msg(&p),
                    })
                });
                gpu.shared.rendezvous.depart(out.is_ok());
                let _ = tx.send((region, out));
            });
            if s.send(job).is_err() {
                // The worker is gone: release the ones already started, unsent ones report
                // `Aborted`.
                self.shared.rendezvous.depart(false);
                reached = region + 1;
                break;
            }
        }
        let mut out: Vec<Option<Result<R, DistError>>> = (0..self.len()).map(|_| None).collect();
        for (r, res) in results.wait() {
            out[r] = Some(res);
        }
        let mut results = Vec::with_capacity(out.len());
        let mut err: Option<DistError> = None;
        for (region, res) in out.into_iter().enumerate() {
            let res = res.unwrap_or_else(|| {
                Err(if region < reached {
                    thread_ended(region)
                } else {
                    DistError::Aborted
                })
            });
            match res {
                Ok(v) => results.push(v),
                Err(e) => {
                    // Keep the first root cause; workers it released report `Aborted`.
                    if err.as_ref().is_none_or(|x| matches!(x, DistError::Aborted)) {
                        err = Some(e);
                    }
                }
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok(results),
        }
    }
}

type Body<'a, S, R> = dyn Fn(&mut Gpu, &mut S) -> Result<R, DistError> + Sync + 'a;

/// A run's workers and whether a call that changes their state failed, which only
/// [`Gated::change`] marks.
pub(crate) struct Gated<S: Send + 'static> {
    workers: Workers<S>,
    /// A call through [`Self::change`] failed part-way: the GPUs may disagree, so every later
    /// call but [`Self::drain`] fails.
    failed: bool,
}

impl<S: Send + 'static> Gated<S> {
    pub(crate) fn new(workers: Workers<S>) -> Self {
        Self {
            workers,
            failed: false,
        }
    }

    /// [`DistError::Poisoned`] once a call through [`Self::change`] has failed.
    pub(crate) fn check(&self) -> Result<(), DistError> {
        if self.failed {
            return Err(DistError::Poisoned);
        }
        Ok(())
    }

    /// `f` on every GPU, writing state a later call reads: refused once a change has failed,
    /// and failing on any GPU fails every later call.
    pub(crate) fn change<R, F>(&mut self, f: F) -> Result<Vec<R>, DistError>
    where
        R: Send + 'static,
        F: Fn(&mut Gpu, &mut S) -> Result<R, DistError> + Sync,
    {
        self.check()?;
        self.failed = true;
        let r = self.workers.run(f)?;
        self.failed = false;
        Ok(r)
    }

    /// `f` on every GPU, writing nothing a later call reads: refused once a change has
    /// failed.
    pub(crate) fn read<R, F>(&self, f: F) -> Result<Vec<R>, DistError>
    where
        R: Send + 'static,
        F: Fn(&mut Gpu, &mut S) -> Result<R, DistError> + Sync,
    {
        self.check()?;
        self.workers.run(f)
    }

    /// Waits for every GPU's queued work, after a failed change too, so that nothing the run
    /// queued still reads or writes its buffers once this returns.
    pub(crate) fn drain(&self) -> Result<(), DistError> {
        self.workers.run(|gpu, _| gpu.synchronize())?;
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.workers.len()
    }
}

/// The channel a run's jobs report on. Dropping it waits until every job has run or been
/// dropped, so a closure the jobs borrow outlives them.
struct Results<T> {
    tx: Option<Sender<T>>,
    rx: Receiver<T>,
}

impl<T> Results<T> {
    fn new() -> Self {
        let (tx, rx) = channel();
        Self { tx: Some(tx), rx }
    }

    fn sender(&self) -> Sender<T> {
        self.tx.clone().expect("sender taken before wait")
    }

    /// Every result, ending once no job is left.
    fn wait(&mut self) -> std::sync::mpsc::Iter<'_, T> {
        self.tx = None;
        self.rx.iter()
    }
}

impl<T> Drop for Results<T> {
    fn drop(&mut self) {
        self.wait().for_each(drop);
    }
}

impl<S: Send + 'static> Drop for Workers<S> {
    /// Peers may still be reading any worker's buffers, so every stream drains before any
    /// thread drops its state.
    fn drop(&mut self) {
        let _ = self.run(|gpu, _| gpu.synchronize());
        self.senders.clear();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

type Ready = Sender<(usize, Result<(), DistError>)>;

fn worker<S, I>(
    region: usize,
    ctx: Arc<Context>,
    shared: Arc<Shared>,
    init: Arc<I>,
    ready: Ready,
    jobs: Receiver<Job<S>>,
) where
    I: Fn(&Gpu) -> Result<S, DistError>,
{
    let setup = || -> Result<Gpu, DistError> {
        ctx.bind_to_thread()?;
        let stream = ctx.new_stream()?;
        let side = Side {
            stream: ctx.new_stream_with_priority(ctx.stream_priority_range()?.greatest())?,
            fork: ctx.new_event_with_flags(None)?,
            join: ctx.new_event_with_flags(None)?,
        };
        let gpu = Gpu {
            region,
            ctx: ctx.clone(),
            stream,
            side,
            shared,
            fences: 0,
        };
        Ok(gpu)
    };
    let started = catch_unwind(AssertUnwindSafe(|| {
        let gpu = setup()?;
        let state = init(&gpu)?;
        Ok((gpu, state))
    }))
    .unwrap_or_else(|p| {
        Err(DistError::Panic {
            region,
            msg: panic_msg(&p),
        })
    });
    let (mut gpu, mut state) = match started {
        Ok((gpu, state)) => {
            let _ = ready.send((region, Ok(())));
            (gpu, state)
        }
        Err(e) => {
            let _ = ready.send((region, Err(e)));
            return;
        }
    };
    drop(ready);
    for job in jobs {
        job(&mut gpu, &mut state);
    }
    let _ = gpu.synchronize();
    drop(state);
}

/// Region `region`'s worker thread ended without reporting, which only a panic outside its
/// `catch_unwind` does.
fn thread_ended(region: usize) -> DistError {
    DistError::Panic {
        region,
        msg: "its worker thread ended".into(),
    }
}

fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic".into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::dist::topo;
    use cuda_core::DeviceBuffer;

    /// The four RTX 5000 Ada of the target machine, NUMA-grouped.
    pub(crate) fn gpus(n: usize) -> Vec<usize> {
        let order = topo::numa_grouped(&topo::devices().unwrap());
        assert!(
            order.len() >= n,
            "test needs {n} GPUs, found {}",
            order.len()
        );
        order[..n].to_vec()
    }

    struct Bufs {
        own: DeviceBuffer<u32>,
        seen: DeviceBuffer<u32>,
    }

    const LEN: usize = 1 << 22;

    /// Each round every GPU overwrites its buffer, fences, then copies every peer's buffer;
    /// after the fence it must see that round's value from every peer, never the previous.
    #[test]
    fn fence_orders_peer_writes_before_peer_reads() {
        let w = Workers::spawn(&gpus(4), |g| {
            Ok(Bufs {
                own: DeviceBuffer::zeroed(g.stream(), LEN)?,
                seen: DeviceBuffer::zeroed(g.stream(), 4 * LEN)?,
            })
        })
        .unwrap();
        let ptrs: Arc<Vec<u64>> = Arc::new(w.run(|_, b| Ok(b.own.cu_deviceptr())).unwrap());
        for round in 1..=4u32 {
            let ptrs = ptrs.clone();
            let got = w
                .run(move |g, b| {
                    let s = g.stream().cu_stream();
                    let value = round * 16 + g.region() as u32;
                    // SAFETY: `own` holds LEN u32 on this device; `seen` holds 4 * LEN.
                    unsafe { sys::cuMemsetD32Async(b.own.cu_deviceptr(), value, LEN, s).result()? };
                    g.fence()?;
                    for (r, &p) in ptrs.iter().enumerate() {
                        let dst = b.seen.cu_deviceptr() + (r * LEN * 4) as u64;
                        // SAFETY: peer access is enabled; both ranges hold LEN u32.
                        unsafe { sys::cuMemcpyDtoDAsync_v2(dst, p, LEN * 4, s).result()? };
                    }
                    // Nobody overwrites `own` until every peer's copy is done.
                    g.fence()?;
                    let seen = b.seen.to_host_vec(g.stream())?;
                    Ok((0..4).all(|r| {
                        seen[r * LEN..(r + 1) * LEN]
                            .iter()
                            .all(|&v| v == round * 16 + r as u32)
                    }))
                })
                .unwrap();
            assert_eq!(got, [true; 4], "round {round}");
        }
    }

    /// The fence covers the side stream: each round every GPU overwrites its buffer eight times
    /// on the side stream, the right value last, fences, copies every peer's buffer on the side
    /// stream, fences, then reads the copies on the main stream.
    #[test]
    fn fence_orders_side_stream_work() {
        let w = Workers::spawn(&gpus(4), |g| {
            Ok(Bufs {
                own: DeviceBuffer::zeroed(g.stream(), LEN)?,
                seen: DeviceBuffer::zeroed(g.stream(), 4 * LEN)?,
            })
        })
        .unwrap();
        let ptrs: Arc<Vec<u64>> = Arc::new(w.run(|_, b| Ok(b.own.cu_deviceptr())).unwrap());
        for round in 1..=4u32 {
            let ptrs = ptrs.clone();
            let got = w
                .run(move |g, b| {
                    let value = round * 16 + g.region() as u32;
                    let own = b.own.cu_deviceptr();
                    g.side(|s| {
                        for i in (0..8u32).rev() {
                            // SAFETY: `own` holds LEN u32 on this device.
                            unsafe {
                                sys::cuMemsetD32Async(own, value | i << 8, LEN, s.cu_stream())
                                    .result()?
                            };
                        }
                        Ok(())
                    })?;
                    g.fence()?;
                    g.side(|s| {
                        for (r, &p) in ptrs.iter().enumerate() {
                            let dst = b.seen.cu_deviceptr() + (r * LEN * 4) as u64;
                            // SAFETY: peer access is enabled; both ranges hold LEN u32.
                            unsafe {
                                sys::cuMemcpyDtoDAsync_v2(dst, p, LEN * 4, s.cu_stream())
                                    .result()?
                            };
                        }
                        Ok(())
                    })?;
                    g.fence()?;
                    let seen = b.seen.to_host_vec(g.stream())?;
                    Ok((0..4).all(|r| {
                        seen[r * LEN..(r + 1) * LEN]
                            .iter()
                            .all(|&v| v == round * 16 + r as u32)
                    }))
                })
                .unwrap();
            assert_eq!(got, [true; 4], "round {round}");
        }
    }

    #[test]
    fn a_failing_worker_releases_the_others() {
        let w = Workers::spawn(&gpus(4), |_| Ok(())).unwrap();
        let err = w
            .run(|g, _| {
                if g.region() == 2 {
                    return Err(DistError::Topology("boom".into()));
                }
                g.fence()?;
                g.fence()
            })
            .unwrap_err();
        assert!(
            matches!(err, DistError::Topology(ref m) if m == "boom"),
            "{err}"
        );

        let err = w
            .run(|g, _| {
                if g.region() == 1 {
                    panic!("kaboom");
                }
                g.fence()
            })
            .unwrap_err();
        assert!(matches!(err, DistError::Panic { region: 1, ref msg } if msg == "kaboom"));

        let err = w
            .run(|g, _| if g.region() == 0 { Ok(()) } else { g.fence() })
            .unwrap_err();
        assert!(matches!(err, DistError::FenceMismatch), "{err}");

        // The workers stay usable after each failure.
        let regions = w.run(|g, _| Ok(g.region())).unwrap();
        assert_eq!(regions, [0, 1, 2, 3]);
    }

    /// `run` borrows its closure, so it must not return, failure included, while a worker
    /// can still reach what the closure borrows.
    #[test]
    fn run_returns_after_every_worker_is_done_with_its_borrows() {
        use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
        let w = Workers::spawn(&gpus(4), |_| Ok(())).unwrap();
        let input: Vec<u32> = (10..14).collect();
        let late: Vec<AtomicU32> = (0..4).map(|_| AtomicU32::new(0)).collect();
        let err = w
            .run(|g, _| {
                let r = g.region();
                if r == 0 {
                    return Err(DistError::Topology("early".into()));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
                late[r].store(input[r], Relaxed);
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(err, DistError::Topology(_)), "{err}");
        let late: Vec<u32> = late.iter().map(|a| a.load(Relaxed)).collect();
        assert_eq!(late, [0, 11, 12, 13]);
    }

    #[test]
    fn init_failure_is_reported() {
        let err = Workers::<()>::spawn(&gpus(2), |g| {
            if g.region() == 1 {
                Err(DistError::Topology("no".into()))
            } else {
                Ok(())
            }
        })
        .err()
        .unwrap();
        assert!(matches!(err, DistError::Topology(_)), "{err}");
        let err = Workers::<()>::spawn(&[], |_| Ok(())).err().unwrap();
        assert!(matches!(err, DistError::Partition(_)), "{err}");
    }
}
