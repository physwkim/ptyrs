//! Angle-parallel ptycho-tomography (DESIGN B4): each GPU reconstructs one angle at a time with
//! no communication, and an angle starts from the final probe of the angle `lag` before it.
//! The source is a rule of the angle index alone, never of which GPU finished first, so the
//! results are bit-identical for any number of GPUs.

use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::dist::{DistError, Gpu, Workers};

/// One angle's reconstruction on a worker's GPU; the engine implements it.
pub trait AngleRecon: Send + Sync + 'static {
    /// Per-GPU state reused across angles (buffers, FFT plans): the `S` of [`Workers`].
    type State: Send + 'static;
    /// Handed from an angle to a later one on any GPU, so a device-resident probe must be
    /// readable from, or copied to, the GPU that takes it.
    type Probe: Clone + Send + 'static;
    type Output: Send + 'static;

    /// Reconstructs `angle` from `probe` and returns its result and final probe. For bit
    /// identity across GPU counts it depends on `angle` and `probe` only.
    fn reconstruct(
        &self,
        gpu: &mut Gpu,
        state: &mut Self::State,
        angle: usize,
        probe: Self::Probe,
    ) -> Result<(Self::Output, Self::Probe), Box<dyn std::error::Error + Send + Sync>>;
}

/// Reconstructs angles `0..nangle` on `workers`, each GPU taking the next angle in index order
/// as it frees. Angle `a` starts from the final probe of angle `a - lag`, or from `initial`
/// when `a < lag`. A `lag` of at least the GPU count keeps every GPU busy; it must not be
/// derived from the GPU count, or the results change with it.
pub fn reconstruct<R: AngleRecon>(
    workers: &Workers<R::State>,
    recon: Arc<R>,
    nangle: usize,
    lag: NonZeroUsize,
    initial: R::Probe,
) -> Result<Vec<R::Output>, DistError> {
    let book = Arc::new(Book::new(nangle, lag.get(), initial));
    let b = book.clone();
    workers.run(move |gpu, state| {
        while let Some((claim, probe)) = b.claim()? {
            let angle = claim.angle;
            let (out, next) = recon
                .reconstruct(gpu, state, angle, probe)
                .map_err(|source| DistError::Angle { angle, source })?;
            claim.finish(out, next);
        }
        Ok(())
    })?;
    let mut st = book.lock();
    Ok(st
        .out
        .drain(..)
        .map(|o| o.expect("a run that succeeded finished every angle"))
        .collect())
}

/// Probe slot `s` holds the starting probe of angle `s`. Slots `0..lag` are seeded with the
/// initial probe and angle `a` fills slot `a + lag`, so every probe is taken at most once.
enum Slot<P> {
    Empty,
    Ready(P),
    Taken,
}

struct State<P, O> {
    /// Next angle to claim; angles are claimed in index order.
    next: usize,
    /// A claimed angle ended without finishing; set only by a dropped [`Claim`].
    failed: bool,
    slots: Vec<Slot<P>>,
    out: Vec<Option<O>>,
}

struct Book<P, O> {
    lag: usize,
    state: Mutex<State<P, O>>,
    cv: Condvar,
}

impl<P: Clone, O> Book<P, O> {
    fn new(nangle: usize, lag: usize, initial: P) -> Self {
        let slots = (0..nangle + lag)
            .map(|s| {
                if s < lag {
                    Slot::Ready(initial.clone())
                } else {
                    Slot::Empty
                }
            })
            .collect();
        Self {
            lag,
            state: Mutex::new(State {
                next: 0,
                failed: false,
                slots,
                out: (0..nangle).map(|_| None).collect(),
            }),
            cv: Condvar::new(),
        }
    }
}

impl<P, O> Book<P, O> {
    fn lock(&self) -> MutexGuard<'_, State<P, O>> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Claims the next angle and waits for its starting probe; `None` when every angle is
    /// claimed. Every angle below the claimed one is already claimed, so the lowest unfinished
    /// angle always has its probe and the waits cannot deadlock.
    fn claim(&self) -> Result<Option<Claimed<'_, P, O>>, DistError> {
        let mut st = self.lock();
        if st.failed {
            return Err(DistError::Aborted);
        }
        let angle = st.next;
        if angle == st.out.len() {
            return Ok(None);
        }
        st.next += 1;
        while matches!(st.slots[angle], Slot::Empty) && !st.failed {
            st = self.cv.wait(st).unwrap_or_else(|p| p.into_inner());
        }
        if st.failed {
            return Err(DistError::Aborted);
        }
        let Slot::Ready(probe) = std::mem::replace(&mut st.slots[angle], Slot::Taken) else {
            unreachable!("slot {angle} has one taker, angle {angle}");
        };
        Ok(Some((
            Claim {
                book: self,
                angle,
                finished: false,
            },
            probe,
        )))
    }
}

/// A claimed angle and its starting probe.
type Claimed<'a, P, O> = (Claim<'a, P, O>, P);

/// A claimed angle. Dropping it unfinished (an error or a panic in the reconstruction) fails
/// the run and releases every worker waiting for a probe.
struct Claim<'a, P, O> {
    book: &'a Book<P, O>,
    angle: usize,
    finished: bool,
}

impl<P, O> Claim<'_, P, O> {
    fn finish(mut self, out: O, probe: P) {
        let mut st = self.book.lock();
        let next = self.angle + self.book.lag;
        assert!(
            matches!(st.slots[next], Slot::Empty),
            "slot {next} is filled only by angle {}",
            self.angle
        );
        st.slots[next] = Slot::Ready(probe);
        st.out[self.angle] = Some(out);
        self.finished = true;
        self.book.cv.notify_all();
    }
}

impl<P, O> Drop for Claim<'_, P, O> {
    fn drop(&mut self) {
        if !self.finished {
            self.book.lock().failed = true;
            self.book.cv.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use cuda_core::{DeviceBuffer, Function, launch_kernel_on_stream};

    use super::*;
    use crate::dist::worker::tests::gpus;

    const WORDS: usize = 4096;
    const THREADS: u32 = 256;
    const STEPS: u32 = 3;
    const SEED: u32 = 0x2545_f491;
    const TWO_M24: f32 = 1.0 / 16_777_216.0;

    fn lowbias32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    fn key(angle: usize, step: u32) -> u32 {
        (angle as u32 * STEPS + step)
            .wrapping_mul(0x9e37_79b1)
            .wrapping_add(SEED)
    }

    /// Host twin of `tomo_step`.
    fn step(p: &mut [f32], key: u32) {
        for (i, v) in p.iter_mut().enumerate() {
            let h = lowbias32((i as u32).wrapping_mul(0x85eb_ca77).wrapping_add(key));
            *v = v.mul_add(0.75, (h >> 8) as f32 * TWO_M24);
        }
    }

    #[derive(Clone)]
    struct Probe {
        /// Angle whose final probe this is; `None` for the initial probe.
        from: Option<usize>,
        v: Vec<f32>,
    }

    struct Done {
        source: Option<usize>,
        /// Position of this angle in the order angles finished.
        rank: usize,
        bits: Vec<u32>,
    }

    #[derive(Clone, Copy)]
    enum Fault {
        Error,
        Panic,
    }

    /// `STEPS` GPU steps of the probe per angle, after a sleep that varies with the angle so
    /// angles finish out of index order.
    struct Synth {
        finished: AtomicUsize,
        fault: Option<(usize, Fault)>,
    }

    impl Synth {
        fn new(fault: Option<(usize, Fault)>) -> Arc<Self> {
            Arc::new(Self {
                finished: AtomicUsize::new(0),
                fault,
            })
        }
    }

    impl AngleRecon for Synth {
        type State = Function;
        type Probe = Probe;
        type Output = Done;

        fn reconstruct(
            &self,
            gpu: &mut Gpu,
            f: &mut Function,
            angle: usize,
            probe: Probe,
        ) -> Result<(Done, Probe), Box<dyn std::error::Error + Send + Sync>> {
            match self.fault {
                Some((a, Fault::Error)) if a == angle => return Err("synthetic failure".into()),
                Some((a, Fault::Panic)) if a == angle => panic!("synthetic panic"),
                _ => {}
            }
            std::thread::sleep(Duration::from_millis(
                u64::from(lowbias32(angle as u32) % 4) * 3,
            ));
            let s = gpu.stream().clone();
            let buf = DeviceBuffer::from_host(&s, &probe.v)?;
            for k in 0..STEPS {
                let mut ptr = buf.cu_deviceptr();
                let mut b = [WORDS as u32, key(angle, k)];
                let mut params: Vec<*mut c_void> = std::iter::once((&raw mut ptr).cast())
                    .chain(b.iter_mut().map(|p| (p as *mut u32).cast()))
                    .collect();
                // SAFETY: one thread per word of `buf`, which holds `WORDS` f32.
                unsafe {
                    launch_kernel_on_stream(
                        f,
                        ((WORDS as u32).div_ceil(THREADS), 1, 1),
                        (THREADS, 1, 1),
                        0,
                        &s,
                        &mut params,
                    )?
                };
            }
            let v = buf.to_host_vec(&s)?;
            let done = Done {
                source: probe.from,
                rank: self.finished.fetch_add(1, Ordering::SeqCst),
                bits: v.iter().map(|x| x.to_bits()).collect(),
            };
            Ok((
                done,
                Probe {
                    from: Some(angle),
                    v,
                },
            ))
        }
    }

    fn initial() -> Probe {
        Probe {
            from: None,
            v: (0..WORDS).map(|i| (i % 97) as f32 / 97.0).collect(),
        }
    }

    fn workers(n: usize) -> Workers<Function> {
        Workers::spawn(&gpus(n), |g| {
            Ok(g.ctx()
                .load_module_from_ptx_src(PTX)?
                .load_function("tomo_step")?)
        })
        .unwrap()
    }

    fn lag(l: usize) -> NonZeroUsize {
        NonZeroUsize::new(l).unwrap()
    }

    /// Each angle's bits equal the host chain in which angle `a` steps the final probe of
    /// angle `a - lag`, on 1, 2 and 4 GPUs.
    #[test]
    fn angles_are_bit_identical_on_1_2_and_4_gpus() {
        let (nangle, l) = (24, 4);
        let mut want: Vec<Vec<f32>> = Vec::with_capacity(nangle);
        for a in 0..nangle {
            let mut p = a
                .checked_sub(l)
                .map_or_else(|| initial().v, |s| want[s].clone());
            for k in 0..STEPS {
                step(&mut p, key(a, k));
            }
            want.push(p);
        }
        for n in [1, 2, 4] {
            let got =
                reconstruct(&workers(n), Synth::new(None), nangle, lag(l), initial()).unwrap();
            assert_eq!(got.len(), nangle);
            for (a, d) in got.iter().enumerate() {
                let w: Vec<u32> = want[a].iter().map(|x| x.to_bits()).collect();
                assert!(d.bits == w, "{n} GPUs, angle {a}");
            }
        }
    }

    /// Angles finish out of index order, yet angle `a` always starts from the probe of angle
    /// `a - lag`, including a lag below the GPU count (GPUs wait on each other) and above it.
    #[test]
    fn each_angle_starts_from_the_probe_lag_angles_before() {
        let nangle = 20;
        let mut reordered = false;
        for n in [1, 2, 4] {
            let w = workers(n);
            for l in [1, 3, 5] {
                let got = reconstruct(&w, Synth::new(None), nangle, lag(l), initial()).unwrap();
                for (a, d) in got.iter().enumerate() {
                    assert_eq!(d.source, a.checked_sub(l), "{n} GPUs, lag {l}, angle {a}");
                }
                reordered |= got.windows(2).any(|p| p[0].rank > p[1].rank);
            }
        }
        assert!(reordered, "no angle finished before a lower one");
    }

    /// A failing or panicking angle fails the run with its own error and releases the workers
    /// waiting for its probe (lag 1 chains every later angle to it).
    #[test]
    fn a_failing_angle_releases_the_waiting_workers() {
        // Built before the clock starts: the limit below times the runs, not context and
        // module setup, which alone can pass a minute while the full suite loads the GPUs.
        let w = workers(2);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let run = |fault| reconstruct(&w, Synth::new(fault), 12, lag(1), initial());
            let _ = tx.send((
                run(Some((5, Fault::Error))).err(),
                run(Some((6, Fault::Panic))).err(),
                run(None).map(|v| v.len()).ok(),
            ));
        });
        // A worker left waiting on the failed angle's probe would hang the run, not fail it.
        let (error, panic, ok) = rx
            .recv_timeout(Duration::from_secs(60))
            .expect("a run hung after an angle failed");
        let error = error.unwrap();
        assert!(
            matches!(error, DistError::Angle { angle: 5, .. }),
            "{error}"
        );
        let panic = panic.unwrap();
        assert!(matches!(panic, DistError::Panic { .. }), "{panic}");
        assert_eq!(ok, Some(12));
    }

    const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// p[i] = fma(p[i], 0.75, (lowbias32(i * 0x85EBCA77 + key) >> 8) * 2^-24) for i < n.
.visible .entry tomo_step(
    .param .u64 p_probe,
    .param .u32 p_n,
    .param .u32 p_key
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [p_probe];
    cvta.to.global.u64 %rd1, %rd1;
    ld.param.u32 %r1, [p_n];
    ld.param.u32 %r2, [p_key];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;
    mad.lo.u32 %r7, %r6, 0x85EBCA77, %r2;
    shr.u32 %r8, %r7, 16;
    xor.b32 %r7, %r7, %r8;
    mul.lo.u32 %r7, %r7, 0x7FEB352D;
    shr.u32 %r8, %r7, 15;
    xor.b32 %r7, %r7, %r8;
    mul.lo.u32 %r7, %r7, 0x846CA68B;
    shr.u32 %r8, %r7, 16;
    xor.b32 %r7, %r7, %r8;
    shr.u32 %r7, %r7, 8;
    cvt.rn.f32.u32 %f1, %r7;
    mul.rn.f32 %f1, %f1, 0f33800000;
    mul.wide.u32 %rd2, %r6, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f2, [%rd3];
    fma.rn.f32 %f3, %f2, 0f3F400000, %f1;
    st.global.f32 [%rd3], %f3;
DONE:
    ret;
}
"#;
}
