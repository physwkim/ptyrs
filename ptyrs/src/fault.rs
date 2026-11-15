//! The exits a call can fail at: part-way, after its LLK record's advance is enqueued, before
//! it counts its frames for one, or mid-update, between a DM or RAAR projection's stacks or an
//! object solve's steps, leaving its waves or object part updated, on one GPU or one of a
//! distributed run's, after which every later call must refuse; or in a read or a build,
//! which must leave the engine as it was.
//! [`at`] is a no-op outside `cfg(test)`; a unit test arms one exit with `arm` to make it
//! fail once, so the failure runs through the call's own wrapper.

/// A point [`at`] marks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Exit {
    /// An AP cycle's, after its record ends and before its update sums are applied.
    ApSums,
    /// An AP cycle's, after its update sums are applied and before its position step.
    ApPositions,
    /// ML's gradient pass's, after its record ends.
    MlGrad,
    /// ML's gradient pass's, before each stack.
    MlStack,
    /// The features' step's, after its LLK pass's record ends and before the step.
    Feature,
    /// A distributed live AP epoch's, after its record's replay and before its parity's
    /// event.
    DistFence,
    /// A distributed live AP run's LLK read's, on each GPU before its ring's flush.
    DistLlk,
    /// A distributed AP cycle's, on each GPU after its object update and before its probe
    /// update.
    DistUpdate,
    /// A distributed live AP run's snapshot's, on each GPU before its copy is queued.
    DistSnapshot,
    /// A DM or RAAR cycle's projection's, before each stack.
    DmStack,
    /// A live AP epoch's, before its ring counts the frames landed since its last count.
    LiveCount,
    /// The full-object object solve's, before each conjugate-gradient step.
    Solve,
    /// An epoch run's build, before its buffers are built.
    RunBuild,
    /// An epoch's, after its stage and before its cycle is enqueued.
    Launch,
    /// The phase ramp's removal's, after the object is written and before the probe is.
    RampWrite,
    /// The features' write in place, after the scale's and before the background's.
    FeatureCopy,
    /// The shared passes' rebuild for another position model's, after they are built.
    Reshare,
    /// A graph's capture's, before it begins.
    Capture,
}

#[cfg(not(test))]
#[inline(always)]
pub(crate) fn at(_: Exit) -> Result<(), crate::gpu::Error> {
    Ok(())
}

#[cfg(test)]
pub(crate) use armed::{REFUSAL, arm, at};

#[cfg(test)]
mod armed {
    use std::sync::{Mutex, MutexGuard, PoisonError};

    use cuda_core::DriverError;

    use super::Exit;

    /// Held by the armed test, so two never arm at once.
    static ARMING: Mutex<()> = Mutex::new(());
    /// The armed exit: the hits it lets pass before failing, and whether it has failed.
    static ARMED: Mutex<Option<(Exit, usize, bool)>> = Mutex::new(None);

    fn armed() -> MutexGuard<'static, Option<(Exit, usize, bool)>> {
        ARMED.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// What an armed exit fails with: a launch the driver refuses for its resources, which
    /// leaves the context usable.
    pub(crate) const REFUSAL: DriverError =
        DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);

    /// [`REFUSAL`] at the armed exit's hit after the ones it lets pass, once.
    pub(crate) fn at(exit: Exit) -> Result<(), crate::gpu::Error> {
        match &mut *armed() {
            Some((e, pass, fired)) if *e == exit && !*fired => match pass.checked_sub(1) {
                Some(left) => *pass = left,
                None => {
                    *fired = true;
                    return Err(REFUSAL.into());
                }
            },
            _ => {}
        }
        Ok(())
    }

    /// Arms `exit` to fail its hit after `pass` others, once, until the guard drops. Armed
    /// for the process, since dist's exits run on its workers' threads, so it panics unless
    /// nextest runs this test in a process of its own: under `cargo test` the exit would fire
    /// in whichever concurrent test hits it.
    pub(crate) fn arm(exit: Exit, pass: usize) -> Armed {
        assert!(
            matches!(
                std::env::var("NEXTEST_EXECUTION_MODE").as_deref(),
                Ok("process-per-test")
            ),
            "fault::arm arms every thread of the process, so it needs a process of the \
             arming test's own: run it under cargo nextest"
        );
        let hold = ARMING.lock().unwrap_or_else(PoisonError::into_inner);
        *armed() = Some((exit, pass, false));
        Armed { _hold: hold }
    }

    /// An armed exit; disarmed on drop.
    pub(crate) struct Armed {
        _hold: MutexGuard<'static, ()>,
    }

    impl Armed {
        /// Whether the exit has failed.
        pub(crate) fn fired(&self) -> bool {
            armed().is_some_and(|(_, _, fired)| fired)
        }
    }

    impl Drop for Armed {
        fn drop(&mut self) {
            *armed() = None;
        }
    }
}
