//! The one way a live graph is captured and replayed (DESIGN.md §3 B2):
//! captured once from leased launches, then replayed on its stream without
//! the host waiting, each replay reacquiring the captured leases.
//!
//! [`LiveGraph::capture`] is the only capture in this crate (clippy.toml
//! disallows the others) and counts each one, so a measurement can tell
//! whether a graph was captured after its warm-up (DESIGN.md §5 inv. 9).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::Stream;

use crate::grow::GrowError;

/// Captures begun in this process. Relaxed: a measurement reads it across
/// a channel or a join from the capturing thread, which orders the two.
static CAPTURES: AtomicU64 = AtomicU64::new(0);

/// Graphs captured in this process so far, failed captures included.
pub fn captures() -> u64 {
    CAPTURES.load(Ordering::Relaxed)
}

/// The capture count when a measurement's warm-up ended, after every graph
/// it replays was captured; a capture after it breaks the measurement.
#[derive(Clone, Copy, Debug)]
pub struct WarmUp(u64);

impl WarmUp {
    pub fn end() -> Self {
        Self(captures())
    }

    /// Graphs captured since the warm-up ended; 0 for a run to count.
    pub fn recaptures(self) -> u64 {
        captures() - self.0
    }
}

pub struct LiveGraph {
    graph: CudaGraph<()>,
}

impl LiveGraph {
    /// Records `f`'s launches on `stream` into a graph; nothing runs yet.
    /// Waits for the work already queued on `stream`.
    pub fn capture<F>(stream: &Arc<Stream>, f: F) -> Result<Self, GrowError>
    where
        F: FnOnce(&ExecutionContext) -> Result<(), ptyrs_gpu::Error> + Send,
    {
        CAPTURES.fetch_add(1, Ordering::Relaxed);
        #[expect(clippy::disallowed_methods, reason = "the counted capture")]
        let graph = CudaGraph::capture(stream.clone(), ptyrs_gpu::record(f))?;
        Ok(Self { graph })
    }

    /// Queues one replay on the capture stream and returns without waiting.
    pub fn replay(&self) -> Result<(), GrowError> {
        let ctx = ExecutionContext::new(self.graph.stream().clone());
        // SAFETY: the output is `()`; dropping `ctx` keeps the replay's
        // leases and owners until it completes on the stream.
        unsafe { self.graph.launch().execute(&ctx) }?;
        Ok(())
    }

    pub fn stream(&self) -> &Arc<Stream> {
        self.graph.stream()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty(s: &Arc<Stream>) -> LiveGraph {
        LiveGraph::capture(s, |_| Ok(())).unwrap()
    }

    /// Replays after the warm-up leave the count alone; one more capture,
    /// failed or not, shows as one recapture.
    #[test]
    fn a_capture_after_the_warm_up_is_a_recapture() {
        let ctx = cuda_core::Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let g = empty(&s);
        let warm = WarmUp::end();
        for _ in 0..3 {
            g.replay().unwrap();
        }
        s.synchronize().unwrap();
        assert_eq!(warm.recaptures(), 0);
        let _again = empty(&s);
        assert_eq!(warm.recaptures(), 1);
        let failed = LiveGraph::capture(&s, |_| Err(ptyrs_gpu::Error::ContextMismatch));
        assert!(failed.is_err());
        assert_eq!(warm.recaptures(), 2);
        assert_eq!(WarmUp::end().recaptures(), 0);
    }
}
