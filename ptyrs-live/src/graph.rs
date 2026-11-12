//! The one way a live graph is captured and replayed (DESIGN.md §3 B2):
//! captured once from leased launches, then replayed on its stream without
//! the host waiting, each replay reacquiring the captured leases.

use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::Stream;

use crate::grow::GrowError;

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
        Ok(Self {
            graph: CudaGraph::capture(stream.clone(), ptyrs_gpu::record(f))?,
        })
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
