//! Spatial multi-GPU split (DESIGN B3, B4): an object tile grid fixed in object coordinates,
//! partitioned into one region per GPU plus a one-probe halo.

pub mod ap;
pub mod exchange;
pub mod feed;
pub mod p2p;
pub mod partition;
#[cfg(test)]
mod synth;
pub mod tomo;
pub mod topo;
pub mod worker;

pub use ap::{ApConfig, ApProblem, DistAp, NLEAF, RegionLoad};
pub use exchange::{Exchange, Layout};
pub use partition::{Csr, DEFAULT_TILE, Partition, Region, Scheme, TileGrid};
pub use worker::{Gpu, Workers};

#[derive(Debug, thiserror::Error)]
pub enum DistError {
    #[error("invalid partition: {0}")]
    Partition(String),
    #[error("CUDA: {0}")]
    Driver(#[from] cuda_core::DriverError),
    #[error("GPU: {0}")]
    Gpu(#[from] ptyrs_gpu::Error),
    #[error("topology: {0}")]
    Topology(String),
    #[error("GPU {from} cannot access GPU {to} peer-to-peer")]
    NoPeerAccess { from: usize, to: usize },
    #[error("peer reads return wrong data for (reader, owner) GPUs {0:?}")]
    PeerReadFault(Vec<(usize, usize)>),
    #[error("region {region} panicked: {msg}")]
    Panic { region: usize, msg: String },
    #[error("another worker failed")]
    Aborted,
    #[error("workers met a different number of fences")]
    FenceMismatch,
    #[error("angle {angle}: {source}")]
    Angle {
        angle: usize,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
