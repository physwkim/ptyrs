//! Spatial multi-GPU split (DESIGN B3, B4): an object tile grid fixed in object coordinates,
//! partitioned into one region per GPU plus a one-probe halo.

pub mod partition;
pub mod topo;
pub mod worker;

pub use partition::{Csr, DEFAULT_TILE, Partition, Region, Scheme, TileGrid};
pub use worker::{Gpu, Workers};

#[derive(Debug, thiserror::Error)]
pub enum DistError {
    #[error("invalid partition: {0}")]
    Partition(String),
    #[error("CUDA: {0}")]
    Driver(#[from] cuda_core::DriverError),
    #[error("topology: {0}")]
    Topology(String),
    #[error("GPU {from} cannot access GPU {to} peer-to-peer")]
    NoPeerAccess { from: usize, to: usize },
    #[error("region {region} panicked: {msg}")]
    Panic { region: usize, msg: String },
    #[error("another worker failed")]
    Aborted,
    #[error("workers met a different number of fences")]
    FenceMismatch,
}
