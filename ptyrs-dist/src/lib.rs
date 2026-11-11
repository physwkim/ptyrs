//! Spatial multi-GPU split (DESIGN B3, B4): an object tile grid fixed in object coordinates,
//! partitioned into one region per GPU plus a one-probe halo.

pub mod partition;
pub mod topo;

pub use partition::{Csr, DEFAULT_TILE, Partition, Region, Scheme, TileGrid};

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
}
