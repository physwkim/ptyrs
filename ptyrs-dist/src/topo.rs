//! GPU topology (PCI bus, NUMA node) and peer access between every pair of devices.

use std::ffi::{CStr, c_char, c_int};
use std::sync::Arc;

use cuda_core::{Context, IntoResult, peer, sys};

use crate::DistError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub ordinal: usize,
    pub name: String,
    /// Lower-case `domain:bus:device.function`, as under `/sys/bus/pci/devices`.
    pub pci_bus_id: String,
    pub numa: Option<u32>,
}

/// Every visible CUDA device, in ordinal order.
pub fn devices() -> Result<Vec<Device>, DistError> {
    // SAFETY: cuInit is idempotent and takes no pointers.
    unsafe { sys::cuInit(0).result()? };
    let n = Context::device_count()?;
    (0..n as usize).map(device).collect()
}

fn device(ordinal: usize) -> Result<Device, DistError> {
    let dev = Context::raw_device(ordinal)?;
    let mut buf = [0 as c_char; 32];
    // SAFETY: the driver writes a NUL-terminated id of at most `len` bytes into `buf`.
    unsafe { sys::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as c_int, dev).result()? };
    // SAFETY: NUL-terminated by the driver within `buf`.
    let pci_bus_id = unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .to_lowercase();
    let numa = std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci_bus_id}/numa_node"))
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .and_then(|n| u32::try_from(n).ok());
    let mut name = [0 as c_char; 256];
    // SAFETY: as above, NUL-terminated within `name`.
    unsafe { sys::cuDeviceGetName(name.as_mut_ptr(), name.len() as c_int, dev).result()? };
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Ok(Device {
        ordinal,
        name,
        pci_bus_id,
        numa,
    })
}

/// Ordinals grouped by NUMA node, then by ordinal, so that region pairs `(2k, 2k + 1)` of a
/// [`crate::Partition`] land on one socket. Devices with unknown NUMA come last.
pub fn numa_grouped(devs: &[Device]) -> Vec<usize> {
    let mut order: Vec<&Device> = devs.iter().collect();
    order.sort_by_key(|d| (d.numa.unwrap_or(u32::MAX), d.ordinal));
    order.iter().map(|d| d.ordinal).collect()
}

/// Driver view of one direction of a device pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerLink {
    pub from: usize,
    pub to: usize,
    pub access: bool,
    pub native_atomics: bool,
    /// Relative link quality reported by the driver; lower is better.
    pub perf_rank: i32,
    pub same_numa: bool,
}

/// Peer attributes of every ordered pair of distinct devices.
pub fn links(devs: &[Device]) -> Result<Vec<PeerLink>, DistError> {
    let attr = |a: sys::CUdevice_P2PAttribute, from: usize, to: usize| {
        let mut v: c_int = 0;
        let (s, d) = (Context::raw_device(from)?, Context::raw_device(to)?);
        // SAFETY: `v` is a valid out-pointer; the devices come from the driver.
        unsafe { sys::cuDeviceGetP2PAttribute(&mut v, a, s, d).result()? };
        Ok::<_, DistError>(v)
    };
    let mut out = Vec::new();
    for a in devs {
        for b in devs.iter().filter(|b| b.ordinal != a.ordinal) {
            let (f, t) = (a.ordinal, b.ordinal);
            out.push(PeerLink {
                from: f,
                to: t,
                access: attr(
                    sys::CUdevice_P2PAttribute_enum_CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
                    f,
                    t,
                )? != 0,
                native_atomics: attr(
                    sys::CUdevice_P2PAttribute_enum_CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                    f,
                    t,
                )? != 0,
                perf_rank: attr(
                    sys::CUdevice_P2PAttribute_enum_CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK,
                    f,
                    t,
                )?,
                same_numa: a.numa.is_some() && a.numa == b.numa,
            });
        }
    }
    Ok(out)
}

/// Enables peer access in both directions for every pair of `ctxs`, then has every device
/// read every peer with a kernel. A pair without P2P, or whose reads arrive corrupted, is an
/// error: halo exchange reads neighbour memory directly and has no host-staged fallback.
/// Access stays enabled when the read check fails.
pub fn enable_all_pairs(ctxs: &[Arc<Context>]) -> Result<(), DistError> {
    for a in ctxs {
        for b in ctxs.iter().filter(|b| b.ordinal() != a.ordinal()) {
            if !peer::can_access_peer(a, b)? {
                return Err(DistError::NoPeerAccess {
                    from: a.ordinal(),
                    to: b.ordinal(),
                });
            }
            peer::enable_peer_access(a, b)?;
        }
    }
    crate::p2p::verify_reads(ctxs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_groups_numa_pairs_and_every_pair_has_p2p() {
        let devs = devices().unwrap();
        assert_eq!(devs.len(), 4);
        let order = numa_grouped(&devs);
        let numa = |o: usize| devs[o].numa;
        assert_eq!(numa(order[0]), numa(order[1]));
        assert_eq!(numa(order[2]), numa(order[3]));
        assert_ne!(numa(order[1]), numa(order[2]));
        let links = links(&devs).unwrap();
        assert_eq!(links.len(), 12);
        assert!(links.iter().all(|l| l.access), "{links:?}");
    }

    #[test]
    fn enables_every_pair() {
        let ctxs: Vec<_> = (0..4).map(|o| Context::new(o).unwrap()).collect();
        enable_all_pairs(&ctxs).unwrap();
        // Enabling again is not an error.
        enable_all_pairs(&ctxs).unwrap();
    }
}
