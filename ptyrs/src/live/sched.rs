//! Thread placement for the receive path: CPU pinning, and the scheduling
//! of threads that idle between frames.

use std::io;

/// Parse a CPU list such as `0-11,48-59`.
pub fn parse_cpu_list(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        let num = |t: &str| {
            t.trim()
                .parse::<usize>()
                .map_err(|e| format!("cpu list {s}: {e}"))
        };
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (num(a)?, num(b)?);
                if a > b {
                    return Err(format!("cpu list {s}: {a}-{b} is descending"));
                }
                out.extend(a..=b);
            }
            None => out.push(num(part)?),
        }
    }
    if out.is_empty() {
        return Err(format!("cpu list {s:?} is empty"));
    }
    Ok(out)
}

/// Restrict the calling thread to `cpus`.
pub fn pin_current(cpus: &[usize]) -> io::Result<()> {
    // SAFETY: cpu_set_t is plain bits; zeroed is the empty set.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in cpus {
        // SAFETY: CPU_SET bounds-checks `c` against the set size.
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    // SAFETY: pid 0 is the calling thread; `set` is a valid cpu_set_t.
    let r = unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) };
    if r == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// `struct sched_attr` up to SCHED_ATTR_SIZE_VER1; libc stops at VER0,
/// before the utilization clamps.
#[repr(C)]
#[derive(Default)]
struct SchedAttr {
    size: u32,
    policy: u32,
    flags: u64,
    nice: i32,
    priority: u32,
    runtime: u64,
    deadline: u64,
    period: u64,
    util_min: u32,
    util_max: u32,
}

/// Fair-class scheduling of a thread that sleeps between frames; a zero
/// field leaves what the thread already has.
#[derive(Clone, Copy, Debug, Default)]
pub struct FairSched {
    /// uclamp min out of 1024: schedutil picks at least this share of peak
    /// capacity whenever the thread runs. A decode worker that sleeps between
    /// frames otherwise never builds the utilization that raises its clock.
    pub util_min: u32,
    /// EEVDF slice. A waking thread preempts the running task only with the
    /// shorter slice (Linux 6.12+), so at the 3 ms base slice any task that
    /// lands on the uploader's CPU held each of its wakeups up to 3 ms.
    pub slice_ns: u64,
}

/// The calling thread's attributes, sized for writing back.
fn get_attr() -> io::Result<SchedAttr> {
    let size = std::mem::size_of::<SchedAttr>() as u32;
    let mut attr = SchedAttr::default();
    // SAFETY: pid 0 is the calling thread; `attr` holds `size` bytes.
    let r = unsafe {
        libc::syscall(
            libc::SYS_sched_getattr,
            0,
            &mut attr as *mut SchedAttr,
            size,
            0,
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    attr.size = size;
    Ok(attr)
}

fn set_attr(attr: &SchedAttr) -> io::Result<()> {
    // SAFETY: pid 0 is the calling thread; `attr` is a valid VER1 sched_attr
    // whose size field matches its layout.
    let r = unsafe { libc::syscall(libc::SYS_sched_setattr, 0, attr as *const SchedAttr, 0) };
    if r == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Apply `s` to the calling thread, keeping its policy and nice.
pub fn set_fair(s: FairSched) -> io::Result<()> {
    if s.util_min == 0 && s.slice_ns == 0 {
        return Ok(());
    }
    let mut attr = get_attr()?;
    if s.slice_ns > 0 {
        attr.runtime = s.slice_ns;
    } else {
        attr.flags |= libc::SCHED_FLAG_KEEP_PARAMS as u64;
    }
    if s.util_min > 0 {
        attr.flags |= libc::SCHED_FLAG_UTIL_CLAMP_MIN as u64;
        attr.util_min = s.util_min;
    }
    set_attr(&attr)
}

/// Move the calling thread to SCHED_FIFO at `prio`; threads it spawns start
/// back in the fair class. Unprivileged, `prio` must not exceed
/// RLIMIT_RTPRIO.
///
/// A waking fair-class thread that has run more than its share waits for
/// the next tick or the running task's slice, and any task on its CPU can
/// preempt it mid-pass; an RT thread runs on wakeup and to its next sleep.
pub fn set_fifo(prio: u32) -> io::Result<()> {
    set_attr(&SchedAttr {
        size: std::mem::size_of::<SchedAttr>() as u32,
        policy: libc::SCHED_FIFO as u32,
        flags: libc::SCHED_FLAG_RESET_ON_FORK as u64,
        priority: prio,
        ..SchedAttr::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists() {
        assert_eq!(parse_cpu_list("0-2,5").unwrap(), vec![0, 1, 2, 5]);
        assert_eq!(parse_cpu_list("7").unwrap(), vec![7]);
        assert!(parse_cpu_list("3-1").is_err());
        assert!(parse_cpu_list("").is_err());
        assert!(parse_cpu_list("a").is_err());
    }

    #[test]
    fn sched_attr_is_ver1_sized() {
        assert_eq!(std::mem::size_of::<SchedAttr>(), 56);
    }

    #[test]
    fn set_fair_changes_only_the_fields_it_is_given() {
        std::thread::spawn(|| {
            // SAFETY: lowers the calling thread's own priority.
            let r = unsafe { libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as u32, 5) };
            assert_eq!(r, 0);
            let base = get_attr().unwrap();
            let fields = |a: SchedAttr| (a.policy, a.nice, a.runtime, a.util_min);
            set_fair(FairSched {
                util_min: 512,
                slice_ns: 0,
            })
            .unwrap();
            assert_eq!(
                fields(get_attr().unwrap()),
                (base.policy, 5, base.runtime, 512)
            );
            set_fair(FairSched {
                util_min: 0,
                slice_ns: 200_000,
            })
            .unwrap();
            assert_eq!(fields(get_attr().unwrap()), (base.policy, 5, 200_000, 512));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn set_fifo_does_not_pass_to_spawned_threads() {
        std::thread::spawn(|| {
            let mut lim = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // SAFETY: `lim` is a valid rlimit to write into.
            assert_eq!(unsafe { libc::getrlimit(libc::RLIMIT_RTPRIO, &mut lim) }, 0);
            if let Err(e) = set_fifo(10) {
                assert_eq!(e.kind(), io::ErrorKind::PermissionDenied, "{e}");
                assert!(
                    lim.rlim_cur < 10,
                    "refused under RLIMIT_RTPRIO {}",
                    lim.rlim_cur
                );
                return;
            }
            let a = get_attr().unwrap();
            assert_eq!((a.policy, a.priority), (libc::SCHED_FIFO as u32, 10));
            let child = std::thread::spawn(|| get_attr().unwrap().policy)
                .join()
                .unwrap();
            assert_eq!(child, libc::SCHED_OTHER as u32);
        })
        .join()
        .unwrap();
    }
}
