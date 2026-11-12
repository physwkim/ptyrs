//! Thread placement for the receive path: CPU pinning, and a frequency
//! floor for threads that idle between frames.

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

/// Ask schedutil to pick at least `min`/1024 of peak capacity whenever the
/// calling thread runs (uclamp min). A decode worker that sleeps between
/// frames otherwise never builds the utilization that raises its clock.
pub fn set_util_min(min: u32) -> io::Result<()> {
    let attr = SchedAttr {
        size: std::mem::size_of::<SchedAttr>() as u32,
        policy: 0,
        flags: (libc::SCHED_FLAG_KEEP_ALL | libc::SCHED_FLAG_UTIL_CLAMP_MIN) as u64,
        nice: 0,
        priority: 0,
        runtime: 0,
        deadline: 0,
        period: 0,
        util_min: min,
        util_max: 0,
    };
    // SAFETY: pid 0 is the calling thread; `attr` is a valid VER1 sched_attr
    // whose size field matches its layout.
    let r = unsafe { libc::syscall(libc::SYS_sched_setattr, 0, &attr as *const SchedAttr, 0) };
    if r == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
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
}
