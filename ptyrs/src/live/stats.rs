//! Clocks, CPU accounting and a fixed-bucket latency histogram.

fn clock_ns(id: libc::clockid_t) -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid out-pointer and `id` is a kernel clock id.
    unsafe { libc::clock_gettime(id, &mut ts) };
    ts.tv_sec * 1_000_000_000 + ts.tv_nsec
}

/// Wall clock, ns since the epoch. Shared by the simulator's
/// `dataTimeStamp` and the receiver, which run on the same host.
pub fn wall_ns() -> i64 {
    clock_ns(libc::CLOCK_REALTIME)
}

pub fn mono_ns() -> i64 {
    clock_ns(libc::CLOCK_MONOTONIC)
}

/// CPU time of the calling thread.
pub fn thread_cpu_ns() -> i64 {
    clock_ns(libc::CLOCK_THREAD_CPUTIME_ID)
}

/// User + system CPU time of the whole process.
pub fn process_cpu_ns() -> i64 {
    // SAFETY: zeroed rusage is a valid out-value for getrusage.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `ru` is a valid out-pointer.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let tv = |t: libc::timeval| t.tv_sec * 1_000_000_000 + t.tv_usec * 1000;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

/// Latency histogram with 1 us buckets up to `max_us`, plus an overflow
/// count. Allocated once.
pub struct Histogram {
    buckets: Vec<u64>,
    over: u64,
    n: u64,
    sum_ns: i128,
    max_ns: i64,
    min_ns: i64,
}

impl Histogram {
    pub fn new(max_us: usize) -> Self {
        Self {
            buckets: vec![0; max_us],
            over: 0,
            n: 0,
            sum_ns: 0,
            max_ns: i64::MIN,
            min_ns: i64::MAX,
        }
    }

    pub fn record(&mut self, ns: i64) {
        let us = (ns.max(0) / 1000) as usize;
        match self.buckets.get_mut(us) {
            Some(b) => *b += 1,
            None => self.over += 1,
        }
        self.n += 1;
        self.sum_ns += ns as i128;
        self.max_ns = self.max_ns.max(ns);
        self.min_ns = self.min_ns.min(ns);
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn merge(&mut self, o: &Histogram) {
        for (a, b) in self.buckets.iter_mut().zip(&o.buckets) {
            *a += b;
        }
        self.over += o.over;
        self.n += o.n;
        self.sum_ns += o.sum_ns;
        self.max_ns = self.max_ns.max(o.max_ns);
        self.min_ns = self.min_ns.min(o.min_ns);
    }

    /// Upper edge of the bucket holding quantile `q`, in us; `None` when
    /// empty or when the quantile falls in the overflow.
    pub fn quantile_us(&self, q: f64) -> Option<u64> {
        if self.n == 0 {
            return None;
        }
        let target = ((q * self.n as f64).ceil() as u64).max(1);
        let mut acc = 0;
        for (i, &b) in self.buckets.iter().enumerate() {
            acc += b;
            if acc >= target {
                return Some(i as u64 + 1);
            }
        }
        None
    }

    pub fn summary(&self) -> String {
        if self.n == 0 {
            return "n=0".into();
        }
        let q = |q| {
            self.quantile_us(q)
                .map_or_else(|| ">max".to_string(), |v| v.to_string())
        };
        format!(
            "n={} min_us={} p50_us={} p90_us={} p99_us={} p999_us={} max_us={} mean_us={:.1}",
            self.n,
            self.min_ns / 1000,
            q(0.5),
            q(0.9),
            q(0.99),
            q(0.999),
            self.max_ns / 1000,
            self.sum_ns as f64 / self.n as f64 / 1000.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_land_on_bucket_edges() {
        let mut h = Histogram::new(1000);
        for us in 0..100 {
            h.record(us * 1000 + 500);
        }
        assert_eq!(h.quantile_us(0.5), Some(50));
        assert_eq!(h.quantile_us(0.99), Some(99));
        h.record(5_000_000);
        assert_eq!(h.quantile_us(1.0), None);
    }
}
