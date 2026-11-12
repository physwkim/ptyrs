//! Scan positions keyed by uniqueId, written by the position monitor and
//! read by the frame workers.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pos {
    pub x: f64,
    pub y: f64,
    /// Arrival time at the receiver, ns.
    pub t_rx: i64,
}

/// Direct-mapped by `uid % capacity`. A slot is rewritten only when a uid
/// `capacity` triggers later arrives, so a reader that sees its uid before
/// and after loading the fields has a consistent entry.
pub struct PosTable {
    id: Box<[AtomicI64]>,
    x: Box<[AtomicU64]>,
    y: Box<[AtomicU64]>,
    t: Box<[AtomicI64]>,
}

impl PosTable {
    pub fn new(capacity: usize) -> Self {
        Self {
            id: (0..capacity).map(|_| AtomicI64::new(-1)).collect(),
            x: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            y: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            t: (0..capacity).map(|_| AtomicI64::new(0)).collect(),
        }
    }

    fn slot(&self, uid: i32) -> usize {
        uid.rem_euclid(self.id.len() as i32) as usize
    }

    pub fn insert(&self, uid: i32, p: Pos) {
        let i = self.slot(uid);
        self.id[i].store(-1, Ordering::Relaxed);
        self.x[i].store(p.x.to_bits(), Ordering::Relaxed);
        self.y[i].store(p.y.to_bits(), Ordering::Relaxed);
        self.t[i].store(p.t_rx, Ordering::Relaxed);
        self.id[i].store(uid as i64, Ordering::Release);
    }

    pub fn get(&self, uid: i32) -> Option<Pos> {
        let i = self.slot(uid);
        if self.id[i].load(Ordering::Acquire) != uid as i64 {
            return None;
        }
        let p = Pos {
            x: f64::from_bits(self.x[i].load(Ordering::Relaxed)),
            y: f64::from_bits(self.y[i].load(Ordering::Relaxed)),
            t_rx: self.t[i].load(Ordering::Relaxed),
        };
        std::sync::atomic::fence(Ordering::Acquire);
        (self.id[i].load(Ordering::Relaxed) == uid as i64).then_some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_misses_until_inserted_and_after_eviction() {
        let t = PosTable::new(4);
        let p = |x, y| Pos { x, y, t_rx: 7 };
        assert_eq!(t.get(5), None);
        t.insert(5, p(1.0, 2.0));
        assert_eq!(t.get(5), Some(p(1.0, 2.0)));
        t.insert(9, p(3.0, 4.0));
        assert_eq!(t.get(5), None);
        assert_eq!(t.get(9), Some(p(3.0, 4.0)));
    }
}
