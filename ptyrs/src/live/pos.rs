//! Scan positions keyed by uniqueId, written by the position monitor and
//! read by the uploader, which lands each frame once its position is in.
//!
//! The table is where a live position is checked: one that is not a scan
//! position ([`Positions::valid`]) is kept as [`Entry::Invalid`], so no
//! reader takes a coordinate from it.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

use crate::base::Positions;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pos {
    pub x: f64,
    pub y: f64,
    /// Arrival time at the receiver, ns.
    pub t_rx: i64,
}

impl Pos {
    /// Whether both coordinates are scan positions.
    pub fn valid(&self) -> bool {
        Positions::valid(self.x) && Positions::valid(self.y)
    }
}

/// What the table holds for a uid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Entry {
    /// No position has arrived, or a later uid took its slot.
    Absent,
    /// The position, [`Pos::valid`].
    Valid(Pos),
    /// A position arrived that is not a scan position; its frame cannot land.
    Invalid,
}

/// Direct-mapped by `uid % capacity`. A slot is rewritten only when a uid
/// `capacity` triggers later arrives, so a reader that sees its uid before
/// and after loading the fields has a consistent entry.
pub struct PosTable {
    id: Box<[AtomicI64]>,
    valid: Box<[AtomicBool]>,
    x: Box<[AtomicU64]>,
    y: Box<[AtomicU64]>,
    t: Box<[AtomicI64]>,
}

impl PosTable {
    pub fn new(capacity: usize) -> Self {
        Self {
            id: (0..capacity).map(|_| AtomicI64::new(-1)).collect(),
            valid: (0..capacity).map(|_| AtomicBool::new(false)).collect(),
            x: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            y: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            t: (0..capacity).map(|_| AtomicI64::new(0)).collect(),
        }
    }

    fn slot(&self, uid: i32) -> usize {
        uid.rem_euclid(self.id.len() as i32) as usize
    }

    /// Records `p` as arrived for `uid`: [`Entry::Valid`] if it is, else
    /// [`Entry::Invalid`].
    pub fn insert(&self, uid: i32, p: Pos) {
        let i = self.slot(uid);
        self.id[i].store(-1, Ordering::Relaxed);
        self.valid[i].store(p.valid(), Ordering::Relaxed);
        self.x[i].store(p.x.to_bits(), Ordering::Relaxed);
        self.y[i].store(p.y.to_bits(), Ordering::Relaxed);
        self.t[i].store(p.t_rx, Ordering::Relaxed);
        self.id[i].store(uid as i64, Ordering::Release);
    }

    pub fn lookup(&self, uid: i32) -> Entry {
        let i = self.slot(uid);
        if self.id[i].load(Ordering::Acquire) != uid as i64 {
            return Entry::Absent;
        }
        let entry = if self.valid[i].load(Ordering::Relaxed) {
            Entry::Valid(Pos {
                x: f64::from_bits(self.x[i].load(Ordering::Relaxed)),
                y: f64::from_bits(self.y[i].load(Ordering::Relaxed)),
                t_rx: self.t[i].load(Ordering::Relaxed),
            })
        } else {
            Entry::Invalid
        };
        std::sync::atomic::fence(Ordering::Acquire);
        if self.id[i].load(Ordering::Relaxed) == uid as i64 {
            entry
        } else {
            Entry::Absent
        }
    }

    /// The position of `uid` if it is [`Entry::Valid`].
    pub fn get(&self, uid: i32) -> Option<Pos> {
        match self.lookup(uid) {
            Entry::Valid(p) => Some(p),
            Entry::Absent | Entry::Invalid => None,
        }
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

    /// Each coordinate at each boundary of [`Positions::valid`]: an entry
    /// outside it is recorded, as invalid, and gives no position.
    #[test]
    fn a_position_outside_the_range_is_recorded_invalid() {
        let inside = 1e10f64;
        let outside = inside.next_up();
        let t = PosTable::new(64);
        let mut uid = 0;
        for (v, valid) in [
            (0.0, true),
            (inside, true),
            (-inside, true),
            (outside, false),
            (-outside, false),
            (f64::NAN, false),
            (f64::INFINITY, false),
            (f64::NEG_INFINITY, false),
        ] {
            for (x, y) in [(v, 1.0), (1.0, v)] {
                let p = Pos { x, y, t_rx: 7 };
                assert_eq!(t.lookup(uid), Entry::Absent);
                t.insert(uid, p);
                let want = if valid {
                    Entry::Valid(p)
                } else {
                    Entry::Invalid
                };
                assert_eq!(t.lookup(uid), want, "({x}, {y})");
                assert_eq!(t.get(uid), valid.then_some(p), "({x}, {y})");
                uid += 1;
            }
        }
    }

    /// A slot's state is its last insert's, whichever way it changes.
    #[test]
    fn a_reused_slot_takes_the_new_entrys_state() {
        let t = PosTable::new(4);
        let p = |x| Pos { x, y: 0.0, t_rx: 0 };
        t.insert(1, p(f64::NAN));
        t.insert(5, p(2.0));
        assert_eq!(
            (t.lookup(1), t.lookup(5)),
            (Entry::Absent, Entry::Valid(p(2.0)))
        );
        t.insert(9, p(f64::INFINITY));
        assert_eq!((t.lookup(5), t.lookup(9)), (Entry::Absent, Entry::Invalid));
    }
}
