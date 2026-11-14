//! Scan positions keyed by uniqueId, written by the position monitor and
//! read by the uploader, which lands each frame once its position is in.
//!
//! A live position is checked by [`Pos::new`]: an arrival it refuses is
//! kept as [`Entry::Invalid`], so no reader takes a coordinate from it.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

use crate::base::Positions;

/// A scan position as it arrived, both coordinates [`Positions::valid`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pos {
    x: f64,
    y: f64,
    t_rx: i64,
}

impl Pos {
    /// `None` if a coordinate is not [`Positions::valid`]; `t_rx` is the
    /// arrival time at the receiver, ns.
    pub fn new(x: f64, y: f64, t_rx: i64) -> Option<Self> {
        (Positions::valid(x) && Positions::valid(y)).then_some(Self { x, y, t_rx })
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }

    /// Arrival time at the receiver, ns.
    pub fn t_rx(&self) -> i64 {
        self.t_rx
    }
}

/// What the table holds for a uid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Entry {
    /// No position has arrived, or a later uid took its slot.
    Absent,
    /// The position.
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

    /// Records an arrival for `uid`: [`Entry::Valid`] for a [`Pos`],
    /// [`Entry::Invalid`] for `None`, one [`Pos::new`] refused.
    pub fn insert(&self, uid: i32, arrival: Option<Pos>) {
        let i = self.slot(uid);
        self.id[i].store(-1, Ordering::Relaxed);
        self.valid[i].store(arrival.is_some(), Ordering::Relaxed);
        if let Some(p) = arrival {
            self.x[i].store(p.x.to_bits(), Ordering::Relaxed);
            self.y[i].store(p.y.to_bits(), Ordering::Relaxed);
            self.t[i].store(p.t_rx, Ordering::Relaxed);
        }
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
        let p = |x, y| Pos::new(x, y, 7);
        assert_eq!(t.get(5), None);
        t.insert(5, p(1.0, 2.0));
        assert_eq!(t.get(5), p(1.0, 2.0));
        t.insert(9, p(3.0, 4.0));
        assert_eq!(t.get(5), None);
        assert_eq!(t.get(9), p(3.0, 4.0));
    }

    /// Each coordinate at each boundary of [`Positions::valid`]: [`Pos::new`]
    /// refuses one outside it, which is recorded, as invalid, and gives no
    /// position.
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
                let p = Pos::new(x, y, 7);
                assert_eq!(p.is_some(), valid, "({x}, {y})");
                assert_eq!(t.lookup(uid), Entry::Absent);
                t.insert(uid, p);
                let want = p.map_or(Entry::Invalid, Entry::Valid);
                assert_eq!(t.lookup(uid), want, "({x}, {y})");
                assert_eq!(t.get(uid), p, "({x}, {y})");
                uid += 1;
            }
        }
    }

    /// A slot's state is its last insert's, whichever way it changes.
    #[test]
    fn a_reused_slot_takes_the_new_entrys_state() {
        let t = PosTable::new(4);
        let p = |x| Pos::new(x, 0.0, 0);
        t.insert(1, p(f64::NAN));
        t.insert(5, p(2.0));
        assert_eq!(
            (t.lookup(1), t.lookup(5)),
            (Entry::Absent, Entry::Valid(p(2.0).unwrap()))
        );
        t.insert(9, p(f64::INFINITY));
        assert_eq!((t.lookup(5), t.lookup(9)), (Entry::Absent, Entry::Invalid));
    }
}
