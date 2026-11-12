//! Bounded frame queue from the sources to the decode workers. Frames leave
//! in arrival order, but a frame wakes the most recently idle worker: a core
//! idle for 1 ms or more drops its clock (uclamp min does not bring it back
//! before a 256² decode ends), and waking the longest-idle worker, as a
//! FIFO waiter list does, gives every worker the longest possible idle.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use crate::live::pipeline::Work;

struct Shared {
    st: Mutex<State>,
    cap: usize,
}

struct State {
    items: VecDeque<Work>,
    /// Workers parked in `recv`, most recently idle last.
    idle: Vec<Thread>,
    senders: usize,
}

/// Queue holding at most `cap` frames.
pub fn work_queue(cap: NonZeroUsize) -> (WorkTx, WorkRx) {
    let sh = Arc::new(Shared {
        st: Mutex::new(State {
            items: VecDeque::with_capacity(cap.get().min(1 << 16)),
            idle: Vec::new(),
            senders: 1,
        }),
        cap: cap.get(),
    });
    (WorkTx(sh.clone()), WorkRx(sh))
}

pub struct WorkTx(Arc<Shared>);

impl WorkTx {
    /// Queue `w` unless the queue is full, in which case `w` comes back.
    /// Never blocks.
    pub fn try_send(&self, w: Work) -> Result<(), Work> {
        let mut s = self.0.st.lock().unwrap();
        if s.items.len() >= self.0.cap {
            return Err(w);
        }
        s.items.push_back(w);
        let t = s.idle.pop();
        drop(s);
        if let Some(t) = t {
            t.unpark();
        }
        Ok(())
    }
}

impl Clone for WorkTx {
    fn clone(&self) -> Self {
        self.0.st.lock().unwrap().senders += 1;
        Self(self.0.clone())
    }
}

impl Drop for WorkTx {
    fn drop(&mut self) {
        let mut s = self.0.st.lock().unwrap();
        s.senders -= 1;
        if s.senders == 0 {
            for t in s.idle.drain(..) {
                t.unpark();
            }
        }
    }
}

#[derive(Clone)]
pub struct WorkRx(Arc<Shared>);

impl WorkRx {
    /// Next frame; `None` once every sender is gone and the queue is empty.
    pub fn recv(&self) -> Option<Work> {
        self.recv_until(None)
    }

    /// As [`recv`](Self::recv), also `None` after `d` without a frame.
    pub fn recv_timeout(&self, d: Duration) -> Option<Work> {
        self.recv_until(Some(Instant::now() + d))
    }

    pub fn try_recv(&self) -> Option<Work> {
        self.0.st.lock().unwrap().items.pop_front()
    }

    pub fn try_iter(&self) -> impl Iterator<Item = Work> + '_ {
        std::iter::from_fn(|| self.try_recv())
    }

    fn recv_until(&self, deadline: Option<Instant>) -> Option<Work> {
        let me = thread::current();
        loop {
            {
                let mut s = self.0.st.lock().unwrap();
                // Only a sender's pop unlists a parked worker; a spurious
                // wake, a timeout or a disconnect leaves it listed.
                if let Some(i) = s.idle.iter().position(|t| t.id() == me.id()) {
                    s.idle.remove(i);
                }
                if let Some(w) = s.items.pop_front() {
                    return Some(w);
                }
                if s.senders == 0 || deadline.is_some_and(|d| Instant::now() >= d) {
                    return None;
                }
                s.idle.push(me.clone());
            }
            match deadline {
                None => thread::park(),
                Some(d) => thread::park_timeout(d.saturating_duration_since(Instant::now())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::codec::Codec;
    use bytes::Bytes;

    fn work(uid: i32) -> Work {
        Work {
            payload: Bytes::new(),
            codec: Codec::None,
            uid,
            t_rx: 0,
            t_pub: 0,
        }
    }

    fn q(cap: usize) -> (WorkTx, WorkRx) {
        work_queue(NonZeroUsize::new(cap).unwrap())
    }

    fn idle(rx: &WorkRx) -> usize {
        rx.0.st.lock().unwrap().idle.len()
    }

    fn until_idle(rx: &WorkRx, n: usize) {
        let end = Instant::now() + Duration::from_secs(10);
        while idle(rx) != n {
            assert!(Instant::now() < end, "{} idle, waited for {n}", idle(rx));
            thread::yield_now();
        }
    }

    #[test]
    fn frames_keep_order_and_a_full_queue_returns_the_frame() {
        let (tx, rx) = q(2);
        assert!(tx.try_send(work(0)).is_ok());
        assert!(tx.try_send(work(1)).is_ok());
        assert_eq!(tx.try_send(work(2)).unwrap_err().uid, 2);
        assert_eq!(rx.try_iter().map(|w| w.uid).collect::<Vec<_>>(), [0, 1]);
    }

    #[test]
    fn a_frame_wakes_the_most_recently_idle_worker() {
        let (tx, rx) = q(4);
        let (dtx, drx) = std::sync::mpsc::channel();
        let spawn = |name: &'static str| {
            let (rx, dtx) = (rx.clone(), dtx.clone());
            thread::spawn(move || {
                while let Some(w) = rx.recv() {
                    dtx.send((name, w.uid)).unwrap();
                }
            })
        };
        let first = spawn("first");
        until_idle(&rx, 1);
        let last = spawn("last");
        until_idle(&rx, 2);
        assert!(tx.try_send(work(0)).is_ok());
        assert_eq!(drx.recv().unwrap(), ("last", 0));
        // `last` is listed again on top before the next frame.
        until_idle(&rx, 2);
        assert!(tx.try_send(work(1)).is_ok());
        assert_eq!(drx.recv().unwrap(), ("last", 1));
        drop(tx);
        first.join().unwrap();
        last.join().unwrap();
    }

    #[test]
    fn last_sender_drop_drains_then_ends_every_worker() {
        let (tx, rx) = q(4);
        let tx2 = tx.clone();
        let w: Vec<_> = (0..2)
            .map(|_| {
                let rx = rx.clone();
                thread::spawn(move || std::iter::from_fn(|| rx.recv()).count())
            })
            .collect();
        until_idle(&rx, 2);
        drop(tx);
        assert!(tx2.try_send(work(0)).is_ok());
        drop(tx2);
        let n: usize = w.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(n, 1);
        assert_eq!(idle(&rx), 0);
    }

    #[test]
    fn timeout_unlists_the_worker() {
        let (_tx, rx) = q(1);
        assert!(rx.recv_timeout(Duration::from_millis(5)).is_none());
        assert_eq!(idle(&rx), 0);
    }
}
