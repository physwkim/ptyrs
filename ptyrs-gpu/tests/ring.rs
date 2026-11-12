//! The LLK ring: records come back in order with the totals of their own partials and the
//! unmasked pixels of their frames, across wraps and copies that overlap later records; a
//! record of the first frames reads only those; a record with a frame not written for it is
//! a stamp error, as is one ended a number of times other than the host counts it, and one
//! with a frame never counted is an uncounted error. A poll returns the records whose copies
//! have completed without waiting for the rest. Each detector pixel's frame count holds the
//! counted frames unmasked there, each once, in any layout and order of counted ranges.

use std::sync::Arc;
use std::sync::mpsc::{self, Sender};

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_gpu::{
    Ended, Error, Kernels, LlkPartial, LlkRing, LlkSums, LlkTotals, MASKED, RingError, StackTable,
    ViewDims,
};

const NFRAME: u32 = 10;
/// Stacks of 4, 4 and 2 frames.
const STACKS: [std::ops::Range<u32>; 3] = [0..4, 4..8, 8..10];

/// One record as a body writes it: which stacks push through its writer, and how many
/// writers end (the first pushes, later ones are empty; none drops it unended). The host
/// counts it once, of frames `0 .. nframe`, with the latest [`Ended`], as it counts a graph
/// replay with the token of the capture.
struct Record {
    push: [bool; 3],
    ends: usize,
    nframe: u32,
}

const WHOLE: Record = Record {
    push: [true; 3],
    ends: 1,
    nframe: NFRAME,
};

/// A record of frames `0 .. nframe` pushed by the stacks `push` asks for, ended once.
fn first(push: [bool; 3], nframe: u32) -> Record {
    Record {
        push,
        ends: 1,
        nframe,
    }
}

/// 8x8 frames with a mask that differs from frame to frame and from pixel to pixel.
fn counts() -> Vec<u16> {
    (0..NFRAME as u64 * 64)
        .map(|i| {
            if (i.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 61) % 3 == 0 {
                MASKED
            } else {
                1
            }
        })
        .collect()
}

/// Each pixel's unmasked frames among `frames` of [`counts`].
fn pixel_frames(frames: impl Iterator<Item = u32> + Clone) -> Vec<u32> {
    let c = counts();
    (0..64)
        .map(|p| {
            frames
                .clone()
                .filter(|&f| c[f as usize * 64 + p] != MASKED)
                .count() as u32
        })
        .collect()
}

fn dims() -> ViewDims {
    ViewDims {
        nframe: NFRAME,
        nmode: 1,
        nprobe: 1,
        nobj: 1,
        frame: [8, 8],
        obj: [8, 8],
        quad_phase: 0.0,
    }
}

/// The unmasked pixels of frames `0 .. n` of [`counts`].
fn nb_obs(n: u32) -> usize {
    counts()[..n as usize * 64]
        .iter()
        .filter(|&&c| c != MASKED)
        .count()
}

/// Deterministic partials of record `r`, spread over magnitudes so that the summation order
/// shows in the bits.
fn partials(r: usize) -> Vec<LlkPartial> {
    let mut s = (r as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let m = (s >> 40) as f32 / (1u64 << 24) as f32;
        m * 10f32.powi((s % 9) as i32 - 3)
    };
    (0..NFRAME)
        .map(|_| LlkPartial {
            poisson: next(),
            gaussian: next(),
            euclidean: next(),
            icalc: next(),
        })
        .collect()
}

fn bits(s: &LlkSums) -> [u64; 4] {
    [s.poisson, s.gaussian, s.euclidean, s.icalc].map(f64::to_bits)
}

struct Fixture {
    stream: Arc<CudaStream>,
    ctx: ExecutionContext,
    dims: ViewDims,
    table: StackTable,
    ring: LlkRing<usize>,
    ended: Option<Ended>,
    counts: DeviceBuffer<u16>,
}

fn fixture(nslot: u32) -> Fixture {
    fixture_counting(nslot, [true; 3])
}

/// A ring of `nslot` records whose device counted the frames of the stacks `counted` asks for,
/// a range per stack.
fn fixture_counting(nslot: u32, counted: [bool; 3]) -> Fixture {
    let cuda = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cuda).expect("load ptyrs-gpu kernels");
    let stream = cuda.new_stream().unwrap();
    let dims = dims();
    let table = StackTable::new(&stream, &dims, &STACKS).unwrap();
    let mut ring = LlkRing::new(&kernels, &stream, &dims, nslot).unwrap();
    let ctx = ExecutionContext::new(stream.clone());
    let dev_counts = DeviceBuffer::from_host(&stream, &counts()).unwrap();
    for (s, _) in counted.iter().enumerate().filter(|(_, c)| **c) {
        ring.device()
            .count(&ctx, STACKS[s].clone(), &dev_counts)
            .unwrap();
    }
    Fixture {
        ctx,
        stream,
        dims,
        table,
        ring,
        ended: None,
        counts: dev_counts,
    }
}

impl Fixture {
    /// Enqueues `records` as numbered from `r0`, with no host sync between them: each stack's
    /// partials from its own staging buffer (uploaded first), then the ends, then the count.
    fn enqueue(&mut self, r0: usize, records: &[Record]) {
        let staged = self.stage(r0, records.len());
        self.enqueue_staged(r0, records, &staged);
    }

    /// Each stack's partials of records `r0 .. r0 + n` in their own staging buffers, uploaded
    /// with a host sync.
    fn stage(&self, r0: usize, n: usize) -> Vec<Vec<DeviceBuffer<LlkPartial>>> {
        let nf_max = self.table.nf_max() as usize;
        (r0..r0 + n)
            .map(|r| {
                let p = partials(r);
                STACKS
                    .iter()
                    .map(|s| {
                        let mut v = p[s.start as usize..s.end as usize].to_vec();
                        v.resize(nf_max, LlkPartial::default());
                        DeviceBuffer::from_host(&self.stream, &v).unwrap()
                    })
                    .collect()
            })
            .collect()
    }

    /// [`Self::enqueue`] from partials [`Self::stage`] uploaded.
    fn enqueue_staged(
        &mut self,
        r0: usize,
        records: &[Record],
        staged: &[Vec<DeviceBuffer<LlkPartial>>],
    ) {
        for (i, rec) in records.iter().enumerate() {
            let dev = self.ring.device();
            let mut writer = dev.record();
            for (s, staging) in staged[i].iter().enumerate() {
                if rec.push[s] {
                    writer
                        .push(&self.ctx, &self.dims, self.table.at(s), staging)
                        .unwrap();
                }
            }
            if rec.ends == 0 {
                drop(writer);
            } else {
                self.ended = Some(writer.end(&self.ctx).unwrap());
                for _ in 1..rec.ends {
                    self.ended = Some(dev.record().end(&self.ctx).unwrap());
                }
            }
            let ended = self.ended.as_ref().expect("an earlier record ended");
            self.ring
                .enqueued(&self.ctx, ended, r0 + i, rec.nframe)
                .unwrap();
        }
    }

    fn flush(&mut self) -> Result<Vec<(usize, LlkTotals)>, RingError> {
        self.ring.flush(&self.ctx)
    }

    fn poll(&mut self) -> Vec<(usize, LlkTotals)> {
        self.ring.poll(&self.ctx).unwrap()
    }

    /// Counts the frames of stack `s` on the ring's device.
    fn count(&mut self, s: usize) {
        self.ring
            .device()
            .count(&self.ctx, STACKS[s].clone(), &self.counts)
            .unwrap();
    }

    fn pixel_frames(&mut self) -> Vec<u32> {
        let dev = self.ring.device();
        dev.pixel_frames().to_host_vec(&self.stream).unwrap()
    }

    /// Holds the stream at this point until the returned sender sends or drops.
    fn hold(&self) -> Sender<()> {
        let (release, held) = mpsc::channel();
        self.stream
            .launch_host_function(move || {
                let _ = held.recv();
            })
            .unwrap();
        release
    }
}

fn check_whole(got: &[(usize, LlkTotals)], tags: std::ops::Range<usize>) {
    assert_eq!(
        got.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        Vec::from_iter(tags)
    );
    for (r, t) in got {
        let want = LlkSums::from_partials(&partials(*r));
        assert_eq!(bits(&t.sums), bits(&want), "record {r}");
        assert_eq!((t.nframe, t.nb_obs), (NFRAME, nb_obs(NFRAME)));
    }
}

/// Flushes give each record once, in order, with the bits of `LlkSums::from_partials` of its
/// own partials and the unmasked pixels of every frame: one slot (every record waits for the
/// last), two (copy after each record), and three and five (wrapping inside a copy), flushing
/// mid-run and at the end.
#[test]
fn records_come_back_in_order_with_their_own_totals() {
    for nslot in [1, 2, 3, 5] {
        let mut f = fixture(nslot);
        let mut got = Vec::new();
        f.enqueue(0, &[WHOLE, WHOLE, WHOLE, WHOLE, WHOLE]);
        got.extend(f.flush().unwrap());
        assert!(
            f.flush().unwrap().is_empty(),
            "a second flush has nothing new"
        );
        f.enqueue(5, &[WHOLE, WHOLE, WHOLE, WHOLE, WHOLE, WHOLE, WHOLE]);
        got.extend(f.flush().unwrap());
        let tags: Vec<usize> = got.iter().map(|(t, _)| *t).collect();
        assert_eq!(tags, (0..12).collect::<Vec<_>>(), "nslot {nslot}");
        for (r, t) in &got {
            let want = LlkSums::from_partials(&partials(*r));
            assert_eq!(bits(&t.sums), bits(&want), "nslot {nslot}, record {r}");
            assert_eq!((t.nframe, t.nb_obs), (NFRAME, nb_obs(NFRAME)));
        }
    }
}

/// A poll never waits. With the stream held before every copy it returns nothing, and with
/// record 4 counted onto record 0's unread slot it leaves record 4 uncopied. Released, it
/// returns records 0 .. 4, whose copies completed, and enqueues record 4's, which the next
/// poll returns; the flush after has nothing left.
#[test]
fn poll_returns_completed_copies_without_waiting() {
    let mut f = fixture(4);
    let staged = f.stage(0, 5);
    let release = f.hold();
    f.enqueue_staged(0, &[WHOLE, WHOLE, WHOLE, WHOLE, WHOLE], &staged);
    assert!(f.poll().is_empty(), "every copy is held");
    release.send(()).unwrap();
    f.stream.synchronize().unwrap();
    check_whole(&f.poll(), 0..4);
    f.stream.synchronize().unwrap();
    check_whole(&f.poll(), 4..5);
    assert!(f.flush().unwrap().is_empty());
}

/// A record of frames `0 .. n` reads only those, whatever the later frames hold: at a stack's
/// end (4), inside a stack (6) and empty (0). A frame below `n` not pushed is still a stamp
/// error, and an `n` past the ring's frames is refused when counted.
#[test]
fn a_record_of_the_first_frames_reads_only_those() {
    let mut f = fixture(3);
    let ns = [4, 6, 0];
    f.enqueue(
        0,
        &[
            first([true, false, false], ns[0]),
            first([true, true, false], ns[1]),
            first([false; 3], ns[2]),
        ],
    );
    let got = f.flush().unwrap();
    assert_eq!(got.len(), 3);
    for (r, t) in &got {
        let n = ns[*r];
        let want = LlkSums::from_partials(&partials(*r)[..n as usize]);
        assert_eq!(bits(&t.sums), bits(&want), "record {r}");
        assert_eq!((t.nframe, t.nb_obs), (n, nb_obs(n)), "record {r}");
    }

    // Record 3 reuses record 0's slot, whose frame 4 nothing has written.
    f.enqueue(3, &[first([true, false, false], 5)]);
    assert_eq!(stamp(f.flush().unwrap_err()), (3, 4, u64::MAX));

    let ended = f.ended.take().unwrap();
    let e = f.ring.enqueued(&f.ctx, &ended, 4, NFRAME + 1).unwrap_err();
    assert!(matches!(e, RingError::Gpu(Error::Shape(_))), "{e}");
}

/// A frame pushed before its unmasked pixels were counted fails the record that reads it,
/// and only that one: a record of the counted frames before it reads.
#[test]
fn a_frame_pushed_uncounted_is_an_error() {
    let mut f = fixture_counting(2, [true, true, false]);
    f.enqueue(0, &[WHOLE]);
    match f.flush().unwrap_err() {
        RingError::Uncounted { record, frame } => assert_eq!((record, frame), (0, 8)),
        e => panic!("expected an uncounted error, got {e}"),
    }
    f.enqueue(1, &[first([true; 3], 8)]);
    let got = f.flush().unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].1.nframe, got[0].1.nb_obs), (8, nb_obs(8)));
}

fn stamp(e: RingError) -> (u64, usize, u64) {
    match e {
        RingError::Stamp {
            record,
            frame,
            found,
        } => (record, frame, found),
        e => panic!("expected a stamp error, got {e}"),
    }
}

/// A stack not pushed leaves its frames with the stamp of the slot's last record, or none.
#[test]
fn skipped_push_is_a_stamp_error() {
    let skip = Record {
        push: [true, false, true],
        ends: 1,
        nframe: NFRAME,
    };
    let mut f = fixture(3);
    f.enqueue(0, &[skip]);
    assert_eq!(stamp(f.flush().unwrap_err()), (0, 4, u64::MAX));

    let mut f = fixture(3);
    let skip = Record {
        push: [true, true, false],
        ends: 1,
        nframe: NFRAME,
    };
    f.enqueue(0, &[WHOLE, WHOLE, WHOLE, WHOLE, skip]);
    assert_eq!(stamp(f.flush().unwrap_err()), (4, 8, 1));
}

/// A body that ends two records, or none, where the host counts one, moves the later records
/// off their slots.
#[test]
fn ends_out_of_step_with_counts_are_a_stamp_error() {
    let twice = Record {
        push: [true; 3],
        ends: 2,
        nframe: NFRAME,
    };
    let mut f = fixture(4);
    f.enqueue(0, &[WHOLE, twice, WHOLE]);
    assert_eq!(stamp(f.flush().unwrap_err()), (2, 0, u64::MAX));

    // Record 2 lands on record 1's slot and stamp, so record 1 reads with record 2's values;
    // only record 2's empty slot shows it, and the flush returns none of them.
    let none = Record {
        push: [true; 3],
        ends: 0,
        nframe: NFRAME,
    };
    let mut f = fixture(4);
    f.enqueue(0, &[WHOLE, none, WHOLE]);
    assert_eq!(stamp(f.flush().unwrap_err()), (2, 0, u64::MAX));
    assert!(f.flush().unwrap().is_empty());
}

/// Counting adds each frame to its unmasked pixels' frame counts once: the first two stacks'
/// frames, then all of them with the first stack counted again, which leaves both its pixels'
/// frame counts and its frames' unmasked pixels as they were. A count of counts too short for
/// the ring's frames is refused, as is a range past its frames or one that ends before it
/// starts.
#[test]
fn pixel_frames_count_each_frame_once() {
    let mut f = fixture_counting(2, [true, true, false]);
    let want = pixel_frames(0..8);
    assert_ne!(want.iter().min(), want.iter().max(), "pixels differ");
    assert_eq!(f.pixel_frames(), want);
    f.count(0);
    f.count(2);
    assert_eq!(f.pixel_frames(), pixel_frames(0..NFRAME));
    f.enqueue(0, &[WHOLE]);
    check_whole(&f.flush().unwrap(), 0..1);

    let half_counts =
        DeviceBuffer::from_host(&f.stream, &counts()[..NFRAME as usize * 32]).unwrap();
    let dev = f.ring.device();
    let refused = [
        dev.count(&f.ctx, 0..4, &half_counts),
        dev.count(&f.ctx, 8..NFRAME + 1, &f.counts),
        dev.count(&f.ctx, std::ops::Range { start: 5, end: 4 }, &f.counts),
    ];
    for e in refused {
        assert!(matches!(e, Err(Error::Shape(_))), "{e:?}");
    }
}

/// The frame counts come out the same whatever ranges count the frames and in whatever
/// order: three ranges forward and backward, one frame per range backward, and one range.
#[test]
fn pixel_frames_are_the_same_in_any_frame_order() {
    let cuda = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cuda).expect("load ptyrs-gpu kernels");
    let stream = cuda.new_stream().unwrap();
    let ctx = ExecutionContext::new(stream.clone());
    let dims = dims();
    let dev_counts = DeviceBuffer::from_host(&stream, &counts()).unwrap();
    let layouts = [
        STACKS.to_vec(),
        STACKS.iter().rev().cloned().collect(),
        (0..NFRAME).rev().map(|f| f..f + 1).collect(),
        std::iter::once(0..NFRAME).collect(),
    ];
    let want = pixel_frames(0..NFRAME);
    for ranges in layouts {
        let mut ring = LlkRing::<()>::new(&kernels, &stream, &dims, 1).unwrap();
        for r in &ranges {
            ring.device().count(&ctx, r.clone(), &dev_counts).unwrap();
        }
        let got = ring.device().pixel_frames().to_host_vec(&stream).unwrap();
        assert_eq!(got, want, "ranges {ranges:?}");
    }
}
