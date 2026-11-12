//! The LLK ring: records come back in order with the totals of their own partials, across
//! wraps and copies that overlap later records, and a record with a frame not written for it
//! is a stamp error.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_gpu::{Kernels, LlkPartial, LlkRing, LlkSums, RingError, StackTable, ViewDims};

const NFRAME: u32 = 10;
/// Stacks of 4, 4 and 2 frames.
const STACKS: [std::ops::Range<u32>; 3] = [0..4, 4..8, 8..10];

/// One record as the epochs write it: which stacks push, and how many advances follow.
struct Record {
    push: [bool; 3],
    advances: usize,
}

const WHOLE: Record = Record {
    push: [true; 3],
    advances: 1,
};

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
}

fn fixture(nslot: u32) -> Fixture {
    let cuda = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&cuda).expect("load ptyrs-gpu kernels");
    let stream = cuda.new_stream().unwrap();
    let dims = ViewDims {
        nframe: NFRAME,
        nmode: 1,
        nprobe: 1,
        nobj: 1,
        frame: [8, 8],
        obj: [8, 8],
        quad_phase: 0.0,
    };
    let table = StackTable::new(&stream, &dims, &STACKS).unwrap();
    let ring = LlkRing::new(&kernels, &stream, NFRAME, nslot).unwrap();
    Fixture {
        ctx: ExecutionContext::new(stream.clone()),
        stream,
        dims,
        table,
        ring,
    }
}

impl Fixture {
    /// Enqueues `records` as numbered from `r0`, with no host sync between them: each stack's
    /// partials from its own staging buffer (uploaded first), then the advances, then the count.
    fn enqueue(&mut self, r0: usize, records: &[Record]) {
        let nf_max = self.table.nf_max() as usize;
        let staged: Vec<Vec<DeviceBuffer<LlkPartial>>> = (r0..r0 + records.len())
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
            .collect();
        for (i, rec) in records.iter().enumerate() {
            let dev = self.ring.device();
            for (s, staging) in staged[i].iter().enumerate() {
                if rec.push[s] {
                    dev.push(&self.ctx, &self.dims, self.table.at(s), staging)
                        .unwrap();
                }
            }
            for _ in 0..rec.advances {
                dev.advance(&self.ctx).unwrap();
            }
            self.ring.enqueued(&self.ctx, r0 + i).unwrap();
        }
    }

    fn flush(&mut self) -> Result<Vec<(usize, LlkSums)>, RingError> {
        self.ring.flush(&self.ctx)
    }
}

/// Flushes give each record once, in order, with the bits of `LlkSums::from_partials` of its
/// own partials: one slot (every record waits for the last), two (copy after each record),
/// and three and five (wrapping inside a copy), flushing mid-run and at the end.
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
        for (r, s) in &got {
            let want = LlkSums::from_partials(&partials(*r));
            assert_eq!(bits(s), bits(&want), "nslot {nslot}, record {r}");
        }
    }
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
        advances: 1,
    };
    let mut f = fixture(3);
    f.enqueue(0, &[skip]);
    assert_eq!(stamp(f.flush().unwrap_err()), (0, 4, u64::MAX));

    let mut f = fixture(3);
    let skip = Record {
        push: [true, true, false],
        advances: 1,
    };
    f.enqueue(0, &[WHOLE, WHOLE, WHOLE, WHOLE, skip]);
    assert_eq!(stamp(f.flush().unwrap_err()), (4, 8, 1));
}

/// A cursor advanced twice, or not at all, within one record moves the later records off
/// their slots.
#[test]
fn cursor_out_of_step_is_a_stamp_error() {
    let twice = Record {
        push: [true; 3],
        advances: 2,
    };
    let mut f = fixture(4);
    f.enqueue(0, &[WHOLE, twice, WHOLE]);
    assert_eq!(stamp(f.flush().unwrap_err()), (2, 0, u64::MAX));

    // Record 1 lands on record 0's slot and stamp, so record 0 reads with record 1's values;
    // only record 1's empty slot shows it, and the flush returns neither.
    let none = Record {
        push: [true; 3],
        advances: 0,
    };
    let mut f = fixture(4);
    f.enqueue(0, &[none, WHOLE]);
    assert_eq!(stamp(f.flush().unwrap_err()), (1, 0, u64::MAX));
    assert!(f.flush().unwrap().is_empty());
}
