//! M6: the CG and BH dot products in their SIMT (`ptyrs-gpu`) and Tile (`ptyrs-tile`) forms.
//! Every call is first checked bit for bit, Tile against SIMT from the same inputs. Each
//! call is then captured as a graph of `reps` calls, one graph per form; each round replays
//! every graph once, in an order rotated per round, between two timing events, and the
//! report is µs per call (median, min, max over the rounds).
//!
//! `ml`: `gpu_time ml`'s scan (400 frames of 256x256, 2 probe modes, one 636x632 object): a
//! CG cycle's `beta_parts` of the object and of the probe and its sums of `2 CG_BLOCKS` and
//! of `gamma_parts` pairs; the start's `norm_parts` of each array and their sum of
//! `CG_BLOCKS` pairs.
//! `bh`: M3's BH scan (1024 frames of 256x256, 2 probe modes, one 852x852 object): `dots`
//! without `v` (batch 96, steepest descent) and with it (full batch, CG), each one launch
//! for the object and one for the probe.
//!
//! `cargo run --release -p ptyrs-tile --example dots_time -- <ml|bh> [reps] [rounds]`

use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as TIMED;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_gpu::{
    BhDisp, BhKernels, C32, CG_BLOCKS, Error, GradKernels, Kernels, ViewDims, bh_parts,
    gamma_parts, record,
};
use ptyrs_tile::TileDots;

/// Deterministic values in `[-scale, scale)`.
fn noise(len: usize, seed: u64, scale: f32) -> Vec<C32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
    };
    (0..len)
        .map(|_| C32 {
            re: next(),
            im: next(),
        })
        .collect()
}

fn flat(v: &[C32]) -> Vec<f32> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

#[derive(Clone, Copy)]
enum Op {
    /// `beta_parts` of array `i` (object 0, probe 1) into its `CG_BLOCKS` pairs.
    Beta(usize),
    /// `sum_parts` of this many pairs.
    Sum(u32),
    /// `norm_parts` of array `i`.
    Norm(usize),
    /// `dots`, with `v` when true.
    Dots(bool),
}

impl Op {
    fn name(self, gamma: u32) -> String {
        const ARRAY: [&str; 2] = ["object", "probe"];
        match self {
            Op::Beta(i) => format!("beta_parts {}", ARRAY[i]),
            Op::Sum(n) if n == gamma => format!("sum_parts {n} (gamma)"),
            Op::Sum(n) => format!("sum_parts {n}"),
            Op::Norm(i) => format!("norm_parts {}", ARRAY[i]),
            Op::Dots(false) => "dots batch 96 (no v)".into(),
            Op::Dots(true) => "dots full batch (v)".into(),
        }
    }
}

struct Impls {
    grad: GradKernels,
    bh: BhKernels,
    tile: TileDots,
    dims: ViewDims,
    lens: [u32; 2],
}

/// Every buffer a call reads or writes; `h` doubles as BH's `v`.
struct Bufs {
    g: [DeviceBuffer<C32>; 2],
    h: [DeviceBuffer<C32>; 2],
    /// `max(bh_parts, 4 CG_BLOCKS)` floats, `CG_BLOCKS`-pair partials in front.
    parts: DeviceBuffer<f32>,
    /// `gamma_parts` pairs.
    gamma: DeviceBuffer<f32>,
    pair: DeviceBuffer<f32>,
}

impl Op {
    fn call(
        self,
        ctx: &ExecutionContext,
        k: &Impls,
        b: &mut Bufs,
        tile: bool,
    ) -> Result<(), Error> {
        match self {
            Op::Beta(i) => {
                let (n, first) = (k.lens[i], i as u32 * CG_BLOCKS);
                let (g, h) = (&b.g[i], &mut b.h[i]);
                if tile {
                    k.tile.beta_parts(ctx, g, h, n, &mut b.parts, first)
                } else {
                    k.grad.beta_parts(ctx, g, h, n, &mut b.parts, first)
                }
            }
            Op::Sum(npart) => {
                let src = if npart > 2 * CG_BLOCKS {
                    &b.gamma
                } else {
                    &b.parts
                };
                if tile {
                    k.tile.sum_parts(ctx, src, npart, &mut b.pair)
                } else {
                    k.grad.sum_parts(ctx, src, npart, &mut b.pair)
                }
            }
            Op::Norm(i) => {
                if tile {
                    k.tile.norm_parts(ctx, &b.g[i], k.lens[i], &mut b.parts)
                } else {
                    k.grad.norm_parts(ctx, &b.g[i], k.lens[i], &mut b.parts)
                }
            }
            Op::Dots(cg) => {
                let d = BhDisp {
                    u: [&b.g[0], &b.g[1]],
                    u_scale: 0.37,
                    v: cg.then_some([&b.h[0], &b.h[1]]),
                };
                if tile {
                    k.tile.dots(ctx, &d, &mut b.parts)
                } else {
                    k.bh.dots(ctx, &k.dims, &d, &mut b.parts)
                }
            }
        }
    }

    /// The floats the call writes.
    fn outputs(self, stream: &Arc<CudaStream>, b: &Bufs) -> Vec<f32> {
        let parts = b.parts.to_host_vec(stream).unwrap();
        match self {
            Op::Beta(i) => {
                let mut v = parts[..4 * CG_BLOCKS as usize].to_vec();
                v.extend(flat(&b.h[i].to_host_vec(stream).unwrap()));
                v
            }
            Op::Sum(_) => b.pair.to_host_vec(stream).unwrap(),
            Op::Norm(_) | Op::Dots(_) => parts[..4 * CG_BLOCKS as usize].to_vec(),
        }
    }
}

fn median_spread(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map_or("ml", String::as_str);
    let reps: usize = args.get(2).map_or(100, |s| s.parse().unwrap());
    let rounds: usize = args.get(3).map_or(21, |s| s.parse().unwrap());
    let (frames, obj) = match case {
        "ml" => (400, [636, 632]),
        "bh" => (1024, [852, 852]),
        c => panic!("case {c:?} is not ml|bh"),
    };
    let dims = ViewDims {
        nframe: frames,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [256, 256],
        obj,
        quad_phase: 0.0,
    };
    let lens = [obj[0] * obj[1], 2 * 256 * 256];
    let gamma = gamma_parts(&dims).unwrap();
    let ops: Vec<Op> = match case {
        "ml" => vec![
            Op::Beta(0),
            Op::Beta(1),
            Op::Sum(2 * CG_BLOCKS),
            Op::Sum(gamma),
            Op::Norm(0),
            Op::Norm(1),
            Op::Sum(CG_BLOCKS),
        ],
        _ => vec![Op::Dots(false), Op::Dots(true)],
    };

    let dev = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&dev).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = dev.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let jit0 = cutile::tile_kernel::jit_compile_count();
    let t = std::time::Instant::now();
    let tile = TileDots::new(&stream, &dims).unwrap();
    println!(
        "{case}: object {} and probe {} values, gamma {gamma} pairs; TileDots::new compiled {} \
         kernels in {:.2} s",
        lens[0],
        lens[1],
        cutile::tile_kernel::jit_compile_count() - jit0,
        t.elapsed().as_secs_f64()
    );
    let k = Impls {
        grad: GradKernels::load(&k).unwrap(),
        bh: BhKernels::load(&k).unwrap(),
        tile,
        dims,
        lens,
    };

    let up = |v: &[C32]| DeviceBuffer::from_host(&stream, v).unwrap();
    let g0 = lens.map(|n| noise(n as usize, 10 + n as u64, 1.0));
    let h0 = lens.map(|n| noise(n as usize, 20 + n as u64, 1.0));
    let parts0 = flat(&noise(
        bh_parts(&dims).max(4 * CG_BLOCKS as usize) / 2,
        1,
        3.0,
    ));
    let mut b = Bufs {
        g: [up(&g0[0]), up(&g0[1])],
        h: [up(&h0[0]), up(&h0[1])],
        parts: DeviceBuffer::from_host(&stream, &parts0).unwrap(),
        gamma: DeviceBuffer::from_host(&stream, &flat(&noise(gamma as usize, 2, 5.0))).unwrap(),
        pair: DeviceBuffer::from_host(&stream, &[0.0f32; 2]).unwrap(),
    };
    let reset = |b: &mut Bufs| {
        for (h, h0) in b.h.iter_mut().zip(&h0) {
            h.copy_from_host(&stream, h0).unwrap();
        }
        b.parts.copy_from_host(&stream, &parts0).unwrap();
        b.pair.copy_from_host(&stream, &[0.0f32; 2]).unwrap();
        stream.synchronize().unwrap();
    };

    let jit = cutile::tile_kernel::jit_compile_count();
    for &op in &ops {
        let [want, got] = [false, true].map(|tile| {
            reset(&mut b);
            op.call(&ex, &k, &mut b, tile).unwrap();
            stream.synchronize().unwrap();
            op.outputs(&stream, &b)
        });
        let bad = got
            .iter()
            .zip(&want)
            .position(|(a, w)| a.to_bits() != w.to_bits());
        assert!(
            bad.is_none(),
            "{}: float {} differs",
            op.name(gamma),
            bad.unwrap()
        );
    }
    println!("every Tile call bit-identical to its SIMT call");

    // Every graph is captured, on a stream of its own, before any replay.
    let side = dev.new_stream().unwrap();
    let mut graphs: Vec<(Op, bool, CudaGraph<()>)> = Vec::new();
    for &op in &ops {
        for tile in [false, true] {
            let g = CudaGraph::capture(
                side.clone(),
                record(|ctx| (0..reps).try_for_each(|_| op.call(ctx, &k, &mut b, tile))),
            )
            .expect("capture");
            graphs.push((op, tile, g));
        }
    }
    reset(&mut b);
    for (_, _, g) in &graphs {
        g.launch().sync_on(&stream).unwrap();
    }

    let mut times = vec![Vec::with_capacity(rounds); graphs.len()];
    for r in 0..rounds {
        let mut events = Vec::with_capacity(graphs.len());
        for j in 0..graphs.len() {
            let i = (j + r) % graphs.len();
            let start = stream.record_event(Some(TIMED)).unwrap();
            let lease = ExecutionContext::new(stream.clone());
            // SAFETY: `lease` holds the replay's leases on every captured buffer until the
            // stream has run it.
            unsafe { graphs[i].2.launch().execute(&lease) }.unwrap();
            let end = stream.record_event(Some(TIMED)).unwrap();
            events.push((i, start, end));
        }
        stream.synchronize().unwrap();
        for (i, start, end) in events {
            times[i].push(1e3 * start.elapsed_ms(&end).unwrap() as f64 / reps as f64);
        }
    }
    assert_eq!(
        cutile::tile_kernel::jit_compile_count(),
        jit,
        "a launch, capture or replay compiled a Tile kernel"
    );

    println!(
        "us per call over {rounds} rounds of a {reps}-call graph (median, min, max); graph \
         order {}",
        graphs
            .iter()
            .map(|(op, tile, _)| format!("{} {}", op.name(gamma), if *tile { "T" } else { "S" }))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "{:28}{:>26}{:>26}{:>9}",
        "call", "SIMT", "Tile", "Tile/SIMT"
    );
    for (pair, t) in graphs.chunks(2).zip(times.chunks_mut(2)) {
        let [s, ti] = t else { unreachable!() };
        let (sm, slo, shi) = median_spread(s);
        let (tm, tlo, thi) = median_spread(ti);
        println!(
            "{:28}{:>10.2} ({:6.2} .. {:6.2}){:>10.2} ({:6.2} .. {:6.2}){:>9.3}",
            pair[0].0.name(gamma),
            sm,
            slo,
            shi,
            tm,
            tlo,
            thi,
            tm / sm
        );
    }
}
