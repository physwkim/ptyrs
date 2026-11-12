//! M6: time per graph-replayed AP epoch with each Tile candidate in its SIMT or Tile form, on
//! the M1 timing scan (`ptyrs::engine` example `gpu_time ap`: 400 frames of 256x256 on a
//! Fermat spiral of step 16 pixels, 2 probe modes, stacks of 16), with its views and object.
//! An epoch is the fused AP cycle of `ptyrs::gpu`'s `fused_time`: per stack, `fused_project`
//! (exit waves, in-kernel FFTs, amplitude projection) and both update sums of its projected
//! waves from the old arrays; then the object (inertia 0.01) and probe (0.001) updates.
//! With `llk`, each `fused_project` also writes its stack's LLK partials and every epoch
//! ends with the totals on the host: Tile `llk_sums` in the graph and a copy of its 4 f64,
//! or SIMT's copy of the partials and `LlkSums::from_partials`.
//!
//! A variant spells the object sums, the probe sums, the object update, the probe update
//! (and with `llk` the totals) as `s` (ptyrs::gpu) or `t` (ptyrs::tile); the default is every
//! steady combination, or `sssss,sssst,tttts,ttttt` with `llk`. SIMT object sums are
//! `fused_project`'s own; Tile ones are `psi_to_obj` after a `fused_project` without them. All graphs are captured before any
//! replay. One epoch of each from the same start is checked against a direct all-SIMT epoch.
//! Each round then replays every variant `epochs` times from that start, in an order rotated
//! per round; its time per epoch is the wall time of the replays, launched back to back with
//! one sync at the end (with `llk`, one sync per epoch for the read).
//!
//! `cargo run --release -p ptyrs --example epoch_graph -- [epochs] [rounds] [steady|llk] [variants]`

use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs::base::sim::fermat;
use ptyrs::base::{AddrTable, Geometry, Positions, required_obj_shape};
use ptyrs::fft::Plan2d;
use ptyrs::gpu::{
    C32, Error, Exit, FarSums, FusedBufs, FusedInputs, FusedSums, Kernels, LlkPartial, LlkSums,
    Model, StackTable, UpdateSums, ViewAddr, ViewDims, Views, record,
};
use ptyrs::tile::{TileKernels, llk_sums_scratch_len};

const N: u32 = 256;
const NPOS: u32 = 400;
const STEP: f64 = 16.0;
const STACK: u32 = 16;
const OBJ_INERTIA: f32 = 0.01;
const PROBE_INERTIA: f32 = 0.001;

/// Which form runs each Tile candidate, `true` for Tile: the object sums, psi_to_probe, the
/// object update, the probe update, the LLK totals.
#[derive(Clone, Copy, PartialEq)]
struct Variant([bool; 5]);

impl Variant {
    const SIMT: Self = Self([false; 5]);

    fn parse(s: &str, len: usize) -> Self {
        assert!(
            s.len() == len && s.bytes().all(|b| b == b's' || b == b't'),
            "variant {s:?} is not {len} of s|t"
        );
        let mut v = [false; 5];
        for (t, b) in v.iter_mut().zip(s.bytes()) {
            *t = b == b't';
        }
        Self(v)
    }

    fn name(&self, len: usize) -> String {
        self.0[..len]
            .iter()
            .map(|&t| if t { 't' } else { 's' })
            .collect()
    }
}

/// Deterministic values in `[0, 1)`.
fn unit(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

/// The views of `gpu_time ap` (`dm_scenario`): `ptyrs::base::sim::fermat(NPOS, STEP)` object
/// pixels apart, rounded to f32 metres, through `pynx_preprocess`, on the object
/// `required_obj_shape` gives (`[636, 632]`) at whole-pixel corners, two probe modes per frame.
fn scan() -> (ViewDims, Vec<ViewAddr>) {
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [N as usize; 2]).unwrap();
    let pixel = g.pixel_size_object();
    let f = fermat(NPOS as usize, STEP * pixel[1]);
    let round = |v: Vec<f64>| v.into_iter().map(|x| x as f32 as f64).collect();
    let pos = Positions::new(round(f.y), round(f.x))
        .unwrap()
        .pynx_preprocess(pixel);
    let obj = required_obj_shape(&pos, pixel, g.frame, 16);
    let addr = AddrTable::new(&pos, pixel, g.frame, obj, 2, 1).unwrap();
    let d = ViewDims {
        nframe: NPOS,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [N, N],
        obj: [obj[0] as u32, obj[1] as u32],
        quad_phase: (PI * pixel[1] * pixel[1] / (g.wavelength * g.distance)) as f32,
    };
    let views = addr
        .views()
        .iter()
        .map(|v| ViewAddr {
            oy: v.oy,
            ox: v.ox,
            fy: 0.0,
            fx: 0.0,
            probe: v.probe,
            obj: v.obj,
        })
        .collect();
    (d, views)
}

/// Mode `m` of a disc probe of radius 80 with a quadratic phase, scaled by `1 / (m + 1)`;
/// `defocus` adds the Gaussian phase `ptyrs::engine`'s `dm_scenario` starts from.
fn probe(defocus: bool) -> Vec<C32> {
    let n2 = (N * N) as usize;
    let sigma = N as f32 / 12.0;
    (0..2 * n2)
        .map(|i| {
            let (m, y, x) = (i / n2, (i / N as usize) % N as usize, i % N as usize);
            let (dy, dx) = (y as f32 - 128.0, x as f32 - 128.0);
            let r2 = dy * dy + dx * dx;
            let a = if r2 < 80.0 * 80.0 { 1.0 } else { 0.05 } / (m + 1) as f32;
            let mut p = 3e-4 * (dy * dy - dx * dx);
            if defocus {
                p -= 0.5 * r2 / (sigma * sigma);
            }
            C32 {
                re: a * p.cos(),
                im: a * p.sin(),
            }
        })
        .collect()
}

/// Unit-modulus object of phase `scale * (u - 0.5)`, `u` uniform.
fn object(len: usize, seed: u64, scale: f32) -> Vec<C32> {
    unit(len, seed)
        .iter()
        .map(|&u| {
            let p = scale * (u - 0.5);
            C32 {
                re: p.cos(),
                im: p.sin(),
            }
        })
        .collect()
}

struct Scan<'a> {
    k: &'a Kernels,
    tk: &'a TileKernels<'a>,
    views: &'a Views,
    table: &'a StackTable,
    counts: &'a DeviceBuffer<u16>,
    probe_len: u32,
}

/// What an epoch reads and writes; all of it allocated before capture.
struct State {
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    /// The fused passes' scratch, one set for every stack in turn.
    bufs: FusedBufs,
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
    /// The scan's LLK partials; `stack_partials[s]` is stack `s`'s slice of it.
    partials: DeviceBuffer<LlkPartial>,
    stack_partials: Vec<DeviceBuffer<LlkPartial>>,
    llk_scratch: DeviceBuffer<f64>,
    llk: DeviceBuffer<f64>,
}

fn epoch(
    ctx: &ExecutionContext,
    scan: &Scan,
    s: &mut State,
    v: Variant,
    llk: bool,
) -> Result<(), Error> {
    let (k, tk, views) = (scan.k, scan.tk, scan.views);
    let [obj_t, probe_t, obj_up_t, probe_up_t, llk_t] = v.0;
    for (i, partials) in s.stack_partials.iter_mut().enumerate() {
        let stack = scan.table.at(i);
        let inputs = FusedInputs {
            views,
            counts: scan.counts,
            obj: &s.obj,
            probe: &s.probe,
            model: Model::default(),
        };
        let exit = Exit::Ap {
            far: FarSums {
                llk: llk.then_some(partials),
                ..FarSums::default()
            },
        };
        let sums = FusedSums {
            obj: (!obj_t).then_some(&mut s.obj_sums),
            ..FusedSums::default()
        };
        k.fused_project(ctx, &inputs, stack, exit, &mut s.bufs, sums)?;
        let work = s.bufs.work();
        if obj_t {
            tk.psi_to_obj(ctx, stack, work, &s.probe, &mut s.obj_sums)?;
        }
        if probe_t {
            tk.psi_to_probe(ctx, stack, work, &s.obj, &mut s.probe_sums)?;
        } else {
            k.psi_to_probe(ctx, views, stack, work, &s.obj, &mut s.probe_sums)?;
        }
    }
    if obj_up_t {
        tk.obj_update().update_obj_from_sums(
            ctx,
            views,
            &mut s.obj,
            &mut s.obj_sums,
            OBJ_INERTIA,
        )?;
    } else {
        k.update_obj_from_sums(ctx, views, &mut s.obj, &mut s.obj_sums, OBJ_INERTIA)?;
    }
    if probe_up_t {
        tk.probe_update().update_probe_from_sums(
            ctx,
            &mut s.probe,
            &mut s.probe_sums,
            PROBE_INERTIA,
        )?;
    } else {
        k.update_from_sums(
            ctx,
            &mut s.probe,
            &mut s.probe_sums,
            scan.probe_len,
            PROBE_INERTIA,
        )?;
    }
    if llk && llk_t {
        tk.llk_sums(ctx, &s.partials, &mut s.llk_scratch, &mut s.llk)?;
    }
    Ok(())
}

/// The LLK totals an epoch of `v` leaves, as `[poisson, gaussian, euclidean, icalc]`.
fn totals(stream: &CudaStream, s: &State, v: Variant) -> [f64; 4] {
    if v.0[4] {
        s.llk.to_host_vec(stream).unwrap().try_into().unwrap()
    } else {
        let t = LlkSums::from_partials(&s.partials.to_host_vec(stream).unwrap());
        [t.poisson, t.gaussian, t.euclidean, t.icalc]
    }
}

/// Largest `|a - b|` over the largest `|b|`.
fn rel_diff(a: &[C32], b: &[C32]) -> f64 {
    let abs = |c: &C32| f64::from(c.re).hypot(f64::from(c.im));
    let num = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            abs(&C32 {
                re: a.re - b.re,
                im: a.im - b.im,
            })
        })
        .fold(0.0, f64::max);
    num / b.iter().map(abs).fold(0.0, f64::max)
}

fn median_spread(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = args.get(1).map_or(50, |s| s.parse().unwrap());
    let rounds: usize = args.get(2).map_or(5, |s| s.parse().unwrap());
    let llk = match args.get(3).map(String::as_str) {
        None | Some("steady") => false,
        Some("llk") => true,
        Some(m) => panic!("mode {m:?} is not steady|llk"),
    };
    let len = if llk { 5 } else { 4 };
    let variants: Vec<Variant> = match args.get(4) {
        Some(list) => list.split(',').map(|s| Variant::parse(s, len)).collect(),
        None if llk => ["sssss", "sssst", "tttts", "ttttt"]
            .iter()
            .map(|s| Variant::parse(s, 5))
            .collect(),
        None => (0..16u32)
            .map(|b| {
                let mut v = [false; 5];
                for (i, t) in v.iter_mut().take(4).enumerate() {
                    *t = b >> (3 - i) & 1 == 1;
                }
                Variant(v)
            })
            .collect(),
    };

    let dev = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&dev).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = dev.new_stream().unwrap();
    let (d, view_list) = scan();
    let views = Views::new(&stream, d, &view_list).unwrap();
    let n2 = (N * N) as usize;
    let obj_len = (d.obj[0] * d.obj[1]) as usize;
    let probe_len = d.nprobe as usize * n2;
    let up = |v: &[C32]| DeviceBuffer::from_host(&stream, v).unwrap();

    // Counts: the far-field intensity of a known object and probe, scaled to 2e4 at most.
    let counts = {
        let whole = StackTable::new(&stream, &d, std::slice::from_ref(&(0..d.nframe))).unwrap();
        let mut plan = Plan2d::new(&stream, N as usize, (d.nframe * d.nmode) as usize).unwrap();
        let mut psi =
            DeviceBuffer::<C32>::zeroed(&stream, (d.nframe * d.nmode) as usize * n2).unwrap();
        let mut icalc = DeviceBuffer::<f32>::zeroed(&stream, d.nframe as usize * n2).unwrap();
        let (truth, true_probe) = (up(&object(obj_len, 5, 1.6)), up(&probe(false)));
        record(|ctx| {
            k.obj_probe_to_psi(ctx, &views, whole.at(0), &truth, &true_probe, &mut psi)?;
            plan.forward_inplace(ctx, &mut psi)?;
            k.mode_intensity(ctx, &d, whole.at(0), &psi, &mut icalc)
        })
        .sync_on(&stream)
        .unwrap();
        let icalc = icalc.to_host_vec(&stream).unwrap();
        let scale = 2e4 / icalc.iter().fold(0f32, |m, &v| m.max(v));
        let counts: Vec<u16> = icalc.iter().map(|&v| (v * scale).round() as u16).collect();
        DeviceBuffer::from_host(&stream, &counts).unwrap()
    };

    let ranges: Vec<_> = (0..d.nframe)
        .step_by(STACK as usize)
        .map(|f0| f0..(f0 + STACK).min(d.nframe))
        .collect();
    assert!(
        d.nframe.is_multiple_of(STACK),
        "every stack's LLK partials are a slice of STACK records"
    );
    let table = StackTable::new(&stream, &d, &ranges).unwrap();
    let tk = TileKernels::new(&stream, &views, &table).unwrap();
    let jit = cutile::tile_kernel::jit_compile_count();
    let scan = Scan {
        k: &k,
        tk: &tk,
        views: &views,
        table: &table,
        counts: &counts,
        probe_len: probe_len as u32,
    };

    let partials = DeviceBuffer::<LlkPartial>::zeroed(&stream, d.nframe as usize).unwrap();
    let rec = size_of::<LlkPartial>();
    let stack_partials = ranges
        .iter()
        .map(|r| {
            let raw = partials
                .storage()
                .slice_bytes(r.start as usize * rec, STACK as usize * rec)
                .unwrap();
            // SAFETY: the bytes are initialized records of `partials`, which outlives the
            // slice in `State`, and every launch on them leases them.
            unsafe { DeviceBuffer::from_storage(raw, STACK as usize) }.unwrap()
        })
        .collect();
    let obj0 = object(obj_len, 7, 1.0);
    let probe0 = probe(true);
    let mut s = State {
        obj: up(&obj0),
        probe: up(&probe0),
        bufs: FusedBufs::new(&stream, &d, table.nf_max()).unwrap(),
        obj_sums: UpdateSums::zeroed(&stream, obj_len).unwrap(),
        probe_sums: UpdateSums::zeroed(&stream, probe_len).unwrap(),
        partials,
        stack_partials,
        llk_scratch: DeviceBuffer::zeroed(&stream, llk_sums_scratch_len(d.nframe)).unwrap(),
        llk: DeviceBuffer::zeroed(&stream, 4).unwrap(),
    };
    stream.synchronize().unwrap();
    println!(
        "{}: {} frames x {} modes of {N}x{N}, object {}x{}, {} stacks of {STACK}, {}",
        dev.device_name().unwrap(),
        d.nframe,
        d.nmode,
        d.obj[0],
        d.obj[1],
        ranges.len(),
        if llk {
            "LLK every epoch"
        } else {
            "steady (no LLK)"
        }
    );

    // Every graph is captured, on a stream of its own, before any Tile kernel launches.
    let side = dev.new_stream().unwrap();
    let graphs: Vec<CudaGraph<()>> = variants
        .iter()
        .map(|&v| {
            CudaGraph::capture(
                side.clone(),
                record(|ctx| epoch(ctx, &scan, &mut s, v, llk)),
            )
            .expect("capture an epoch")
        })
        .collect();

    let reset = |s: &mut State| {
        s.obj.copy_from_host(&stream, &obj0).unwrap();
        s.probe.copy_from_host(&stream, &probe0).unwrap();
        stream.synchronize().unwrap();
    };
    let read = |s: &State| {
        (
            s.obj.to_host_vec(&stream).unwrap(),
            s.probe.to_host_vec(&stream).unwrap(),
        )
    };
    reset(&mut s);
    record(|ctx| epoch(ctx, &scan, &mut s, Variant::SIMT, llk))
        .sync_on(&stream)
        .unwrap();
    let (want_obj, want_probe) = read(&s);
    let want_llk = llk.then(|| totals(&stream, &s, Variant::SIMT));
    println!("one epoch from the start against a direct all-SIMT epoch (max |diff| / max |SIMT|):");
    for (v, g) in variants.iter().zip(&graphs) {
        reset(&mut s);
        g.launch().sync_on(&stream).unwrap();
        let (obj, probe) = read(&s);
        let llk_note = match want_llk {
            Some(want) => {
                let got = totals(&stream, &s, *v);
                assert!(
                    got.iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{}: LLK totals {got:?}, SIMT {want:?}",
                    v.name(len)
                );
                ", LLK totals bit-equal"
            }
            None => "",
        };
        for (what, sums) in [("object", &s.obj_sums), ("probe", &s.probe_sums)] {
            let num = sums.num.to_host_vec(&stream).unwrap();
            let den = sums.den.to_host_vec(&stream).unwrap();
            assert!(
                num.iter()
                    .all(|c| c.re.to_bits() == 0 && c.im.to_bits() == 0)
                    && den.iter().all(|v| v.to_bits() == 0),
                "{}: {what} sums not consumed",
                v.name(len)
            );
        }
        println!(
            "  {}: object {:.2e}, probe {:.2e}{llk_note}",
            v.name(len),
            rel_diff(&obj, &want_obj),
            rel_diff(&probe, &want_probe)
        );
    }

    let mut times = vec![Vec::with_capacity(rounds); variants.len()];
    for r in 0..rounds {
        for j in 0..variants.len() {
            let i = (j + r) % variants.len();
            reset(&mut s);
            let t = Instant::now();
            if llk {
                for _ in 0..epochs {
                    graphs[i].launch().sync_on(&stream).unwrap();
                    std::hint::black_box(totals(&stream, &s, variants[i]));
                }
            } else {
                for _ in 0..epochs {
                    let ex = ExecutionContext::new(stream.clone());
                    // SAFETY: `ex` holds the replay's leases on every captured buffer until
                    // the stream has run it.
                    unsafe { graphs[i].launch().execute(&ex) }.unwrap();
                }
                stream.synchronize().unwrap();
            }
            times[i].push(1e3 * t.elapsed().as_secs_f64() / epochs as f64);
        }
    }
    assert_eq!(
        cutile::tile_kernel::jit_compile_count(),
        jit,
        "a capture or replay compiled a Tile kernel"
    );

    println!("ms per epoch over {rounds} rounds of {epochs} replays (median, min, max):");
    let mut med = Vec::with_capacity(variants.len());
    for (v, t) in variants.iter().zip(&mut times) {
        let (m, lo, hi) = median_spread(t);
        med.push(m);
        let tag = match v.0[..len].iter().filter(|&&t| t).count() {
            0 => "  all-SIMT",
            n if n == len => "  all-Tile",
            _ => "",
        };
        println!("  {}: {m:.3} ({lo:.3} .. {hi:.3}){tag}", v.name(len));
    }
    let mixed = variants
        .iter()
        .zip(&med)
        .filter(|(v, _)| (1..len).contains(&v.0[..len].iter().filter(|&&t| t).count()))
        .min_by(|a, b| a.1.total_cmp(b.1));
    if let Some((v, m)) = mixed {
        println!("fastest mixed: {} at {m:.3} ms", v.name(len));
    }
}
