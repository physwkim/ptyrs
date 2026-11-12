//! Per-cycle time of the M1 and the fused AP and DM cycles on CUDA device 0.
//!
//! `cargo run --release -p ptyrs-gpu --example fused_time -- [ncycle] [sizes] [paths]`
//! (defaults 50, `8,16,20,25,32,40` and `ap-m1,ap-fused,dm-m1,dm-fused`) builds the scan of
//! the engine's `gpu_time` (400 Fermat positions 16 object pixels apart, 256 x 256 frames,
//! 2 probe modes, 1 object mode, counts from the model) and, per comma-separated stack size,
//! runs each selected path, the engine's M1 cycles (object inertia 0.01, probe inertia 0.001,
//! DM alpha 0.02 and one object-then-probe update, no LLK) or the fused passes: 5 untimed
//! cycles, then `ncycle` timed ones, eagerly and as replays of a graph captured from one
//! steady cycle.

use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_core::{AddrTable, Geometry, Positions, required_obj_shape};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    C32, Error, Exit, FusedBufs, FusedInputs, FusedSums, Kernels, StackTable, UpdateSums, ViewAddr,
    ViewDims, Views, record,
};

const WARMUP: usize = 5;
const OBJ_INERTIA: f32 = 0.01;
const PROBE_INERTIA: f32 = 0.001;
const ALPHA: f32 = 0.02;
const NPOS: usize = 400;
const FRAME: usize = 256;
const NPROBE: usize = 2;
const STEP: f64 = 16.0;

/// What every run reads: the views, counts and starting arrays.
struct Scene {
    views: Views,
    counts: DeviceBuffer<u16>,
    obj: Vec<C32>,
    probe: Vec<C32>,
}

#[derive(Clone, Copy, PartialEq)]
enum Algo {
    Ap,
    Dm,
}

/// One run's buffers, laid out as the engine lays out its M1 stacks.
struct Run {
    table: StackTable,
    /// Per stack, its plan in `plans` and its exit waves' length in values.
    plan_of: Vec<usize>,
    len: Vec<u32>,
    plans: Vec<Plan2d>,
    psi: Vec<DeviceBuffer<C32>>,
    scratch: DeviceBuffer<C32>,
    bufs: FusedBufs,
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
}

impl Run {
    fn new(
        stream: &Arc<CudaStream>,
        d: &ViewDims,
        obj: &[C32],
        probe: &[C32],
        size: u32,
    ) -> Result<Self, Error> {
        let (n, nmode) = (d.frame[0] as usize * d.frame[1] as usize, d.nmode as usize);
        let ranges: Vec<_> = (0..d.nframe)
            .step_by(size as usize)
            .map(|f0| f0..d.nframe.min(f0 + size))
            .collect();
        let table = StackTable::new(stream, d, &ranges)?;
        let mut lens: Vec<u32> = Vec::new();
        let mut plan_of = Vec::new();
        for r in &ranges {
            let nf = r.end - r.start;
            plan_of.push(lens.iter().position(|&l| l == nf).unwrap_or_else(|| {
                lens.push(nf);
                lens.len() - 1
            }));
        }
        let plans = lens
            .iter()
            .map(|&nf| Plan2d::new(stream, d.frame[0] as usize, nf as usize * nmode))
            .collect::<Result<Vec<_>, _>>()?;
        let buf_len = table.nf_max() as usize * nmode * n;
        let psi = ranges
            .iter()
            .map(|_| DeviceBuffer::zeroed(stream, buf_len))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            len: ranges
                .iter()
                .map(|r| (r.end - r.start) * d.nmode * n as u32)
                .collect(),
            plan_of,
            plans,
            psi,
            scratch: DeviceBuffer::zeroed(stream, buf_len)?,
            bufs: FusedBufs::new(stream, d, table.nf_max())?,
            obj: DeviceBuffer::from_host(stream, obj)?,
            probe: DeviceBuffer::from_host(stream, probe)?,
            obj_sums: UpdateSums::zeroed(stream, obj.len())?,
            probe_sums: UpdateSums::zeroed(stream, probe.len())?,
            table,
        })
    }

    /// Engine `ap_cycle` without LLK.
    fn m1_ap(&mut self, ctx: &ExecutionContext, k: &Kernels, sc: &Scene) -> Result<(), Error> {
        let (v, d) = (&sc.views, *sc.views.dims());
        for s in 0..self.table.len() {
            let (stack, psi) = (self.table.at(s), &mut self.psi[s]);
            let plan = &mut self.plans[self.plan_of[s]];
            k.obj_probe_to_psi(ctx, v, stack, &self.obj, &self.probe, psi)?;
            plan.forward_inplace(ctx, psi)?;
            k.apply_amplitude(ctx, &d, stack, &sc.counts, psi)?;
            plan.inverse_inplace(ctx, psi)?;
            k.psi_to_obj(ctx, v, stack, psi, &self.probe, &mut self.obj_sums)?;
            k.psi_to_probe(ctx, v, stack, psi, &self.obj, &mut self.probe_sums)?;
        }
        self.update_obj(ctx, k, sc)?;
        self.update_probe(ctx, k)
    }

    /// The AP cycle with `fused_project`, which adds both sums.
    fn fused_ap(&mut self, ctx: &ExecutionContext, k: &Kernels, sc: &Scene) -> Result<(), Error> {
        for s in 0..self.table.len() {
            let stack = self.table.at(s);
            let inputs = FusedInputs {
                views: &sc.views,
                counts: &sc.counts,
                obj: &self.obj,
                probe: &self.probe,
            };
            let exit = Exit::Ap { llk: None };
            let sums = FusedSums {
                obj: Some(&mut self.obj_sums),
                probe: Some(&mut self.probe_sums),
            };
            k.fused_project(ctx, &inputs, stack, exit, &mut self.bufs, sums)?;
        }
        self.update_obj(ctx, k, sc)?;
        self.update_probe(ctx, k)
    }

    /// Engine `dm_cycle` with one object-then-probe update.
    fn m1_dm(
        &mut self,
        ctx: &ExecutionContext,
        k: &Kernels,
        sc: &Scene,
        first: bool,
    ) -> Result<(), Error> {
        let (v, d) = (&sc.views, *sc.views.dims());
        for s in 0..self.table.len() {
            let (stack, psi) = (self.table.at(s), &mut self.psi[s]);
            let plan = &mut self.plans[self.plan_of[s]];
            let po = if first { &mut *psi } else { &mut self.scratch };
            k.obj_probe_to_psi(ctx, v, stack, &self.obj, &self.probe, po)?;
            if !first {
                k.dm_reflect(ctx, psi, &mut self.scratch, self.len[s], ALPHA)?;
            }
            plan.forward_inplace(ctx, psi)?;
            k.apply_amplitude(ctx, &d, stack, &sc.counts, psi)?;
            plan.inverse_inplace(ctx, psi)?;
            if !first {
                k.add_c32(ctx, psi, &self.scratch, self.len[s])?;
            }
        }
        for s in 0..self.table.len() {
            let stack = self.table.at(s);
            k.psi_to_obj(ctx, v, stack, &self.psi[s], &self.probe, &mut self.obj_sums)?;
        }
        self.update_obj(ctx, k, sc)?;
        self.dm_probe(ctx, k, sc)
    }

    /// The DM cycle with `fused_project`, whose object sums replace the object-only pass.
    fn fused_dm(
        &mut self,
        ctx: &ExecutionContext,
        k: &Kernels,
        sc: &Scene,
        first: bool,
    ) -> Result<(), Error> {
        for s in 0..self.table.len() {
            let inputs = FusedInputs {
                views: &sc.views,
                counts: &sc.counts,
                obj: &self.obj,
                probe: &self.probe,
            };
            let psi = &mut self.psi[s];
            let exit = if first {
                Exit::DmFirst { psi }
            } else {
                Exit::Dm { psi, alpha: ALPHA }
            };
            let sums = FusedSums {
                obj: Some(&mut self.obj_sums),
                probe: None,
            };
            k.fused_project(ctx, &inputs, self.table.at(s), exit, &mut self.bufs, sums)?;
        }
        self.update_obj(ctx, k, sc)?;
        self.dm_probe(ctx, k, sc)
    }

    /// DM's probe-only pass, from the new object.
    fn dm_probe(&mut self, ctx: &ExecutionContext, k: &Kernels, sc: &Scene) -> Result<(), Error> {
        for s in 0..self.table.len() {
            let stack = self.table.at(s);
            k.psi_to_probe(
                ctx,
                &sc.views,
                stack,
                &self.psi[s],
                &self.obj,
                &mut self.probe_sums,
            )?;
        }
        self.update_probe(ctx, k)
    }

    fn update_obj(&mut self, ctx: &ExecutionContext, k: &Kernels, sc: &Scene) -> Result<(), Error> {
        k.update_obj_from_sums(
            ctx,
            &sc.views,
            &mut self.obj,
            &mut self.obj_sums,
            OBJ_INERTIA,
        )
    }

    fn update_probe(&mut self, ctx: &ExecutionContext, k: &Kernels) -> Result<(), Error> {
        let n = self.probe.len() as u32;
        k.update_from_sums(ctx, &mut self.probe, &mut self.probe_sums, n, PROBE_INERTIA)
    }

    fn cycle(
        &mut self,
        ctx: &ExecutionContext,
        k: &Kernels,
        sc: &Scene,
        (algo, fused): (Algo, bool),
        first: bool,
    ) -> Result<(), Error> {
        match (algo, fused) {
            (Algo::Ap, false) => self.m1_ap(ctx, k, sc),
            (Algo::Ap, true) => self.fused_ap(ctx, k, sc),
            (Algo::Dm, false) => self.m1_dm(ctx, k, sc, first),
            (Algo::Dm, true) => self.fused_dm(ctx, k, sc, first),
        }
    }
}

/// The engine's `gpu_time` scan: its geometry, views and object shape, with a unit-modulus
/// random-phase object, the simulated probe and counts of their far field.
fn scene(stream: &Arc<CudaStream>, k: &Kernels) -> Result<Scene, Box<dyn std::error::Error>> {
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [FRAME, FRAME])?;
    let pixel = g.pixel_size_object();
    let f = ptyrs_core::sim::fermat(NPOS, STEP * pixel[1]);
    let round = |v: Vec<f64>| v.into_iter().map(|x| x as f32 as f64).collect();
    let positions = Positions::new(round(f.y), round(f.x))?.pynx_preprocess(pixel);
    let shape = required_obj_shape(&positions, pixel, g.frame, 16);
    let addr = AddrTable::new(&positions, pixel, g.frame, shape, NPROBE, 1)?;
    let dims = ViewDims {
        nframe: NPOS as u32,
        nmode: NPROBE as u32,
        nprobe: NPROBE as u32,
        nobj: 1,
        frame: [FRAME as u32; 2],
        obj: shape.map(|s| s as u32),
        quad_phase: (PI * pixel[1] * pixel[1] / (g.wavelength * g.distance)) as f32,
    };
    let addrs: Vec<ViewAddr> = addr
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
    let views = Views::new(stream, dims, &addrs)?;

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let obj: Vec<C32> = (0..shape[0] * shape[1])
        .map(|_| {
            let t = uniform() - 0.5;
            C32 {
                re: t.cos() as f32,
                im: t.sin() as f32,
            }
        })
        .collect();
    // Scaled so a frame's far field holds about 2e5 photons.
    let scale = (2e5f64).sqrt();
    let probe: Vec<C32> = ptyrs_core::sim::probe(g.frame, NPROBE)?
        .as_slice()
        .iter()
        .map(|c| C32 {
            re: (c.re * scale) as f32,
            im: (c.im * scale) as f32,
        })
        .collect();

    // Counts: the model's intensity, rounded, one 16-frame stack at a time.
    let (n, size) = (FRAME * FRAME, 16);
    let mut run = Run::new(stream, &dims, &obj, &probe, size)?;
    let mut icalc = DeviceBuffer::<f32>::zeroed(stream, size as usize * n)?;
    let mut counts = Vec::with_capacity(NPOS * n);
    for s in 0..run.table.len() {
        let ctx = ExecutionContext::new(stream.clone());
        let stack = run.table.at(s);
        let psi = &mut run.psi[s];
        k.obj_probe_to_psi(&ctx, &views, stack, &run.obj, &run.probe, psi)?;
        run.plans[run.plan_of[s]].forward_inplace(&ctx, psi)?;
        k.mode_intensity(&ctx, &dims, stack, psi, &mut icalc)?;
        stream.synchronize()?;
        drop(ctx);
        let pixels = run.len[s] as usize / NPROBE;
        let host = icalc.to_host_vec(stream)?;
        counts.extend(
            host[..pixels]
                .iter()
                .map(|&c| c.round().min(60000.0) as u16),
        );
    }
    Ok(Scene {
        views,
        counts: DeviceBuffer::from_host(stream, &counts)?,
        obj,
        probe,
    })
}

/// Milliseconds per cycle of `ncycle` cycles, each enqueued by `cycle` on a fresh context
/// as the engine enqueues its own, synchronized once at the end.
fn time(
    stream: &Arc<CudaStream>,
    ncycle: usize,
    mut cycle: impl FnMut(&ExecutionContext) -> Result<(), Error>,
) -> Result<f64, Box<dyn std::error::Error>> {
    stream.synchronize()?;
    let t = Instant::now();
    for _ in 0..ncycle {
        cycle(&ExecutionContext::new(stream.clone()))?;
    }
    stream.synchronize()?;
    Ok(1e3 * t.elapsed().as_secs_f64() / ncycle as f64)
}

fn rel_diff(a: &[C32], b: &[C32]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (a, b) in a.iter().zip(b) {
        let (dr, di) = ((a.re - b.re) as f64, (a.im - b.im) as f64);
        num += dr * dr + di * di;
        den += (a.re as f64).powi(2) + (a.im as f64).powi(2);
    }
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: fused_time [ncycle] [sizes] [paths]";
    let mut args = std::env::args().skip(1);
    let ncycle: usize = args.next().map_or(Ok(50), |s| s.parse())?;
    let sizes = args.next().unwrap_or_else(|| "8,16,20,25,32,40".into());
    let sizes = sizes
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<u32>, _>>()
        .map_err(|_| usage)?;
    let paths = args
        .next()
        .unwrap_or_else(|| "ap-m1,ap-fused,dm-m1,dm-fused".into());
    let paths = paths
        .split(',')
        .map(|p| match p {
            "ap-m1" => Ok((Algo::Ap, false)),
            "ap-fused" => Ok((Algo::Ap, true)),
            "dm-m1" => Ok((Algo::Dm, false)),
            "dm-fused" => Ok((Algo::Dm, true)),
            _ => Err(usage),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ncycle == 0 || sizes.contains(&0) || args.next().is_some() {
        return Err(usage.into());
    }

    let cuda = CudaContext::new(0)?;
    let k = Kernels::load(&cuda)?;
    let stream = cuda.default_stream();
    let side = cuda.new_stream()?;
    let sc = scene(&stream, &k)?;
    let d = sc.views.dims();
    eprintln!(
        "{} frames {:?}, {} probe modes, object {:?}",
        d.nframe, d.frame, d.nprobe, d.obj
    );

    for size in sizes {
        for algo in [Algo::Ap, Algo::Dm] {
            let name = if algo == Algo::Ap { "ap" } else { "dm" };
            let mut objs = Vec::new();
            for fused in [false, true] {
                let path = (algo, fused);
                if !paths.contains(&path) {
                    continue;
                }
                let label = if fused { "fused" } else { "m1" };

                let mut run = Run::new(&stream, d, &sc.obj, &sc.probe, size)?;
                let mut ic = 0;
                time(&stream, WARMUP, |ctx| {
                    ic += 1;
                    run.cycle(ctx, &k, &sc, path, ic == 1)
                })?;
                let eager = time(&stream, ncycle, |ctx| run.cycle(ctx, &k, &sc, path, false))?;
                objs.push(run.obj.to_host_vec(&stream)?);

                let mut run = Run::new(&stream, d, &sc.obj, &sc.probe, size)?;
                let mut ic = 0;
                time(&stream, WARMUP, |ctx| {
                    ic += 1;
                    run.cycle(ctx, &k, &sc, path, ic == 1)
                })?;
                let graph = CudaGraph::capture(
                    side.clone(),
                    record(|ctx| run.cycle(ctx, &k, &sc, path, false)),
                )?;
                // SAFETY (both replays): `ctx` holds the replay's leases on every captured
                // buffer until the stream has run it, so no other stream touches them.
                time(&stream, 1, |ctx| {
                    Ok(unsafe { graph.launch().execute(ctx) }?)
                })?;
                let replay = time(&stream, ncycle, |ctx| {
                    Ok(unsafe { graph.launch().execute(ctx) }?)
                })?;
                println!(
                    "{name} size {size} {label}: eager {eager:.3} graph {replay:.3} ms/cycle over {ncycle} cycles"
                );
            }
            if let [m1, fused] = &objs[..] {
                println!(
                    "{name} size {size}: object rel L2 m1 vs fused after {} eager cycles {:.2e}",
                    WARMUP + ncycle,
                    rel_diff(m1, fused)
                );
            }
        }
    }
    Ok(())
}
