//! Launches the four Tile kernels, or their SIMT counterparts, `reps` times each on a
//! 256x256, two-mode scan of `side`² frames as one stack, for device-time comparison under
//! `nsys` (one process per variant, so equal kernel names do not mix). The LLK totals'
//! SIMT counterpart is the host path (copy the partials, `LlkSums::from_partials`); its wall
//! time, and the Tile kernel's with the copy of its four totals, are printed here.
//!
//! `cargo run --release -p ptyrs-tile --example tile_vs_simt -- <tile|simt> [reps] [side]`

use std::sync::Arc;
use std::time::Instant;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_gpu::{
    C32, Kernels, LlkPartial, LlkSums, StackTable, UpdateSums, ViewAddr, ViewDims, Views,
};
use ptyrs_tile::{TileKernels, llk_sums_scratch_len};

const N: u32 = 256;
const STEP: u32 = 24;

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

fn median_spread(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tile = match args.get(1).map(String::as_str) {
        Some("tile") => true,
        Some("simt") => false,
        _ => panic!("usage: tile_vs_simt <tile|simt> [reps] [side]"),
    };
    let reps: usize = args.get(2).map_or(30, |s| s.parse().unwrap());
    let side: u32 = args.get(3).map_or(63, |s| s.parse().unwrap());

    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.new_stream().unwrap();

    // A raster of `side`² frames, `STEP` pixels apart with up to 3 pixels of jitter and a
    // sub-pixel offset, on an object of whole 16-pixel tiles.
    let nobj_side = ((side - 1) * STEP + 3 + N + 1).div_ceil(16) * 16;
    let d = ViewDims {
        nframe: side * side,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [N, N],
        obj: [nobj_side, nobj_side],
        quad_phase: 0.0,
    };
    let r = unit(4 * d.nframe as usize, 1);
    let views: Vec<ViewAddr> = (0..d.nframe)
        .flat_map(|f| {
            let i = 4 * f as usize;
            let oy = STEP * (f / side) + (3.0 * r[i]) as u32;
            let ox = STEP * (f % side) + (3.0 * r[i + 1]) as u32;
            let (fy, fx) = (r[i + 2], r[i + 3]);
            (0..d.nmode).map(move |m| ViewAddr {
                oy,
                ox,
                fy,
                fx,
                probe: m,
                obj: 0,
            })
        })
        .collect();
    let views = Views::new(&stream, d, &views).unwrap();
    let (n2, nk) = ((N * N) as usize, (d.nframe * d.nmode) as usize);
    let obj_len = (d.obj[0] * d.obj[1]) as usize;
    let probe_len = d.nprobe as usize * n2;

    let phase = unit(obj_len, 2);
    let obj_host: Vec<C32> = phase
        .iter()
        .map(|&p| C32 {
            re: (3.0 * p).cos(),
            im: (3.0 * p).sin(),
        })
        .collect();
    let probe_host: Vec<C32> = (0..probe_len)
        .map(|i| {
            let (m, y, x) = (i / n2, (i / N as usize) % N as usize, i % N as usize);
            let (dy, dx) = (y as f32 - 128.0, x as f32 - 128.0);
            let a = if dy * dy + dx * dx < 80.0 * 80.0 {
                1.0
            } else {
                0.05
            } / (m + 1) as f32;
            let p = 3e-4 * (dy * dy - dx * dx);
            C32 {
                re: a * p.cos(),
                im: a * p.sin(),
            }
        })
        .collect();
    let mut obj = DeviceBuffer::from_host(&stream, &obj_host).unwrap();
    let mut probe = DeviceBuffer::from_host(&stream, &probe_host).unwrap();
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, nk * n2).unwrap();
    let table = StackTable::new(&stream, &d, std::slice::from_ref(&(0..d.nframe))).unwrap();
    let tk = TileKernels::new(&stream, &views, &table).unwrap();
    // Leases every buffer the launches below touch until it drops after their sync.
    let ex = ExecutionContext::new(stream.clone());
    k.obj_probe_to_psi(&ex, &views, table.at(0), &obj, &probe, &mut psi)
        .unwrap();
    let partials_host: Vec<LlkPartial> = unit(4 * d.nframe as usize, 3)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| LlkPartial {
            poisson: 1e3 * c[0],
            gaussian: 1e2 * c[1],
            euclidean: 1e2 * c[2],
            icalc: 1e5 * c[3],
        })
        .collect();
    let partials = DeviceBuffer::from_host(&stream, &partials_host).unwrap();
    let mut obj_sums = UpdateSums::zeroed(&stream, obj_len).unwrap();
    let mut probe_sums = UpdateSums::zeroed(&stream, probe_len).unwrap();
    let mut llk = DeviceBuffer::<f64>::zeroed(&stream, 4).unwrap();
    let mut llk_scratch =
        DeviceBuffer::<f64>::zeroed(&stream, llk_sums_scratch_len(d.nframe)).unwrap();
    stream.synchronize().unwrap();
    println!(
        "{} {}: {} frames x {} modes of {N}x{N}, object {}x{}, {reps} reps",
        ctx.device_name().unwrap(),
        if tile { "tile" } else { "simt" },
        d.nframe,
        d.nmode,
        d.obj[0],
        d.obj[1]
    );

    // Two warm-up launches and `reps` timed ones of each kernel, in the
    // order the nsys trace is read: psi_to_obj, psi_to_probe, update_obj_from_sums,
    // update_from_sums on the probe, then the LLK totals.
    let total = 2 + reps;
    for _ in 0..total {
        if tile {
            tk.psi_to_obj(&ex, 0, &psi, &probe, &mut obj_sums).unwrap();
        } else {
            k.psi_to_obj(&ex, &views, table.at(0), &psi, &probe, &mut obj_sums)
                .unwrap();
        }
    }
    for _ in 0..total {
        if tile {
            tk.psi_to_probe(&ex, 0, &psi, &obj, &mut probe_sums)
                .unwrap();
        } else {
            k.psi_to_probe(&ex, &views, table.at(0), &psi, &obj, &mut probe_sums)
                .unwrap();
        }
    }
    for _ in 0..total {
        if tile {
            tk.update_obj_from_sums(&ex, &mut obj, &mut obj_sums, 0.01)
                .unwrap();
        } else {
            k.update_obj_from_sums(&ex, &views, &mut obj, &mut obj_sums, 0.01)
                .unwrap();
        }
    }
    for _ in 0..total {
        if tile {
            tk.update_probe_from_sums(&ex, &mut probe, &mut probe_sums, 0.01)
                .unwrap();
        } else {
            k.update_from_sums(&ex, &mut probe, &mut probe_sums, probe_len as u32, 0.01)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    drop(ex);

    // LLK totals to the host, wall time from an idle stream.
    let mut wall = Vec::with_capacity(reps);
    for i in 0..total {
        stream.synchronize().unwrap();
        let t = Instant::now();
        let totals = if tile {
            let ex = ExecutionContext::new(stream.clone());
            tk.llk_sums(&ex, &partials, &mut llk_scratch, &mut llk)
                .unwrap();
            llk.to_host_vec(&stream).unwrap()
        } else {
            let s = LlkSums::from_partials(&partials.to_host_vec(&stream).unwrap());
            vec![s.poisson, s.gaussian, s.euclidean, s.icalc]
        };
        let us = t.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(totals);
        if i >= 2 {
            wall.push(us);
        }
    }
    let (med, lo, hi) = median_spread(&mut wall);
    println!(
        "llk totals to host ({}): median {med:.1} us, min {lo:.1}, max {hi:.1}",
        if tile {
            "Tile llk_sums + copy of 4 f64"
        } else {
            "copy of partials + LlkSums::from_partials"
        }
    );
}
