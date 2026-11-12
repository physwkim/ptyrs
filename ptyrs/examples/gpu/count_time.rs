//! Time of [`ptyrs::gpu::RingDevice::count`] on CUDA device 0: the first count of each frame,
//! every frame new, and a repeat, every frame already counted; then the copy floors for the
//! same bytes.
//!
//! `cargo run --release -p ptyrs --example count_time -- [nframes] [range] [frame] [reps]`
//! (defaults `400,100000`, 0, 256, 10) counts frames `0 .. nframe` of `frame` x `frame` pixels
//! in calls of `range` frames each, or in one call with `range` 0, as the engine's setup does;
//! live counts each landed range. No pixel is masked, as in the engine's `gpu_time` scan, so
//! every frame adds to every pixel's frame count. Each rep takes a fresh ring and counts all
//! frames twice, each call from its own execution context; each pass is timed on the host from
//! its first launch to a stream sync, and on the device between events around its launches.
//! Prints the medians over the reps, per call and per frame, then the rates of a 256 MiB copy
//! host to device from pinned memory and device to device (read plus write bytes).
//! The device counts of `nframe` frames must fit on the card.

use std::time::Instant;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, DeviceBuffer, PinnedHostBuffer};
use ptyrs::gpu::{Kernels, LlkRing, ViewDims};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: count_time [nframes] [range] [frame] [reps]";
    let mut args = std::env::args().skip(1);
    let nframes = args.next().unwrap_or_else(|| "400,100000".into());
    let nframes = nframes
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<u32>, _>>()?;
    let mut next = |default: u32| args.next().map_or(Ok(default), |s| s.parse());
    let (range, frame, reps) = (next(0)?, next(256)?, next(10)?);
    if args.next().is_some() || reps == 0 {
        return Err(usage.into());
    }

    let cuda = CudaContext::new(0)?;
    let kernels = Kernels::load(&cuda)?;
    let stream = cuda.new_stream()?;
    let (e0, e1) = (cuda.new_event()?, cuda.new_event()?);
    for nframe in nframes {
        let dims = ViewDims {
            nframe,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [frame; 2],
            obj: [frame; 2],
            quad_phase: 0.0,
        };
        let step = if range == 0 { nframe } else { range };
        let ranges: Vec<_> = (0..nframe)
            .step_by(step as usize)
            .map(|f| f..(f + step).min(nframe))
            .collect();
        let counts = DeviceBuffer::<u16>::zeroed(&stream, nframe as usize * frame.pow(2) as usize)?;
        // Host then device µs of a pass, of the first pass then the repeat.
        let mut us = [[(); 2].map(|_| Vec::new()), [(); 2].map(|_| Vec::new())];
        for _ in 0..reps {
            let mut ring = LlkRing::<()>::new(&kernels, &stream, &dims, 1)?;
            stream.synchronize()?;
            for [host, dev] in &mut us {
                let t = Instant::now();
                e0.record(&stream)?;
                for r in &ranges {
                    let ctx = ExecutionContext::new(stream.clone());
                    ring.device().count(&ctx, r.clone(), &counts)?;
                }
                e1.record(&stream)?;
                stream.synchronize()?;
                host.push(1e6 * t.elapsed().as_secs_f64());
                dev.push(1e3 * f64::from(e0.elapsed_time(&e1)?));
            }
        }
        let [first, repeat] = us.map(|pass| pass.map(median));
        let (calls, frames) = (ranges.len() as f64, f64::from(nframe));
        println!(
            "count nframe {nframe} range {step} frame {frame}: {} calls; first host {:.2} device {:.2} us/call, device {:.4} us/frame; repeat host {:.2} device {:.2} us/call; median of {reps}",
            ranges.len(),
            first[0] / calls,
            first[1] / calls,
            first[1] / frames,
            repeat[0] / calls,
            repeat[1] / calls,
        );
    }

    let len = 128 << 20;
    let host = PinnedHostBuffer::<u16>::zeroed(&cuda, len)?;
    let mut a = DeviceBuffer::<u16>::zeroed(&stream, len)?;
    let b = DeviceBuffer::<u16>::zeroed(&stream, len)?;
    let mut rates = [Vec::new(), Vec::new()];
    for _ in 0..reps {
        for (k, rate) in rates.iter_mut().enumerate() {
            e0.record(&stream)?;
            if k == 0 {
                // SAFETY: `host` outlives the copy, which the sync below completes.
                unsafe { a.copy_from_host_async_unchecked(&stream, host.as_slice())? };
            } else {
                a.copy_from_device_async(&b, &stream)?;
            }
            e1.record(&stream)?;
            stream.synchronize()?;
            let bytes = (2 * len * (k + 1)) as f64;
            rate.push(bytes / (1e6 * f64::from(e0.elapsed_time(&e1)?)));
        }
    }
    let [h2d, d2d] = rates.map(median);
    println!(
        "copy 256 MiB: host to device {h2d:.1} GB/s, device to device {d2d:.1} GB/s read plus write; median of {reps}"
    );
    Ok(())
}
