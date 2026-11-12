//! Time per call of [`ptyrs_gpu::RingDevice::count`] on CUDA device 0: the first count of
//! each stack, every frame new, and a repeat, every frame already counted.
//!
//! `cargo run --release -p ptyrs-gpu --example count_time -- [nframes] [stack] [frame] [reps] [shared|fresh]`
//! (defaults `400,10000,100000`, 16, 256, 10, `shared`) counts `nframe` frames of `frame` x `frame`
//! pixels in stacks of `stack` frames. No pixel is masked, as in the engine's `gpu_time` scan,
//! so every frame adds to every pixel's frame count. Each rep takes a fresh ring and counts
//! all its stacks twice; each pass is timed on the host from its first launch to a stream
//! sync, and on the device between events around its launches, each divided by the stacks.
//! Prints the medians over the reps. `shared` launches a pass from one execution context, as
//! the engine's setup does, so each launch also checks the leases the pass took before it;
//! `fresh` takes a context per call.

use std::time::Instant;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, DeviceBuffer};
use ptyrs_gpu::{Kernels, LlkRing, StackTable, ViewDims};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: count_time [nframes] [stack] [frame] [reps] [shared|fresh]";
    let mut args = std::env::args().skip(1);
    let nframes = args.next().unwrap_or_else(|| "400,10000,100000".into());
    let nframes = nframes
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<u32>, _>>()?;
    let mut next = |default: u32| args.next().map_or(Ok(default), |s| s.parse());
    let (stack, frame, reps) = (next(16)?, next(256)?, next(10)?);
    let fresh = match args.next().as_deref() {
        None | Some("shared") => false,
        Some("fresh") => true,
        Some(_) => return Err(usage.into()),
    };
    if args.next().is_some() || stack == 0 || reps == 0 {
        return Err(usage.into());
    }

    let cuda = CudaContext::new(0)?;
    let kernels = Kernels::load(&cuda)?;
    let stream = cuda.new_stream()?;
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
        let stacks: Vec<_> = (0..nframe)
            .step_by(stack as usize)
            .map(|f| f..(f + stack).min(nframe))
            .collect();
        let table = StackTable::new(&stream, &dims, &stacks)?;
        let counts = DeviceBuffer::<u16>::zeroed(&stream, nframe as usize * frame.pow(2) as usize)?;
        let (e0, e1) = (cuda.new_event()?, cuda.new_event()?);
        // Host then device µs per call, of the first pass then the repeat.
        let mut us = [[(); 2].map(|_| Vec::new()), [(); 2].map(|_| Vec::new())];
        for _ in 0..reps {
            let mut ring = LlkRing::<()>::new(&kernels, &stream, &dims, 1)?;
            stream.synchronize()?;
            for [host, dev] in &mut us {
                let t = Instant::now();
                e0.record(&stream)?;
                let mut ctx = ExecutionContext::new(stream.clone());
                for s in 0..table.len() {
                    if fresh {
                        ctx = ExecutionContext::new(stream.clone());
                    }
                    ring.device().count(&ctx, &dims, table.at(s), &counts)?;
                }
                e1.record(&stream)?;
                stream.synchronize()?;
                let calls = table.len() as f64;
                host.push(1e6 * t.elapsed().as_secs_f64() / calls);
                dev.push(1e3 * f64::from(e0.elapsed_time(&e1)?) / calls);
            }
        }
        let [first, repeat] = us.map(|pass| pass.map(median));
        println!(
            "count {} nframe {nframe} stack {stack} frame {frame}: first host {:.2} device {:.2} us/call, repeat host {:.2} device {:.2} us/call, {} calls per pass, median of {reps}",
            if fresh { "fresh" } else { "shared" },
            first[0],
            first[1],
            repeat[0],
            repeat[1],
            table.len()
        );
    }
    Ok(())
}
