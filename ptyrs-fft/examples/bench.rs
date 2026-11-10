//! Throughput baseline: forward+inverse of 256² × 2 modes per frame, timed with CUDA events.
use cuda_core::{Context, DeviceBuffer};
use ptyrs_fft::{C32, Plan2d};

fn main() {
    let (n, modes, iters) = (256usize, 2usize, 200);
    let ctx = Context::new(0).unwrap();
    println!("{}", ctx.device_name().unwrap());
    let stream = ctx.new_stream().unwrap();
    for frames in [32usize, 128, 512] {
        let batch = frames * modes;
        let plan = Plan2d::new(&stream, n, batch).unwrap();
        let mut a = DeviceBuffer::<C32>::zeroed(&stream, n * n * batch).unwrap();
        let mut b = DeviceBuffer::<C32>::zeroed(&stream, n * n * batch).unwrap();
        for _ in 0..10 {
            plan.forward(&a, &mut b).unwrap();
            plan.inverse(&b, &mut a).unwrap();
        }
        stream.synchronize().unwrap();
        let start = ctx.new_event().unwrap();
        start.record(&stream).unwrap();
        for _ in 0..iters {
            plan.forward(&a, &mut b).unwrap();
            plan.inverse(&b, &mut a).unwrap();
        }
        let end = ctx.new_event().unwrap();
        end.record(&stream).unwrap();
        let ms = start.elapsed_ms(&end).unwrap() as f64;
        let pairs = (frames * iters) as f64 / (ms / 1e3);
        println!(
            "frames {frames:4} (batch {batch:4} = {modes} modes): {:.3} ms/iter, {pairs:.0} frame fwd+inv pairs/s, {:.0} 2D-FFTs/s",
            ms / iters as f64,
            pairs * 2.0 * modes as f64
        );
    }
}
