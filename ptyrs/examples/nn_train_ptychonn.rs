//! Trains the PtychoNN predictor on scans from `ptyrs::engine::sim::simulate` and compares its
//! patches with the flat start on held-out scans.
//!
//! `cargo run --release -p ptyrs --features nn --example nn_train_ptychonn -- <out_dir> [frame] [bin] [nscan] [epochs] [width] [per_scan] [structured]`
//! (defaults 256, 4, 10000, 30, 32, 10, 0). The training set holds `nscan` scans of
//! `random_object` and `structured` more of `structured_object`. Each training scan contributes
//! `per_scan` of its 100 frames, drawn at random: neighbouring frames of a scan see nearly the
//! same object pixels, so fewer frames from more scans give more distinct objects for the same
//! epoch time. Writes the predictor to `out_dir`, and after every epoch the training state to
//! `out_dir/checkpoint`, from which the same command resumes a stopped run.

#[path = "common/nn/mod.rs"]
mod common;

use std::path::PathBuf;
use std::time::Instant;

use burn::prelude::*;
use common::scan;
use ptyrs::base::{Object, Probe, sim};
use ptyrs::nn::data::{random_object, structured_object};
use ptyrs::nn::predictor::{
    Patches, PredictorConfig, Samples, TrainOptions, mean_frame_total, object_patches, train,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// First seed of the structured training scans, past the random ones, the validation scans and
/// the held-out ones.
const STRUCTURED: u64 = 3_000_000;

/// `per_scan` frames of each of the scans `seeds` of the objects `draw` gives for them,
/// simulated on all host threads.
fn samples(
    config: &PredictorConfig,
    probe: &Probe,
    seeds: Vec<u64>,
    per_scan: usize,
    draw: fn([usize; 2], u64) -> Object,
) -> Samples {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let shape = common::object_shape(config.frame[0]);
    let parts: Vec<Samples> = std::thread::scope(|s| {
        let handles: Vec<_> = seeds
            .chunks(seeds.len().div_ceil(threads).max(1))
            .map(|chunk| {
                s.spawn(move || {
                    let mut part = Samples::new(config);
                    for &seed in chunk {
                        let (scan, object, addr) =
                            scan(config.frame[0], probe, seed, Some(draw(shape, seed)));
                        let mut all = Samples::new(config);
                        all.push_scan(config, &scan.counts, &object, &addr).unwrap();
                        part.extend(pick(all, per_scan, seed));
                    }
                    part
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut all = Samples::new(config);
    for p in parts {
        all.extend(p);
    }
    all
}

/// `n` of `s`'s pairs drawn by `seed`, in order; all of them when `n` covers them.
fn pick(s: Samples, n: usize, seed: u64) -> Samples {
    if n >= s.len() {
        return s;
    }
    // The scan's object and noise draw from stream 0 of this seed.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(1);
    let mut idx = rand::seq::index::sample(&mut rng, s.len(), n).into_vec();
    idx.sort_unstable();
    let [_, py, px] = s.target.dims;
    let plane = py * px;
    let gather = |v: &[f32]| -> Vec<f32> {
        idx.iter()
            .flat_map(|&i| &v[i * plane..(i + 1) * plane])
            .copied()
            .collect()
    };
    Samples {
        input: gather(&s.input),
        target: Patches {
            dims: [n, py, px],
            amp: gather(&s.target.amp),
            phase: gather(&s.target.phase),
        },
    }
}

fn l1(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| f64::from((a - b).abs()))
        .sum::<f64>()
        / a.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: train_ptychonn <out_dir> [frame] [bin] [nscan] [epochs] [width] [per_scan] [structured]";
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().ok_or(usage)?);
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (frame, bin, nscan, epochs, width, per_scan, structured) = (
        next(256)?,
        next(4)?,
        next(10000)?,
        next(30)?,
        next(32)?,
        next(10)?,
        next(0)?,
    );

    let device = Device::cuda(0);
    let probe = sim::probe([frame, frame], 1)?;
    let config = PredictorConfig::new([frame, frame], bin).with_width(width);
    let t = Instant::now();
    let mut train_set = samples(
        &config,
        &probe,
        (0..nscan as u64).collect(),
        per_scan,
        random_object,
    );
    train_set.extend(samples(
        &config,
        &probe,
        (STRUCTURED..STRUCTURED + structured as u64).collect(),
        per_scan,
        structured_object,
    ));
    let valid_set = samples(
        &config,
        &probe,
        (1_000_000..1_000_010).collect(),
        100,
        random_object,
    );
    eprintln!(
        "simulated {} + {} frames in {:.1} s",
        train_set.len(),
        valid_set.len(),
        t.elapsed().as_secs_f64()
    );

    let options = TrainOptions {
        epochs,
        batch: 32,
        lr: 1e-3,
        seed: 1,
        checkpoint: Some(out.join("checkpoint")),
    };
    let t = Instant::now();
    let predictor = train(
        &config,
        &options,
        &train_set,
        Some(&valid_set),
        &device,
        |e| {
            eprintln!(
                "epoch {:3}  train {:.4}  valid {:.4}  {:.0} s",
                e.epoch,
                e.train,
                e.valid.unwrap_or(f64::NAN),
                t.elapsed().as_secs_f64()
            )
        },
    )?;
    predictor.save(&out)?;

    for (name, _, scan, truth, addr, _) in common::held_out(frame, &probe) {
        let patches = predictor.predict(&scan.counts, mean_frame_total(&scan.counts))?;
        let want = object_patches(&truth, &addr, config.patch());
        let ones = vec![1.0; want.amp.len()];
        let zeros = vec![0.0; want.amp.len()];
        println!(
            "{name:12} amp L1 pred {:.4} flat {:.4}   phase L1 pred {:.4} flat {:.4}",
            l1(&patches.amp, &want.amp),
            l1(&ones, &want.amp),
            l1(&patches.phase, &want.phase),
            l1(&zeros, &want.phase)
        );
    }
    Ok(())
}
