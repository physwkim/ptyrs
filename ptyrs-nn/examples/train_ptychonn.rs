//! Trains the PtychoNN predictor on scans from `ptyrs_engine::sim::simulate` and compares its
//! patches with the flat start on held-out scans.
//!
//! `cargo run --release -p ptyrs-nn --example train_ptychonn -- <out_dir> [frame] [bin] [nscan] [epochs] [width]`
//! (defaults 256, 4, 200, 30, 32). Writes the predictor to `out_dir`.

use std::path::PathBuf;
use std::time::Instant;

use burn::prelude::*;
use ptyrs_core::{AddrTable, Geometry, Object, Probe, Scan, required_obj_shape, sim};
use ptyrs_nn::data::random_object;
use ptyrs_nn::predictor::{
    PredictorConfig, Samples, TrainOptions, mean_frame_total, object_patches, train,
};

/// The engine's parity geometry (`ptyrs-engine/tests/common`).
fn geometry(frame: usize) -> Geometry {
    Geometry::new(1.5e-10, 1.0, 75e-6, [frame, frame]).unwrap()
}

/// A 100-point Fermat scan, 4 object pixels apart, of `object` (random when `None`).
fn scan(
    frame: usize,
    probe: &Probe,
    seed: u64,
    object: Option<Object>,
) -> (Scan, Object, AddrTable) {
    let g = geometry(frame);
    let pixel = g.pixel_size_object();
    let positions = sim::fermat(100, 4.0 * pixel[1]);
    let pre = positions.pynx_preprocess(pixel);
    let shape = required_obj_shape(&pre, pixel, g.frame, 8);
    let object = object.unwrap_or_else(|| random_object(shape, seed));
    let addr = AddrTable::new(&pre, pixel, g.frame, object.plane_shape(), 1, 1).unwrap();
    let scan = ptyrs_engine::sim::simulate(g, positions, &object, probe, 1e6, None, seed).unwrap();
    (scan, object, addr)
}

/// Samples of the scans `seeds`, simulated on all host threads.
fn samples(config: &PredictorConfig, probe: &Probe, seeds: Vec<u64>) -> Samples {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let parts: Vec<Samples> = std::thread::scope(|s| {
        let handles: Vec<_> = seeds
            .chunks(seeds.len().div_ceil(threads))
            .map(|chunk| {
                s.spawn(move || {
                    let mut part = Samples::new(config);
                    for &seed in chunk {
                        let (scan, object, addr) = scan(config.frame[0], probe, seed, None);
                        part.push_scan(config, &scan.counts, &object, &addr)
                            .unwrap();
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

fn l1(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| f64::from((a - b).abs()))
        .sum::<f64>()
        / a.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: train_ptychonn <out_dir> [frame] [bin] [nscan] [epochs] [width]";
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().ok_or(usage)?);
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (frame, bin, nscan, epochs, width) =
        (next(256)?, next(4)?, next(200)?, next(30)?, next(32)?);

    let device = Device::cuda(0);
    let probe = sim::probe([frame, frame], 1)?;
    let config = PredictorConfig::new([frame, frame], bin).with_width(width);
    let t = Instant::now();
    let train_set = samples(&config, &probe, (0..nscan as u64).collect());
    let valid_set = samples(&config, &probe, (1_000_000..1_000_010).collect());
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

    // Held-out scans: random objects and the engine's test object.
    let shape = {
        let (_, o, _) = scan(frame, &probe, 0, None);
        o.plane_shape()
    };
    let held: Vec<(&str, Option<Object>, u64)> = vec![
        ("random", None, 2_000_000),
        ("random", None, 2_000_001),
        ("sim::object", Some(sim::object(shape)), 2_000_002),
    ];
    for (name, object, seed) in held {
        let (scan, truth, addr) = scan(frame, &probe, seed, object);
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
