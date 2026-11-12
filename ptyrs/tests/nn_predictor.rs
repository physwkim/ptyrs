//! The predictor on scans from `ptyrs::engine::sim::simulate` (GPU).

use std::path::{Path, PathBuf};

use burn::prelude::*;
use ptyrs::base::{AddrTable, Geometry, Object, Probe, Scan, required_obj_shape, sim};
use ptyrs::nn::NnError;
use ptyrs::nn::data::random_object;
use ptyrs::nn::predictor::{
    Predictor, PredictorConfig, Samples, TrainOptions, mean_frame_total, object_patches, train,
};
use ptyrs::nn::ptychonn::PtychoNN;

const FRAME: usize = 64;

fn geometry() -> Geometry {
    Geometry::new(1.5e-10, 1.0, 75e-6, [FRAME, FRAME]).unwrap()
}

/// A Fermat scan of a random object, its truth and address table.
fn scan(probe: &Probe, seed: u64) -> (Scan, Object, AddrTable) {
    let g = geometry();
    let pixel = g.pixel_size_object();
    let positions = sim::fermat(40, 4.0 * pixel[1]);
    let pre = positions.pynx_preprocess(pixel);
    let shape = required_obj_shape(&pre, pixel, g.frame, 8);
    let object = random_object(shape, seed);
    let addr = AddrTable::new(&pre, pixel, g.frame, shape, 1, 1).unwrap();
    let scan = ptyrs::engine::sim::simulate(g, positions, &object, probe, 1e5, None, seed).unwrap();
    (scan, object, addr)
}

fn config() -> PredictorConfig {
    PredictorConfig::new([FRAME, FRAME], 2).with_width(16)
}

fn samples(config: &PredictorConfig, probe: &Probe, seeds: std::ops::Range<u64>) -> Samples {
    let mut s = Samples::new(config);
    for seed in seeds {
        let (scan, object, addr) = scan(probe, seed);
        s.push_scan(config, &scan.counts, &object, &addr).unwrap();
    }
    s
}

fn l1(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| f64::from((a - b).abs()))
        .sum::<f64>()
        / a.len() as f64
}

#[test]
fn ptychonn_maps_images_to_amplitude_and_phase_of_the_same_size() {
    let device = Device::cuda(0);
    let net = PtychoNN::new(8, 3, &device);
    assert_eq!(net.granule(), 8);
    let x = Tensor::<4>::random(
        [2, 1, 32, 24],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (amp, phase) = net.forward(x);
    assert_eq!(amp.dims(), [2, 1, 32, 24]);
    assert_eq!(phase.dims(), [2, 1, 32, 24]);
    let amp = amp.into_data().try_into_vec::<f32>().unwrap();
    let phase = phase.into_data().try_into_vec::<f32>().unwrap();
    assert!(amp.iter().all(|&a| a > 0.0 && a < 2.0));
    assert!(phase.iter().all(|&p| p.abs() < std::f32::consts::PI));
}

#[test]
fn config_rejects_frames_the_network_cannot_tile() {
    let device = Device::cuda(0);
    assert!(PredictorConfig::new([64, 64], 2).init(&device).is_ok());
    assert!(PredictorConfig::new([64, 60], 2).init(&device).is_err());
    assert!(PredictorConfig::new([64, 64], 0).init(&device).is_err());
}

#[test]
fn trained_predictor_keeps_its_fit_at_inference_and_after_reload() {
    const SCANS: u64 = 5;
    const EPOCHS: usize = 30;
    let device = Device::cuda(0);
    let probe = sim::probe([FRAME, FRAME], 1).unwrap();
    let config = config();
    let train_set = samples(&config, &probe, 0..SCANS);
    let options = TrainOptions {
        epochs: EPOCHS,
        batch: 32,
        lr: 1e-3,
        seed: 7,
        checkpoint: None,
    };
    let mut losses = Vec::new();
    let predictor = train(&config, &options, &train_set, None, &device, |e| {
        losses.push(e.train)
    })
    .unwrap();
    assert_eq!(losses.len(), EPOCHS);

    // A training scan through `predict` (its own norm, the stored input statistics, batch-norm
    // running statistics) lands well below the flat start 1 + 0i. Beating it on unseen scans
    // takes ~10^4 training frames; `examples/nn_train_ptychonn` measures that.
    let (scan, truth, addr) = scan(&probe, 0);
    let norm = mean_frame_total(&scan.counts);
    let patches = predictor.predict(&scan.counts, norm).unwrap();
    let want = object_patches(&truth, &addr, config.patch());
    assert_eq!(patches.dims, want.dims);
    let flat =
        l1(&vec![1.0; want.amp.len()], &want.amp) + l1(&vec![0.0; want.phase.len()], &want.phase);
    let pred = l1(&patches.amp, &want.amp) + l1(&patches.phase, &want.phase);
    assert!(
        pred < 0.5 * flat,
        "predicted {pred}, flat {flat}, epoch losses {losses:?}"
    );

    let object = predictor.initial_object(&scan.counts, norm, &addr).unwrap();
    assert_eq!(object.dims(), truth.dims());

    // A saved and reloaded predictor predicts the same patches.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ptyrs-nn-predictor");
    predictor.save(&dir).unwrap();
    let back = Predictor::load(&dir, &device).unwrap();
    assert_eq!(back.config().frame, config.frame);
    assert_eq!(back.predict(&scan.counts, norm).unwrap(), patches);
}

#[test]
fn a_resumed_run_continues_from_its_last_epoch() {
    let device = Device::cuda(0);
    let probe = sim::probe([FRAME, FRAME], 1).unwrap();
    let config = config();
    let train_set = samples(&config, &probe, 0..2);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ptyrs-nn-checkpoint");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let options = |epochs, seed, checkpoint: Option<&Path>| TrainOptions {
        epochs,
        batch: 16,
        lr: 1e-3,
        seed,
        checkpoint: checkpoint.map(Path::to_path_buf),
    };
    let run = |options: &TrainOptions, samples: &Samples| -> Result<_, NnError> {
        let mut losses = Vec::new();
        let p = train(&config, options, samples, None, &device, |e| {
            losses.push((e.epoch, e.train))
        })?;
        Ok((losses, p))
    };

    let (whole, straight) = run(&options(3, 7, None), &train_set).unwrap();
    let (first, _) = run(&options(2, 7, Some(&dir)), &train_set).unwrap();
    let (rest, resumed) = run(&options(3, 7, Some(&dir)), &train_set).unwrap();
    assert_eq!([first, rest].concat(), whole);
    let (scan, ..) = scan(&probe, 0);
    let norm = mean_frame_total(&scan.counts);
    assert_eq!(
        resumed.predict(&scan.counts, norm).unwrap(),
        straight.predict(&scan.counts, norm).unwrap()
    );
    let mut kept: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    kept.sort();
    assert_eq!(kept, ["epoch-3", "latest"]);

    // A checkpoint of other settings or other samples is refused.
    assert!(run(&options(4, 8, Some(&dir)), &train_set).is_err());
    let others = samples(&config, &probe, 2..4);
    assert_eq!(others.len(), train_set.len());
    assert!(run(&options(4, 7, Some(&dir)), &others).is_err());
}
