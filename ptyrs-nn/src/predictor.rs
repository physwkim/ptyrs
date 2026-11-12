//! PtychoNN-style initial-value predictor: detector frames in, object amplitude and phase patches
//! out, stitched into an object for the engine to start from.
//!
//! A frame of `[ny, nx]` detector pixels binned by `bin` becomes a `[ny/bin, nx/bin]` amplitude
//! image; binning halves the real-space field of view the detector samples, so the network
//! predicts the central `[ny/bin, nx/bin]` object pixels of the frame's window. A trained model
//! holds for the probe and geometry of its training scans.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use ptyrs_core::{AddrTable, C64, Counts, Object, Stack};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::ptychonn::PtychoNN;
use crate::{NnError, burn_err};

/// Shape of a predictor: the detector frame it reads, the binning, and the network size.
#[derive(Config, Debug, PartialEq)]
pub struct PredictorConfig {
    pub frame: [usize; 2],
    pub bin: usize,
    #[config(default = 32)]
    pub width: usize,
    #[config(default = 3)]
    pub depth: usize,
}

impl PredictorConfig {
    /// Side of the network image and of the predicted object patch.
    pub fn patch(&self) -> [usize; 2] {
        self.frame.map(|n| n / self.bin)
    }

    fn check(&self) -> Result<(), NnError> {
        let granule = self.bin << self.depth;
        if self.bin == 0 || self.frame.iter().any(|n| n % granule != 0) {
            return Err(NnError::Invalid(format!(
                "frame {:?} is not a multiple of bin {} × 2^depth {}",
                self.frame, self.bin, self.depth
            )));
        }
        Ok(())
    }

    pub fn init(&self, device: &Device) -> Result<PtychoNN, NnError> {
        self.check()?;
        Ok(PtychoNN::new(self.width, self.depth, device))
    }
}

/// Object patches `[n, py, px]` as amplitude and phase.
#[derive(Clone, Debug, PartialEq)]
pub struct Patches {
    pub dims: [usize; 3],
    pub amp: Vec<f32>,
    pub phase: Vec<f32>,
}

impl Patches {
    fn empty(patch: [usize; 2]) -> Self {
        Self {
            dims: [0, patch[0], patch[1]],
            amp: Vec::new(),
            phase: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.dims[0]
    }

    pub fn is_empty(&self) -> bool {
        self.dims[0] == 0
    }

    fn append(&mut self, other: Patches) {
        assert_eq!(self.dims[1..], other.dims[1..]);
        self.dims[0] += other.dims[0];
        self.amp.extend(other.amp);
        self.phase.extend(other.phase);
    }
}

/// Mean over frames of the unmasked counts per frame: the scale [`diffraction_input`] divides
/// out, so the predictor reads amplitude relative to the scan rather than the beam flux.
pub fn mean_frame_total(frames: &Counts) -> f64 {
    let total: f64 = frames
        .as_slice()
        .iter()
        .filter_map(|&v| Counts::intensity(v))
        .sum();
    total / frames.nframe() as f64
}

/// Network input `[nframe, ny/bin, nx/bin]` from detector-order frames: each `bin × bin` block
/// summed (a masked pixel counts as the block's unmasked mean, a fully masked block as 0),
/// divided by `norm` ([`mean_frame_total`] of the scan) and by the binned pixel share, and
/// square-rooted, so a frame of mean intensity has mean square amplitude 1.
pub fn diffraction_input(frames: &Counts, bin: usize, norm: f64) -> Result<Vec<f32>, NnError> {
    let [ny, nx] = frames.frame_shape();
    if bin == 0 || ny % bin != 0 || nx % bin != 0 {
        return Err(NnError::Invalid(format!(
            "frame [{ny}, {nx}] is not a multiple of bin {bin}"
        )));
    }
    if norm.is_nan() || norm <= 0.0 {
        return Err(NnError::Invalid(format!("norm {norm} must be positive")));
    }
    let (by, bx) = (ny / bin, nx / bin);
    let scale = (by * bx) as f64 / norm;
    let mut out = Vec::with_capacity(frames.nframe() * by * bx);
    for f in 0..frames.nframe() {
        let frame = frames.frame(f);
        for y in 0..by {
            for x in 0..bx {
                let (mut sum, mut n) = (0.0, 0usize);
                for dy in 0..bin {
                    let row = &frame[(y * bin + dy) * nx + x * bin..][..bin];
                    for v in row.iter().filter_map(|&v| Counts::intensity(v)) {
                        sum += v;
                        n += 1;
                    }
                }
                let block = if n == 0 {
                    0.0
                } else {
                    sum * (bin * bin) as f64 / n as f64
                };
                out.push((block * scale).sqrt() as f32);
            }
        }
    }
    Ok(out)
}

/// Top-left corner in the object of frame `f`'s patch: the centre `patch` of its view window.
fn patch_corner(addr: &AddrTable, f: usize, patch: [usize; 2]) -> [usize; 2] {
    let v = addr.frame_views(f)[0];
    let frame = addr.frame_shape();
    [
        v.oy as usize + (frame[0] - patch[0]) / 2,
        v.ox as usize + (frame[1] - patch[1]) / 2,
    ]
}

/// The true patches of object mode 0 at every frame of `addr`: the predictor's training target.
pub fn object_patches(object: &Object, addr: &AddrTable, patch: [usize; 2]) -> Patches {
    let nxo = object.plane_shape()[1];
    let plane = object.plane(0);
    let mut out = Patches::empty(patch);
    for f in 0..addr.nframe() {
        let [cy, cx] = patch_corner(addr, f, patch);
        for y in 0..patch[0] {
            for c in &plane[(cy + y) * nxo + cx..][..patch[1]] {
                out.amp.push(c.norm() as f32);
                out.phase.push(c.arg() as f32);
            }
        }
    }
    out.dims[0] = addr.nframe();
    out
}

const FLAT: C64 = C64::new(1.0, 0.0);

/// The flat initial object `1 + 0i`, one mode of `addr`'s shape: the start a predicted object
/// is measured against ([`crate::metric`]).
pub fn flat_object(addr: &AddrTable) -> Object {
    let [ny, nx] = addr.obj_shape();
    Object::new(Stack::filled([1, ny, nx], FLAT))
}

/// Stitches per-frame patches into a single-mode object of `addr`'s shape. Overlaps average
/// `A·e^{iφ}` under a `sin²` taper that favours patch centres; pixels no patch covers stay
/// flat, as [`flat_object`].
pub fn stitch(patches: &Patches, addr: &AddrTable) -> Result<Object, NnError> {
    let [n, py, px] = patches.dims;
    if n != addr.nframe() {
        return Err(NnError::Invalid(format!(
            "{n} patches for {} frames",
            addr.nframe()
        )));
    }
    let [nyo, nxo] = addr.obj_shape();
    let taper = |i: usize, len: usize| (PI * (i as f64 + 0.5) / len as f64).sin().powi(2);
    let wy: Vec<f64> = (0..py).map(|i| taper(i, py)).collect();
    let wx: Vec<f64> = (0..px).map(|i| taper(i, px)).collect();
    let mut sum = vec![C64::default(); nyo * nxo];
    let mut weight = vec![0.0; nyo * nxo];
    for f in 0..n {
        let [cy, cx] = patch_corner(addr, f, [py, px]);
        for (y, wy) in wy.iter().enumerate() {
            for (x, wx) in wx.iter().enumerate() {
                let k = (f * py + y) * px + x;
                let w = wy * wx;
                let o = (cy + y) * nxo + cx + x;
                sum[o] +=
                    C64::from_polar(f64::from(patches.amp[k]), f64::from(patches.phase[k])) * w;
                weight[o] += w;
            }
        }
    }
    let data = sum
        .into_iter()
        .zip(weight)
        .map(|(s, w)| if w > 0.0 { s / w } else { FLAT })
        .collect();
    Ok(Object::new(Stack::from_vec([1, nyo, nxo], data)?))
}

/// Training pairs: network inputs `[n, py, px]` and the true patches.
#[derive(Clone, Debug)]
pub struct Samples {
    pub input: Vec<f32>,
    pub target: Patches,
}

impl Samples {
    pub fn new(config: &PredictorConfig) -> Self {
        Self {
            input: Vec::new(),
            target: Patches::empty(config.patch()),
        }
    }

    /// The pairs of one simulated scan: `frames` normalised by their own [`mean_frame_total`],
    /// targets from `object` through `addr`, the scan's address table.
    pub fn push_scan(
        &mut self,
        config: &PredictorConfig,
        frames: &Counts,
        object: &Object,
        addr: &AddrTable,
    ) -> Result<(), NnError> {
        if frames.frame_shape() != config.frame || addr.frame_shape() != config.frame {
            return Err(NnError::Invalid(format!(
                "scan frame {:?} differs from predictor frame {:?}",
                frames.frame_shape(),
                config.frame
            )));
        }
        if frames.nframe() != addr.nframe() {
            return Err(NnError::Invalid(format!(
                "{} frames for {} addressed frames",
                frames.nframe(),
                addr.nframe()
            )));
        }
        let input = diffraction_input(frames, config.bin, mean_frame_total(frames))?;
        self.input.extend(input);
        self.target
            .append(object_patches(object, addr, config.patch()));
        Ok(())
    }

    /// Appends `other`'s pairs.
    pub fn extend(&mut self, other: Samples) {
        self.input.extend(other.input);
        self.target.append(other.target);
    }

    pub fn len(&self) -> usize {
        self.target.len()
    }

    pub fn is_empty(&self) -> bool {
        self.target.is_empty()
    }
}

/// Optimiser settings for [`train`].
#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub seed: u64,
    /// Directory [`train`] writes its state to after every epoch and resumes from when it holds
    /// one, so that a stopped run continues from its last finished epoch.
    pub checkpoint: Option<PathBuf>,
}

/// The run a checkpoint belongs to and the epochs it has finished.
#[derive(Config, Debug, PartialEq)]
struct Checkpoint {
    config: PredictorConfig,
    batch: usize,
    lr: f64,
    seed: u64,
    epochs: usize,
}

impl Checkpoint {
    /// Writes the state into `dir/epoch-<epochs>`, then points `dir/latest` at it and removes
    /// older states. `latest` is renamed into place only once the state is complete, so a run
    /// killed at any point leaves the previous checkpoint readable.
    fn save(&self, dir: &Path, model: &Model, optim: &ModuleOptimizer) -> Result<(), NnError> {
        let name = format!("epoch-{}", self.epochs);
        let state = dir.join(&name);
        std::fs::create_dir_all(&state)?;
        Config::save(self, state.join("checkpoint.json"))?;
        // `valid` folds in the last batch-norm running-statistics update, which burn holds back
        // until the next read; saving `model` itself would store the value before it.
        model
            .valid()
            .save_file(state.join("model.bpk"))
            .map_err(burn_err)?;
        optim.save(state.join("optim.bpk")).map_err(burn_err)?;
        let tmp = dir.join("latest.tmp");
        std::fs::write(&tmp, &name)?;
        std::fs::rename(&tmp, dir.join("latest"))?;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let old = entry.file_name();
            if old.to_string_lossy().starts_with("epoch-") && old != *name {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
        Ok(())
    }

    /// The epochs finished, model and optimiser of the state `dir/latest` points at; 0 and the
    /// given ones when there is none. The state must come from a run of the same settings
    /// (`self`, whatever its `epochs`) on the same samples, which `model`'s input statistics
    /// stand for.
    fn resume(
        &self,
        dir: &Path,
        model: Model,
        optim: ModuleOptimizer,
    ) -> Result<(usize, Model, ModuleOptimizer), NnError> {
        let name = match std::fs::read_to_string(dir.join("latest")) {
            Ok(name) => name,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((0, model, optim)),
            Err(e) => return Err(e.into()),
        };
        let state = dir.join(name);
        let saved = Checkpoint::load(state.join("checkpoint.json")).map_err(burn_err)?;
        let own = Checkpoint {
            config: self.config.clone(),
            epochs: saved.epochs,
            ..*self
        };
        if saved != own {
            return Err(NnError::Invalid(format!(
                "checkpoint {} is of {saved:?}, not {own:?}",
                state.display()
            )));
        }
        let stats = |m: &Model| -> Result<_, NnError> {
            Ok((host(m.mean.val())?, host(m.inv_spread.val())?))
        };
        let want = stats(&model)?;
        let model = model
            .try_load_file(state.join("model.bpk"))
            .map_err(burn_err)?;
        if stats(&model)? != want {
            return Err(NnError::Invalid(format!(
                "checkpoint {} was trained on other samples",
                state.display()
            )));
        }
        let optim = optim.load(state.join("optim.bpk")).map_err(burn_err)?;
        Ok((saved.epochs, model, optim))
    }
}

/// Mean L1 loss (amplitude plus phase) of one epoch.
#[derive(Clone, Copy, Debug)]
pub struct EpochLoss {
    pub epoch: usize,
    pub train: f64,
    pub valid: Option<f64>,
}

fn tensor(values: &[f32], dims: [usize; 3], device: &Device) -> Tensor<4> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [dims[0], 1, dims[1], dims[2]]),
        device,
    )
}

/// A tensor's values on the host.
pub(crate) fn host<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, NnError> {
    t.try_into_data()
        .map_err(burn_err)?
        .try_into_vec::<f32>()
        .map_err(burn_err)
}

/// [`PtychoNN`] behind a fixed standardisation of its input: the per-pixel mean of the training
/// inputs is subtracted and the result multiplied by the inverse RMS spread about that mean.
/// Every frame carries the probe's far field and the object only modulates it; removing the
/// shared pattern lets training leave the mean patch several times sooner. The statistics are
/// parameters only so that they are saved with the weights; they take no gradient.
#[derive(Module, Debug)]
struct Model {
    mean: Param<Tensor<4>>,
    inv_spread: Param<Tensor<4>>,
    net: PtychoNN,
}

impl Model {
    /// `mean` is the `[py, px]` patch-shaped mean input.
    fn new(
        config: &PredictorConfig,
        mean: Vec<f32>,
        inv_spread: f32,
        device: &Device,
    ) -> Result<Self, NnError> {
        let [py, px] = config.patch();
        let fixed = |values: Vec<f32>, shape: [usize; 4]| {
            Param::from_tensor(Tensor::from_data(TensorData::new(values, shape), device))
                .set_require_grad(false)
        };
        Ok(Self {
            net: config.init(device)?,
            mean: fixed(mean, [1, 1, py, px]),
            inv_spread: fixed(vec![inv_spread], [1, 1, 1, 1]),
        })
    }

    fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        self.net
            .forward((x - self.mean.val()) * self.inv_spread.val())
    }
}

/// Per-pixel mean of `samples`' inputs and the inverse RMS spread about it.
fn input_stats(samples: &Samples) -> Result<(Vec<f32>, f32), NnError> {
    let [n, py, px] = samples.target.dims;
    let plane = py * px;
    let mut mean = vec![0.0f64; plane];
    for frame in samples.input.chunks(plane) {
        for (m, &x) in mean.iter_mut().zip(frame) {
            *m += f64::from(x) / n as f64;
        }
    }
    let var = samples
        .input
        .chunks(plane)
        .flat_map(|frame| {
            frame
                .iter()
                .zip(&mean)
                .map(|(&x, m)| (f64::from(x) - m).powi(2))
        })
        .sum::<f64>()
        / (n * plane) as f64;
    if var.is_nan() || var == 0.0 {
        return Err(NnError::Invalid(format!(
            "training inputs have spread {var}"
        )));
    }
    Ok((
        mean.into_iter().map(|m| m as f32).collect(),
        (1.0 / var.sqrt()) as f32,
    ))
}

fn l1(model: &Model, x: Tensor<4>, amp: Tensor<4>, phase: Tensor<4>) -> Tensor<1> {
    let (a, p) = model.forward(x);
    (a - amp).abs().mean() + (p - phase).abs().mean()
}

/// Trains a fresh network of `config` with Adam on L1 loss, reporting each epoch to `log`, or
/// resumes the run in `options.checkpoint`. `options.seed` seeds the device's generator for the
/// initial weights and the shuffle of each epoch. `device` is a plain device; autodiff is enabled
/// on it here.
pub fn train(
    config: &PredictorConfig,
    options: &TrainOptions,
    train: &Samples,
    valid: Option<&Samples>,
    device: &Device,
    mut log: impl FnMut(&EpochLoss),
) -> Result<Predictor, NnError> {
    if train.is_empty() || options.batch == 0 {
        return Err(NnError::Invalid("no training samples or zero batch".into()));
    }
    let ad = device.clone().autodiff();
    ad.seed(options.seed);
    let (mean, inv_spread) = input_stats(train)?;
    let mut model = Model::new(config, mean, inv_spread, &ad)?;
    let mut optim = AdamConfig::new().init();
    let mut checkpoint = Checkpoint {
        config: config.clone(),
        batch: options.batch,
        lr: options.lr,
        seed: options.seed,
        epochs: 0,
    };
    let mut start = 0;
    if let Some(dir) = &options.checkpoint {
        (start, model, optim) = checkpoint.resume(dir, model, optim)?;
    }
    let dims = train.target.dims;
    let x = tensor(&train.input, dims, &ad);
    let amp = tensor(&train.target.amp, dims, &ad);
    let phase = tensor(&train.target.phase, dims, &ad);
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut order: Vec<i64> = (0..dims[0] as i64).collect();
    // Each epoch shuffles the previous order: replay the finished ones.
    for _ in 0..start {
        order.shuffle(&mut rng);
    }
    for epoch in start..options.epochs {
        order.shuffle(&mut rng);
        let mut sum = Tensor::<1>::zeros([1], &device.clone().inner());
        for batch in order.chunks(options.batch) {
            let idx =
                Tensor::<1, Int>::from_data(TensorData::new(batch.to_vec(), [batch.len()]), &ad);
            let loss = l1(
                &model,
                x.clone().select(0, idx.clone()),
                amp.clone().select(0, idx.clone()),
                phase.clone().select(0, idx),
            );
            sum = sum + loss.clone().detach().inner() * batch.len() as f64;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(options.lr, model, grads);
        }
        let train_loss =
            f64::from(sum.try_into_scalar::<f32>().map_err(burn_err)?) / dims[0] as f64;
        let valid_loss = valid
            .map(|v| {
                Predictor {
                    config: config.clone(),
                    model: model.valid(),
                }
                .loss(v)
            })
            .transpose()?;
        if let Some(dir) = &options.checkpoint {
            checkpoint.epochs = epoch + 1;
            checkpoint.save(dir, &model, &optim)?;
        }
        log(&EpochLoss {
            epoch,
            train: train_loss,
            valid: valid_loss,
        });
    }
    Ok(Predictor {
        config: config.clone(),
        model: model.valid(),
    })
}

/// Frames in, object patches out.
#[derive(Debug)]
pub struct Predictor {
    config: PredictorConfig,
    model: Model,
}

/// Frames per network call.
const CHUNK: usize = 512;

impl Predictor {
    pub fn config(&self) -> &PredictorConfig {
        &self.config
    }

    fn device(&self) -> Device {
        self.model.devices()[0].clone()
    }

    /// Patches `[nframe, py, px]` for detector-order `frames` of a scan whose
    /// [`mean_frame_total`] is `norm`.
    pub fn predict(&self, frames: &Counts, norm: f64) -> Result<Patches, NnError> {
        if frames.frame_shape() != self.config.frame {
            return Err(NnError::Invalid(format!(
                "frame {:?} differs from predictor frame {:?}",
                frames.frame_shape(),
                self.config.frame
            )));
        }
        let input = diffraction_input(frames, self.config.bin, norm)?;
        let patch = self.config.patch();
        let plane = patch[0] * patch[1];
        let device = self.device();
        let mut out = Patches::empty(patch);
        for chunk in input.chunks(CHUNK * plane) {
            let n = chunk.len() / plane;
            let (a, p) = self
                .model
                .forward(tensor(chunk, [n, patch[0], patch[1]], &device));
            out.append(Patches {
                dims: [n, patch[0], patch[1]],
                amp: host(a)?,
                phase: host(p)?,
            });
        }
        Ok(out)
    }

    /// The object initial value for a scan: [`predict`](Self::predict) then [`stitch`].
    pub fn initial_object(
        &self,
        frames: &Counts,
        norm: f64,
        addr: &AddrTable,
    ) -> Result<Object, NnError> {
        stitch(&self.predict(frames, norm)?, addr)
    }

    /// Mean L1 loss over `samples`, the training loss.
    pub fn loss(&self, samples: &Samples) -> Result<f64, NnError> {
        let device = self.device();
        let [n, py, px] = samples.target.dims;
        let plane = py * px;
        let mut sum = 0.0;
        for start in (0..n).step_by(CHUNK) {
            let m = CHUNK.min(n - start);
            let r = start * plane..(start + m) * plane;
            let dims = [m, py, px];
            let loss = l1(
                &self.model,
                tensor(&samples.input[r.clone()], dims, &device),
                tensor(&samples.target.amp[r.clone()], dims, &device),
                tensor(&samples.target.phase[r], dims, &device),
            );
            sum += f64::from(loss.try_into_scalar::<f32>().map_err(burn_err)?) * m as f64;
        }
        Ok(sum / n as f64)
    }

    /// Writes `dir/predictor.json` (the config) and `dir/predictor.bpk` (the input statistics
    /// and the weights).
    pub fn save(&self, dir: &Path) -> Result<(), NnError> {
        std::fs::create_dir_all(dir)?;
        self.config.save(dir.join("predictor.json"))?;
        self.model
            .clone()
            .save_file(dir.join("predictor.bpk"))
            .map_err(burn_err)
    }

    /// Reads a predictor [`save`](Self::save) wrote, onto `device`.
    pub fn load(dir: &Path, device: &Device) -> Result<Self, NnError> {
        let config = PredictorConfig::load(dir.join("predictor.json")).map_err(burn_err)?;
        let [py, px] = config.patch();
        let model = Model::new(&config, vec![0.0; py * px], 1.0, device)?
            .try_load_file(dir.join("predictor.bpk"))
            .map_err(burn_err)?;
        Ok(Self { config, model })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptyrs_core::{MASKED, Positions};

    #[test]
    fn diffraction_input_bins_fills_masked_and_normalises() {
        // One 4×4 frame, bin 2: blocks [1,2;5,6], [3,4;7,8], [9,MASKED;13,14], all-masked.
        #[rustfmt::skip]
        let frame = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, MASKED, MASKED, MASKED,
            13, 14, MASKED, MASKED,
        ];
        let counts = Counts::from_raw([1, 4, 4], frame).unwrap();
        let norm = 8.0;
        let got = diffraction_input(&counts, 2, norm).unwrap();
        let block = [14.0, 22.0, 36.0 * 4.0 / 3.0, 0.0];
        let want: Vec<f32> = block
            .iter()
            .map(|b: &f64| (b * 4.0 / norm).sqrt() as f32)
            .collect();
        assert_eq!(got, want);
        assert!(diffraction_input(&counts, 3, norm).is_err());
        assert!(diffraction_input(&counts, 2, 0.0).is_err());
    }

    #[test]
    fn mean_frame_total_skips_masked() {
        let counts = Counts::from_raw([2, 1, 2], vec![3, MASKED, 4, 5]).unwrap();
        assert_eq!(mean_frame_total(&counts), 6.0);
    }

    #[test]
    fn stitching_true_patches_restores_the_covered_object() {
        let frame = [16, 16];
        let positions = Positions::new(vec![0.0, 0.0, 3.0], vec![-3.0, 3.0, 0.0]).unwrap();
        let addr = AddrTable::new(&positions, [1.0, 1.0], frame, [32, 32], 1, 1).unwrap();
        let object = crate::data::random_object([32, 32], 4);
        let patches = object_patches(&object, &addr, [8, 8]);
        assert_eq!(patches.dims, [3, 8, 8]);
        let back = stitch(&patches, &addr).unwrap();
        let mut covered = vec![false; 32 * 32];
        for f in 0..3 {
            let [cy, cx] = patch_corner(&addr, f, [8, 8]);
            for y in cy..cy + 8 {
                covered[y * 32 + cx..][..8].fill(true);
            }
        }
        assert!(covered.iter().any(|c| !c));
        for (i, (b, o)) in back.as_slice().iter().zip(object.as_slice()).enumerate() {
            let want = if covered[i] { *o } else { C64::new(1.0, 0.0) };
            assert!((b - want).norm() < 1e-6, "pixel {i}: {b} vs {want}");
        }
        let short = Patches {
            dims: [2, 8, 8],
            amp: patches.amp[..128].to_vec(),
            phase: patches.phase[..128].to_vec(),
        };
        assert!(stitch(&short, &addr).is_err());
    }
}
