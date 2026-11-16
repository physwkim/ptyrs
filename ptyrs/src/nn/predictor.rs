//! PtychoNN-style initial-value predictor: detector frames in, object amplitude and phase patches
//! out, stitched into an object for the engine to start from.
//!
//! A frame of `[ny, nx]` detector pixels binned by `bin` becomes a `[ny/bin, nx/bin]` amplitude
//! image; binning halves the real-space field of view the detector samples, so the network
//! predicts the central `[ny/bin, nx/bin]` object pixels of the frame's window. Binning also
//! averages out the fringes between object points more than a `bin`-th of the window apart, which
//! are what places structure finer than that inside the patch, so the network reads the unbinned
//! centre of the frame as a second image ([`Input`]). A trained model holds for the probe and
//! geometry of its training scans.
//!
//! The phase output is the phase detail, [`phase_detail`]: one frame does not fix the phase
//! level of its patch, and a start that carries a level without the structure on it slows rPIE
//! below the flat start. The stitched start therefore has zero phase at scales beyond
//! [`DETAIL_SIGMA`] of the patch.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use crate::base::{AddrTable, C64, Counts, Frame, Object, Stack};
use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn_cubecl::Cube;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::nn::grid::{check_operands, fits};
use crate::nn::ptychonn::{Activations, Folded, PtychoNN};
use crate::nn::{NnError, burn_err};

/// The images [`diffraction_input`] makes of each frame.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// The binned frame alone, what predictors read before the centre was added.
    Binned,
    /// The binned frame, then its unbinned centre.
    BinnedCentre,
}

impl Input {
    /// Images per frame.
    pub fn images(self) -> usize {
        match self {
            Input::Binned => 1,
            Input::BinnedCentre => 2,
        }
    }
}

/// Shape of a predictor: the detector frame it reads, the binning, its input and the network
/// size.
#[derive(Config, Debug, PartialEq)]
pub struct PredictorConfig {
    pub frame: [usize; 2],
    pub bin: usize,
    #[config(default = "Input::BinnedCentre")]
    pub input: Input,
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

    /// Fails unless [`init`](Self::init) builds a network the nn kernels run on one frame; that
    /// frame's activations otherwise.
    fn check(&self) -> Result<Activations, NnError> {
        let Self {
            frame,
            bin,
            width,
            depth,
            ..
        } = *self;
        let invalid = |what: String| Err(NnError::Invalid(what));
        if width < 2 || depth == 0 {
            return invalid(format!("width {width} below 2 or depth {depth} below 1"));
        }
        // `n · 2^k`, none past `usize`.
        let scaled = |n: usize, k: usize| n.checked_mul(1usize.checked_shl(k.try_into().ok()?)?);
        // The widest weight, the last encoder stage's `[top, top, 3, 3]`.
        if !scaled(width, depth - 1).is_some_and(|top| fits(&[top, top, 3, 3])) {
            return invalid(format!(
                "width {width} × 2^(depth {depth} - 1) makes a weight of 2³² elements or more"
            ));
        }
        let granule = scaled(bin, depth).filter(|&g| g > 0);
        if !granule.is_some_and(|g| frame.iter().all(|&n| n > 0 && n % g == 0)) {
            return invalid(format!(
                "frame {frame:?} is not a positive multiple of bin {bin} × 2^depth {depth}"
            ));
        }
        let act = match PtychoNN::activations(self.input.images(), width, depth, self.patch()) {
            Some(act) if act.largest <= u128::from(u32::MAX) => act,
            _ => {
                return invalid(format!(
                    "width {width} over patch {:?} makes an activation of 2³² elements or more",
                    self.patch()
                ));
            }
        };
        Frame::new(frame).map_err(|e| NnError::Invalid(e.to_string()))?;
        Ok(act)
    }

    /// Frames one `pass` on `device` holds: each tensor under 2³² elements, the nn kernels'
    /// bound, and within one of cubecl's buffers, and all the pass's tensors within half the
    /// device's memory. Fails unless the config passes its check and one frame fits.
    fn frames_per_call(&self, device: &Device, pass: Pass) -> Result<usize, NnError> {
        let act = self.check()?;
        let (buffer, memory) = capacity(device)?;
        match most_frames(act, pass, buffer, memory) {
            0 => Err(NnError::Invalid(format!(
                "one frame of {self:?} does not fit a device of {memory} bytes"
            ))),
            frames => Ok(frames),
        }
    }

    fn geometry(&self) -> InputGeometry {
        InputGeometry {
            frame: self.frame,
            bin: self.bin,
            input: self.input,
        }
    }

    pub fn init(&self, device: &Device) -> Result<PtychoNN, NnError> {
        self.check()?;
        Ok(PtychoNN::new(
            self.input.images(),
            self.width,
            self.depth,
            device,
        ))
    }
}

/// Object patches `[n, py, px]` as amplitude and phase detail ([`phase_detail`]), `n · py · px`
/// values each.
#[derive(Clone, Debug, PartialEq)]
pub struct Patches {
    dims: [usize; 3],
    amp: Vec<f32>,
    phase: Vec<f32>,
}

impl Patches {
    /// Fails unless `amp` and `phase` each hold the `n · py · px` values of `dims`.
    pub fn new(dims: [usize; 3], amp: Vec<f32>, phase: Vec<f32>) -> Result<Self, NnError> {
        let len = dims.iter().try_fold(1usize, |p, &d| p.checked_mul(d));
        if len != Some(amp.len()) || len != Some(phase.len()) {
            return Err(NnError::Invalid(format!(
                "{} amplitudes and {} phases for patches {dims:?}",
                amp.len(),
                phase.len()
            )));
        }
        Ok(Self { dims, amp, phase })
    }

    fn empty(patch: [usize; 2]) -> Self {
        Self {
            dims: [0, patch[0], patch[1]],
            amp: Vec::new(),
            phase: Vec::new(),
        }
    }

    /// `[n, py, px]`.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn amp(&self) -> &[f32] {
        &self.amp
    }

    pub fn phase(&self) -> &[f32] {
        &self.phase
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

    /// The patches `picks`, in that order; panics if a pick is not below [`len`](Self::len).
    fn select(&self, picks: &[usize]) -> Self {
        let [_, py, px] = self.dims;
        Self {
            dims: [picks.len(), py, px],
            amp: gather(&self.amp, picks, py * px),
            phase: gather(&self.phase, picks, py * px),
        }
    }
}

/// The runs `picks` of `size` values of `values`, in that order.
fn gather(values: &[f32], picks: &[usize], size: usize) -> Vec<f32> {
    picks
        .iter()
        .flat_map(|&i| &values[i * size..(i + 1) * size])
        .copied()
        .collect()
}

/// Mean over frames of the unmasked counts per frame: the scale [`diffraction_input`] divides
/// out, so the predictor reads amplitude relative to the scan rather than the beam flux.
pub fn mean_frame_total(frames: &Counts) -> f64 {
    frames.total() / frames.nframe() as f64
}

/// Network input `[nframe, input.images(), ny/bin, nx/bin]` from detector-order frames. Image 0
/// is the binned frame: each `bin × bin` block summed (a masked pixel counts as the block's
/// unmasked mean, a fully masked block as 0), divided by `norm` ([`mean_frame_total`] of the
/// scan) and by the binned pixel share, and square-rooted, so a frame of mean intensity has mean
/// square amplitude 1. Image 1 of [`Input::BinnedCentre`] is the unbinned `[ny/bin, nx/bin]`
/// pixels about the zero frequency `[ny/2, nx/2]` on the same scale (a masked pixel is 0).
pub fn diffraction_input(
    frames: &Counts,
    input: Input,
    bin: usize,
    norm: f64,
) -> Result<Vec<f32>, NnError> {
    frames_input(frames.as_slice(), frames.frame_shape(), input, bin, norm)
}

/// [`diffraction_input`] of whole `[ny, nx]` frames in one slice.
fn frames_input(
    frames: &[u16],
    frame: Frame,
    input: Input,
    bin: usize,
    norm: f64,
) -> Result<Vec<f32>, NnError> {
    let scale = InputScale::new(frame, input, bin, norm)?;
    let mut out = vec![0.0; frames.len() / scale.area() * scale.len()];
    for (f, o) in frames
        .chunks_exact(scale.area())
        .zip(out.chunks_exact_mut(scale.len()))
    {
        scale.frame(f, o);
    }
    Ok(out)
}

/// [`diffraction_input`] one frame at a time, for frames of one shape, input, binning and norm.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InputScale {
    frame: [usize; 2],
    bin: usize,
    /// `frame`'s area.
    area: usize,
    len: usize,
    scale: f64,
    fine: f64,
}

impl InputScale {
    pub(crate) fn new(frame: Frame, input: Input, bin: usize, norm: f64) -> Result<Self, NnError> {
        let [ny, nx] = frame.sides();
        if bin == 0 || ny % bin != 0 || nx % bin != 0 {
            return Err(NnError::Invalid(format!(
                "frame [{ny}, {nx}] is not a positive multiple of bin {bin}"
            )));
        }
        if norm.is_nan() || norm <= 0.0 {
            return Err(NnError::Invalid(format!("norm {norm} must be positive")));
        }
        let area = frame.area();
        // At most `area`.
        let binned = (ny / bin) * (nx / bin);
        let len = input.images().checked_mul(binned).ok_or_else(|| {
            NnError::Invalid(format!(
                "{input:?} of frame [{ny}, {nx}] at bin {bin} has values past usize"
            ))
        })?;
        Ok(Self {
            frame: [ny, nx],
            bin,
            area,
            len,
            scale: binned as f64 / norm,
            fine: area as f64 / norm,
        })
    }

    /// Pixels per frame.
    pub(crate) fn area(&self) -> usize {
        self.area
    }

    /// Input values per frame.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The input of one frame into `out`, [`len`](Self::len) values.
    pub(crate) fn frame(&self, frame: &[u16], out: &mut [f32]) {
        let ([ny, nx], bin) = (self.frame, self.bin);
        let (by, bx) = (ny / bin, nx / bin);
        let (cy, cx) = (ny / 2 - by / 2, nx / 2 - bx / 2);
        let (binned, centre) = out.split_at_mut(by * bx);
        for (i, o) in binned.iter_mut().enumerate() {
            let (y, x) = (i / bx, i % bx);
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
            *o = (block * self.scale).sqrt() as f32;
        }
        // Empty for `Input::Binned`.
        for (y, o) in centre.chunks_exact_mut(bx).enumerate() {
            let row = &frame[(cy + y) * nx + cx..][..bx];
            for (o, &v) in o.iter_mut().zip(row) {
                *o = (Counts::intensity(v).unwrap_or(0.0) * self.fine).sqrt() as f32;
            }
        }
    }
}

/// Top-left corner in the object of each frame's patch: the centre `patch` of its view window.
/// Fails unless `patch` fits in `addr`'s frame.
fn patch_corners(addr: &AddrTable, patch: [usize; 2]) -> Result<Vec<[usize; 2]>, NnError> {
    let frame = addr.frame_shape();
    if patch[0] > frame[0] || patch[1] > frame[1] {
        return Err(NnError::Invalid(format!(
            "patch {patch:?} wider than frame {frame:?}"
        )));
    }
    let corner = |f| {
        let v = addr.frame_views(f)[0];
        [
            v.oy as usize + (frame[0] - patch[0]) / 2,
            v.ox as usize + (frame[1] - patch[1]) / 2,
        ]
    };
    Ok((0..addr.nframe()).map(corner).collect())
}

/// Sigma of the blur [`phase_detail`] removes, as a fraction of the patch side: at bin 4 the
/// intensity width of `sim::probe`, the spot over which a frame leaves the phase level open.
pub const DETAIL_SIGMA: f64 = 1.0 / 4.0;

/// The phase of object mode 0 minus its Gaussian blur of `sigma` pixels, the blur renormalised
/// over the pixels inside the object at its edges.
pub fn phase_detail(object: &Object, sigma: f64) -> Vec<f32> {
    let [ny, nx] = object.plane_shape();
    let phase: Vec<f64> = object.plane(0).iter().map(|c| c.arg()).collect();
    let r = (4.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = (0..=2 * r)
        .map(|i| (-((i as f64 - r as f64) / sigma).powi(2) / 2.0).exp())
        .collect();
    // One pass along a line of `len` pixels `step` apart.
    let pass = |src: &[f64], dst: &mut [f64], start: usize, step: usize, len: usize| {
        for i in 0..len {
            let (lo, hi) = (i.saturating_sub(r), (i + r).min(len - 1));
            let (mut sum, mut weight) = (0.0, 0.0);
            for j in lo..=hi {
                let k = kernel[j + r - i];
                sum += k * src[start + j * step];
                weight += k;
            }
            dst[start + i * step] = sum / weight;
        }
    };
    let mut rows = vec![0.0; ny * nx];
    for y in 0..ny {
        pass(&phase, &mut rows, y * nx, 1, nx);
    }
    let mut blur = vec![0.0; ny * nx];
    for x in 0..nx {
        pass(&rows, &mut blur, x, nx, ny);
    }
    phase
        .iter()
        .zip(blur)
        .map(|(p, b)| (p - b) as f32)
        .collect()
}

/// The true patches of object mode 0 at every frame of `addr`: the predictor's training target.
/// The detail is taken on the whole object before the patches are cut. Fails unless `object` is
/// of `addr`'s object shape and `patch` fits in its frame.
pub fn object_patches(
    object: &Object,
    addr: &AddrTable,
    patch: [usize; 2],
) -> Result<Patches, NnError> {
    let corners = patch_corners(addr, patch)?;
    if object.plane_shape() != addr.obj_shape() {
        return Err(NnError::Invalid(format!(
            "object {:?} for addressed object {:?}",
            object.plane_shape(),
            addr.obj_shape()
        )));
    }
    let nxo = object.plane_shape()[1];
    let plane = object.plane(0);
    let sigma = DETAIL_SIGMA * patch[0].min(patch[1]) as f64;
    let detail = phase_detail(object, sigma);
    let (mut amp, mut phase) = (Vec::new(), Vec::new());
    for [cy, cx] in corners {
        for y in 0..patch[0] {
            let row = (cy + y) * nxo + cx;
            amp.extend(plane[row..][..patch[1]].iter().map(|c| c.norm() as f32));
            phase.extend(&detail[row..][..patch[1]]);
        }
    }
    Patches::new([addr.nframe(), patch[0], patch[1]], amp, phase)
}

const FLAT: C64 = C64::new(1.0, 0.0);

/// The flat initial object `1 + 0i`, one mode of `addr`'s shape: the start a predicted object
/// is measured against ([`crate::nn::metric`]).
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
    let corners = patch_corners(addr, [py, px])?;
    let [nyo, nxo] = addr.obj_shape();
    let taper = |i: usize, len: usize| (PI * (i as f64 + 0.5) / len as f64).sin().powi(2);
    let wy: Vec<f64> = (0..py).map(|i| taper(i, py)).collect();
    let wx: Vec<f64> = (0..px).map(|i| taper(i, px)).collect();
    let mut sum = vec![C64::default(); nyo * nxo];
    let mut weight = vec![0.0; nyo * nxo];
    for (f, [cy, cx]) in corners.into_iter().enumerate() {
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

/// What a config makes of a frame: the frame, binning and [`Input`] that [`Samples`] are made
/// with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InputGeometry {
    frame: [usize; 2],
    bin: usize,
    input: Input,
}

/// Training pairs: network inputs `[n, kind.images(), py, px]` of the [`Input`] `kind` and the
/// true patches. The frame, binning and kind are those of the config the samples were made for,
/// and every predictor that reads them must have the same.
#[derive(Clone, Debug)]
pub struct Samples {
    geometry: InputGeometry,
    input: Vec<f32>,
    target: Patches,
}

impl Samples {
    /// Fails unless `config` passes its check.
    pub fn new(config: &PredictorConfig) -> Result<Self, NnError> {
        config.check()?;
        Ok(Self {
            geometry: config.geometry(),
            input: Vec::new(),
            target: Patches::empty(config.patch()),
        })
    }

    pub fn kind(&self) -> Input {
        self.geometry.input
    }

    /// The network inputs, `[n, kind.images(), py, px]`.
    pub fn input(&self) -> &[f32] {
        &self.input
    }

    pub fn target(&self) -> &Patches {
        &self.target
    }

    /// Fails unless samples of `geometry` are these samples'.
    fn check(&self, geometry: InputGeometry) -> Result<(), NnError> {
        if geometry != self.geometry {
            return Err(NnError::Invalid(format!(
                "samples of {:?} for {geometry:?}",
                self.geometry
            )));
        }
        Ok(())
    }

    /// Fails unless a predictor of `config` reads these samples.
    fn check_config(&self, config: &PredictorConfig) -> Result<(), NnError> {
        self.check(config.geometry())
    }

    /// The pairs of one simulated scan: `frames` normalised by their own [`mean_frame_total`],
    /// targets from `object` through `addr`, the scan's address table.
    pub fn push_scan(
        &mut self,
        frames: &Counts,
        object: &Object,
        addr: &AddrTable,
    ) -> Result<(), NnError> {
        let InputGeometry { frame, bin, input } = self.geometry;
        if frames.frame_shape() != frame || addr.frame_shape() != frame {
            return Err(NnError::Invalid(format!(
                "scan frame {:?} differs from samples' frame {frame:?}",
                frames.frame_shape(),
            )));
        }
        if frames.nframe() != addr.nframe() {
            return Err(NnError::Invalid(format!(
                "{} frames for {} addressed frames",
                frames.nframe(),
                addr.nframe()
            )));
        }
        let input = diffraction_input(frames, input, bin, mean_frame_total(frames))?;
        let [_, py, px] = self.target.dims;
        let target = object_patches(object, addr, [py, px])?;
        self.input.extend(input);
        self.target.append(target);
        Ok(())
    }

    /// Appends `other`'s pairs, which must be of the same frame, binning and kind.
    pub fn extend(&mut self, other: Samples) -> Result<(), NnError> {
        self.check(other.geometry)?;
        self.input.extend(other.input);
        self.target.append(other.target);
        Ok(())
    }

    /// The pairs `picks`, in that order.
    ///
    /// # Panics
    ///
    /// If a pick is not below [`len`](Self::len).
    pub fn select(&self, picks: &[usize]) -> Samples {
        let [_, py, px] = self.target.dims;
        Samples {
            geometry: self.geometry,
            input: gather(&self.input, picks, self.kind().images() * py * px),
            target: self.target.select(picks),
        }
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
    ///
    /// The state holds the weights with their batch-norm running statistics, Adam's moments and
    /// step, and the epochs finished; the resumed run replays those epochs' shuffles from `seed`,
    /// so it continues the same optimisation batch for batch. Resumed in the same process, it
    /// reproduces the uninterrupted run's losses to the bit. A new process tunes its kernels
    /// afresh and may pick other convolutions, which round differently, so its losses are not
    /// bit-equal to the uninterrupted run's.
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
        let file = state.join("checkpoint.json");
        let mut json = read_json(&file)?;
        let config = json.get_mut("config").ok_or_else(|| {
            NnError::Invalid(format!("{} holds no predictor config", file.display()))
        })?;
        settle_input(config, &state.join("model.bpk"))?;
        let saved: Checkpoint = serde_json::from_value(json).map_err(burn_err)?;
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

/// The JSON value of the file `path`.
fn read_json(path: &Path) -> Result<serde_json::Value, NnError> {
    serde_json::from_slice(&std::fs::read(path)?)
        .map_err(|e| NnError::Invalid(format!("{}: {e}", path.display())))
}

/// Fills in the `input` of a [`PredictorConfig`]'s JSON `config` from the images the input mean
/// in `weights` holds, when the file predates the field: the configs of the one-image and the
/// two-image input read alike there.
fn settle_input(config: &mut serde_json::Value, weights: &Path) -> Result<(), NnError> {
    let Some(fields) = config.as_object_mut() else {
        return Err(NnError::Invalid(format!("predictor config {config}")));
    };
    if fields.contains_key("input") {
        return Ok(());
    }
    let mean = burn_pack::Reader::from_file(weights)
        .and_then(burn_pack::Reader::into_tensors)
        .map_err(burn_err)?
        .into_iter()
        .find(|t| t.name == "mean")
        .ok_or_else(|| NnError::Invalid(format!("{} holds no input mean", weights.display())))?;
    let input = match mean.shape.as_slice() {
        [_, 1, ..] => Input::Binned,
        [_, 2, ..] => Input::BinnedCentre,
        shape => {
            return Err(NnError::Invalid(format!(
                "{} holds an input mean of shape {shape:?}",
                weights.display()
            )));
        }
    };
    fields.insert(
        "input".into(),
        serde_json::to_value(input).map_err(burn_err)?,
    );
    Ok(())
}

/// `values` as `[n, channels, py, px]` for `dims = [n, py, px]`.
fn tensor(values: &[f32], channels: usize, dims: [usize; 3], device: &Device) -> Tensor<4> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [dims[0], channels, dims[1], dims[2]]),
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
/// inputs is subtracted and the result multiplied by each image's inverse RMS spread about that
/// mean.
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
    /// `mean` is the `[images, py, px]` mean input, `inv_spread` one value per image.
    fn new(
        config: &PredictorConfig,
        mean: Vec<f32>,
        inv_spread: Vec<f32>,
        device: &Device,
    ) -> Result<Self, NnError> {
        let net = config.init(device)?;
        let [py, px] = config.patch();
        let images = config.input.images();
        let fixed = |values: Vec<f32>, shape: [usize; 4]| {
            Param::from_tensor(Tensor::from_data(TensorData::new(values, shape), device))
                .set_require_grad(false)
        };
        Ok(Self {
            net,
            mean: fixed(mean, [1, images, py, px]),
            inv_spread: fixed(inv_spread, [1, images, 1, 1]),
        })
    }

    fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        self.net.forward(self.standardise(x))
    }

    /// `x`, `[batch, images, py, px]`, less the mean input, times each image's inverse spread.
    fn standardise(&self, x: Tensor<4>) -> Tensor<4> {
        let (mean, inv_spread) = (self.mean.val(), self.inv_spread.val());
        let [_, images, py, px] = x.dims();
        check_operands(
            &x,
            &[
                (&mean, &[1, images, py, px]),
                (&inv_spread, &[1, images, 1, 1]),
            ],
        );
        (x - mean) * inv_spread
    }
}

/// Per-pixel mean of `samples`' inputs and each image's inverse RMS spread about it.
fn input_stats(samples: &Samples) -> Result<(Vec<f32>, Vec<f32>), NnError> {
    let [n, py, px] = samples.target.dims;
    let images = samples.kind().images();
    let plane = py * px;
    let mut mean = vec![0.0f64; images * plane];
    for frame in samples.input.chunks(mean.len()) {
        for (m, &x) in mean.iter_mut().zip(frame) {
            *m += f64::from(x) / n as f64;
        }
    }
    let mut var = vec![0.0f64; images];
    for frame in samples.input.chunks(mean.len()) {
        for ((x, m), v) in frame
            .chunks(plane)
            .zip(mean.chunks(plane))
            .zip(var.iter_mut())
        {
            *v += x
                .iter()
                .zip(m)
                .map(|(&x, m)| (f64::from(x) - m).powi(2))
                .sum::<f64>()
                / (n * plane) as f64;
        }
    }
    if let Some(v) = var.iter().find(|v| v.is_nan() || **v == 0.0) {
        return Err(NnError::Invalid(format!("training inputs have spread {v}")));
    }
    Ok((
        mean.into_iter().map(|m| m as f32).collect(),
        var.iter().map(|v| (1.0 / v.sqrt()) as f32).collect(),
    ))
}

/// The training loss of output `a`, `p` against targets `amp`, `phase`.
fn l1(a: Tensor<4>, p: Tensor<4>, amp: Tensor<4>, phase: Tensor<4>) -> Tensor<1> {
    (a - amp).abs().mean() + (p - phase).abs().mean()
}

/// Trains a fresh network of `config` with Adam on L1 loss, reporting each epoch to `log`, or
/// resumes the run in `options.checkpoint`. `options.seed` seeds the device's generator for the
/// initial weights and the shuffle of each epoch. `device` is a plain device; autodiff is enabled
/// on it here. `train` and `valid` must be samples of `config`'s input and patch, and one
/// training step on `device` must hold a batch, or every sample when there are fewer.
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
    train.check_config(config)?;
    if let Some(v) = valid {
        v.check_config(config)?;
    }
    let frames = config.frames_per_call(device, Pass::Train)?;
    let batch = options.batch.min(train.len());
    if batch > frames {
        return Err(NnError::Invalid(format!(
            "a batch of {batch} frames past the {frames} one training step holds: a split batch \
             would change its batch-norm statistics"
        )));
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
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut order: Vec<usize> = (0..train.len()).collect();
    // Each epoch shuffles the previous order: replay the finished ones.
    for _ in 0..start {
        order.shuffle(&mut rng);
    }
    for epoch in start..options.epochs {
        order.shuffle(&mut rng);
        let mut sum = Tensor::<1>::zeros([1], &device.clone().inner());
        for batch in order.chunks(options.batch) {
            // Uploaded batch by batch: a tensor of every sample would escape the batch's bound.
            let picked = train.select(batch);
            let dims = picked.target.dims;
            let (a, p) = model.forward(tensor(&picked.input, train.kind().images(), dims, &ad));
            let loss = l1(
                a,
                p,
                tensor(&picked.target.amp, 1, dims, &ad),
                tensor(&picked.target.phase, 1, dims, &ad),
            );
            sum = sum + loss.clone().detach().inner() * batch.len() as f64;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(options.lr, model, grads);
        }
        let train_loss =
            f64::from(sum.try_into_scalar::<f32>().map_err(burn_err)?) / train.len() as f64;
        let valid_loss = valid
            .map(|v| Predictor::new(config.clone(), model.valid())?.loss(v))
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
    Predictor::new(config.clone(), model.valid())
}

/// What one network call runs.
#[derive(Clone, Copy, Debug)]
enum Pass {
    /// The folded network's forward pass.
    Predict,
    /// A training step, whose forward pass keeps every tensor for the backward pass, which
    /// gives each a gradient.
    Train,
}

/// Bytes on `device` of cubecl's largest buffer and of the device's memory.
fn capacity(device: &Device) -> Result<(u64, u64), NnError> {
    let probe = Tensor::<1>::empty([1], device)
        .inner()
        .try_into_primitive::<Cube>()
        .map_err(burn_err)?;
    let memory = &probe.client.properties().memory;
    let total = memory
        .max_memory()
        .ok_or_else(|| NnError::Invalid("a device that states no memory".into()))?;
    Ok((memory.max_page_size, total))
}

/// Frames of `act` one `pass` holds with each tensor under 2³² elements and `buffer` bytes, and
/// all of them within half of `memory` bytes: the other half holds the weights, Adam's moments,
/// the memory pool's slack and whatever else shares the device.
fn most_frames(act: Activations, pass: Pass, buffer: u64, memory: u64) -> usize {
    let copies = match pass {
        Pass::Predict => 1,
        Pass::Train => 2,
    };
    let bytes = size_of::<f32>() as u128;
    let frames = (u128::from(u32::MAX) / act.largest)
        .min(u128::from(buffer) / (bytes * act.largest))
        .min(u128::from(memory) / 2 / (copies * bytes * act.total));
    usize::try_from(frames).unwrap_or(usize::MAX)
}

/// Frames in, object patches out.
#[derive(Debug)]
pub struct Predictor {
    config: PredictorConfig,
    /// `config`'s frame.
    frame: Frame,
    /// What [`save`](Self::save) writes.
    model: Model,
    /// What predicts: `model`'s network folded.
    net: Folded,
    /// Frames one network call holds, [`Pass::Predict`]'s on the model's device.
    frames: usize,
}

impl Predictor {
    fn new(config: PredictorConfig, model: Model) -> Result<Self, NnError> {
        let frames = config.frames_per_call(&model.devices()[0], Pass::Predict)?;
        let net = model.net.fold();
        Ok(Self {
            frame: Frame::new(config.frame)?,
            config,
            model,
            net,
            frames,
        })
    }

    pub fn config(&self) -> &PredictorConfig {
        &self.config
    }

    fn device(&self) -> Device {
        self.model.devices()[0].clone()
    }

    /// The model's output for raw input `x`.
    fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        self.net.forward(self.model.standardise(x))
    }

    /// Patches `[nframe, py, px]` for detector-order `frames` of a scan whose
    /// [`mean_frame_total`] is `norm`.
    pub fn predict(&self, frames: &Counts, norm: f64) -> Result<Patches, NnError> {
        self.predict_raw(frames.as_slice(), frames.frame_shape().sides(), norm)
    }

    /// [`predict`](Self::predict) for whole `frame`-shaped frames in one slice, a masked pixel
    /// holding [`crate::base::MASKED`].
    pub fn predict_raw(
        &self,
        frames: &[u16],
        frame: [usize; 2],
        norm: f64,
    ) -> Result<Patches, NnError> {
        self.check_frames(frames, frame)?;
        let config = &self.config;
        let input = frames_input(frames, self.frame, config.input, config.bin, norm)?;
        let plane = config.patch().iter().product::<usize>();
        let mut out = Patches::empty(config.patch());
        for chunk in input.chunks(self.frames * config.input.images() * plane) {
            out.append(self.network(chunk)?);
        }
        Ok(out)
    }

    /// Fails unless `frames` is whole frames of this predictor's `frame`.
    pub(crate) fn check_frames(&self, frames: &[u16], frame: [usize; 2]) -> Result<(), NnError> {
        if frame != self.frame || !frames.len().is_multiple_of(self.frame.area()) {
            return Err(NnError::Invalid(format!(
                "{} counts of frame {frame:?} for predictor frame {:?}",
                frames.len(),
                self.config.frame
            )));
        }
        Ok(())
    }

    /// Fails unless one network call holds `n` frames.
    pub(crate) fn check_call(&self, n: usize) -> Result<(), NnError> {
        if n > self.frames {
            return Err(NnError::Invalid(format!(
                "{n} frames past the {} one network call holds",
                self.frames
            )));
        }
        Ok(())
    }

    /// The patches of `input`, whole frames of [`diffraction_input`], in one network call, which
    /// must hold them ([`check_call`](Self::check_call)).
    pub(crate) fn network(&self, input: &[f32]) -> Result<Patches, NnError> {
        let [py, px] = self.config.patch();
        let images = self.config.input.images();
        let n = input.len() / (images * py * px);
        self.check_call(n)?;
        let (a, p) = self.forward(tensor(input, images, [n, py, px], &self.device()));
        Patches::new([n, py, px], host(a)?, host(p)?)
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

    /// Mean L1 loss over `samples`, the training loss; they must be of this predictor's input
    /// and patch.
    pub fn loss(&self, samples: &Samples) -> Result<f64, NnError> {
        samples.check_config(&self.config)?;
        let device = self.device();
        let [n, py, px] = samples.target.dims;
        let plane = py * px;
        let images = samples.kind().images();
        let mut sum = 0.0;
        for start in (0..n).step_by(self.frames) {
            let m = self.frames.min(n - start);
            let r = start * plane..(start + m) * plane;
            let dims = [m, py, px];
            let (a, p) = self.forward(tensor(
                &samples.input[images * r.start..images * r.end],
                images,
                dims,
                &device,
            ));
            let loss = l1(
                a,
                p,
                tensor(&samples.target.amp[r.clone()], 1, dims, &device),
                tensor(&samples.target.phase[r], 1, dims, &device),
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
        let weights = dir.join("predictor.bpk");
        let mut json = read_json(&dir.join("predictor.json"))?;
        settle_input(&mut json, &weights)?;
        let config: PredictorConfig = serde_json::from_value(json).map_err(burn_err)?;
        config.check()?;
        let [py, px] = config.patch();
        let images = config.input.images();
        let model = Model::new(
            &config,
            vec![0.0; images * py * px],
            vec![1.0; images],
            device,
        )?
        .try_load_file(weights)
        .map_err(burn_err)?;
        Self::new(config, model)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::base::{MASKED, Positions};
    use burn::tensor::DType;

    /// A predictor of `config` at its initial weights, standardising nothing.
    pub(crate) fn untrained(config: &PredictorConfig, device: &Device) -> Predictor {
        let [py, px] = config.patch();
        let images = config.input.images();
        let model = Model::new(
            config,
            vec![0.0; images * py * px],
            vec![1.0; images],
            device,
        );
        Predictor::new(config.clone(), model.unwrap()).unwrap()
    }

    /// py's probe: a 1024² detector at bin 2 with the default width and depth.
    pub(crate) fn wide_frames() -> PredictorConfig {
        PredictorConfig::new([1024, 1024], 2)
    }

    /// Frames per call stop at the first of the 2³² element, buffer and memory bounds, and a
    /// training step's at half the memory's.
    #[test]
    fn frames_per_call_stop_at_the_tightest_bound() {
        let act = |largest, total| Activations { largest, total };
        let ample = u64::MAX;
        // py's widest activation: width 32 over a 512² patch, 2²³ elements a frame.
        let wide = act(1 << 23, 1 << 23);
        // 511 · 2²³ elements are below 2³², 512 · 2²³ not.
        assert_eq!(most_frames(wide, Pass::Predict, ample, ample), 511);
        // 8190 MiB holds 255 frames of 32 MiB, 8192 MiB 256.
        assert_eq!(most_frames(wide, Pass::Predict, 8190 << 20, ample), 255);
        assert_eq!(most_frames(wide, Pass::Predict, 8192 << 20, ample), 256);
        // Half of 32 GiB holds 512 frames of 32 MiB of tensors, a byte less 511; of a training
        // step's tensors and their gradients, 256.
        let many = act(1, 1 << 23);
        assert_eq!(most_frames(many, Pass::Predict, ample, 32 << 30), 512);
        assert_eq!(most_frames(many, Pass::Predict, ample, (32 << 30) - 1), 511);
        assert_eq!(most_frames(many, Pass::Train, ample, 32 << 30), 256);
    }

    /// py's probe: 256 frames of [`wide_frames`] in one call passed cubecl's 8190 MiB buffer and
    /// 512 the kernels' 2³² elements. They run in calls of the frames one holds, and a frame's
    /// patch is the one it gets alone.
    #[test]
    fn predict_splits_frames_past_one_call() {
        let device = crate::nn::cuda(0);
        let config = wide_frames();
        let predictor = untrained(&config, &device);
        assert!(predictor.frames < 256, "{}", predictor.frames);
        let frame = 1024 * 1024;
        let counts: Vec<u16> = (0..512 * frame).map(|i| (i % 97) as u16).collect();
        let norm = 48.0 * frame as f64;
        let some = predictor.predict_raw(&counts[..256 * frame], config.frame, norm);
        assert_eq!(some.unwrap().dims, [256, 512, 512]);
        let all = predictor.predict_raw(&counts, config.frame, norm).unwrap();
        assert_eq!(all.dims, [512, 512, 512]);
        let last = predictor
            .predict_raw(&counts[511 * frame..], config.frame, norm)
            .unwrap();
        let plane = 512 * 512;
        assert_eq!(all.amp[511 * plane..], last.amp[..]);
        assert_eq!(all.phase[511 * plane..], last.phase[..]);
    }

    /// One network call runs the frames it holds and refuses one more.
    #[test]
    fn a_call_past_its_frames_is_refused() {
        let device = crate::nn::cuda(0);
        let config = wide_frames();
        let predictor = untrained(&config, &device);
        let len = config.input.images() * 512 * 512;
        let input = vec![0.0; (predictor.frames + 1) * len];
        let got = predictor.network(&input);
        assert!(matches!(got, Err(NnError::Invalid(_))), "{got:?}");
        assert!(predictor.network(&input[len..]).is_ok());
    }

    /// py's probe: a batch of 512 at width 2048 on 64² patches passed the check, then panicked
    /// past 2³² elements. Split, it would train on other batch-norm statistics, so it is refused.
    #[test]
    fn train_refuses_a_batch_one_step_cannot_hold() {
        let device = crate::nn::cuda(0);
        let config = PredictorConfig::new([64, 64], 1)
            .with_width(2048)
            .with_depth(1);
        let (n, plane) = (512, 64 * 64);
        let samples = Samples {
            geometry: config.geometry(),
            input: (0..n * config.input.images() * plane)
                .map(|i| (i % 13) as f32)
                .collect(),
            target: Patches::new([n, 64, 64], vec![1.0; n * plane], vec![0.0; n * plane]).unwrap(),
        };
        let options = TrainOptions {
            batch: n,
            ..options()
        };
        let got = train(&config, &options, &samples, None, &device, |_| {});
        assert!(
            matches!(&got, Err(NnError::Invalid(m)) if m.contains("batch-norm")),
            "{got:?}"
        );
    }

    /// A frame with a side of no pixels is refused rather than divided by.
    #[test]
    fn a_frame_of_no_pixels_is_refused() {
        assert!(Counts::from_raw([3, 0, 4], vec![]).is_err());
        let net = PredictorConfig::new([0, 16], 1).init(&crate::nn::cuda(0));
        assert!(matches!(net, Err(NnError::Invalid(_))));
    }

    /// A config `PtychoNN::new` panicked on, or `check` itself overflowed on, is refused.
    #[test]
    fn a_network_too_narrow_or_shallow_is_refused() {
        let device = crate::nn::cuda(0);
        let at = |width, depth| {
            PredictorConfig::new([16, 16], 2)
                .with_width(width)
                .with_depth(depth)
        };
        for config in [at(1, 3), at(32, 0), at(32, 64), at(1 << 62, 3)] {
            let net = config.init(&device);
            assert!(matches!(net, Err(NnError::Invalid(_))), "{config:?}");
        }
    }

    /// A weight or a one-frame activation of 2³² elements, which the nn kernels refuse, is
    /// refused with its config.
    #[test]
    fn a_network_past_the_kernels_is_refused() {
        let at = |frame, width| {
            PredictorConfig::new(frame, 1)
                .with_width(width)
                .with_depth(1)
        };
        // 9 · 21845² and 2 · 65534 · 32768 are below 2³², 9 · 21846² and 2 · 65536 · 32768 not.
        assert!(at([2, 2], 21845).check().is_ok());
        assert!(matches!(
            at([2, 2], 21846).check(),
            Err(NnError::Invalid(_))
        ));
        assert!(at([65534, 32768], 2).check().is_ok());
        assert!(matches!(
            at([65536, 32768], 2).check(),
            Err(NnError::Invalid(_))
        ));
    }

    /// A patch whose area passes `usize`, or whose activation count passes `u128`, is refused:
    /// wrapped, it counted no elements, which passed the check and divided by zero in
    /// `most_frames`.
    #[test]
    fn an_activation_past_usize_is_refused() {
        for (side, depth) in [(1usize << 33, 1), (1 << 36, 3), (1 << 63, 1)] {
            let config = PredictorConfig::new([side, side], 1).with_depth(depth);
            assert!(
                matches!(config.check(), Err(NnError::Invalid(_))),
                "{config:?}"
            );
        }
    }

    /// A frame of 2⁶⁶ pixels binned to a patch of 4 is refused by the config's check and by
    /// `Counts`, and a frame whose two input images pass `usize` by `diffraction_input`:
    /// unchecked, the area passed the check, then overflowed in `InputScale::new` or, wrapped
    /// to 0 in release, divided by zero.
    #[test]
    fn a_frame_area_past_usize_is_refused() {
        let config = PredictorConfig::new([1 << 33, 1 << 33], 1 << 32).with_depth(1);
        assert!(matches!(config.check(), Err(NnError::Invalid(_))));
        assert!(Counts::from_raw([0, config.frame[0], config.frame[1]], Vec::new()).is_err());
        let counts = Counts::from_raw([0, 1 << 32, 1 << 31], Vec::new()).unwrap();
        let got = diffraction_input(&counts, Input::BinnedCentre, 1, 1.0);
        assert!(matches!(got, Err(NnError::Invalid(_))), "{got:?}");
    }

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
        let got = diffraction_input(&counts, Input::BinnedCentre, 2, norm).unwrap();
        let block = [14.0, 22.0, 36.0 * 4.0 / 3.0, 0.0];
        // The unbinned centre: rows 1..3, columns 1..3.
        let centre = [6.0, 7.0, 0.0, 0.0];
        let want: Vec<f32> = block
            .iter()
            .map(|b: &f64| (b * 4.0 / norm).sqrt() as f32)
            .chain(centre.iter().map(|c: &f64| (c * 16.0 / norm).sqrt() as f32))
            .collect();
        assert_eq!(got, want);
        let binned = diffraction_input(&counts, Input::Binned, 2, norm).unwrap();
        assert_eq!(binned, want[..4]);
        assert!(diffraction_input(&counts, Input::BinnedCentre, 3, norm).is_err());
        assert!(diffraction_input(&counts, Input::BinnedCentre, 2, 0.0).is_err());
    }

    /// A small predictor of `input` and the samples of one three-frame scan it reads.
    fn small(input: Input) -> (PredictorConfig, Samples) {
        let config = PredictorConfig::new([16, 16], 2)
            .with_input(input)
            .with_width(4);
        let positions = Positions::new(vec![0.0, 0.0, 3.0], vec![-3.0, 3.0, 0.0]).unwrap();
        let addr = AddrTable::new(&positions, [1.0, 1.0], [16, 16], [32, 32], 1, 1).unwrap();
        let object = crate::nn::data::random_object([32, 32], 4);
        let counts = (0..3 * 256).map(|i| (i * 7 % 50) as u16).collect();
        let counts = Counts::from_raw([3, 16, 16], counts).unwrap();
        let mut samples = Samples::new(&config).unwrap();
        samples.push_scan(&counts, &object, &addr).unwrap();
        (config, samples)
    }

    fn options() -> TrainOptions {
        TrainOptions {
            epochs: 1,
            batch: 2,
            lr: 1e-3,
            seed: 1,
            checkpoint: None,
        }
    }

    /// Validation samples of the other input gave a wrong loss of one image per frame read as
    /// two, or a slice past the end of two read as one.
    #[test]
    fn train_refuses_validation_samples_of_another_input() {
        let device = crate::nn::cuda(0);
        let (one, ones) = small(Input::Binned);
        let (two, twos) = small(Input::BinnedCentre);
        assert!(train(&one, &options(), &ones, Some(&twos), &device, |_| {}).is_err());
        assert!(train(&two, &options(), &twos, Some(&ones), &device, |_| {}).is_err());
    }

    /// A config of bin 0 divided by zero before `train` reached its network.
    #[test]
    fn train_refuses_a_config_it_cannot_build() {
        let device = crate::nn::cuda(0);
        let (config, samples) = small(Input::Binned);
        let config = PredictorConfig { bin: 0, ..config };
        let got = train(&config, &options(), &samples, None, &device, |_| {});
        assert!(matches!(got, Err(NnError::Invalid(_))));
    }

    /// Training samples of the other input gave wrong input statistics, then a panic.
    #[test]
    fn train_refuses_training_samples_of_another_input() {
        let device = crate::nn::cuda(0);
        let (one, ones) = small(Input::Binned);
        let (two, twos) = small(Input::BinnedCentre);
        assert!(train(&two, &options(), &ones, None, &device, |_| {}).is_err());
        assert!(train(&one, &options(), &twos, None, &device, |_| {}).is_err());
    }

    #[test]
    fn loss_refuses_samples_of_another_input() {
        let device = crate::nn::cuda(0);
        let (one, ones) = small(Input::Binned);
        let (_, twos) = small(Input::BinnedCentre);
        let predictor = train(&one, &options(), &ones, None, &device, |_| {}).unwrap();
        assert!(predictor.loss(&ones).is_ok());
        assert!(predictor.loss(&twos).is_err());
    }

    /// Samples of the unbinned half frame, whose patch is the same `[8, 8]` as `small`'s.
    fn unbinned_half_frame() -> (PredictorConfig, Samples) {
        let config = PredictorConfig::new([8, 8], 1)
            .with_input(Input::Binned)
            .with_width(4);
        let positions = Positions::new(vec![0.0, 0.0, 3.0], vec![-3.0, 3.0, 0.0]).unwrap();
        let addr = AddrTable::new(&positions, [1.0, 1.0], [8, 8], [16, 16], 1, 1).unwrap();
        let object = crate::nn::data::random_object([16, 16], 4);
        let counts = Counts::from_raw([3, 8, 8], vec![1; 3 * 64]).unwrap();
        let mut samples = Samples::new(&config).unwrap();
        samples.push_scan(&counts, &object, &addr).unwrap();
        (config, samples)
    }

    /// Neither a scan of another frame nor samples of another input, or of another frame and
    /// binning to the same patch, join the samples.
    #[test]
    fn samples_refuse_scans_and_samples_of_another_geometry() {
        let (_, mut ones) = small(Input::Binned);
        let (_, twos) = small(Input::BinnedCentre);
        let (_, halves) = unbinned_half_frame();
        let object = crate::nn::data::random_object([16, 16], 4);
        let positions = Positions::new(vec![0.0, 0.0, 3.0], vec![-3.0, 3.0, 0.0]).unwrap();
        let addr = AddrTable::new(&positions, [1.0, 1.0], [8, 8], [16, 16], 1, 1).unwrap();
        let counts = Counts::from_raw([3, 8, 8], vec![1; 3 * 64]).unwrap();
        assert!(ones.push_scan(&counts, &object, &addr).is_err());
        assert!(ones.extend(twos).is_err());
        assert!(ones.extend(halves).is_err());
        assert_eq!((ones.len(), ones.input().len()), (3, 3 * 64));
        let (_, more) = small(Input::Binned);
        ones.extend(more).unwrap();
        assert_eq!((ones.len(), ones.input().len()), (6, 6 * 64));
    }

    /// Samples of `small`'s patch and input but another frame and binning gave a predictor of one
    /// binning inputs of another.
    #[test]
    fn train_and_loss_refuse_samples_of_another_frame_and_binning() {
        let device = crate::nn::cuda(0);
        let (config, samples) = small(Input::Binned);
        let (half, halves) = unbinned_half_frame();
        assert!(train(&half, &options(), &samples, None, &device, |_| {}).is_err());
        assert!(
            train(
                &config,
                &options(),
                &samples,
                Some(&halves),
                &device,
                |_| {}
            )
            .is_err()
        );
        let predictor = train(&config, &options(), &samples, None, &device, |_| {}).unwrap();
        assert!(predictor.loss(&halves).is_err());
    }

    /// burn's elementwise ops launch in the input's float type, so they read an f16 mean as f32.
    #[test]
    #[should_panic(expected = "an operand in the input's float type")]
    fn a_mean_of_another_float_type() {
        let device = crate::nn::cuda(0);
        let (config, _) = small(Input::Binned);
        let mut predictor = untrained(&config, &device);
        let mean = predictor.model.mean.val().cast(DType::F16);
        predictor.model.mean = Param::from_tensor(mean);
        let [py, px] = config.patch();
        let shape = [1, config.input.images(), py, px];
        predictor.forward(Tensor::zeros(shape, (&device, DType::F32)));
    }

    #[test]
    fn samples_of_a_config_it_cannot_build_are_refused() {
        let config = PredictorConfig::new([16, 16], 0).with_width(4);
        assert!(matches!(Samples::new(&config), Err(NnError::Invalid(_))));
    }

    /// A scan refused part way leaves no inputs without their targets.
    #[test]
    fn a_scan_of_another_object_leaves_the_samples_whole() {
        let (_, mut samples) = small(Input::Binned);
        let positions = Positions::new(vec![0.0, 0.0, 3.0], vec![-3.0, 3.0, 0.0]).unwrap();
        let addr = AddrTable::new(&positions, [1.0, 1.0], [16, 16], [32, 32], 1, 1).unwrap();
        let object = crate::nn::data::random_object([32, 33], 4);
        let counts = Counts::from_raw([3, 16, 16], vec![1; 3 * 256]).unwrap();
        let got = samples.push_scan(&counts, &object, &addr);
        assert!(matches!(got, Err(NnError::Invalid(_))));
        assert_eq!((samples.len(), samples.input().len()), (3, 3 * 64));
    }

    #[test]
    fn select_takes_whole_pairs_in_order() {
        let (_, twos) = small(Input::BinnedCentre);
        let picked = twos.select(&[2, 0]);
        let frame = |s: &Samples, i: usize| s.input()[i * 128..(i + 1) * 128].to_vec();
        let patch = |s: &Samples, i: usize| s.target().amp[i * 64..(i + 1) * 64].to_vec();
        assert_eq!(picked.target().dims, [2, 8, 8]);
        assert_eq!(picked.input().len(), 2 * 128);
        for (k, i) in [(0, 2), (1, 0)] {
            assert_eq!(frame(&picked, k), frame(&twos, i));
            assert_eq!(patch(&picked, k), patch(&twos, i));
        }
    }

    #[test]
    fn phase_detail_removes_the_level_and_keeps_a_step() {
        let phase = |x: usize| if x < 32 { 0.3 } else { 0.9 };
        let data = (0..64 * 64).map(|i| C64::from_polar(0.7, phase(i % 64)));
        let object = Object::new(Stack::from_vec([1, 64, 64], data.collect()).unwrap());
        let d = phase_detail(&object, 4.0);
        // Far from the step the level is gone; across it half the step stays on each side.
        assert!(d[10 * 64 + 5].abs() < 1e-6 && d[10 * 64 + 60].abs() < 1e-6);
        assert!((d[10 * 64 + 31] + 0.27).abs() < 0.02, "{}", d[10 * 64 + 31]);
        assert!((d[10 * 64 + 32] - 0.27).abs() < 0.02, "{}", d[10 * 64 + 32]);
    }

    #[test]
    fn mean_frame_total_skips_masked() {
        let counts = Counts::from_raw([2, 2, 2], vec![3, MASKED, 0, 0, 4, 5, 0, 0]).unwrap();
        assert_eq!(mean_frame_total(&counts), 6.0);
    }

    #[test]
    fn stitching_true_patches_restores_the_covered_object() {
        let frame = [16, 16];
        let positions = Positions::new(vec![0.0, 0.0, 3.0], vec![-3.0, 3.0, 0.0]).unwrap();
        let addr = AddrTable::new(&positions, [1.0, 1.0], frame, [32, 32], 1, 1).unwrap();
        let object = crate::nn::data::random_object([32, 32], 4);
        let patches = object_patches(&object, &addr, [8, 8]).unwrap();
        assert_eq!(patches.dims, [3, 8, 8]);
        let detail = phase_detail(&object, DETAIL_SIGMA * 8.0);
        let back = stitch(&patches, &addr).unwrap();
        let mut covered = vec![false; 32 * 32];
        for [cy, cx] in patch_corners(&addr, [8, 8]).unwrap() {
            for y in cy..cy + 8 {
                covered[y * 32 + cx..][..8].fill(true);
            }
        }
        assert!(covered.iter().any(|c| !c));
        for (i, (b, o)) in back.as_slice().iter().zip(object.as_slice()).enumerate() {
            let want = if covered[i] {
                C64::from_polar(o.norm(), f64::from(detail[i]))
            } else {
                C64::new(1.0, 0.0)
            };
            assert!((b - want).norm() < 1e-6, "pixel {i}: {b} vs {want}");
        }
    }

    /// Patches of too few values, of other than one per frame, wider than the frame or of
    /// another object are refused rather than indexed past.
    #[test]
    fn patches_that_do_not_fit_the_scan_are_refused() {
        let positions = Positions::new(vec![0.0, 0.0, 3.0], vec![-3.0, 3.0, 0.0]).unwrap();
        let addr = AddrTable::new(&positions, [1.0, 1.0], [16, 16], [32, 32], 1, 1).unwrap();
        let object = crate::nn::data::random_object([32, 32], 4);
        let patches = object_patches(&object, &addr, [8, 8]).unwrap();
        fn invalid<T>(r: Result<T, NnError>) -> bool {
            matches!(r, Err(NnError::Invalid(_)))
        }
        let cut = |dims: [usize; 3], len: usize| {
            Patches::new(
                dims,
                patches.amp[..len].to_vec(),
                patches.phase[..len].to_vec(),
            )
        };
        assert!(invalid(cut([3, 8, 8], 128)));
        assert!(invalid(Patches::new(
            [3, 8, 8],
            patches.amp.clone(),
            vec![]
        )));
        assert!(invalid(stitch(&cut([2, 8, 8], 128).unwrap(), &addr)));
        let wide = Patches::new([3, 32, 32], vec![1.0; 3 << 10], vec![0.0; 3 << 10]).unwrap();
        assert!(invalid(stitch(&wide, &addr)));
        assert!(invalid(object_patches(&object, &addr, [32, 32])));
        assert!(invalid(object_patches(&object, &addr, [8, 17])));
        let other = crate::nn::data::random_object([32, 33], 4);
        assert!(invalid(object_patches(&other, &addr, [8, 8])));
    }
}
