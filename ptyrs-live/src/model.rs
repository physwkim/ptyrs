//! The frames pty-sim publishes and the receiver checks against: `synth`
//! (default), the M0 detector-sized speckle frames, or `ptycho`, the M5 scan
//! simulator of [`crate::farfield`]. [`ModelCfg`] comes from `--model` and
//! its flags, or from a `[model.synth]` / `[model.ptycho]` TOML table.

use serde::Deserialize;

use crate::args::Args;
use crate::farfield::{Ptycho, PtychoSpec, ScanPath};
use crate::pipeline::Truth;
use crate::synth::{Region, Synth};

pub enum Model {
    Synth(Synth),
    Ptycho(Ptycho),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub enum ModelCfg {
    Synth(Synth),
    Ptycho(PtychoCfg),
}

impl Default for ModelCfg {
    fn default() -> Self {
        ModelCfg::Synth(Synth::default())
    }
}

impl ModelCfg {
    /// `--model` and the flags of that model.
    pub fn from_args(a: &Args) -> Self {
        match a.str("model", "synth").as_str() {
            "synth" => {
                let d = Synth::default();
                ModelCfg::Synth(Synth {
                    width: a.get("width", d.width),
                    height: a.get("height", d.height),
                    seed: a.get("seed", d.seed),
                    flux: a.get("flux", d.flux),
                    pool: a.get("pool", d.pool),
                })
            }
            "ptycho" => ModelCfg::Ptycho(PtychoCfg::from_args(a)),
            m => panic!("--model {m}: expected synth or ptycho"),
        }
    }

    pub fn build(&self) -> Model {
        match self {
            ModelCfg::Synth(s) => Model::Synth(*s),
            ModelCfg::Ptycho(p) => Model::Ptycho(p.build()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathKind {
    Raster,
    Fermat,
}

impl std::str::FromStr for PathKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "raster" => Ok(PathKind::Raster),
            "fermat" => Ok(PathKind::Fermat),
            _ => Err("expected raster or fermat".into()),
        }
    }
}

/// [`PtychoSpec`] with the flags' names and units.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct PtychoCfg {
    pub n: usize,
    pub path: PathKind,
    pub points: usize,
    /// Raster row length; ⌈√points⌉ when unset. A Fermat spiral has no rows.
    pub cols: Option<usize>,
    pub step_px: f64,
    pub photons: f64,
    pub seed: u64,
    pub wavelength_m: f64,
    pub distance_m: f64,
    pub det_pixel_m: f64,
}

impl Default for PtychoCfg {
    /// 8 keV, Eiger pixels 5 m away: 40 nm object pixels at n = 256.
    fn default() -> Self {
        Self {
            n: 256,
            path: PathKind::Fermat,
            points: 4096,
            cols: None,
            step_px: 20.0,
            photons: 1e6,
            seed: 1,
            wavelength_m: 1.5498e-10,
            distance_m: 5.0,
            det_pixel_m: 75e-6,
        }
    }
}

impl PtychoCfg {
    /// `--n --path --points --cols --step-px --photons --seed --wavelength-m
    /// --distance-m --det-pixel-m`.
    pub fn from_args(a: &Args) -> Self {
        let d = Self::default();
        Self {
            n: a.get("n", d.n),
            path: a.get("path", d.path),
            points: a.get("points", d.points),
            cols: a.opt("cols"),
            step_px: a.get("step-px", d.step_px),
            photons: a.get("photons", d.photons),
            seed: a.get("seed", d.seed),
            wavelength_m: a.get("wavelength-m", d.wavelength_m),
            distance_m: a.get("distance-m", d.distance_m),
            det_pixel_m: a.get("det-pixel-m", d.det_pixel_m),
        }
    }

    pub fn build(&self) -> Ptycho {
        let points = self.points;
        Ptycho::new(PtychoSpec {
            n: self.n,
            path: match self.path {
                PathKind::Raster => ScanPath::Raster {
                    cols: self.cols.unwrap_or((points as f64).sqrt().ceil() as usize),
                },
                PathKind::Fermat => ScanPath::Fermat,
            },
            points,
            step: self.step_px,
            photons: self.photons,
            seed: self.seed,
            wavelength: self.wavelength_m,
            distance: self.distance_m,
            det_pixel: self.det_pixel_m,
        })
    }
}

impl Model {
    /// One line for the logs.
    pub fn describe(&self) -> String {
        match self {
            Model::Synth(s) => format!("model=synth seed={} flux={}", s.seed, s.flux),
            Model::Ptycho(p) => p.describe(),
        }
    }

    fn truth(&self) -> &dyn Truth {
        match self {
            Model::Synth(s) => s,
            Model::Ptycho(p) => p,
        }
    }
}

impl Truth for Model {
    fn fill(&self, uid: i32, r: Region, out: &mut [u16]) {
        self.truth().fill(uid, r, out)
    }

    fn position(&self, uid: i32) -> (f64, f64) {
        self.truth().position(uid)
    }

    fn frame_size(&self) -> (usize, usize) {
        self.truth().frame_size()
    }

    fn period(&self) -> usize {
        self.truth().period()
    }
}
