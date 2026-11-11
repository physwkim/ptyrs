//! The frames pty-sim publishes and pty-recv checks against: `--model
//! synth` (default), the M0 detector-sized speckle frames, or `--model
//! ptycho`, the M5 scan simulator of [`crate::farfield`].

use crate::args::Args;
use crate::farfield::{Ptycho, PtychoSpec, ScanPath};
use crate::pipeline::Truth;
use crate::synth::{Region, Synth};

pub enum Model {
    Synth(Synth),
    Ptycho(Ptycho),
}

impl Model {
    pub fn from_args(a: &Args) -> Self {
        match a.str("model", "synth").as_str() {
            "synth" => Model::Synth(Synth {
                width: a.get("width", 1028),
                height: a.get("height", 1062),
                seed: a.get("seed", 1),
                flux: a.get("flux", 1.0),
                pool: a.get("pool", 64),
            }),
            "ptycho" => Model::Ptycho(ptycho_from_args(a)),
            m => panic!("--model {m}: expected synth or ptycho"),
        }
    }

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

/// `--n --path --points --cols --step-px --photons --seed --wavelength-m
/// --distance-m --det-pixel-m`.
pub fn ptycho_from_args(a: &Args) -> Ptycho {
    let points: usize = a.get("points", 4096);
    let path = match a.str("path", "fermat").as_str() {
        "raster" => ScanPath::Raster {
            cols: a.get("cols", (points as f64).sqrt().ceil() as usize),
        },
        "fermat" => ScanPath::Fermat,
        p => panic!("--path {p}: expected raster or fermat"),
    };
    Ptycho::new(PtychoSpec {
        n: a.get("n", 256),
        path,
        points,
        step: a.get("step-px", 20.0),
        photons: a.get("photons", 1e6),
        seed: a.get("seed", 1),
        // 8 keV, Eiger pixels 5 m away: 40 nm object pixels at n = 256.
        wavelength: a.get("wavelength-m", 1.5498e-10),
        distance: a.get("distance-m", 5.0),
        det_pixel: a.get("det-pixel-m", 75e-6),
    })
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
