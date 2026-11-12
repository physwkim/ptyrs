use std::sync::Arc;

use ptyrs_core::C64;
use rustfft::{Fft, FftPlanner};

/// Unitary 2D FFT over every `[ny, nx]` plane of a buffer (numpy `fftn` / `ifftn` with
/// `axes=(-2, -1), norm="ortho"`): forward `X[k] = Σ_n x[n] e^{-2πi k·n/N} / √(ny·nx)`, inverse
/// with `e^{+2πi …}` and the same scale. No shift is applied; `psi` stays FFT-shifted.
pub struct Fft2 {
    shape: [usize; 2],
    /// `[forward, inverse]` along x (length `nx`) and along y (length `ny`).
    x: [Arc<dyn Fft<f64>>; 2],
    y: [Arc<dyn Fft<f64>>; 2],
}

impl Fft2 {
    pub fn new(shape: [usize; 2]) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            shape,
            x: [
                planner.plan_fft_forward(shape[1]),
                planner.plan_fft_inverse(shape[1]),
            ],
            y: [
                planner.plan_fft_forward(shape[0]),
                planner.plan_fft_inverse(shape[0]),
            ],
        }
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Forward transform of every plane in `planes` (length a multiple of `ny * nx`).
    pub fn forward(&self, planes: &mut [C64]) {
        self.run(planes, 0);
    }

    /// Inverse transform of every plane in `planes`.
    pub fn inverse(&self, planes: &mut [C64]) {
        self.run(planes, 1);
    }

    fn run(&self, planes: &mut [C64], dir: usize) {
        let [ny, nx] = self.shape;
        let n = ny * nx;
        assert_eq!(
            planes.len() % n,
            0,
            "buffer is not a whole number of {ny}x{nx} planes"
        );
        if planes.is_empty() {
            return;
        }
        let scale = 1.0 / (n as f64).sqrt();
        let mut t = vec![C64::default(); n];
        for plane in planes.chunks_exact_mut(n) {
            self.x[dir].process(plane);
            for y in 0..ny {
                for x in 0..nx {
                    t[x * ny + y] = plane[y * nx + x];
                }
            }
            self.y[dir].process(&mut t);
            for y in 0..ny {
                for x in 0..nx {
                    plane[y * nx + x] = t[x * ny + y] * scale;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    fn signal(n: usize, seed: f64) -> Vec<C64> {
        (0..n)
            .map(|i| {
                let t = i as f64 + seed;
                C64::new((0.37 * t).sin() + 0.1 * t.cos(), (1.3 * t).cos() - 0.2)
            })
            .collect()
    }

    #[test]
    fn matches_direct_dft_with_ortho_scale() {
        let (ny, nx) = (4, 6);
        let x = signal(ny * nx, 0.0);
        let mut got = x.clone();
        Fft2::new([ny, nx]).forward(&mut got);
        for ky in 0..ny {
            for kx in 0..nx {
                let mut s = C64::default();
                for y in 0..ny {
                    for xx in 0..nx {
                        let ph = -2.0
                            * PI
                            * ((ky * y) as f64 / ny as f64 + (kx * xx) as f64 / nx as f64);
                        s += x[y * nx + xx] * C64::from_polar(1.0, ph);
                    }
                }
                s /= ((ny * nx) as f64).sqrt();
                assert!((got[ky * nx + kx] - s).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn inverse_undoes_forward() {
        // The adjoint pairing is checked by registering `Propagator::Fourier`.
        let f = Fft2::new([8, 6]);
        let a = signal(96, 0.0);
        let mut back = a.clone();
        f.forward(&mut back);
        f.inverse(&mut back);
        for (u, v) in back.iter().zip(&a) {
            assert!((u - v).norm() < 1e-13);
        }
    }
}
