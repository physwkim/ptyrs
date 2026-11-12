//! Linear operators paired with their adjoints, admitted only through a dot test (DESIGN B7
//! extension rule, invariant 4).

use ptyrs_core::{C64, Object, Probe, Stack, View};

use crate::{EngineError, Fft2, accumulate_obj, accumulate_probe, obj_probe_to_psi};

/// A linear map `A` from `domain_len()` to `range_len()` complex values with its adjoint
/// `Aᴴ`, `⟨A x, y⟩ = ⟨x, Aᴴ y⟩` under `⟨a, b⟩ = Σ a·conj(b)`.
pub trait LinearOp {
    fn domain_len(&self) -> usize;
    fn range_len(&self) -> usize;
    /// `y = A x`; `y` is overwritten.
    fn forward(&self, x: &[C64], y: &mut [C64]);
    /// `x = Aᴴ y`; `x` is overwritten.
    fn adjoint(&self, y: &[C64], x: &mut [C64]);
}

/// An operator whose adjoint passed [`register`]'s dot test; `register` is its only
/// constructor.
pub struct Registered<A>(A);

impl<A: LinearOp> LinearOp for Registered<A> {
    fn domain_len(&self) -> usize {
        self.0.domain_len()
    }
    fn range_len(&self) -> usize {
        self.0.range_len()
    }
    fn forward(&self, x: &[C64], y: &mut [C64]) {
        self.0.forward(x, y)
    }
    fn adjoint(&self, y: &[C64], x: &mut [C64]) {
        self.0.adjoint(y, x)
    }
}

/// Bound on `|⟨A x, y⟩ − ⟨x, Aᴴ y⟩|` relative to `max(‖A x‖·‖y‖, ‖x‖·‖Aᴴ y‖)`, the
/// Cauchy–Schwarz scale of either side; f64 rounding over the CPU operators stays below 1e-13.
pub const DOT_TOL: f64 = 1e-10;

/// Admits `op` if it passes the dot test on fixed pseudo-random `x` and `y` with independent
/// real and imaginary parts, which catches a missing conjugate, transpose or scale.
pub fn register<A: LinearOp>(op: A) -> Result<Registered<A>, EngineError> {
    let x = noise(op.domain_len(), 0x5eed_0001);
    let y = noise(op.range_len(), 0x5eed_0002);
    let mut ax = vec![C64::default(); y.len()];
    let mut ahy = vec![C64::default(); x.len()];
    op.forward(&x, &mut ax);
    op.adjoint(&y, &mut ahy);
    let (lhs, rhs) = (dot(&ax, &y), dot(&x, &ahy));
    let scale = (norm(&ax) * norm(&y)).max(norm(&x) * norm(&ahy));
    let err = (lhs - rhs).norm() / scale;
    if err <= DOT_TOL {
        Ok(Registered(op))
    } else {
        Err(EngineError::Invalid(format!(
            "{} fails the dot test: <Ax, y> = {lhs}, <x, Aᴴy> = {rhs}, relative error {err:.3e}",
            std::any::type_name::<A>()
        )))
    }
}

fn dot(a: &[C64], b: &[C64]) -> C64 {
    a.iter().zip(b).map(|(u, v)| u * v.conj()).sum()
}

fn norm(a: &[C64]) -> f64 {
    a.iter().map(|u| u.norm_sqr()).sum::<f64>().sqrt()
}

/// SplitMix64 values in `[-1, 1)` for both parts.
fn noise(n: usize, seed: u64) -> Vec<C64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    (0..n).map(|_| C64::new(next(), next())).collect()
}

/// `O ↦ psi`: [`obj_probe_to_psi`] with the probe held fixed; adjoint [`accumulate_obj`]'s
/// numerator.
pub struct ObjectToPsi<'a> {
    pub views: &'a [View],
    pub obj_dims: [usize; 3],
    pub probe: &'a Probe,
    pub quad: &'a [C64],
}

impl LinearOp for ObjectToPsi<'_> {
    fn domain_len(&self) -> usize {
        self.obj_dims.iter().product()
    }
    fn range_len(&self) -> usize {
        self.views.len() * self.probe.plane_len()
    }
    fn forward(&self, x: &[C64], y: &mut [C64]) {
        let object = Object::new(Stack::from_vec(self.obj_dims, x.to_vec()).expect("object"));
        obj_probe_to_psi(self.views, &object, self.probe, self.quad, y);
    }
    fn adjoint(&self, y: &[C64], x: &mut [C64]) {
        let (num, _) = accumulate_obj(self.views, self.obj_dims, self.probe, self.quad, y);
        x.copy_from_slice(num.as_slice());
    }
}

/// `P ↦ psi`: [`obj_probe_to_psi`] with the object held fixed; adjoint [`accumulate_probe`]'s
/// numerator.
pub struct ProbeToPsi<'a> {
    pub views: &'a [View],
    pub object: &'a Object,
    pub probe_dims: [usize; 3],
    pub quad: &'a [C64],
}

impl LinearOp for ProbeToPsi<'_> {
    fn domain_len(&self) -> usize {
        self.probe_dims.iter().product()
    }
    fn range_len(&self) -> usize {
        self.views.len() * self.probe_dims[1] * self.probe_dims[2]
    }
    fn forward(&self, x: &[C64], y: &mut [C64]) {
        let probe = Probe::new(Stack::from_vec(self.probe_dims, x.to_vec()).expect("probe"));
        obj_probe_to_psi(self.views, self.object, &probe, self.quad, y);
    }
    fn adjoint(&self, y: &[C64], x: &mut [C64]) {
        let (num, _) = accumulate_probe(self.views, self.object, self.probe_dims, self.quad, y);
        x.copy_from_slice(num.as_slice());
    }
}

/// [`Fft2::forward`] over `nplane` planes; adjoint [`Fft2::inverse`] (unitary).
pub struct FarField<'a> {
    pub fft: &'a Fft2,
    pub nplane: usize,
}

impl LinearOp for FarField<'_> {
    fn domain_len(&self) -> usize {
        let [ny, nx] = self.fft.shape();
        self.nplane * ny * nx
    }
    fn range_len(&self) -> usize {
        self.domain_len()
    }
    fn forward(&self, x: &[C64], y: &mut [C64]) {
        y.copy_from_slice(x);
        self.fft.forward(y);
    }
    fn adjoint(&self, y: &[C64], x: &mut [C64]) {
        x.copy_from_slice(y);
        self.fft.inverse(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagonal `y = d·x`; the correct adjoint multiplies by `conj(d)`.
    struct Diag {
        d: Vec<C64>,
        conj: bool,
        scale: f64,
    }

    impl LinearOp for Diag {
        fn domain_len(&self) -> usize {
            self.d.len()
        }
        fn range_len(&self) -> usize {
            self.d.len()
        }
        fn forward(&self, x: &[C64], y: &mut [C64]) {
            for ((y, x), d) in y.iter_mut().zip(x).zip(&self.d) {
                *y = d * x;
            }
        }
        fn adjoint(&self, y: &[C64], x: &mut [C64]) {
            for ((x, y), d) in x.iter_mut().zip(y).zip(&self.d) {
                let d = if self.conj { d.conj() } else { *d };
                *x = d * y * self.scale;
            }
        }
    }

    #[test]
    fn register_admits_a_true_adjoint_and_rejects_wrong_ones() {
        let d = noise(64, 7);
        let op = |conj, scale| Diag {
            d: d.clone(),
            conj,
            scale,
        };
        assert!(register(op(true, 1.0)).is_ok());
        let missing_conj = register(op(false, 1.0)).err().unwrap().to_string();
        assert!(
            missing_conj.contains("fails the dot test"),
            "{missing_conj}"
        );
        assert!(register(op(true, 1.0 + 1e-6)).is_err(), "scale error");
    }

    #[test]
    fn far_field_registers() {
        let fft = Fft2::new([8, 6]);
        register(FarField {
            fft: &fft,
            nplane: 3,
        })
        .unwrap();
    }
}
