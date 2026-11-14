//! The predictor as the live fill's [`PatchPredictor`].

use crate::live::fill::PatchPredictor;
use crate::nn::NnError;
use crate::nn::predictor::Predictor;

/// A [`Predictor`] with the frame scale fixed for the scan. `norm` is the mean frame total of a
/// scan taken with this beam and exposure (a previous scan, a pre-scan, the simulator's known
/// scan): the stream's own mean exists only at its end. A frame's patch then depends on its own
/// counts and `norm` alone, not on the batch it lands in or on the batches before it.
pub struct LivePredictor {
    predictor: Predictor,
    norm: f64,
}

impl LivePredictor {
    pub fn new(predictor: Predictor, norm: f64) -> Result<Self, NnError> {
        if norm.is_nan() || norm <= 0.0 {
            return Err(NnError::Invalid(format!("norm {norm} must be positive")));
        }
        Ok(Self { predictor, norm })
    }
}

impl PatchPredictor for LivePredictor {
    fn patch(&self) -> [usize; 2] {
        self.predictor.config().patch()
    }

    fn predict(
        &mut self,
        counts: &[u16],
        frame: [usize; 2],
        out: &mut [[f32; 2]],
    ) -> Result<(), String> {
        let patches = self
            .predictor
            .predict_raw(counts, frame, self.norm)
            .map_err(|e| e.to_string())?;
        if out.len() != patches.amp.len() {
            return Err(format!(
                "{} values for {} patch pixels",
                out.len(),
                patches.amp.len()
            ));
        }
        for (o, (a, p)) in out.iter_mut().zip(patches.amp.iter().zip(&patches.phase)) {
            *o = [a * p.cos(), a * p.sin()];
        }
        Ok(())
    }
}
