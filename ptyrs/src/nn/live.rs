//! The predictor as the live fill's [`PatchPredictor`].

use rayon::prelude::*;

use crate::live::fill::PatchPredictor;
use crate::nn::NnError;
use crate::nn::predictor::{InputScale, Predictor};

/// Threads preparing a batch's network input.
const THREADS: usize = 16;

/// Network calls [`LivePredictor::new`] makes before it returns: the first tunes the
/// convolutions, the next few still run slow.
const WARM_CALLS: usize = 8;

/// A [`Predictor`] with the frame scale fixed for the scan. `norm` is the mean frame total of a
/// scan taken with this beam and exposure (a previous scan, a pre-scan, the simulator's known
/// scan): the stream's own mean exists only at its end. Every call runs the network on `batch`
/// frames, the fill's `max_batch`, padding a short batch: cubecl tunes its convolution kernels
/// per batch size, on the first call of a size, so one size tunes once, in [`new`](Self::new),
/// and a frame's patch depends on its own counts and `norm` alone, not on the batch it lands in
/// or on the batches before it. Another predictor or process may tune to other kernels, whose
/// patches differ in rounding.
pub struct LivePredictor {
    predictor: Predictor,
    scale: InputScale,
    /// Prepares the input of a batch, one task per frame.
    pool: rayon::ThreadPool,
    /// `batch` frames of input; past a short batch, earlier frames' input, whose patches are
    /// dropped.
    input: Vec<f32>,
}

impl LivePredictor {
    /// Tunes and warms the network on `batch` frames before it returns, which takes seconds;
    /// fails unless one network call holds them.
    pub fn new(predictor: Predictor, norm: f64, batch: usize) -> Result<Self, NnError> {
        if batch == 0 {
            return Err(NnError::Invalid("batch of 0 frames".into()));
        }
        predictor.check_call(batch)?;
        let config = predictor.config();
        let scale = InputScale::new(predictor.frame(), config.input, config.bin, norm)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(THREADS)
            .build()
            .map_err(std::io::Error::other)?;
        let input = vec![0.0; batch * scale.len()];
        for _ in 0..WARM_CALLS {
            predictor.network(&input)?;
        }
        Ok(Self {
            predictor,
            scale,
            pool,
            input,
        })
    }
}

/// The network input of `counts`, whole frames of `scale`'s, into `input`: one task per frame on
/// `pool`.
fn par_input(pool: &rayon::ThreadPool, scale: &InputScale, counts: &[u16], input: &mut [f32]) {
    pool.install(|| {
        input
            .par_chunks_exact_mut(scale.len())
            .zip(counts.par_chunks_exact(scale.area()))
            .with_max_len(1)
            .for_each(|(o, f)| scale.frame(f, o))
    });
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
        self.predictor
            .check_frames(counts, frame)
            .map_err(|e| e.to_string())?;
        let n = counts.len() / self.scale.area();
        let batch = self.input.len() / self.scale.len();
        if n > batch {
            return Err(format!("{n} frames for a batch of {batch}"));
        }
        let [py, px] = self.patch();
        if out.len() != n * py * px {
            return Err(format!(
                "{} values for {} patch pixels",
                out.len(),
                n * py * px
            ));
        }
        par_input(
            &self.pool,
            &self.scale,
            counts,
            &mut self.input[..n * self.scale.len()],
        );
        let patches = self
            .predictor
            .network(&self.input)
            .map_err(|e| e.to_string())?;
        for (o, (a, p)) in out
            .iter_mut()
            .zip(patches.amp().iter().zip(patches.phase()))
        {
            *o = [a * p.cos(), a * p.sin()];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{Counts, Frame, MASKED};
    use crate::nn::predictor::tests::{untrained, wide_frames};
    use crate::nn::predictor::{Input, diffraction_input};

    /// A batch past the frames one network call holds is refused before its input is made; it
    /// ran in one call that failed past cubecl's buffer or the kernels' 2³² elements.
    #[test]
    fn a_batch_past_one_call_is_refused() {
        let predictor = untrained(&wide_frames(), &crate::nn::cuda(0));
        let got = LivePredictor::new(predictor, 1e6, 512);
        assert!(matches!(got, Err(NnError::Invalid(_))));
    }

    #[test]
    fn par_input_is_diffraction_input() {
        let ([n, ny, nx], bin, norm) = ([37, 16, 24], 4, 1234.5);
        let mut state = 1u64;
        let data: Vec<u16> = (0..n * ny * nx)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match state >> 54 {
                    0..8 => MASKED,
                    r => r as u16,
                }
            })
            .collect();
        let want = diffraction_input(
            &Counts::from_raw([n, ny, nx], data.clone()).unwrap(),
            Input::BinnedCentre,
            bin,
            norm,
        )
        .unwrap();
        let frame = Frame::new([ny, nx]).unwrap();
        let scale = InputScale::new(frame, Input::BinnedCentre, bin, norm).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(THREADS)
            .build()
            .unwrap();
        let mut got = vec![f32::NAN; want.len()];
        par_input(&pool, &scale, &data, &mut got);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
