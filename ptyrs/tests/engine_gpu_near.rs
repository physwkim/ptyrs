//! The GPU passes that run a propagator's plane stage besides LSQML's and BH's, on the cases
//! of `common/near.rs`.

use ptyrs::engine::{Features, RpieParams};

#[path = "common/near.rs"]
mod near;
use near::{cases, drawn_offsets};

const RPIE: RpieParams = RpieParams {
    obj_step: Some(1.0),
    probe_step: Some(1.0),
    obj_alpha: 0.5,
    probe_alpha: 0.5,
    batch_size: 4,
    calc_llk: 1,
};

/// rPIE under each propagator, with fixed positions and refining drawn offsets: every epoch
/// lowers the LLK the device records, the last is the CPU's of the arrays it ends on (fixed
/// positions), and the graphs replay the eager run bit for bit.
#[test]
fn rpie_descends_and_replays_under_every_propagator() {
    const NEPOCH: usize = 6;
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    for c in cases() {
        for features in [None, Some(&moved)] {
            let what = format!("{} positions {}", c.name, features.is_some());
            let run = |graph| {
                let mut gpu = c.gpu(graph);
                if let Some(f) = features {
                    gpu.set_features(f).unwrap();
                }
                let start = gpu.llk().unwrap().poisson;
                gpu.rpie(&RPIE, NEPOCH).unwrap();
                let trace: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
                let arrays = (gpu.object().unwrap(), gpu.probe().unwrap());
                (start, trace, arrays, gpu.features().unwrap().position)
            };
            let (start, trace, arrays, offsets) = run(false);
            eprintln!("{what}: start {start:.4e}, epochs {trace:.4?}");
            assert_eq!(trace.len(), NEPOCH, "{what}");
            let mut last = start;
            for (e, &v) in trace.iter().enumerate() {
                assert!(v < last, "{what} epoch {e}: {v:e} after {last:e}");
                last = v;
            }
            if features.is_none() {
                let want = c.llk(&arrays.0, &arrays.1);
                let err = (last - want).abs() / want;
                eprintln!("{what}: CPU LLK of the result {want:.6e}, {err:.2e}");
                assert!(err < 1e-5, "{what}: recorded {last:e}, CPU {want:e}");
            }
            let graphs = run(true);
            assert_eq!(graphs.1, trace, "{what}: LLK from graphs");
            assert_eq!(graphs.2.0.as_slice(), arrays.0.as_slice(), "{what}: object");
            assert_eq!(graphs.2.1.as_slice(), arrays.1.as_slice(), "{what}: probe");
            assert_eq!(graphs.3, offsets, "{what}: offsets");
        }
    }
}
