//! [`GpuOptions::psi_f16`] against f32 carried waves over 300 cycles of the M1 simulation
//! (`engine_llk_trace`'s 256-pixel frames, two probe modes, 400 positions 16 object pixels
//! apart), the bounds of PyNX with its carried waves rounded to f16 (DESIGN §8): RAAR's LLK
//! within 2e-5 of f32's at every record, and DM's, whose f32 runs themselves diverge within
//! about 100 cycles, within the spread of f32 runs from starts perturbed once by f16's
//! rounding error.

use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{C64, Object, Probe, Scan, Stack};
use ptyrs::engine::{
    Backend, DmParams, EngineError, GpuOptions, GpuPtycho, Params, Ptycho, RaarParams, Step,
};

#[path = "common/engine/mod.rs"]
mod common;

const NCYCLE: usize = 300;

/// The relative RMS error of one f16 rounding of a complex value, 2.06e-4, is √(2/3) of this.
const KICK: f64 = 2.52e-4;

const F32: GpuOptions = GpuOptions {
    stack_size: None,
    graph: true,
    llk_slots: 8,
    fused: true,
    psi_f16: false,
};

const F16: GpuOptions = GpuOptions {
    psi_f16: true,
    ..F32
};

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA device 0")
}

/// The M1 scan and its start, scaled on the CPU as `engine_llk_trace` writes `init.h5`.
fn m1() -> (Scan, Object, Probe) {
    let (scan, object, probe) = common::dm_scenario(256, 400, 2, 2e5, 16.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let (object, probe) = p.into_modes();
    (scan, object, probe)
}

/// `object` with every value x made x (1 + KICK (u + i v)), u and v uniform in [-1, 1) from
/// `seed`: cycle 0's exit waves `P O` perturbed as f16 rounds them, once.
fn kicked(object: &Object, seed: u64) -> Object {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        2.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    let data = object
        .as_slice()
        .iter()
        .map(|&x| x * (1.0 + KICK * C64::new(uniform(), uniform())))
        .collect();
    Object::new(Stack::from_vec(object.dims(), data).unwrap())
}

/// The Poisson LLK of every cycle of `params` from `object` and `probe`.
fn poisson(
    ctx: &Arc<CudaContext>,
    scan: &Scan,
    (object, probe): (Object, Probe),
    options: GpuOptions,
    params: Params,
) -> Vec<f64> {
    let mut gpu = GpuPtycho::new(ctx, scan, object, probe, options).unwrap();
    gpu.run(&Step {
        params,
        count: NCYCLE,
    })
    .unwrap();
    gpu.history().iter().map(|r| r.llk.poisson).collect()
}

#[test]
fn raar_f16_llk_is_f32s_within_2e_5() {
    let (scan, object, probe) = m1();
    let ctx = ctx();
    let raar = Params::Raar(RaarParams {
        calc_llk: 1,
        ..RaarParams::default()
    });
    let start = || (object.clone(), probe.clone());
    let f32 = poisson(&ctx, &scan, start(), F32, raar);
    let f16 = poisson(&ctx, &scan, start(), F16, raar);
    assert_eq!((f32.len(), f16.len()), (NCYCLE, NCYCLE));
    let (cycle, rel) = f32
        .iter()
        .zip(&f16)
        .map(|(a, b)| (b - a).abs() / a)
        .enumerate()
        .fold((0, 0.0), |w, (c, r)| if r > w.1 { (c, r) } else { w });
    eprintln!("RAAR f16 against f32: worst LLK {rel:.2e} at cycle {cycle}");
    assert!(rel <= 2e-5, "cycle {cycle}: LLK {rel:e} > 2e-5");
}

#[test]
fn dm_f16_llk_is_inside_the_f32_spread() {
    let (scan, object, probe) = m1();
    let ctx = ctx();
    let dm = Params::Dm(DmParams {
        calc_llk: 1,
        ..DmParams::default()
    });
    // DM's LLK swings from cycle to cycle, so each run is its mean over the last 50
    let tail = |llk: Vec<f64>| llk[NCYCLE - 50..].iter().sum::<f64>() / 50.0;
    let mut f32 = vec![tail(poisson(
        &ctx,
        &scan,
        (object.clone(), probe.clone()),
        F32,
        dm,
    ))];
    for seed in 1..=8 {
        let start = (kicked(&object, seed), probe.clone());
        f32.push(tail(poisson(&ctx, &scan, start, F32, dm)));
    }
    let f16 = tail(poisson(&ctx, &scan, (object, probe), F16, dm));
    let lo = f32.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = f32.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("DM mean LLK of the last 50 cycles: f16 {f16:.6e}, f32 {lo:.6e}..{hi:.6e}");
    assert!(
        (lo..=hi).contains(&f16),
        "f16 {f16:e} outside f32's {lo:e}..{hi:e}"
    );
}

#[test]
fn psi_f16_without_fused_is_refused() {
    let (scan, object, probe) = common::dm_scenario(32, 8, 1, 2e5, 4.0);
    let options = GpuOptions {
        fused: false,
        ..F16
    };
    match GpuPtycho::new(&ctx(), &scan, object, probe, options) {
        Err(EngineError::Invalid(m)) => assert!(m.contains("psi_f16 needs"), "{m}"),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("psi_f16 without fused passes was accepted"),
    }
}
