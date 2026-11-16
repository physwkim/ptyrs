//! GPU LSQML against the CPU's `f64` [`Lsqml`] from the same scaled start, without and with
//! a probe variation (OPR), without and with a modelled position (drawn offsets), which moves
//! the probe or samples the object: each epoch's steps, the arrays, the offsets' and the
//! variation's changes after the first epochs, eagerly and from graphs, and the LLK after
//! each of enough epochs to reach a kept preconditioner; Gaussian and Poisson costs, one
//! batch of every frame, batches of 12 (two stacks each but the last) and of 4 (the last of
//! 2). With a position, without and with a probe variation, the run is bit-identical across
//! stack splits and eager or from graphs. Frames are on the fused passes' smallest side, two incoherent probe modes, one
//! masked row and column.

use std::slice;
use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{C64, Counts, MASKED, Object, Probe, Scan, Stack};
use ptyrs::engine::{
    Algorithm, BhParams, Cost, Feature, Features, GpuOptions, GpuPtycho, LlkRecord, Lsqml,
    LsqmlParams, LsqmlStep, Optics, ProbeVariation, Ptycho, Sampling,
};
use ptyrs::gpu::FUSED_SIDES;

#[path = "common/engine/mod.rs"]
mod common;
#[path = "common/opr.rs"]
mod opr;
#[path = "common/worst.rs"]
mod worst;
use opr::{Refined, variation, variation_rel};
use worst::worst_of;

/// 30 positions: stacks of at most 8 cut them into 8, 8, 7 and 7 frames, and batches of 12 or
/// 4 leave a shorter last one.
const NPOS: usize = 30;

const OPTIONS: GpuOptions = GpuOptions {
    stack_size: Some(8),
    graph: true,
    llk_slots: 3,
    fused: true,
    psi_f16: false,
};

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA device 0")
}

/// The parity scenario on `frame`-pixel frames, scaled on the CPU, with row 3 and column 20
/// of every frame masked.
fn start(frame: usize) -> (Scan, Object, Probe) {
    let (mut scan, object, probe) = common::dm_scenario(frame, NPOS, 2, 2e5, frame as f64 / 8.0);
    let dims = scan.counts.dims();
    let [_, ny, nx] = dims;
    let data = scan
        .counts
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let (y, x) = ((i / nx) % ny, i % nx);
            if y == 3 || x == 20 { MASKED } else { c }
        })
        .collect();
    scan.counts = Counts::from_raw(dims, data).unwrap();
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let (object, probe) = p.into_modes();
    (scan, object, probe)
}

/// `|a − b| / |b|` over the whole array.
fn rel_l2(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        e += (x - y).norm_sqr();
        t += y.norm_sqr();
    }
    (e / t).sqrt()
}

/// `|a − b| / |b|`, or `|a|` where `b` is 0.
fn rel(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a.abs()
    } else {
        (a - b).abs() / b.abs()
    }
}

fn runs() -> impl Iterator<Item = (Cost, usize)> {
    [Cost::Amplitude, Cost::Poisson]
        .into_iter()
        .flat_map(|cost| [NPOS, 12, 4].map(|b| (cost, b)))
}

/// Offsets drawn about their mean, up to 0.4 pixel per axis.
fn drawn_offsets() -> Vec<[f64; 2]> {
    let mut d: Vec<[f64; 2]> = (0..NPOS)
        .map(|f| {
            let f = f as f64;
            [0.4 * (1.7 * f + 0.3).sin(), 0.4 * (2.3 * f).cos()]
        })
        .collect();
    for axis in 0..2 {
        let mean = d.iter().map(|d| d[axis]).sum::<f64>() / NPOS as f64;
        d.iter_mut().for_each(|d| d[axis] -= mean);
    }
    d
}

/// The features modelling `position`, or none.
fn moved(position: Option<Vec<[f64; 2]>>) -> Features {
    Features {
        position: position.map(Feature::refined),
        ..Features::default()
    }
}

/// `max |Δa − Δb| / max |Δb|` of the offsets' changes from `start`'s; 0 without a position.
fn offsets_rel(a: &Features, b: &Features, start: &Features) -> f64 {
    let (Some(a), Some(b), Some(s)) = (&a.position, &b.position, &start.position) else {
        assert_eq!(a.position.is_some(), b.position.is_some());
        return 0.0;
    };
    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
        d.iter()
            .zip(&s.values)
            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
            .collect()
    };
    let (a, b) = (delta(&a.values), delta(&b.values));
    let d = worst_of(a.iter().zip(&b).map(|(a, b)| (a - b).abs()));
    d / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
}

#[test]
fn lsqml_steps_match_cpu() {
    steps_match_cpu(None, Sampling::Patch, None);
}

#[test]
fn lsqml_position_steps_match_cpu() {
    steps_match_cpu(Some(drawn_offsets()), Sampling::Patch, None);
}

#[test]
fn lsqml_full_object_position_steps_match_cpu() {
    steps_match_cpu(Some(drawn_offsets()), Sampling::FullObject, None);
}

#[test]
fn lsqml_opr_steps_match_cpu() {
    steps_match_cpu(None, Sampling::Patch, Some((3, Refined::Weights)));
}

#[test]
fn lsqml_opr_position_steps_match_cpu() {
    let opr = Some((2, Refined::Both));
    steps_match_cpu(Some(drawn_offsets()), Sampling::Patch, opr);
}

#[test]
fn lsqml_opr_full_object_position_steps_match_cpu() {
    let opr = Some((3, Refined::Intensity));
    steps_match_cpu(Some(drawn_offsets()), Sampling::FullObject, opr);
}

/// The start's [`variation`] of `opr`'s eigenmodes and refined weights, or none.
fn start_variation(probe: &Probe, opr: Option<(usize, Refined)>) -> ProbeVariation {
    match opr {
        Some((k, refined)) => variation(probe, NPOS, k, refined),
        None => ProbeVariation::none(probe.plane_shape()),
    }
}

/// After a fresh GPU run of each of the first epochs (the preconditioner built, then
/// updated), the last epoch's steps (frames, `a_far`, `a_obj`, `a_probe`), the arrays, the
/// offsets' and the variation's changes, the position entered as `sampling`, the probe
/// varied by `opr`'s eigenmodes and refined weights.
fn steps_match_cpu(
    position: Option<Vec<[f64; 2]>>,
    sampling: Sampling,
    opr: Option<(usize, Refined)>,
) {
    const NEPOCH: usize = 3;
    let features = moved(position);
    let frame = FUSED_SIDES[0] as usize;
    for (graph, (cost, batch_size)) in [false, true]
        .into_iter()
        .flat_map(|g| runs().map(move |r| (g, r)))
    {
        let options = GpuOptions { graph, ..OPTIONS };
        let params = LsqmlParams {
            batch_size,
            cost,
            seed: 7,
        };
        let (scan, object, probe) = start(frame);
        let at = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let optics = Optics::far_field(&scan.geometry);
        let (mut o, mut p) = (object.clone(), probe.clone());
        let start = start_variation(&probe, opr);
        let mut var = start.clone();
        let mut cpu = Lsqml::new(params)
            .unwrap()
            .with_features(features.clone())
            .unwrap()
            .with_sampling(sampling);
        let mut worst = [0.0f64; 9];
        for nepoch in 1..=NEPOCH {
            let want = cpu.epoch(
                at.addr(),
                at.observed(),
                &optics,
                slice::from_mut(&mut o),
                &mut p,
                &mut var,
            );
            let mut gpu =
                GpuPtycho::new(&ctx(), &scan, object.clone(), probe.clone(), options).unwrap();
            gpu.set_features(&features).unwrap();
            gpu.set_sampling(sampling).unwrap();
            let mut gvar = start.clone();
            gpu.lsqml(&params, &mut gvar, nepoch, 0).unwrap();
            let got = gpu.lsqml_steps().unwrap();
            let frames = |s: &[LsqmlStep]| -> Vec<Vec<usize>> {
                s.iter().map(|s| s.frames.clone()).collect()
            };
            assert_eq!(frames(&got), frames(&want), "epoch {nepoch}");
            for (w, e) in worst[..3].iter_mut().zip(steps_rel(&got, &want)) {
                *w = worst_of([*w, e]);
            }
            worst[3] = worst_of([worst[3], rel_l2(&gpu.object().unwrap(), &o)]);
            worst[4] = worst_of([worst[4], rel_l2(&gpu.probe().unwrap(), &p)]);
            let offsets = offsets_rel(&gpu.features().unwrap(), cpu.features(), &features);
            worst[5] = worst_of([worst[5], offsets]);
            for (w, e) in worst[6..]
                .iter_mut()
                .zip(variation_rel(&gvar, &var, &start))
            {
                *w = worst_of([*w, e]);
            }
        }
        eprintln!(
            "lsqml steps {cost:?} graph {graph} batch {batch_size} position {} {sampling:?} \
             opr {opr:?}: a_far {:.2e} a_obj {:.2e} a_probe {:.2e} object {:.2e} probe {:.2e} \
             offsets {:.2e} eigenmodes {:.2e} weights {:.2e} intensity {:.2e}",
            features.position.is_some(),
            worst[0],
            worst[1],
            worst[2],
            worst[3],
            worst[4],
            worst[5],
            worst[6],
            worst[7],
            worst[8]
        );
        let names = [
            "a_far",
            "a_obj",
            "a_probe",
            "object",
            "probe",
            "offsets",
            "eigenmodes",
            "weights",
            "intensity",
        ];
        for (what, e) in names.into_iter().zip(worst) {
            assert!(
                e < 1e-4,
                "{cost:?} graph {graph} batch {batch_size} {sampling:?} opr {opr:?} {what}: {e:e}"
            );
        }
    }
}

#[test]
fn lsqml_matches_cpu() {
    matches_cpu(None, Sampling::Patch, None);
}

#[test]
fn lsqml_position_matches_cpu() {
    matches_cpu(Some(drawn_offsets()), Sampling::Patch, None);
}

#[test]
fn lsqml_full_object_position_matches_cpu() {
    matches_cpu(Some(drawn_offsets()), Sampling::FullObject, None);
}

#[test]
fn lsqml_opr_matches_cpu() {
    matches_cpu(None, Sampling::Patch, Some((3, Refined::Both)));
}

#[test]
fn lsqml_opr_position_matches_cpu() {
    let opr = Some((3, Refined::Weights));
    matches_cpu(Some(drawn_offsets()), Sampling::Patch, opr);
}

#[test]
fn lsqml_opr_full_object_position_matches_cpu() {
    let opr = Some((2, Refined::Both));
    matches_cpu(Some(drawn_offsets()), Sampling::FullObject, opr);
}

/// The LLK after each of 12 epochs, tagged LSQML (epoch 11 keeps the preconditioner of
/// epoch 10), [`Lsqml::llk`]'s at the frames' own probes, the final arrays, the offsets' and
/// the variation's changes, the position entered as `sampling`, the probe varied by `opr`'s
/// eigenmodes and refined weights.
fn matches_cpu(position: Option<Vec<[f64; 2]>>, sampling: Sampling, opr: Option<(usize, Refined)>) {
    const NEPOCH: usize = 12;
    let features = moved(position);
    let frame = FUSED_SIDES[0] as usize;
    for (cost, batch_size) in runs() {
        let params = LsqmlParams {
            batch_size,
            cost,
            seed: 7,
        };
        let (scan, object, probe) = start(frame);
        let at = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let optics = Optics::far_field(&scan.geometry);
        let (mut o, mut p) = (object.clone(), probe.clone());
        let start = start_variation(&probe, opr);
        let mut var = start.clone();
        let mut cpu = Lsqml::new(params)
            .unwrap()
            .with_features(features.clone())
            .unwrap()
            .with_sampling(sampling);
        let want: Vec<_> = (0..NEPOCH)
            .map(|cycle| {
                cpu.epoch(
                    at.addr(),
                    at.observed(),
                    &optics,
                    slice::from_mut(&mut o),
                    &mut p,
                    &mut var,
                );
                let (addr, obs) = (at.addr(), at.observed());
                let llk = cpu.llk(addr, obs, &optics, slice::from_ref(&o), &p, &var);
                LlkRecord {
                    cycle,
                    algorithm: Algorithm::Lsqml,
                    llk,
                }
            })
            .collect();
        let mut gvar = start.clone();
        let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, OPTIONS).unwrap();
        gpu.set_features(&features).unwrap();
        gpu.set_sampling(sampling).unwrap();
        gpu.lsqml(&params, &mut gvar, NEPOCH, 1).unwrap();
        let got = gpu.history();
        assert_eq!(got.len(), want.len());
        let mut llk = 0.0f64;
        for (g, w) in got.iter().zip(&want) {
            assert_eq!((g.cycle, g.algorithm), (w.cycle, w.algorithm));
            for (a, b) in [
                (g.llk.poisson, w.llk.poisson),
                (g.llk.gaussian, w.llk.gaussian),
                (g.llk.euclidian, w.llk.euclidian),
                (g.llk.nb_photons_calc, w.llk.nb_photons_calc),
            ] {
                llk = worst_of([llk, rel(a, b)]);
            }
        }
        let [eigenmodes, weights, intensity] = variation_rel(&gvar, &var, &start);
        let errs = [
            llk,
            rel_l2(&gpu.object().unwrap(), &o),
            rel_l2(&gpu.probe().unwrap(), &p),
            offsets_rel(&gpu.features().unwrap(), cpu.features(), &features),
            eigenmodes,
            weights,
            intensity,
        ];
        eprintln!(
            "lsqml {cost:?} batch {batch_size} position {} {sampling:?} opr {opr:?}: llk {:.2e} \
             object {:.2e} probe {:.2e} offsets {:.2e} eigenmodes {:.2e} weights {:.2e} \
             intensity {:.2e}, poisson llk {:.4e} -> {:.4e}",
            features.position.is_some(),
            errs[0],
            errs[1],
            errs[2],
            errs[3],
            errs[4],
            errs[5],
            errs[6],
            want[0].llk.poisson,
            want[NEPOCH - 1].llk.poisson
        );
        let names = [
            "llk",
            "object",
            "probe",
            "offsets",
            "eigenmodes",
            "weights",
            "intensity",
        ];
        for (what, e) in names.into_iter().zip(errs) {
            assert!(
                e < 1e-4,
                "{cost:?} batch {batch_size} {sampling:?} opr {opr:?} {what}: {e:e}"
            );
        }
    }
}

/// The worst `max |a − b| / max |b|` of a batch's sizes of `a_far`, `a_obj` and `a_probe`;
/// the frames equal.
// Per frame, a probe size of 2.0e-4 in a batch of largest 0.91 read 2.4e-3 for an absolute
// error of 4.8e-7, the other sizes' own.
fn steps_rel(got: &[LsqmlStep], want: &[LsqmlStep]) -> [f64; 3] {
    assert_eq!(got.len(), want.len());
    let rel = |a: &[f64], b: &[f64]| {
        assert_eq!(a.len(), b.len());
        let d = a.iter().zip(b).map(|(a, b)| (a - b).abs());
        worst_of(d) / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
    };
    let mut worst = [0.0f64; 3];
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.frames, w.frames);
        let now = [
            rel(&g.a_far, &w.a_far),
            rel(&g.a_obj[0], &w.a_obj[0]),
            rel(&g.a_probe, &w.a_probe),
        ];
        for (w, n) in worst.iter_mut().zip(now) {
            *w = worst_of([*w, n]);
        }
    }
    worst
}

/// Every LSQML call starts a fresh [`Lsqml`], as on the CPU: its first epoch's steps match a
/// fresh CPU run from the arrays it starts at, after another LSQML call (whose graphs it
/// replays), after a BH call (which drops them), after modelling a position and after
/// modelling none again (each of which rebuilds the LSQML state). A zero batch size, a
/// modelled scale or background, held or refined, and frames off the fused sides fail.
#[test]
fn lsqml_restarts_and_rejects() {
    let frame = FUSED_SIDES[0] as usize;
    let params = LsqmlParams {
        batch_size: 4,
        cost: Cost::Poisson,
        seed: 7,
    };
    let bh = BhParams {
        batch_size: 4,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    };
    let (scan, object, probe) = start(frame);
    let optics = Optics::far_field(&scan.geometry);
    let mut none = ProbeVariation::none(probe.plane_shape());
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, OPTIONS).unwrap();
    gpu.lsqml(&params, &mut none, 2, 0).unwrap();
    for (case, position) in [
        ("after lsqml", None),
        ("after bh", None),
        ("position", Some(drawn_offsets())),
        ("no position again", None),
    ] {
        if case == "after bh" {
            gpu.bh(&bh, 1, 0).unwrap();
        }
        let features = moved(position);
        gpu.set_features(&features).unwrap();
        let (mut o, mut p) = (gpu.object().unwrap(), gpu.probe().unwrap());
        let at = Ptycho::new(&scan, o.clone(), p.clone()).unwrap();
        let mut var = ProbeVariation::none(p.plane_shape());
        let mut cpu = Lsqml::new(params)
            .unwrap()
            .with_features(features.clone())
            .unwrap();
        let want = cpu.epoch(
            at.addr(),
            at.observed(),
            &optics,
            slice::from_mut(&mut o),
            &mut p,
            &mut var,
        );
        gpu.lsqml(&params, &mut none, 1, 0).unwrap();
        let errs = [
            worst_of(steps_rel(&gpu.lsqml_steps().unwrap(), &want)),
            rel_l2(&gpu.object().unwrap(), &o),
            rel_l2(&gpu.probe().unwrap(), &p),
            offsets_rel(&gpu.features().unwrap(), cpu.features(), &features),
        ];
        eprintln!(
            "lsqml restart {case}: steps {:.2e} object {:.2e} probe {:.2e} offsets {:.2e}",
            errs[0], errs[1], errs[2], errs[3]
        );
        for (what, e) in ["steps", "object", "probe", "offsets"]
            .into_iter()
            .zip(errs)
        {
            assert!(e < 1e-4, "{case} {what}: {e:e}");
        }
    }

    let zero = LsqmlParams {
        batch_size: 0,
        ..params
    };
    assert!(gpu.lsqml(&zero, &mut none, 1, 1).is_err());
    for features in [
        Features {
            scale: Some(Feature::refined(vec![1.0; NPOS])),
            ..Features::default()
        },
        Features {
            background: Some(Feature::refined(vec![1.0; frame * frame])),
            ..Features::default()
        },
        Features {
            scale: Some(Feature::held(vec![1.0; NPOS])),
            ..Features::default()
        },
        Features {
            background: Some(Feature::held(vec![1.0; frame * frame])),
            ..Features::default()
        },
    ] {
        gpu.set_features(&features).unwrap();
        let err = gpu.lsqml(&params, &mut none, 1, 0).unwrap_err().to_string();
        assert!(
            err.contains("LSQML models no intensity scale or background"),
            "{features:?}: {err}"
        );
    }

    let (scan, object, probe) = start(32);
    let m1 = GpuOptions {
        fused: false,
        ..OPTIONS
    };
    let mut none = ProbeVariation::none(probe.plane_shape());
    let mut off = GpuPtycho::new(&ctx(), &scan, object, probe, m1).unwrap();
    assert!(off.lsqml(&params, &mut none, 1, 0).is_err());
}

/// A variation whose planes are not the probe's, of eigenmodes or of none, or whose weights
/// are not one row per frame fails before a run, leaving the variation, the arrays and the
/// history as they were; one on the M1 passes fails as LSQML there does.
#[test]
fn lsqml_refuses_variations_off_the_probe_or_frames() {
    let params = LsqmlParams {
        batch_size: 4,
        cost: Cost::Poisson,
        seed: 7,
    };
    let (scan, object, probe) = start(FUSED_SIDES[0] as usize);
    let [ny, nx] = probe.plane_shape();
    let wide = Probe::new(Stack::filled([2, ny, nx + 2], C64::new(1.0, 0.0)));
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe.clone(), OPTIONS).unwrap();
    let (o, p) = (gpu.object().unwrap(), gpu.probe().unwrap());
    let planes = format!("planes on a probe of {:?}", [ny, nx]);
    let rows = format!(
        "{} OPR weights for 2 eigenmodes over {NPOS} frames",
        2 * (NPOS - 1)
    );
    for (what, var, err) in [
        ("none", ProbeVariation::none([ny, nx + 2]), &planes),
        (
            "eigenmodes",
            variation(&wide, NPOS, 2, Refined::Weights),
            &planes,
        ),
        ("rows", variation(&probe, NPOS - 1, 2, Refined::Both), &rows),
    ] {
        let mut v = var.clone();
        let e = gpu.lsqml(&params, &mut v, 1, 1).unwrap_err().to_string();
        assert!(e.contains(err.as_str()), "{what}: {e}");
        assert_eq!(v, var, "{what}");
        assert!(gpu.object().unwrap() == o, "{what}");
        assert!(gpu.probe().unwrap() == p, "{what}");
        assert!(gpu.history().is_empty(), "{what}");
    }

    let (scan, object, probe) = start(32);
    let m1 = GpuOptions {
        fused: false,
        ..OPTIONS
    };
    let mut var = variation(&probe, NPOS, 2, Refined::Weights);
    let mut off = GpuPtycho::new(&ctx(), &scan, object, probe, m1).unwrap();
    assert!(off.lsqml(&params, &mut var, 1, 0).is_err());
}

/// The bits of a GPU LSQML run with a position.
#[derive(Debug, PartialEq)]
struct RunBits {
    variation: ProbeVariation,
    object: Vec<[u64; 2]>,
    probe: Vec<[u64; 2]>,
    offsets: Vec<u64>,
    /// Per batch, its frames, then `a_far`, `a_obj` and `a_probe`.
    steps: Vec<(Vec<usize>, Vec<u64>)>,
    llk: Vec<[u64; 4]>,
}

fn bits(s: &Stack<C64>) -> Vec<[u64; 2]> {
    s.as_slice()
        .iter()
        .map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect()
}

fn run_bits(g: &GpuPtycho, variation: ProbeVariation) -> RunBits {
    let f64s = |v: &[f64]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    RunBits {
        variation,
        object: bits(&g.object().unwrap()),
        probe: bits(&g.probe().unwrap()),
        offsets: f64s(&g.features().unwrap().position.unwrap().values.concat()),
        steps: g
            .lsqml_steps()
            .unwrap()
            .into_iter()
            .map(|s| {
                (
                    s.frames,
                    f64s(&[s.a_far, s.a_obj.concat(), s.a_probe].concat()),
                )
            })
            .collect(),
        llk: g
            .history()
            .iter()
            .map(|r| {
                let l = r.llk;
                [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc].map(f64::to_bits)
            })
            .collect(),
    }
}

/// With a position, without and with a probe variation, 3 epochs leave the same bits in
/// stacks of 8, 4 and every frame, and eagerly in stacks of 8.
#[test]
fn lsqml_position_is_bit_identical_across_stack_splits() {
    let frame = FUSED_SIDES[0] as usize;
    let features = moved(Some(drawn_offsets()));
    let opr = [None, Some((3, Refined::Both))];
    for (opr, (cost, batch_size)) in opr.into_iter().flat_map(|o| runs().map(move |r| (o, r))) {
        let params = LsqmlParams {
            batch_size,
            cost,
            seed: 7,
        };
        let run = |stack_size, graph| {
            let (scan, object, probe) = start(frame);
            let options = GpuOptions {
                stack_size: Some(stack_size),
                graph,
                ..OPTIONS
            };
            let mut var = start_variation(&probe, opr);
            let mut g = GpuPtycho::new(&ctx(), &scan, object, probe, options).unwrap();
            g.set_features(&features).unwrap();
            g.lsqml(&params, &mut var, 3, 1).unwrap();
            run_bits(&g, var)
        };
        let eight = run(8, true);
        assert_eq!(eight.llk.len(), 3);
        for (stack_size, graph) in [(4, true), (NPOS, true), (8, false)] {
            let other = run(stack_size, graph);
            let what = format!(
                "{cost:?} batch {batch_size} opr {opr:?} stacks of {stack_size} graph {graph}"
            );
            assert!(other.steps == eight.steps, "{what}: steps differ");
            assert!(other.llk == eight.llk, "{what}: LLK records differ");
            assert!(other.object == eight.object, "{what}: object differs");
            assert!(other.probe == eight.probe, "{what}: probe differs");
            assert!(other.offsets == eight.offsets, "{what}: offsets differ");
            assert!(
                other.variation == eight.variation,
                "{what}: variation differs"
            );
        }
    }
}

/// A second OPR call re-uploads the variation it is given; after a call without one and under
/// another shape too. Each call's first epoch against a fresh CPU run from the arrays and
/// variation it starts at, with its LLK.
#[test]
fn lsqml_opr_restarts_match_cpu() {
    let frame = FUSED_SIDES[0] as usize;
    let params = LsqmlParams {
        batch_size: 12,
        cost: Cost::Poisson,
        seed: 7,
    };
    let (scan, object, probe) = start(frame);
    let optics = Optics::far_field(&scan.geometry);
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe.clone(), OPTIONS).unwrap();
    let mut gvar = variation(&probe, NPOS, 3, Refined::Both);
    gpu.lsqml(&params, &mut gvar, 2, 0).unwrap();
    for case in [
        "opr again",
        "after none",
        "other shape",
        "other shape again",
    ] {
        if case == "after none" {
            let mut none = ProbeVariation::none(probe.plane_shape());
            gpu.lsqml(&params, &mut none, 1, 0).unwrap();
        }
        let (mut o, mut p) = (gpu.object().unwrap(), gpu.probe().unwrap());
        if case == "other shape" {
            gvar = variation(&p, NPOS, 2, Refined::Weights);
        }
        let start = gvar.clone();
        let at = Ptycho::new(&scan, o.clone(), p.clone()).unwrap();
        let mut var = start.clone();
        let mut cpu = Lsqml::new(params).unwrap();
        let want = cpu.epoch(
            at.addr(),
            at.observed(),
            &optics,
            slice::from_mut(&mut o),
            &mut p,
            &mut var,
        );
        let llk = cpu.llk(
            at.addr(),
            at.observed(),
            &optics,
            slice::from_ref(&o),
            &p,
            &var,
        );
        let mut g = start.clone();
        gpu.lsqml(&params, &mut g, 1, 1).unwrap();
        let got = gpu.history().last().unwrap().llk;
        let [e, w, c] = variation_rel(&g, &var, &start);
        let errs = [
            worst_of(steps_rel(&gpu.lsqml_steps().unwrap(), &want)),
            rel_l2(&gpu.object().unwrap(), &o),
            rel_l2(&gpu.probe().unwrap(), &p),
            rel(got.poisson, llk.poisson),
            e,
            w,
            c,
        ];
        eprintln!(
            "lsqml opr restart {case}: steps {:.2e} object {:.2e} probe {:.2e} llk {:.2e} \
             eigenmodes {:.2e} weights {:.2e} intensity {:.2e}",
            errs[0], errs[1], errs[2], errs[3], errs[4], errs[5], errs[6]
        );
        for (what, e) in [
            "steps",
            "object",
            "probe",
            "llk",
            "eigenmodes",
            "weights",
            "intensity",
        ]
        .into_iter()
        .zip(errs)
        {
            assert!(e < 1e-4, "{case} {what}: {e:e}");
        }
        gvar = g;
    }
}
