//! CPU DM and RAAR under [`Sampling::FullObject`] on the near-field cases of
//! `tests/common/near_cases.rs`, the Rust side of `refs/py/ptychi_full_object_dm.py`.
//!
//! `cargo run --release -p ptyrs --example engine_fullobj_dump -- <out> [ncycle...]`
//! (default 8 200) writes per case `<out>/<case>/`: `meta.txt` (`key value` lines),
//! `counts.f64` `[nframe, ny, nx]` natural layout, `corners.f64` `[nframe, 2]`, `obj0.c128`,
//! `probe0.c128`, `truth.c128`, `probe_truth.c128`, `offsets0.f64` `[nframe, 2]` (the drawn
//! offsets), `offsets_whole.f64` (whole-pixel ones: `round(1.4 sin(1.7f + 0.3))`,
//! `round(1.4 cos(2.3f))`, mean kept) and `counts_truth_<drawn|whole>.f64`, the truth's
//! intensities at the corners plus those offsets under `FullObject`, rounded half away from
//! zero. Per run, `<alg>_<label>_<ncycle>_{llk,offsets}.f64`, `_{obj,probe}.c128`: the LLK
//! history (each cycle's start) then the LLK of the arrays the run ends on. From the start
//! object and probe on `counts.f64`, label `fixed` (no offsets), `position` (the drawn offsets,
//! refined) and `held` (the drawn offsets, held); from the truth on
//! `counts_truth_<drawn|whole>.f64`, `truth_<drawn|whole>` (those offsets, held).
//! Little-endian f64; complex as `re im` pairs. `tests/engine_gpu_near.rs`'s DM and RAAR
//! without the closing `zero_phase_ramp`, which runs after the last cycle only and, with
//! offsets, changes the LLK of the state it ends on.

use std::fs;
use std::path::Path;
use std::slice;
use std::time::Instant;

use ptyrs::base::{C64, Counts, Object, Probe, Scan, sim};
use ptyrs::engine::{DmParams, Feature, Features, Ptycho, RaarParams, Sampling, SliceWaves, shift};

#[path = "../tests/common/near_cases.rs"]
mod near;

const DM: DmParams = DmParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    alpha: 0.02,
    loop_obj_probe: 1,
    calc_llk: 1,
    zero_phase_ramp: false,
};

const RAAR: RaarParams = RaarParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    beta: 0.9,
    loop_obj_probe: 1,
    calc_llk: 1,
    zero_phase_ramp: false,
};

fn put(path: &Path, v: impl IntoIterator<Item = f64>) {
    let b: Vec<u8> = v.into_iter().flat_map(f64::to_le_bytes).collect();
    fs::write(path, b).unwrap();
}

fn put_c(path: &Path, v: &[C64]) {
    put(path, v.iter().flat_map(|c| [c.re, c.im]));
}

fn whole_offsets(n: usize) -> Vec<[f64; 2]> {
    let d = |f: f64| [1.4 * (1.7 * f + 0.3).sin(), 1.4 * (2.3 * f).cos()];
    (0..n).map(|f| d(f as f64).map(f64::round)).collect()
}

/// `c`'s scan with the counts of `object` and `probe` at the corners plus `offsets`.
fn truth_scan(c: &near::Case, object: &Object, probe: &Probe, offsets: &[[f64; 2]]) -> Scan {
    let optics = c.optics();
    let views = c.addr.views();
    let objects = slice::from_ref(object);
    let at = Some(offsets);
    let waves = SliceWaves::with_sampling(&optics, views, objects, probe, at, Sampling::FullObject);
    let mut psi = waves.psi().to_vec();
    optics.exit().forward(&mut psi);
    let [nframe, ny, nx] = c.scan.counts.dims();
    let (n, nmode) = (ny * nx, c.addr.nmode());
    let mut counts = vec![0.0; nframe * n];
    for (f, block) in psi.chunks_exact(nmode * n).enumerate() {
        for j in 0..n {
            let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
            counts[f * n + shift(j, ny, nx)] = icalc.round();
        }
    }
    let counts = Counts::from_f64([nframe, ny, nx], &counts, None).unwrap();
    Scan {
        counts,
        ..c.scan.clone()
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(
        args.next()
            .expect("usage: engine_fullobj_dump <out> [ncycle...]"),
    );
    let mut ncycles: Vec<usize> = args.map(|a| a.parse().unwrap()).collect();
    if ncycles.is_empty() {
        ncycles = vec![8, 200];
    }
    let slugs = [("as", 1e-4), ("fresnel", 5e-3), ("fresnel_back", -1e-2)];
    for ((slug, distance), c) in slugs.into_iter().zip(near::cases()) {
        let dir = out.join(slug);
        fs::create_dir_all(&dir).unwrap();
        let (nframe, nmode) = (c.addr.nframe(), c.addr.nmode());
        let [_, oy, ox] = c.object.dims();
        let [_, ny, nx] = c.scan.counts.dims();
        let meta = format!(
            "name {}\nwavelength {:e}\npixel {:e}\ndistance {:e}\nnframe {nframe}\nnmode {nmode}\n\
             ny {ny}\nnx {nx}\noy {oy}\nox {ox}\nnb_obs {}\n",
            c.name.replace(' ', "_"),
            near::WAVELENGTH,
            near::PIXEL,
            distance,
            c.obs.nb_obs()
        );
        fs::write(dir.join("meta.txt"), meta).unwrap();
        let put_counts = |name: &str, scan: &Scan| {
            let counts = (0..nframe).flat_map(|f| {
                (scan.counts.frame(f).iter()).map(|&v| Counts::intensity(v).unwrap())
            });
            put(&dir.join(name), counts);
        };
        put_counts("counts.f64", &c.scan);
        let corners = (0..nframe).flat_map(|f| {
            let v = &c.addr.frame_views(f)[0];
            [v.oy as f64, v.ox as f64]
        });
        put(&dir.join("corners.f64"), corners);
        put_c(&dir.join("obj0.c128"), c.object.as_slice());
        put_c(&dir.join("probe0.c128"), c.probe.as_slice());
        let truth = sim::object([oy, ox]);
        put_c(&dir.join("truth.c128"), truth.as_slice());
        let mut truth_probe = sim::probe([ny, nx], nmode).unwrap();
        (truth_probe.as_mut_slice().iter_mut()).for_each(|v| *v *= near::PHOTONS.sqrt());
        put_c(&dir.join("probe_truth.c128"), truth_probe.as_slice());
        let (drawn, whole) = (near::drawn_offsets(), whole_offsets(nframe));
        put(&dir.join("offsets0.f64"), drawn.iter().flatten().copied());
        put(
            &dir.join("offsets_whole.f64"),
            whole.iter().flatten().copied(),
        );
        let truth_drawn = truth_scan(&c, &truth, &truth_probe, &drawn);
        let truth_whole = truth_scan(&c, &truth, &truth_probe, &whole);
        put_counts("counts_truth_drawn.f64", &truth_drawn);
        put_counts("counts_truth_whole.f64", &truth_whole);
        let start = (&c.object, &c.probe);
        let from_truth = (&truth, &truth_probe);
        let runs = [
            ("fixed", &c.scan, start, None),
            ("position", &c.scan, start, Some(Feature::refined(&drawn))),
            ("held", &c.scan, start, Some(Feature::held(&drawn))),
            (
                "truth_drawn",
                &truth_drawn,
                from_truth,
                Some(Feature::held(&drawn)),
            ),
            (
                "truth_whole",
                &truth_whole,
                from_truth,
                Some(Feature::held(&whole)),
            ),
        ];
        for &ncycle in &ncycles {
            for alg in ["dm", "raar"] {
                for (label, scan, (object, probe), position) in runs {
                    let (object, probe) = (object.clone(), probe.clone());
                    let mut cpu = Ptycho::with_optics(scan, c.optics(), object, probe).unwrap();
                    cpu.set_sampling(Sampling::FullObject);
                    if let Some(p) = position {
                        let f = Features {
                            position: Some(p.map(Vec::clone)),
                            ..Features::default()
                        };
                        cpu.set_features(f).unwrap();
                    }
                    let start = cpu.llk().poisson;
                    let t = Instant::now();
                    match alg {
                        "dm" => cpu.dm(&DM, ncycle),
                        _ => cpu.raar(&RAAR, ncycle),
                    }
                    let secs = t.elapsed().as_secs_f64();
                    let mut llk: Vec<f64> = cpu.history().iter().map(|r| r.llk.poisson).collect();
                    assert_eq!(llk.len(), ncycle);
                    assert_eq!(llk[0], start, "{slug} {alg} {label}: first record");
                    llk.push(cpu.llk().poisson);
                    let stem = format!("{alg}_{label}_{ncycle}");
                    put(&dir.join(format!("{stem}_llk.f64")), llk.iter().copied());
                    put_c(
                        &dir.join(format!("{stem}_obj.c128")),
                        cpu.object().as_slice(),
                    );
                    put_c(
                        &dir.join(format!("{stem}_probe.c128")),
                        cpu.probe().as_slice(),
                    );
                    let d = cpu
                        .features()
                        .position
                        .clone()
                        .map_or(vec![[0.0; 2]; nframe], |p| p.values);
                    put(
                        &dir.join(format!("{stem}_offsets.f64")),
                        d.into_iter().flatten(),
                    );
                    let min = llk.iter().copied().fold(f64::INFINITY, f64::min);
                    println!(
                        "{slug} {stem}: llk {:.4} -> {:.4} (min {min:.4} at {}), {secs:.2} s",
                        llk[0],
                        llk[ncycle],
                        llk.iter().position(|&v| v == min).unwrap()
                    );
                }
            }
        }
    }
}
