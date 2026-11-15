//! CPU DM and RAAR under [`Sampling::FullObject`] on the near-field cases of
//! `tests/common/near_cases.rs`, the Rust side of `refs/py/ptychi_full_object_dm.py`.
//!
//! `cargo run --release -p ptyrs --example engine_fullobj_dump -- <out> [ncycle...]`
//! (default 8 200) writes per case `<out>/<case>/`: `meta.txt` (`key value` lines),
//! `counts.f64` `[nframe, ny, nx]` natural layout, `corners.f64` `[nframe, 2]`, `obj0.c128`,
//! `probe0.c128`, `truth.c128`, `probe_truth.c128`, `offsets0.f64` `[nframe, 2]` (the drawn
//! offsets) and per run, without and with the drawn offsets as `position`,
//! `<alg>_<fixed|position>_<ncycle>_{llk,offsets}.f64`, `_{obj,probe}.c128`: the LLK history
//! (each cycle's start) then the LLK of the arrays the run ends on. Little-endian f64; complex
//! as `re im` pairs. `tests/engine_gpu_near.rs`'s DM and RAAR without the closing
//! `zero_phase_ramp`, which runs after the last cycle only and, under `position`, changes the
//! LLK of the state it ends on.

use std::fs;
use std::path::Path;
use std::time::Instant;

use ptyrs::base::{C64, sim};
use ptyrs::engine::{DmParams, Features, Ptycho, RaarParams, Sampling};

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
        let counts = (0..nframe).flat_map(|f| {
            (c.scan.counts.frame(f).iter()).map(|&v| ptyrs::base::Counts::intensity(v).unwrap())
        });
        put(&dir.join("counts.f64"), counts);
        let corners = (0..nframe).flat_map(|f| {
            let v = &c.addr.frame_views(f)[0];
            [v.oy as f64, v.ox as f64]
        });
        put(&dir.join("corners.f64"), corners);
        put_c(&dir.join("obj0.c128"), c.object.as_slice());
        put_c(&dir.join("probe0.c128"), c.probe.as_slice());
        put_c(&dir.join("truth.c128"), sim::object([oy, ox]).as_slice());
        let mut truth_probe = sim::probe([ny, nx], nmode).unwrap();
        (truth_probe.as_mut_slice().iter_mut()).for_each(|v| *v *= near::PHOTONS.sqrt());
        put_c(&dir.join("probe_truth.c128"), truth_probe.as_slice());
        put(
            &dir.join("offsets0.f64"),
            near::drawn_offsets().into_iter().flatten(),
        );
        for &ncycle in &ncycles {
            for alg in ["dm", "raar"] {
                for label in ["fixed", "position"] {
                    let (object, probe) = (c.object.clone(), c.probe.clone());
                    let mut cpu = Ptycho::with_optics(&c.scan, c.optics(), object, probe).unwrap();
                    cpu.set_sampling(Sampling::FullObject);
                    if label == "position" {
                        let f = Features {
                            position: Some(near::drawn_offsets()),
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
                        .unwrap_or(vec![[0.0; 2]; nframe]);
                    put(
                        &dir.join(format!("{stem}_offsets.f64")),
                        d.into_iter().flatten(),
                    );
                    println!(
                        "{slug} {stem}: llk {:.4} -> {:.4} (min {:.4} at {}), {secs:.2} s",
                        llk[0],
                        llk[ncycle],
                        llk.iter().copied().fold(f64::INFINITY, f64::min),
                        llk.iter()
                            .position(|&v| v == llk.iter().copied().fold(f64::INFINITY, f64::min))
                            .unwrap()
                    );
                }
            }
        }
    }
}
