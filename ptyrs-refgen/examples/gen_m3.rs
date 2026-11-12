//! The M3 reference datasets (DESIGN §7 M3) for the pty-chi runs in `refs/ptychi/`:
//!
//!     cargo run --release -p ptyrs-refgen --example gen_m3 -- <out_dir> [photons_per_frame]
//!
//! 256² frames, 1024 frames, 2 probe modes, step 16 object pixels; one feature per dataset.

use ptyrs_core::Geometry;
use ptyrs_refgen::{Background, Pattern, Spec, generate, write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .ok_or("usage: gen_m3 <out_dir> [photons_per_frame]")?;
    let photons: f64 = args.next().map_or(Ok(1e6), |s| s.parse())?;
    let geometry = Geometry::new(1.5e-10, 1.0, 75e-6, [256, 256])?;
    let base = Spec {
        name: String::new(),
        geometry,
        pattern: Pattern::Fermat { n: 1024 },
        step_px: 16.0,
        nmode: 2,
        init_nmode: 2,
        photons_per_frame: photons,
        seed: 0,
        position_error_px: 0.0,
        intensity_sigma: 0.0,
        background: None,
        fly_px: vec![0],
        opr_amplitude: 0.0,
    };
    let raster = Pattern::Raster { ny: 32, nx: 32 };
    let specs = [
        Spec {
            name: "raster_clean".into(),
            pattern: raster,
            seed: 1,
            ..base.clone()
        },
        Spec {
            name: "fermat_clean".into(),
            seed: 2,
            ..base.clone()
        },
        Spec {
            name: "fermat_poserr".into(),
            seed: 3,
            position_error_px: 2.0,
            ..base.clone()
        },
        Spec {
            name: "fermat_intvar".into(),
            seed: 4,
            intensity_sigma: 0.15,
            ..base.clone()
        },
        Spec {
            name: "fermat_bg".into(),
            seed: 5,
            background: Some(Background {
                flat: 0.5,
                peak: 10.0,
                width_px: 30.0,
            }),
            ..base.clone()
        },
        Spec {
            name: "raster_fly".into(),
            pattern: raster,
            seed: 6,
            fly_px: vec![-4, -2, 0, 2, 4],
            init_nmode: 6,
            ..base.clone()
        },
        Spec {
            name: "fermat_opr".into(),
            seed: 7,
            opr_amplitude: 0.3,
            ..base.clone()
        },
    ];
    for spec in &specs {
        let t = std::time::Instant::now();
        let g = generate(spec)?;
        write(&out, &g)?;
        let m = &g.meta;
        println!(
            "{:14} frames {} obj {} max count {} mean/frame {:.4e} llk_truth poisson {:.5} ({:.1}s)",
            spec.name,
            m["nframe"],
            m["obj_shape"],
            m["counts"]["max"],
            m["counts"]["mean_per_frame"].as_f64().unwrap_or(f64::NAN),
            m["llk_truth"]["poisson"].as_f64().unwrap_or(f64::NAN),
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
