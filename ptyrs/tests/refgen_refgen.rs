use std::f64::consts::PI;

use ptyrs::base::{C64, Counts, Geometry, MASKED, sim};
use ptyrs::refgen::{
    Background, Far, Pattern, Spec, TILE, generate, llk, quad_phase, textured_object, write,
};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// `Far` against the centred DFT it implements: detector pixel `k` holds
/// `|Σ_m a[m] e^{-2πi (k-n/2)(m-n/2)/n}|² / N` per axis, `a = P·O_win·q`.
#[test]
fn far_matches_centred_dft() {
    let g = Geometry::new(1e-10, 0.5, 55e-6, [6, 8]).unwrap();
    let [ny, nx] = g.frame;
    let (nyo, nxo) = (9, 11);
    let mut s = 7;
    let mut c =
        |n: usize| -> Vec<C64> { (0..n).map(|_| C64::new(lcg(&mut s), lcg(&mut s))).collect() };
    let probes = [c(ny * nx), c(ny * nx)];
    let object = c(nyo * nxo);
    let corner = [2, 3];
    let mut out = vec![0.5; ny * nx];
    let refs: Vec<&[C64]> = probes.iter().map(|p| p.as_slice()).collect();
    Far::new(&g).accumulate(&refs, &object, nxo, corner, 2.0, &mut out);
    let q = quad_phase(&g);
    for ky in 0..ny {
        for kx in 0..nx {
            let mut want = 0.0;
            for p in &probes {
                let mut acc = C64::default();
                for y in 0..ny {
                    for x in 0..nx {
                        let i = y * nx + x;
                        let a = p[i] * object[(corner[0] + y) * nxo + corner[1] + x] * q[i];
                        let ph = -2.0
                            * PI
                            * ((ky as f64 - (ny / 2) as f64) * (y as f64 - (ny / 2) as f64)
                                / ny as f64
                                + (kx as f64 - (nx / 2) as f64) * (x as f64 - (nx / 2) as f64)
                                    / nx as f64);
                        acc += a * C64::from_polar(1.0, ph);
                    }
                }
                want += acc.norm_sqr() / (ny * nx) as f64;
            }
            let got = out[ky * nx + kx];
            assert!(
                (got - (0.5 + 2.0 * want)).abs() < 1e-12,
                "({ky},{kx}) {got} vs {want}"
            );
        }
    }
}

/// PyNX formulas term by term; the masked pixel is skipped and not counted in `nb_obs`.
#[test]
fn llk_skips_masked_and_normalises() {
    let counts = Counts::from_raw([1, 1, 3], vec![4, 0, MASKED]).unwrap();
    let icalc = [2.0, 0.5, 9.0];
    let l = llk(&icalc, &counts);
    assert_eq!(l.nb_obs, 2);
    let p = (2.0 - 4.0 + 4.0 * (4.0f64 / 2.0).ln()) + 0.5;
    let g = (4.0 - 2.0f64).powi(2) / 5.0 + 0.25 / 1.0;
    let e = 4.0 * (2.0 - 2f64.sqrt()).powi(2) + 4.0 * 0.5;
    assert!((l.poisson - p / 2.0).abs() < 1e-15);
    assert!((l.gaussian - g / 2.0).abs() < 1e-15);
    assert!((l.euclidian - e / 2.0).abs() < 1e-15);
}

/// The tile is flat at its border, so tiling it leaves no seam.
#[test]
fn tile_border_is_flat() {
    let t = sim::object([TILE, TILE]);
    let t = t.plane(0);
    let edge = (0..TILE).flat_map(|i| [i, (TILE - 1) * TILE + i, i * TILE, i * TILE + TILE - 1]);
    for i in edge {
        assert!(
            (t[i] - C64::new(0.95, 0.0)).norm() < 3.4e-3,
            "pixel {i}: {}",
            t[i]
        );
    }
    let o = textured_object([2 * TILE + 4, TILE + 2]);
    assert_eq!(o.plane_shape(), [2 * TILE + 4, TILE + 2]);
}

fn small(name: &str) -> Spec {
    Spec {
        name: name.into(),
        geometry: Geometry::new(1.5e-10, 1.0, 75e-6, [32, 32]).unwrap(),
        pattern: Pattern::Raster { ny: 3, nx: 4 },
        step_px: 5.0,
        nmode: 2,
        init_nmode: 3,
        photons_per_frame: 1e4,
        seed: 11,
        position_error_px: 1.5,
        intensity_sigma: 0.2,
        background: Some(Background {
            flat: 0.5,
            peak: 2.0,
            width_px: 4.0,
        }),
        fly_px: vec![-1, 0, 1],
        opr_amplitude: 0.2,
    }
}

/// Every feature on: deterministic, true positions on whole pixels after PyNX preprocessing,
/// nominal ones off them, and the four files read back as written.
#[test]
fn generate_writes_consistent_files() {
    let spec = small("small");
    let a = generate(&spec).unwrap();
    let b = generate(&spec).unwrap();
    assert_eq!(a.scan.counts, b.scan.counts);
    let m = &a.meta;
    let nframe = 12;
    assert_eq!(m["nframe"], nframe);
    let px =
        |k: &str| -> Vec<[f64; 2]> { serde_json::from_value(m["positions"][k].clone()).unwrap() };
    assert!(px("true_pp_px").iter().flatten().all(|v| v.fract() == 0.0));
    assert!(
        px("nominal_pp_px")
            .iter()
            .flatten()
            .any(|v| v.fract().abs() > 1e-3)
    );
    let scales: Vec<f64> = serde_json::from_value(m["intensity_scale"].clone()).unwrap();
    assert_eq!(scales.len(), nframe);
    assert!(scales.iter().any(|s| (s - 1.0).abs() > 0.05));
    assert_eq!(m["opr"]["weights"].as_array().unwrap().len(), nframe);
    // Truth probe in counts: Σ|P|²·⟨|O|²⟩ is the clean mean frame, here ≈ photons_per_frame.
    let p2: f64 = a.truth.1.0.as_slice().iter().map(|c| c.norm_sqr()).sum();
    assert!(p2 > 1e4 && p2 < 1e5, "truth probe power {p2}");
    assert_eq!(a.init.1.nmode(), 3);
    assert_eq!(a.init.0.plane_shape(), a.truth.0.plane_shape());

    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("refgen_small");
    write(&dir, &a).unwrap();
    let scan = ptyrs::io::read_cxi(dir.join("small.cxi")).unwrap();
    assert_eq!(scan.counts, a.scan.counts);
    for (r, w) in scan.positions.x.iter().zip(&a.scan.positions.x) {
        assert!((r - w).abs() <= 1e-7 * w.abs().max(1e-12), "{r} vs {w}");
    }
    let (o, p) = ptyrs::io::read_modes(dir.join("small_truth.h5")).unwrap();
    assert_eq!((o, p), a.truth);
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("small.json")).unwrap()).unwrap();
    assert_eq!(&json, m);
}
