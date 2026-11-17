//! rPIE's position step floors the largest denominator of its batch at pty-chi's `1e-6`
//! times the propagator's intensity scale on the GPU as on the CPU. The case is one where the
//! floor's units decide the step: a phase object `1e-4` of the Siemens star's, so weak that
//! every batch's largest denominator (1.1e-4 to 3.4e-4) lies below `1e-6 ny nx` (4.1e-3 at
//! 64 x 64) and above `1e-6 k`; the GPU's floor at the frame's pixel count moved its offsets
//! up to 0.19 px off the CPU's under each propagator. The CPU's own floor binds only where the
//! rounded counts' noise drives every step to the limit, which hides the floor's value.

use std::slice;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Counts, Geometry, Object, Probe, Scan, Stack, required_obj_shape, sim,
};
use ptyrs::engine::{
    Feature, Features, GpuOptions, GpuPtycho, Optics, Ptycho, RpieParams, SliceWaves, Wavefield,
    shift,
};
use ptyrs::gpu::FUSED_SIDES;

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const NPOS: usize = 30;
const NPROBE: usize = 2;
const WAVELENGTH: f64 = 1e-10;
const PIXEL: f64 = 50e-9;
/// Photons the probe sends through a frame.
const PHOTONS: f64 = 2e5;
/// The truth's phase over the Siemens star's.
const PHASE: f64 = 1e-4;

/// Positions only: the object and probe stay the truth's, so every epoch's denominators are
/// the first's.
const RPIE: RpieParams = RpieParams {
    obj_step: None,
    probe_step: None,
    obj_alpha: 0.5,
    probe_alpha: 0.5,
    batch_size: 4,
    calc_llk: 1,
    seed: 7,
};

/// The scan of the weak truth at `distance`, its optics, the truth and the probe; counts the
/// truth's intensities rounded.
fn weak(distance: f64) -> (Scan, Optics, Object, Probe) {
    let frame = FUSED_SIDES[0] as usize;
    let pixel = [PIXEL, PIXEL];
    let field = Wavefield::new([frame, frame], WAVELENGTH, pixel, distance).unwrap();
    let optics = Optics::free_space(&field);
    let raw = sim::fermat(NPOS, frame as f64 / 8.0 * PIXEL);
    let positions = raw.pynx_preprocess(pixel);
    let shape = required_obj_shape(&positions, pixel, [frame, frame], 16);
    let addr = AddrTable::new(&positions, pixel, [frame, frame], shape, NPROBE, 1).unwrap();
    let star = sim::object(shape);
    let data = (star.as_slice().iter())
        .map(|v| C64::from_polar(1.0, PHASE * v.arg()))
        .collect();
    let truth = Object::new(Stack::from_vec(star.dims(), data).unwrap());
    let mut probe = sim::probe([frame, frame], NPROBE).unwrap();
    (probe.as_mut_slice().iter_mut()).for_each(|v| *v *= PHOTONS.sqrt());

    let waves = SliceWaves::new(&optics, addr.views(), slice::from_ref(&truth), &probe, None);
    let mut psi = waves.psi().to_vec();
    optics.exit().forward(&mut psi);
    let n = frame * frame;
    let mut counts = vec![0.0; addr.nframe() * n];
    for (f, block) in psi.chunks_exact(NPROBE * n).enumerate() {
        for j in 0..n {
            let icalc: f64 = (0..NPROBE).map(|m| block[m * n + j].norm_sqr()).sum();
            counts[f * n + shift(j, frame, frame)] = icalc.round();
        }
    }
    let counts = Counts::from_f64([addr.nframe(), frame, frame], &counts, None).unwrap();
    let geometry = Geometry::new(WAVELENGTH, 1.0, PIXEL, [frame, frame]).unwrap();
    let scan = Scan::new(geometry, raw, counts).unwrap();
    (scan, optics, truth, probe)
}

/// Two epochs from drawn offsets; the GPU's offsets against the CPU's, over the CPU's moves.
#[test]
fn rpie_position_floor_matches_cpu_under_every_propagator() {
    // Measured worst: 4.8e-3 (angular spectrum), 1.3e-3, 3.0e-3; 0.97 under angular
    // spectrum with the GPU's floor at the frame's pixel count.
    const TOL: f64 = 1.5e-2;
    let start: Vec<[f64; 2]> = (0..NPOS)
        .map(|f| {
            let f = f as f64;
            [0.4 * (1.7 * f + 0.3).sin(), 0.4 * (2.3 * f).cos()]
        })
        .collect();
    let moved = Features {
        position: Some(Feature::refined(start.clone())),
        ..Features::default()
    };
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let options = GpuOptions {
        stack_size: Some(8),
        graph: false,
        llk_slots: 3,
        fused: true,
        psi_f16: false,
    };
    let cases = [
        ("angular spectrum", 1e-4),
        ("Fresnel", 5e-3),
        ("Fresnel back", -1e-2),
    ];
    for (name, distance) in cases {
        let (scan, optics, truth, probe) = weak(distance);
        let mut gpu =
            GpuPtycho::with_optics(&ctx, &scan, &optics, truth.clone(), probe.clone(), options)
                .unwrap();
        gpu.set_features(&moved).unwrap();
        gpu.rpie(&RPIE, 2).unwrap();
        let mut cpu = Ptycho::with_optics(&scan, optics, truth, probe).unwrap();
        cpu.set_features(moved.clone()).unwrap();
        cpu.rpie(&RPIE, 2).unwrap();

        let got = gpu.features().unwrap().position.unwrap().values;
        let want = cpu.features().position.clone().unwrap().values;
        let flat = |d: &[[f64; 2]]| d.iter().flatten().copied().collect::<Vec<_>>();
        let (got, want, start) = (flat(&got), flat(&want), flat(&start));
        let off = worst_of(got.iter().zip(&want).map(|(g, w)| (g - w).abs()));
        let steps = worst_of(want.iter().zip(&start).map(|(w, s)| (w - s).abs()));
        eprintln!("{name}: offsets {:.2e} of the CPU's moves", off / steps);
        assert!(off <= TOL * steps, "{name}: offsets {off:.2e}");
    }
}
