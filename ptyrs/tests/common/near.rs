//! The near-field cases the GPU engines are held to, on the fused passes' smallest side: an
//! angular-spectrum propagator and a Fresnel transform forward and back. Two incoherent probe
//! modes, 30 Fermat positions, counts the truth's intensities rounded.

use std::slice;
use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Counts, Geometry, Object, Probe, Scan, Stack, required_obj_shape,
};
use ptyrs::engine::{GpuOptions, GpuPtycho, Observed, Optics, SliceWaves, Wavefield, llk, shift};
use ptyrs::gpu::FUSED_SIDES;

pub const NPOS: usize = 30;
pub const NPROBE: usize = 2;
pub const WAVELENGTH: f64 = 1e-10;
pub const PIXEL: f64 = 50e-9;
/// Photons the probe sends through a frame.
pub const PHOTONS: f64 = 2e5;

pub const OPTIONS: GpuOptions = GpuOptions {
    stack_size: Some(8),
    graph: true,
    llk_slots: 3,
    fused: true,
    psi_f16: false,
};

pub struct Case {
    pub name: &'static str,
    pub scan: Scan,
    pub addr: AddrTable,
    pub obs: Observed,
    /// From the object to the detector, in metres.
    distance: f64,
    pub object: Object,
    pub probe: Probe,
}

/// Detector-plane exit waves of every view.
pub fn propagated(addr: &AddrTable, optics: &Optics, object: &Object, probe: &Probe) -> Vec<C64> {
    let mut psi = SliceWaves::new(optics, addr.views(), slice::from_ref(object), probe, None)
        .psi()
        .to_vec();
    optics.exit().forward(&mut psi);
    psi
}

fn free_space(distance: f64) -> Optics {
    let frame = FUSED_SIDES[0] as usize;
    let field = Wavefield::new([frame, frame], WAVELENGTH, [PIXEL, PIXEL], distance).unwrap();
    Optics::free_space(&field)
}

pub fn case(name: &'static str, distance: f64) -> Case {
    let frame = FUSED_SIDES[0] as usize;
    let pixel = [PIXEL, PIXEL];
    let optics = free_space(distance);
    let raw = ptyrs::base::sim::fermat(NPOS, frame as f64 / 8.0 * PIXEL);
    let positions = raw.pynx_preprocess(pixel);
    let shape = required_obj_shape(&positions, pixel, [frame, frame], 16);
    let addr = AddrTable::new(&positions, pixel, [frame, frame], shape, NPROBE, 1).unwrap();
    let truth = ptyrs::base::sim::object(shape);
    let mut probe = ptyrs::base::sim::probe([frame, frame], NPROBE).unwrap();
    probe
        .as_mut_slice()
        .iter_mut()
        .for_each(|v| *v *= PHOTONS.sqrt());

    let psi = propagated(&addr, &optics, &truth, &probe);
    let (n, nmode, nframe) = (frame * frame, addr.nmode(), addr.nframe());
    let mut counts = vec![0.0; nframe * n];
    for (f, block) in psi.chunks_exact(nmode * n).enumerate() {
        for j in 0..n {
            let icalc: f64 = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
            counts[f * n + shift(j, frame, frame)] = icalc.round();
        }
    }
    let counts = Counts::from_f64([nframe, frame, frame], &counts, None).unwrap();

    // The start: unit modulus and a random phase in -1/4..1/4 rad, the probe's defocus cut
    // by a third.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let data = (truth.as_slice().iter())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            C64::from_polar(1.0, 0.5 * (u - 0.5))
        })
        .collect();
    let object = Object::new(Stack::from_vec(truth.dims(), data).unwrap());
    let sigma = frame as f64 / 12.0;
    for m in 0..probe.nmode() {
        for (i, v) in probe.plane_mut(m).iter_mut().enumerate() {
            let u = ((i % frame) as f64 - frame as f64 / 2.0) / sigma;
            let w = ((i / frame) as f64 - frame as f64 / 2.0) / sigma;
            *v *= C64::from_polar(1.0, -0.5 * (u * u + w * w));
        }
    }
    let geometry = Geometry::new(WAVELENGTH, 1.0, PIXEL, [frame, frame]).unwrap();
    Case {
        name,
        addr,
        obs: Observed::from_counts(&counts),
        scan: Scan::new(geometry, raw, counts).unwrap(),
        distance,
        object,
        probe,
    }
}

pub fn cases() -> [Case; 3] {
    [
        case("angular spectrum", 1e-4),
        case("Fresnel", 5e-3),
        // Not nearer: at -5e-3 and -2e-3 the CPU's second CG step has H(D, D) < 0, after
        // which its own steps from an f32 start are 2e-6 to 2e-5 off and the GPU's 15x that.
        case("Fresnel back", -1e-2),
    ]
}

impl Case {
    pub fn optics(&self) -> Optics {
        free_space(self.distance)
    }

    /// The normalised Poisson LLK of `object` and `probe` over every frame.
    pub fn llk(&self, object: &Object, probe: &Probe) -> f64 {
        let psi = propagated(&self.addr, &self.optics(), object, probe);
        llk(&self.obs, 0..self.addr.nframe(), self.addr.nmode(), &psi)
            .normalised(self.obs.nb_obs())
            .poisson
    }

    pub fn gpu(&self, graph: bool) -> GpuPtycho {
        let ctx: Arc<CudaContext> = CudaContext::new(0).expect("CUDA device 0");
        let options = GpuOptions { graph, ..OPTIONS };
        let (object, probe) = (self.object.clone(), self.probe.clone());
        GpuPtycho::with_optics(&ctx, &self.scan, &self.optics(), object, probe, options).unwrap()
    }
}

pub fn drawn_offsets() -> Vec<[f64; 2]> {
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
