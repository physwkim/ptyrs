//! `ptyrs_tile::psi_to_obj` against the SIMT `ptyrs_gpu::Kernels::psi_to_obj` and an f64
//! reference, on the geometry of `ptyrs-gpu`'s `update_sums` test.

use std::ops::Range;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs_gpu::{C32, Kernels, UpdateSums, ViewAddr, ViewDims, Views};

type Z = (f64, f64);

/// Deterministic values in `[-scale, scale)`.
fn noise(len: usize, seed: u64, scale: f32) -> Vec<C32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
    };
    (0..len)
        .map(|_| C32 {
            re: next(),
            im: next(),
        })
        .collect()
}

fn up<T: DeviceCopy>(stream: &CudaStream, host: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(stream, host).unwrap()
}

/// 24x32 frames on two 50x60 object modes and two probe modes (four views per frame);
/// frames 1 and 3 wrap their `+ 1` neighbours, and `fy`, `fx` cover zero and non-zero.
fn dims() -> (ViewDims, Vec<ViewAddr>) {
    let d = ViewDims {
        nframe: 5,
        nmode: 4,
        nprobe: 2,
        nobj: 2,
        frame: [24, 32],
        obj: [50, 60],
        quad_phase: 0.003,
    };
    let offsets = [
        (0, 0, 0.0, 0.0),
        (26, 28, 0.25, 0.0),
        (13, 7, 0.0, 0.7),
        (26, 28, 0.6, 0.9),
        (5, 20, 0.5, 0.5),
    ];
    let views = offsets
        .iter()
        .flat_map(|&(oy, ox, fy, fx)| {
            (0..d.nmode).map(move |m| ViewAddr {
                oy,
                ox,
                fy,
                fx,
                probe: m % d.nprobe,
                obj: m / d.nprobe,
            })
        })
        .collect();
    (d, views)
}

/// Adds the object sums of `views` (from view 0 of `psi`), scattering each probe pixel to
/// its four bilinear corners in f64.
fn obj_sums_f64(
    d: &ViewDims,
    views: &[ViewAddr],
    psi: &[C32],
    probe: &[C32],
    num: &mut [Z],
    den: &mut [f64],
) {
    let [ny, nx] = [d.frame[0] as usize, d.frame[1] as usize];
    let [nyo, nxo] = [d.obj[0] as usize, d.obj[1] as usize];
    let n = ny * nx;
    for (kl, v) in views.iter().enumerate() {
        for y in 0..ny {
            for x in 0..nx {
                let p = probe[v.probe as usize * n + y * nx + x];
                let p = (p.re as f64, p.im as f64);
                let (dy, dx) = (y as f64 - (ny / 2) as f64, x as f64 - (nx / 2) as f64);
                let a = d.quad_phase as f64 * (dy * dy + dx * dx);
                let pq = (p.0 * a.cos() - p.1 * a.sin(), p.0 * a.sin() + p.1 * a.cos());
                let s = psi[kl * n + ((y + ny / 2) % ny) * nx + (x + nx / 2) % nx];
                let u = (
                    pq.0 * s.re as f64 + pq.1 * s.im as f64,
                    pq.0 * s.im as f64 - pq.1 * s.re as f64,
                );
                let (fy, fx) = (v.fy as f64, v.fx as f64);
                let (y0, x0) = (v.oy as usize + y, v.ox as usize + x);
                let at = |yy: usize, xx: usize| (v.obj as usize * nyo + yy % nyo) * nxo + xx % nxo;
                for (j, w) in [
                    (at(y0, x0), (1.0 - fy) * (1.0 - fx)),
                    (at(y0, x0 + 1), (1.0 - fy) * fx),
                    (at(y0 + 1, x0), fy * (1.0 - fx)),
                    (at(y0 + 1, x0 + 1), fy * fx),
                ] {
                    num[j] = (num[j].0 + w * u.0, num[j].1 + w * u.1);
                    den[j] += w * (p.0 * p.0 + p.1 * p.1);
                }
            }
        }
    }
}

/// The sums as host `f32`s, real and imaginary parts of `num` interleaved.
fn host(stream: &CudaStream, s: &UpdateSums) -> (Vec<f32>, Vec<f32>) {
    let num = s.num.to_host_vec(stream).unwrap();
    let num = num.iter().flat_map(|c| [c.re, c.im]).collect();
    (num, s.den.to_host_vec(stream).unwrap())
}

/// `sqrt(sum (got - want)^2 / sum want^2)`.
fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (e, r) = got.iter().zip(want).fold((0.0, 0.0), |(e, r), (&g, &w)| {
        (e + (g as f64 - w).powi(2), r + w * w)
    });
    (e / r).sqrt()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn psi_to_obj_matches_simt_and_ignores_stack_split() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let (d, view_list) = dims();
    let n = (d.frame[0] * d.frame[1]) as usize;
    let nk = |f: Range<u32>| ((f.end - f.start) * d.nmode) as usize;
    let views = Views::new(&stream, d, &view_list).unwrap();
    let probe_host = noise(d.nprobe as usize * n, 2, 1.0);
    let probe = up(&stream, &probe_host);
    let psi_host = noise(nk(0..5) * n, 3, 1.0);
    let psi = up(&stream, &psi_host);
    let split = nk(0..2) * n;
    let psi_parts = [
        up(&stream, &psi_host[..split]),
        up(&stream, &psi_host[split..]),
    ];
    // Non-zero starting sums: the kernels add to what is there.
    let len = d.nobj as usize * (d.obj[0] * d.obj[1]) as usize;
    let num0 = noise(len, 10, 1.0);
    let den0: Vec<f32> = noise(len, 11, 1.0).iter().map(|c| c.re.abs()).collect();
    let start = || UpdateSums {
        num: up(&stream, &num0),
        den: up(&stream, &den0),
    };

    let mut simt = start();
    k.psi_to_obj(&stream, &views, 0..5, &psi, &probe, &mut simt)
        .unwrap();
    let simt = host(&stream, &simt);
    let tile_whole = || {
        let mut s = start();
        ptyrs_tile::psi_to_obj(&stream, &views, 0..5, &psi, &probe, &mut s).unwrap();
        host(&stream, &s)
    };
    let tile = tile_whole();
    let mut parts = start();
    ptyrs_tile::psi_to_obj(&stream, &views, 0..2, &psi_parts[0], &probe, &mut parts).unwrap();
    ptyrs_tile::psi_to_obj(&stream, &views, 2..5, &psi_parts[1], &probe, &mut parts).unwrap();
    let parts = host(&stream, &parts);

    let mut num: Vec<Z> = num0.iter().map(|c| (c.re as f64, c.im as f64)).collect();
    let mut den: Vec<f64> = den0.iter().map(|&v| v as f64).collect();
    obj_sums_f64(&d, &view_list, &psi_host, &probe_host, &mut num, &mut den);
    let num: Vec<f64> = num.iter().flat_map(|&(a, b)| [a, b]).collect();
    let as_f64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();

    for (what, t, s, want) in [
        ("num", &tile.0, &simt.0, &num),
        ("den", &tile.1, &simt.1, &den),
    ] {
        let (vs_simt, vs_f64, simt_f64) = (rel_l2(t, &as_f64(s)), rel_l2(t, want), rel_l2(s, want));
        let same = t
            .iter()
            .zip(s.iter())
            .filter(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "psi_to_obj {what}: rel L2 Tile-SIMT {vs_simt:.2e}, Tile-f64 {vs_f64:.2e}, \
             SIMT-f64 {simt_f64:.2e}; {} of {} values bit-equal to SIMT",
            same.count(),
            t.len()
        );
        assert!(vs_simt < 1e-6, "{what}: Tile vs SIMT rel L2 {vs_simt:.2e}");
        assert!(vs_f64 < 1e-5, "{what}: Tile vs f64 rel L2 {vs_f64:.2e}");
    }
    assert_eq!(
        bits(&parts.0),
        bits(&tile.0),
        "num depends on the stack split"
    );
    assert_eq!(
        bits(&parts.1),
        bits(&tile.1),
        "den depends on the stack split"
    );
    let again = tile_whole();
    assert_eq!(bits(&again.0), bits(&tile.0), "num differs run to run");
    assert_eq!(bits(&again.1), bits(&tile.1), "den differs run to run");
}

#[test]
fn psi_to_obj_checks_frames_and_lengths() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let (d, view_list) = dims();
    let n = (d.frame[0] * d.frame[1]) as usize;
    let views = Views::new(&stream, d, &view_list).unwrap();
    let probe = up(&stream, &noise(d.nprobe as usize * n, 2, 1.0));
    let psi = up(&stream, &noise(2 * d.nmode as usize * n, 3, 1.0));
    let len = d.nobj as usize * (d.obj[0] * d.obj[1]) as usize;
    let mut sums = UpdateSums::zeroed(&stream, len).unwrap();
    for frames in [1..1, 4..6] {
        let r = ptyrs_tile::psi_to_obj(&stream, &views, frames.clone(), &psi, &probe, &mut sums);
        assert!(
            matches!(r, Err(ptyrs_tile::Error::Gpu(ptyrs_gpu::Error::Shape(_)))),
            "frames {frames:?}: {r:?}"
        );
    }
    // Two frames of psi do not hold three.
    let r = ptyrs_tile::psi_to_obj(&stream, &views, 0..3, &psi, &probe, &mut sums);
    assert!(
        matches!(r, Err(ptyrs_tile::Error::Gpu(ptyrs_gpu::Error::Shape(_)))),
        "{r:?}"
    );
}
