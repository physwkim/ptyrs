//! The M1 pixel kernels against f64 CPU references of the contract in the `ptyrs_gpu`
//! crate docs (PyNX `cpu_operator.py` semantics, orthonormal FFT).

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    C32, Error, Kernels, LlkPartial, LlkSums, MASKED, UpdateSums, ViewAddr, ViewDims, Views,
};

type Z = (f64, f64);

fn setup() -> (Arc<CudaStream>, Kernels) {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let kernels = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    (ctx.default_stream(), kernels)
}

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

fn z(c: C32) -> Z {
    (c.re as f64, c.im as f64)
}

fn mul(a: Z, b: Z) -> Z {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `conj(a) * b`.
fn cmul_conj(a: Z, b: Z) -> Z {
    (a.0 * b.0 + a.1 * b.1, a.0 * b.1 - a.1 * b.0)
}

fn norm2(a: Z) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

/// `sqrt(sum |got - want|^2 / sum |want|^2)`.
fn rel_l2(got: &[C32], want: &[Z]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (e, r) = got.iter().zip(want).fold((0.0, 0.0), |(e, r), (g, w)| {
        (
            e + norm2((g.re as f64 - w.0, g.im as f64 - w.1)),
            r + norm2(*w),
        )
    });
    (e / r).sqrt()
}

fn rel_l2_re(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (e, r) = got.iter().zip(want).fold((0.0, 0.0), |(e, r), (&g, &w)| {
        (e + (g as f64 - w).powi(2), r + w * w)
    });
    (e / r).sqrt()
}

/// A scan: frame `f` sits at `offsets[f]` for all its views; mode `m` is probe
/// `m % nprobe` times object `m / nprobe`.
struct Scan {
    d: ViewDims,
    views: Vec<ViewAddr>,
    obj: Vec<C32>,
    probe: Vec<C32>,
}

impl Scan {
    fn new(d: ViewDims, offsets: &[(u32, u32, f32, f32)], probe_scale: f32) -> Self {
        assert_eq!(offsets.len(), d.nframe as usize);
        assert_eq!(d.nmode, d.nprobe * d.nobj);
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
        let nobj = d.nobj as usize * d.obj[0] as usize * d.obj[1] as usize;
        let nprobe = d.nprobe as usize * d.frame[0] as usize * d.frame[1] as usize;
        Self {
            d,
            views,
            obj: noise(nobj, 1, 1.0),
            probe: noise(nprobe, 2, probe_scale),
        }
    }

    fn n(&self) -> usize {
        self.d.frame[0] as usize * self.d.frame[1] as usize
    }

    fn nk(&self, frames: &Range<u32>) -> usize {
        (frames.end - frames.start) as usize * self.d.nmode as usize
    }

    fn views_of(&self, frames: &Range<u32>) -> &[ViewAddr] {
        let m = self.d.nmode as usize;
        &self.views[frames.start as usize * m..frames.end as usize * m]
    }

    /// `(y, x, y * nx + x)` over a frame.
    fn pixels(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        let [ny, nx] = [self.d.frame[0] as usize, self.d.frame[1] as usize];
        (0..ny).flat_map(move |y| (0..nx).map(move |x| (y, x, y * nx + x)))
    }

    /// Origin-at-`[0, 0]` offset of natural pixel `(y, x)`; also the FFT-order frequency of
    /// detector pixel `(y, x)`.
    fn shifted(&self, y: usize, x: usize) -> usize {
        let [ny, nx] = [self.d.frame[0] as usize, self.d.frame[1] as usize];
        ((y + ny / 2) % ny) * nx + (x + nx / 2) % nx
    }

    fn quad(&self, y: usize, x: usize) -> Z {
        let dy = y as f64 - (self.d.frame[0] / 2) as f64;
        let dx = x as f64 - (self.d.frame[1] / 2) as f64;
        let a = self.d.quad_phase as f64 * (dy * dy + dx * dx);
        (a.cos(), a.sin())
    }

    /// Object indices and bilinear weights of view `v`'s sample at probe pixel `(y, x)`.
    fn corners(&self, v: &ViewAddr, y: usize, x: usize) -> [(usize, f64); 4] {
        let [nyo, nxo] = [self.d.obj[0] as usize, self.d.obj[1] as usize];
        let (fy, fx) = (v.fy as f64, v.fx as f64);
        let (y0, x0) = (v.oy as usize + y, v.ox as usize + x);
        let at = |yy: usize, xx: usize| (v.obj as usize * nyo + yy % nyo) * nxo + xx % nxo;
        [
            (at(y0, x0), (1.0 - fy) * (1.0 - fx)),
            (at(y0, x0 + 1), (1.0 - fy) * fx),
            (at(y0 + 1, x0), fy * (1.0 - fx)),
            (at(y0 + 1, x0 + 1), fy * fx),
        ]
    }

    fn sample(&self, obj: &[C32], v: &ViewAddr, y: usize, x: usize) -> Z {
        self.corners(v, y, x).iter().fold((0.0, 0.0), |s, &(i, w)| {
            (s.0 + w * obj[i].re as f64, s.1 + w * obj[i].im as f64)
        })
    }

    /// Exit waves of `frames` for object `obj` and the scan's probe.
    fn psi(&self, frames: Range<u32>, obj: &[C32]) -> Vec<Z> {
        let n = self.n();
        let mut out = vec![(0.0, 0.0); self.nk(&frames) * n];
        for (kl, v) in self.views_of(&frames).iter().enumerate() {
            for (y, x, i) in self.pixels() {
                let o = self.sample(obj, v, y, x);
                let p = z(self.probe[v.probe as usize * n + i]);
                out[kl * n + self.shifted(y, x)] = mul(mul(o, p), self.quad(y, x));
            }
        }
        out
    }

    /// Adds the stack's object sums, scattered from each probe pixel to its four corners.
    fn obj_sums(&self, frames: Range<u32>, psi: &[C32], num: &mut [Z], den: &mut [f64]) {
        let n = self.n();
        for (kl, v) in self.views_of(&frames).iter().enumerate() {
            for (y, x, i) in self.pixels() {
                let p = z(self.probe[v.probe as usize * n + i]);
                let u = cmul_conj(mul(p, self.quad(y, x)), z(psi[kl * n + self.shifted(y, x)]));
                for (j, w) in self.corners(v, y, x) {
                    num[j] = (num[j].0 + w * u.0, num[j].1 + w * u.1);
                    den[j] += w * norm2(p);
                }
            }
        }
    }

    /// Adds the stack's probe sums.
    fn probe_sums(&self, frames: Range<u32>, psi: &[C32], num: &mut [Z], den: &mut [f64]) {
        let n = self.n();
        for (kl, v) in self.views_of(&frames).iter().enumerate() {
            for (y, x, i) in self.pixels() {
                let o = self.sample(&self.obj, v, y, x);
                let u = cmul_conj(mul(o, self.quad(y, x)), z(psi[kl * n + self.shifted(y, x)]));
                let j = v.probe as usize * n + i;
                num[j] = (num[j].0 + u.0, num[j].1 + u.1);
                den[j] += norm2(o);
            }
        }
    }
}

/// 24x32 frames on two 50x60 object modes and two probe modes (four views per frame).
/// Frames 1 and 3 end on the object's last row and column, so their `+ 1` neighbours wrap;
/// the subpixel offsets cover `fy` and `fx` zero and non-zero.
fn scan() -> Scan {
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
    Scan::new(d, &offsets, 1.0)
}

/// PyNX `LLKAll` terms of one unmasked pixel: Poisson, Gaussian, Euclidean, calc.
fn llk_terms(obs: f64, calc: f64) -> [f64; 4] {
    let poisson = if obs == 0.0 {
        calc
    } else {
        calc - obs + obs * (obs / calc).ln()
    };
    [
        poisson,
        (obs - calc).powi(2) / (obs + 1.0),
        4.0 * (obs.sqrt() - calc.sqrt()).powi(2),
        calc,
    ]
}

fn assert_llk(what: &str, got: &[LlkPartial], want: &[[f64; 4]]) {
    assert_eq!(got.len(), want.len());
    for (f, (g, w)) in got.iter().zip(want).enumerate() {
        let g = [g.poisson, g.gaussian, g.euclidean, g.icalc];
        for (j, (&g, &w)) in g.iter().zip(w).enumerate() {
            let rel = (g as f64 - w).abs() / w.abs().max(1.0);
            assert!(
                rel < 1e-5,
                "{what} frame {f} term {j}: {g} vs {w} (rel {rel:.2e})"
            );
        }
    }
    let sums = LlkSums::from_partials(got);
    let total = |j: usize| want.iter().map(|w| w[j]).sum::<f64>();
    let pairs = [
        (sums.poisson, total(0)),
        (sums.gaussian, total(1)),
        (sums.euclidean, total(2)),
        (sums.icalc, total(3)),
    ];
    for (g, w) in pairs {
        assert!(
            (g - w).abs() / w.abs().max(1.0) < 1e-5,
            "{what} total {g} vs {w}"
        );
    }
}

#[test]
fn obj_probe_to_psi_matches_f64() {
    let (stream, k) = setup();
    let s = scan();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let (obj, probe) = (up(&stream, &s.obj), up(&stream, &s.probe));
    for frames in [0..5, 2..5] {
        let mut psi = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&frames) * s.n()).unwrap();
        k.obj_probe_to_psi(&stream, &views, frames.clone(), &obj, &probe, &mut psi)
            .unwrap();
        let err = rel_l2(
            &psi.to_host_vec(&stream).unwrap(),
            &s.psi(frames.clone(), &s.obj),
        );
        println!("obj_probe_to_psi {frames:?}: rel L2 {err:.2e}");
        assert!(err < 2e-6, "frames {frames:?}: rel L2 {err:.2e}");
    }
}

/// Uploads `num`, `den` as sums.
fn sums(stream: &CudaStream, start: &(Vec<C32>, Vec<f32>)) -> UpdateSums {
    UpdateSums {
        num: up(stream, &start.0),
        den: up(stream, &start.1),
    }
}

/// `whole` matches the f64 sums, and `parts` (the same frames in two stacks) matches
/// `whole` bit for bit.
fn check_sums(
    what: &str,
    stream: &CudaStream,
    whole: &UpdateSums,
    parts: &UpdateSums,
    want: (&[Z], &[f64]),
) {
    let num = whole.num.to_host_vec(stream).unwrap();
    let den = whole.den.to_host_vec(stream).unwrap();
    let (en, ed) = (rel_l2(&num, want.0), rel_l2_re(&den, want.1));
    println!("{what} sums: num rel L2 {en:.2e}, den {ed:.2e}");
    assert!(en < 1e-5 && ed < 1e-5, "{what}: num {en:.2e}, den {ed:.2e}");
    let bits = |v: &[C32]| -> Vec<u32> {
        v.iter()
            .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
            .collect()
    };
    let fbits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(
        bits(&num),
        bits(&parts.num.to_host_vec(stream).unwrap()),
        "{what} num depends on the stack split"
    );
    assert_eq!(
        fbits(&den),
        fbits(&parts.den.to_host_vec(stream).unwrap()),
        "{what} den depends on the stack split"
    );
}

#[test]
fn update_sums_match_f64_and_ignore_stack_split() {
    let (stream, k) = setup();
    let s = scan();
    let n = s.n();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let (obj, probe) = (up(&stream, &s.obj), up(&stream, &s.probe));
    let psi_host = noise(s.nk(&(0..5)) * n, 3, 1.0);
    let psi = up(&stream, &psi_host);
    let split = s.nk(&(0..2)) * n;
    let psi_parts = [
        up(&stream, &psi_host[..split]),
        up(&stream, &psi_host[split..]),
    ];
    // Non-zero starting sums: the kernels add to what is there.
    let start = |len: usize, seed: u64| {
        let den = noise(len, seed + 1, 1.0)
            .iter()
            .map(|c| c.re.abs())
            .collect();
        (noise(len, seed, 1.0), den)
    };
    let f64_start = |st: &(Vec<C32>, Vec<f32>)| -> (Vec<Z>, Vec<f64>) {
        (
            st.0.iter().map(|&c| z(c)).collect(),
            st.1.iter().map(|&v| v as f64).collect(),
        )
    };

    let st = start(s.obj.len(), 10);
    let (mut whole, mut parts) = (sums(&stream, &st), sums(&stream, &st));
    k.psi_to_obj(&stream, &views, 0..5, &psi, &probe, &mut whole)
        .unwrap();
    k.psi_to_obj(&stream, &views, 0..2, &psi_parts[0], &probe, &mut parts)
        .unwrap();
    k.psi_to_obj(&stream, &views, 2..5, &psi_parts[1], &probe, &mut parts)
        .unwrap();
    let (mut num, mut den) = f64_start(&st);
    s.obj_sums(0..5, &psi_host, &mut num, &mut den);
    check_sums("object", &stream, &whole, &parts, (&num, &den));

    let st = start(s.probe.len(), 20);
    let (mut whole, mut parts) = (sums(&stream, &st), sums(&stream, &st));
    k.psi_to_probe(&stream, &views, 0..5, &psi, &obj, &mut whole)
        .unwrap();
    k.psi_to_probe(&stream, &views, 0..2, &psi_parts[0], &obj, &mut parts)
        .unwrap();
    k.psi_to_probe(&stream, &views, 2..5, &psi_parts[1], &obj, &mut parts)
        .unwrap();
    let (mut num, mut den) = f64_start(&st);
    s.probe_sums(0..5, &psi_host, &mut num, &mut den);
    check_sums("probe", &stream, &whole, &parts, (&num, &den));
}

#[test]
fn psi_to_obj_is_the_adjoint_of_obj_probe_to_psi() {
    let (stream, k) = setup();
    let s = scan();
    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let probe = up(&stream, &s.probe);
    let o = noise(s.obj.len(), 7, 1.0);
    // <A o, A o> = <o, A^H A o>: no cancellation, so f32 rounding stays small.
    let mut a_o = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..5)) * s.n()).unwrap();
    k.obj_probe_to_psi(&stream, &views, 0..5, &up(&stream, &o), &probe, &mut a_o)
        .unwrap();
    let mut ah = UpdateSums::zeroed(&stream, o.len()).unwrap();
    k.psi_to_obj(&stream, &views, 0..5, &a_o, &probe, &mut ah)
        .unwrap();
    let dot = |a: &[C32], b: &[C32]| {
        a.iter().zip(b).fold((0.0, 0.0), |s, (&a, &b)| {
            let t = cmul_conj(z(a), z(b));
            (s.0 + t.0, s.1 + t.1)
        })
    };
    let a_o = a_o.to_host_vec(&stream).unwrap();
    let lhs = dot(&a_o, &a_o);
    let rhs = dot(&o, &ah.num.to_host_vec(&stream).unwrap());
    let rel = norm2((lhs.0 - rhs.0, lhs.1 - rhs.1)).sqrt() / norm2(lhs).sqrt();
    println!("adjoint: <Ao,Ao> {lhs:?}, <o,A^H Ao> {rhs:?}, rel {rel:.2e}");
    assert!(rel < 1e-5, "rel {rel:.2e}");
}

/// Scan counts: every 7th pixel masked, every 5th of the rest zero, others 1..=60.
fn counts(len: usize) -> Vec<u16> {
    (0..len)
        .map(|i| match i {
            _ if i % 7 == 3 => MASKED,
            _ if i % 5 == 1 => 0,
            _ => 1 + ((i * 37) % 60) as u16,
        })
        .collect()
}

#[test]
fn intensity_amplitude_and_llk_match_f64() {
    let (stream, k) = setup();
    let s = scan();
    let (d, n, nm) = (s.d, s.n(), s.d.nmode as usize);
    let frames = 1..4;
    let nf = 3;
    // Far-field stack scaled so the calculated intensity is near the counts.
    let psi_host = noise(s.nk(&frames) * n, 5, 50.0);
    let counts_host = counts(d.nframe as usize * n);

    let mut icalc_ref = vec![0.0; nf * n];
    let mut proj_ref: Vec<Z> = psi_host.iter().map(|&c| z(c)).collect();
    let mut llk_ref = vec![[0.0; 4]; nf];
    for fl in 0..nf {
        for (y, x, i) in s.pixels() {
            let at = s.shifted(y, x);
            let plane = |m: usize| (fl * nm + m) * n + at;
            let calc = (0..nm).map(|m| norm2(z(psi_host[plane(m)]))).sum::<f64>() / n as f64;
            icalc_ref[fl * n + i] = calc;
            let obs = counts_host[(frames.start as usize + fl) * n + i];
            let scale = if obs == MASKED {
                1.0 / n as f64
            } else {
                for (l, t) in llk_ref[fl].iter_mut().zip(llk_terms(obs as f64, calc)) {
                    *l += t;
                }
                (obs as f64 / calc.max(1e-12)).sqrt() / n as f64
            };
            for m in 0..nm {
                let v = &mut proj_ref[plane(m)];
                *v = (v.0 * scale, v.1 * scale);
            }
        }
    }

    let counts = up(&stream, &counts_host);
    let mut psi = up(&stream, &psi_host);
    let mut icalc = DeviceBuffer::<f32>::zeroed(&stream, nf * n).unwrap();
    k.mode_intensity(&stream, &d, frames.clone(), &psi, &mut icalc)
        .unwrap();
    let mut llk = DeviceBuffer::<LlkPartial>::zeroed(&stream, nf).unwrap();
    k.llk(&stream, &d, frames.clone(), &counts, &psi, &mut llk)
        .unwrap();
    k.apply_amplitude(&stream, &d, frames.clone(), &counts, &mut psi)
        .unwrap();

    let err = rel_l2_re(&icalc.to_host_vec(&stream).unwrap(), &icalc_ref);
    println!("mode_intensity: rel L2 {err:.2e}");
    assert!(err < 2e-6, "mode_intensity rel L2 {err:.2e}");
    assert_llk("llk", &llk.to_host_vec(&stream).unwrap(), &llk_ref);
    let err = rel_l2(&psi.to_host_vec(&stream).unwrap(), &proj_ref);
    println!("apply_amplitude: rel L2 {err:.2e}");
    assert!(err < 2e-6, "apply_amplitude rel L2 {err:.2e}");
}

#[test]
fn update_from_sums_matches_f64() {
    let (stream, k) = setup();
    let inertia = 0.01f32;
    // 300_001 values make each block of the max pass stride; its maximum is the last value.
    for (len, seed) in [(6000usize, 20u64), (300_001, 30)] {
        let x_host = noise(len, seed, 1.0);
        let num_host = noise(len, seed + 1, 3.0);
        let mut den_host: Vec<f32> = noise(len, seed + 2, 2.0)
            .iter()
            .map(|c| c.re.abs())
            .collect();
        den_host[len - 1] = 7.5;
        let reg = inertia as f64 * den_host.iter().fold(0.0f64, |m, &v| m.max(v as f64));
        let want: Vec<Z> = (0..len)
            .map(|i| {
                let (a, b) = (z(num_host[i]), z(x_host[i]));
                let d = (den_host[i] as f64 + reg).max(1e-12);
                ((a.0 + reg * b.0) / d, (a.1 + reg * b.1) / d)
            })
            .collect();
        let mut x = up(&stream, &x_host);
        let sums = UpdateSums {
            num: up(&stream, &num_host),
            den: up(&stream, &den_host),
        };
        k.update_from_sums(&stream, &mut x, &sums, len as u32, inertia)
            .unwrap();
        let got = x.to_host_vec(&stream).unwrap();
        let err = rel_l2(&got, &want);
        println!("update_from_sums n={len}: rel L2 {err:.2e}");
        assert!(err < 1e-6, "n={len}: rel L2 {err:.2e}");

        // Bit for bit the plain f32 formula, each operation rounded on its own (what a Tile
        // or CPU port computes), not a mix of fused and separate roundings.
        let reg = inertia * den_host.iter().fold(0.0f32, |m, &v| m.max(v));
        let mismatches = (0..len)
            .filter(|&i| {
                let d = (den_host[i] + reg).max(1e-12);
                let part = |a: f32, b: f32| (a + reg * b) / d;
                let (a, b) = (num_host[i], x_host[i]);
                let w = (part(a.re, b.re), part(a.im, b.im));
                (got[i].re.to_bits(), got[i].im.to_bits()) != (w.0.to_bits(), w.1.to_bits())
            })
            .count();
        assert_eq!(
            mismatches, 0,
            "n={len}: {mismatches} values differ from plain f32"
        );
    }
}

#[test]
fn dm_reflect_and_add_match_f64() {
    let (stream, k) = setup();
    let (len, alpha) = (5000usize, 0.02f32);
    let a = alpha as f64;
    let (psi_host, po_host) = (noise(len, 40, 1.0), noise(len, 41, 1.0));
    let psi1: Vec<Z> = psi_host
        .iter()
        .zip(&po_host)
        .map(|(&p, &o)| {
            let (p, o) = (z(p), z(o));
            (
                (2.0 - a) * o.0 - (1.0 - a) * p.0,
                (2.0 - a) * o.1 - (1.0 - a) * p.1,
            )
        })
        .collect();
    let po1: Vec<Z> = po_host
        .iter()
        .zip(&psi1)
        .map(|(&o, p1)| {
            (
                (1.0 - a) * (o.re as f64 - p1.0),
                (1.0 - a) * (o.im as f64 - p1.1),
            )
        })
        .collect();
    let sum: Vec<Z> = psi1
        .iter()
        .zip(&po1)
        .map(|(p, o)| (p.0 + o.0, p.1 + o.1))
        .collect();

    let (mut psi, mut po) = (up(&stream, &psi_host), up(&stream, &po_host));
    k.dm_reflect(&stream, &mut psi, &mut po, len as u32, alpha)
        .unwrap();
    let e_psi = rel_l2(&psi.to_host_vec(&stream).unwrap(), &psi1);
    let e_po = rel_l2(&po.to_host_vec(&stream).unwrap(), &po1);
    k.add_c32(&stream, &mut psi, &po, len as u32).unwrap();
    let e_sum = rel_l2(&psi.to_host_vec(&stream).unwrap(), &sum);
    println!("dm_reflect: psi {e_psi:.2e}, po {e_po:.2e}; add_c32 {e_sum:.2e}");
    assert!(e_psi < 1e-6 && e_po < 1e-6 && e_sum < 1e-6);

    // First DM cycle: psi = PO is a fixed point, and the add-back term vanishes.
    let (mut psi, mut po) = (up(&stream, &po_host), up(&stream, &po_host));
    k.dm_reflect(&stream, &mut psi, &mut po, len as u32, alpha)
        .unwrap();
    let want: Vec<Z> = po_host.iter().map(|&c| z(c)).collect();
    let e_fix = rel_l2(&psi.to_host_vec(&stream).unwrap(), &want);
    let rest = po
        .to_host_vec(&stream)
        .unwrap()
        .iter()
        .fold(0.0f32, |m, c| m.max(c.re.abs()).max(c.im.abs()));
    println!("dm_reflect fixed point: psi rel L2 {e_fix:.2e}, max |po| {rest:.2e}");
    assert!(e_fix < 1e-6 && rest < 1e-6);
}

/// f64 orthonormal 2D DFT of an `n x n` plane (forward kernel `exp(-2 pi i jk / n)`).
fn dft2_ortho(plane: &[Z], n: usize, inverse: bool) -> Vec<Z> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let tw: Vec<Z> = (0..n)
        .map(|j| {
            let a = sign * 2.0 * PI * j as f64 / n as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let line = |get: &dyn Fn(usize) -> Z, k: usize| {
        (0..n).fold((0.0, 0.0), |s, j| {
            let t = mul(get(j), tw[(j * k) % n]);
            (s.0 + t.0, s.1 + t.1)
        })
    };
    let mut rows = vec![(0.0, 0.0); n * n];
    for r in 0..n {
        for k in 0..n {
            rows[r * n + k] = line(&|j| plane[r * n + j], k);
        }
    }
    let mut out = vec![(0.0, 0.0); n * n];
    for c in 0..n {
        for k in 0..n {
            let v = line(&|j| rows[j * n + c], k);
            out[k * n + c] = (v.0 / n as f64, v.1 / n as f64);
        }
    }
    out
}

/// `obj_probe_to_psi`, cuFFT, `llk`, `apply_amplitude`, inverse cuFFT against PyNX's
/// orthonormal `IFT * ApplyAmplitude(calc_llk) * FT * ObjProbe2Psi` in f64.
#[test]
fn fourier_projection_matches_pynx_orthonormal_f64() {
    const N: usize = 32;
    let (stream, k) = setup();
    let d = ViewDims {
        nframe: 3,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [N as u32, N as u32],
        obj: [48, 48],
        quad_phase: 0.002,
    };
    let s = Scan::new(
        d,
        &[(0, 0, 0.0, 0.0), (16, 3, 0.3, 0.0), (10, 16, 0.5, 0.25)],
        10.0,
    );
    let (n, nm, nf) = (N * N, 2, 3);

    // PyNX in f64: FT, counts near 1.2x the model, LLK, amplitude projection, IFT.
    let mut far: Vec<Z> = s
        .psi(0..3, &s.obj)
        .chunks(n)
        .flat_map(|p| dft2_ortho(p, N, false))
        .collect();
    let mut counts_host = vec![0u16; nf * n];
    let mut llk_ref = vec![[0.0; 4]; nf];
    for f in 0..nf {
        for (y, x, i) in s.pixels() {
            let at = |m: usize| (f * nm + m) * n + s.shifted(y, x);
            let calc: f64 = (0..nm).map(|m| norm2(far[at(m)])).sum();
            let obs = match i {
                _ if i % 11 == 4 => MASKED,
                _ if i % 13 == 2 => 0,
                _ => (1.2 * calc).round().min(60000.0) as u16,
            };
            counts_host[f * n + i] = obs;
            if obs == MASKED {
                continue;
            }
            for (l, t) in llk_ref[f].iter_mut().zip(llk_terms(obs as f64, calc)) {
                *l += t;
            }
            let r = (obs as f64).sqrt() / calc.sqrt().max(1e-12);
            for m in 0..nm {
                far[at(m)] = (far[at(m)].0 * r, far[at(m)].1 * r);
            }
        }
    }
    let want: Vec<Z> = far.chunks(n).flat_map(|p| dft2_ortho(p, N, true)).collect();

    let views = Views::new(&stream, d, &s.views).unwrap();
    let plan = Plan2d::new(&stream, N, nf * nm).unwrap();
    let counts = up(&stream, &counts_host);
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, nf * nm * n).unwrap();
    k.obj_probe_to_psi(
        &stream,
        &views,
        0..3,
        &up(&stream, &s.obj),
        &up(&stream, &s.probe),
        &mut psi,
    )
    .unwrap();
    plan.forward_inplace(&mut psi).unwrap();
    let mut llk = DeviceBuffer::<LlkPartial>::zeroed(&stream, nf).unwrap();
    k.llk(&stream, &d, 0..3, &counts, &psi, &mut llk).unwrap();
    k.apply_amplitude(&stream, &d, 0..3, &counts, &mut psi)
        .unwrap();
    plan.inverse_inplace(&mut psi).unwrap();

    assert_llk("chain llk", &llk.to_host_vec(&stream).unwrap(), &llk_ref);
    let err = rel_l2(&psi.to_host_vec(&stream).unwrap(), &want);
    println!("FT, apply_amplitude, IFT vs PyNX ortho f64: rel L2 {err:.2e}");
    assert!(err < 1e-5, "rel L2 {err:.2e}");
}

#[test]
fn views_and_frame_ranges_are_checked() {
    let (stream, k) = setup();
    let s = scan();
    let shape_err = |r: Result<Views, Error>| matches!(r, Err(Error::Shape(_)));
    assert!(
        shape_err(Views::new(&stream, s.d, &s.views[1..])),
        "wrong view count"
    );
    let with = |f: fn(&mut ViewAddr)| {
        let mut v = s.views.clone();
        f(&mut v[6]);
        Views::new(&stream, s.d, &v)
    };
    assert!(
        shape_err(with(|v| v.oy = 27)),
        "view past the object's last row"
    );
    assert!(
        shape_err(with(|v| v.ox = 29)),
        "view past the object's last column"
    );
    assert!(shape_err(with(|v| v.fy = 1.0)), "fy = 1");
    assert!(shape_err(with(|v| v.fx = -0.1)), "fx < 0");
    assert!(shape_err(with(|v| v.probe = 2)), "probe mode out of range");
    assert!(shape_err(with(|v| v.obj = 2)), "object mode out of range");
    let odd = ViewDims {
        frame: [24, 31],
        ..s.d
    };
    assert!(
        shape_err(Views::new(&stream, odd, &s.views)),
        "odd frame side"
    );

    let views = Views::new(&stream, s.d, &s.views).unwrap();
    let (obj, probe) = (up(&stream, &s.obj), up(&stream, &s.probe));
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..5)) * s.n()).unwrap();
    for frames in [3..3, 4..6] {
        let r = k.obj_probe_to_psi(&stream, &views, frames.clone(), &obj, &probe, &mut psi);
        assert!(matches!(r, Err(Error::Shape(_))), "frames {frames:?}");
    }
    let mut short = DeviceBuffer::<C32>::zeroed(&stream, s.nk(&(0..5)) * s.n() - 1).unwrap();
    let r = k.obj_probe_to_psi(&stream, &views, 0..5, &obj, &probe, &mut short);
    assert!(matches!(r, Err(Error::Shape(_))), "short psi");
}
