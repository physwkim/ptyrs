//! `ptyrs_tile::update_from_sums` and `update_obj_from_sums` against their SIMT
//! `ptyrs_gpu::Kernels` counterparts and an f64 reference.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use ptyrs_gpu::{C32, Kernels, OBJ_TILE, UpdateSums, ViewAddr, ViewDims, Views, view_tiles};

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

fn flat(v: &[C32]) -> Vec<f32> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
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
fn update_from_sums_matches_simt() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let inertia = 0.01f32;
    // 300_001 values span several max programs; their maximum is the last value. The 1500
    // have no positive `den` (negatives, -0, NaN), so `reg` is 0 and the update is exact.
    for (len, seed, positive) in [
        (6000usize, 20u64, true),
        (300_001, 30, true),
        (1500, 40, false),
    ] {
        let x_host = noise(len, seed, 1.0);
        let num_host = noise(len, seed + 1, 3.0);
        let mut den_host: Vec<f32> = noise(len, seed + 2, 2.0)
            .iter()
            .map(|c| c.re.abs())
            .collect();
        if positive {
            den_host[len - 1] = 7.5;
        } else {
            den_host.iter_mut().for_each(|v| *v = -*v);
            den_host[100] = f32::NAN;
            den_host[200] = -0.0;
        }
        let reg = inertia as f64 * den_host.iter().fold(0.0f64, |m, &v| m.max(v as f64));
        let want: Vec<f64> = (0..len)
            .flat_map(|i| {
                let (a, b) = (num_host[i], x_host[i]);
                let d = (den_host[i] as f64 + reg).max(1e-12);
                [
                    (a.re as f64 + reg * b.re as f64) / d,
                    (a.im as f64 + reg * b.im as f64) / d,
                ]
            })
            .collect();
        let mut sums =
            UpdateSums::new(up(&stream, &num_host), up(&stream, &den_host), &stream).unwrap();
        let mut simt = up(&stream, &x_host);
        k.update_from_sums(&ex, &mut simt, &mut sums, len as u32, inertia)
            .unwrap();
        let simt = flat(&simt.to_host_vec(&stream).unwrap());
        // Stale scratch above every `den`: the op must zero it itself.
        sums.max_bits = up(&stream, &[1e30f32.to_bits()]);
        let mut run = || {
            let mut x = up(&stream, &x_host);
            ptyrs_tile::update_from_sums(&ex, &mut x, &mut sums, len as u32, inertia).unwrap();
            flat(&x.to_host_vec(&stream).unwrap())
        };
        let tile = run();

        let simt64: Vec<f64> = simt.iter().map(|&v| v as f64).collect();
        let (vs_simt, vs_f64) = (rel_l2(&tile, &simt64), rel_l2(&tile, &want));
        let same = tile
            .iter()
            .zip(&simt)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "update_from_sums n={len}: rel L2 Tile-SIMT {vs_simt:.2e}, Tile-f64 {vs_f64:.2e}; \
             {same} of {} values bit-equal to SIMT",
            tile.len()
        );
        assert!(vs_simt < 1e-6, "n={len}: Tile vs SIMT rel L2 {vs_simt:.2e}");
        assert!(vs_f64 < 1e-6, "n={len}: Tile vs f64 rel L2 {vs_f64:.2e}");
        if !positive {
            assert_eq!(bits(&tile), bits(&simt), "n={len}: reg 0 is not exact");
        }
        assert_eq!(bits(&run()), bits(&tile), "n={len}: differs run to run");
    }
}

/// The object update reads and writes only the tiles with views: their sums poisoned
/// (`num` NaN, `den` 1e30, above every lit `den`), those tiles keep `obj`, `num` and `den`
/// bit for bit, and every other pixel gets the SIMT kernels' bits, `max(den)` included,
/// with negative and NaN `den` among the lit pixels and stale `max_bits`.
#[test]
fn update_obj_from_sums_matches_simt_on_tiles_with_views() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let inertia = 0.01f32;
    // Two 82x100 object modes, one view each: tiles rows 0-1 x columns 0-1 and rows 3-5 x
    // columns 4-6 (the partial last row and column) have views, the rest none.
    let d = ViewDims {
        nframe: 2,
        nmode: 1,
        nprobe: 1,
        nobj: 2,
        frame: [24, 32],
        obj: [82, 100],
        quad_phase: 0.0,
    };
    let view = |oy, ox, fy, fx, obj| ViewAddr {
        oy,
        ox,
        fy,
        fx,
        probe: 0,
        obj,
    };
    let view_list = [view(0, 0, 0.0, 0.0, 0), view(57, 67, 0.6, 0.9, 1)];
    let views = Views::new(&stream, d, &view_list).unwrap();
    let [nyo, nxo] = d.obj.map(|v| v as usize);
    let [nty, ntx] = d.tiles().map(|v| v as usize);
    let mut lit_tiles = vec![false; nty * ntx];
    for v in &view_list {
        for [ty, tx] in view_tiles(&d, v) {
            lit_tiles[ty as usize * ntx + tx as usize] = true;
        }
    }
    let t = OBJ_TILE as usize;
    let len = d.nobj as usize * nyo * nxo;
    let lit: Vec<bool> = (0..len)
        .map(|i| {
            let p = i % (nyo * nxo);
            lit_tiles[(p / nxo / t) * ntx + (p % nxo) / t]
        })
        .collect();
    assert_eq!(lit_tiles.iter().filter(|&&l| l).count(), 4 + 9);

    let x_host = noise(len, 50, 1.0);
    let mut num_host = noise(len, 51, 3.0);
    let mut den_host: Vec<f32> = noise(len, 52, 2.0).iter().map(|c| c.re.abs()).collect();
    let nan = C32 {
        re: f32::NAN,
        im: f32::NAN,
    };
    for i in 0..len {
        if !lit[i] {
            num_host[i] = nan;
            den_host[i] = 1e30;
        } else if i % 7 == 3 {
            den_host[i] = -den_host[i];
        }
    }
    // Lit: pixel (0, 0) of mode 0, the far corner of mode 1.
    let corner = len - 1;
    assert!(lit[0] && lit[corner] && !lit[nxo - 1]);
    den_host[0] = f32::NAN;
    den_host[corner] = 7.5;

    let max = (0..len)
        .filter(|&i| lit[i])
        .fold(0.0f64, |m, i| m.max(den_host[i] as f64));
    assert_eq!(max, 7.5);
    let reg = inertia as f64 * max;
    let want: Vec<f64> = (0..len)
        .flat_map(|i| {
            let (a, b) = (num_host[i], x_host[i]);
            if !lit[i] {
                return [b.re as f64, b.im as f64];
            }
            let d = (den_host[i] as f64 + reg).max(1e-12);
            [
                (a.re as f64 + reg * b.re as f64) / d,
                (a.im as f64 + reg * b.im as f64) / d,
            ]
        })
        .collect();

    let stale = || up(&stream, &[1e30f32.to_bits()]);
    let run = |simt: bool| {
        let mut sums =
            UpdateSums::new(up(&stream, &num_host), up(&stream, &den_host), &stream).unwrap();
        sums.max_bits = stale();
        let mut x = up(&stream, &x_host);
        if simt {
            k.update_obj_from_sums(&ex, &views, &mut x, &mut sums, inertia)
                .unwrap();
        } else {
            ptyrs_tile::update_obj_from_sums(&ex, &views, &mut x, &mut sums, inertia).unwrap();
        }
        let x = x.to_host_vec(&stream).unwrap();
        let num = sums.num.to_host_vec(&stream).unwrap();
        let den = sums.den.to_host_vec(&stream).unwrap();
        (x, num, den)
    };
    let simt = run(true);
    let tile = run(false);
    let (tx, sx) = (flat(&tile.0), flat(&simt.0));
    let vs_f64 = rel_l2(&tx, &want);
    println!("update_obj_from_sums: rel L2 Tile-f64 {vs_f64:.2e}");
    assert_eq!(bits(&tx), bits(&sx), "obj differs from SIMT");
    assert!(vs_f64 < 1e-6, "Tile vs f64 rel L2 {vs_f64:.2e}");
    for i in (0..len).filter(|&i| !lit[i]) {
        let (x, n) = (flat(&tile.0[i..=i]), flat(&tile.1[i..=i]));
        assert_eq!(
            bits(&x),
            bits(&flat(&x_host[i..=i])),
            "obj {i}, without views"
        );
        assert_eq!(
            bits(&n),
            bits(&flat(&num_host[i..=i])),
            "num {i}, without views"
        );
        assert_eq!(
            tile.2[i].to_bits(),
            den_host[i].to_bits(),
            "den {i}, without views"
        );
    }
    assert_eq!(bits(&flat(&run(false).0)), bits(&tx), "differs run to run");
}
