//! `ptyrs::tile::TileProbeUpdate::update_probe_from_sums` and
//! `TileObjUpdate::update_obj_from_sums` against their SIMT `ptyrs::gpu::Kernels` counterparts
//! and an f64 reference.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use cutile::tile_kernel::jit_compile_count;
use ptyrs::gpu::{
    C32, Kernels, OBJ_TILE, StackTable, UpdateSums, ViewAddr, ViewDims, Views, view_tiles,
};
use ptyrs::tile::{TileKernels, TileObjUpdate};

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

/// A scan whose probe is `nprobe` modes of `frame`: one frame, viewing each mode at the
/// origin of a one-mode object of the frame's size, and its one-stack table.
fn probe_scan(stream: &Arc<CudaStream>, nprobe: u32, frame: [u32; 2]) -> (Views, StackTable) {
    let d = ViewDims {
        nframe: 1,
        nmode: nprobe,
        nprobe,
        nobj: 1,
        frame,
        obj: frame,
        quad_phase: 0.0,
    };
    let views: Vec<ViewAddr> = (0..nprobe)
        .map(|probe| ViewAddr {
            probe,
            ..ViewAddr::default()
        })
        .collect();
    let views = Views::new(stream, d, &views).unwrap();
    let table = StackTable::new(stream, &d, std::slice::from_ref(&(0..1))).unwrap();
    (views, table)
}

#[test]
fn update_probe_from_sums_matches_simt() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let inertia = 0.01f32;
    // 300_000 values span several max programs; their maximum is the last value. The 1500
    // have no positive `den` (negatives, -0, NaN), so `reg` is 0 and the update is exact.
    for (nprobe, frame, seed, positive) in [
        (3, [40, 50], 20u64, true),
        (1, [500, 600], 30, true),
        (1, [30, 50], 40, false),
    ] {
        let len = (nprobe * frame[0] * frame[1]) as usize;
        let (views, table) = probe_scan(&stream, nprobe, frame);
        let tk = TileKernels::new(&stream, &views, &table).unwrap();
        let jit = jit_compile_count();
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
        let sums =
            || UpdateSums::new(up(&stream, &num_host), up(&stream, &den_host), &stream).unwrap();
        // The bits of `x` and of the sums the update leaves.
        let after = |x: DeviceBuffer<C32>, sums: UpdateSums| {
            let num = flat(&sums.num.to_host_vec(&stream).unwrap());
            let den = sums.den.to_host_vec(&stream).unwrap();
            (
                flat(&x.to_host_vec(&stream).unwrap()),
                bits(&num),
                bits(&den),
            )
        };
        let mut simt = up(&stream, &x_host);
        let mut simt_sums = sums();
        k.update_from_sums(&ex, &mut simt, &mut simt_sums, len as u32, inertia)
            .unwrap();
        let (simt, simt_num, simt_den) = after(simt, simt_sums);
        let run = || {
            let mut sums = sums();
            // Stale scratch above every `den`: the op must zero it itself.
            sums.max_bits = up(&stream, &[1e30f32.to_bits()]);
            let mut x = up(&stream, &x_host);
            tk.probe_update()
                .update_probe_from_sums(&ex, &mut x, &mut sums, inertia)
                .unwrap();
            after(x, sums)
        };
        let (tile, num, den) = run();
        // Both consume every sum.
        assert_eq!(num, vec![0; 2 * len], "n={len}: num not consumed");
        assert_eq!(den, vec![0; len], "n={len}: den not consumed");
        assert_eq!((&simt_num, &simt_den), (&num, &den), "n={len}: SIMT sums");

        let simt64: Vec<f64> = simt.iter().map(|&v| v as f64).collect();
        let (vs_simt, vs_f64) = (rel_l2(&tile, &simt64), rel_l2(&tile, &want));
        let same = tile
            .iter()
            .zip(&simt)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "update_probe_from_sums n={len}: rel L2 Tile-SIMT {vs_simt:.2e}, Tile-f64 {vs_f64:.2e}; \
             {same} of {} values bit-equal to SIMT",
            tile.len()
        );
        assert!(vs_simt < 1e-6, "n={len}: Tile vs SIMT rel L2 {vs_simt:.2e}");
        assert!(vs_f64 < 1e-6, "n={len}: Tile vs f64 rel L2 {vs_f64:.2e}");
        if !positive {
            assert_eq!(bits(&tile), bits(&simt), "n={len}: reg 0 is not exact");
        }
        assert_eq!(bits(&run().0), bits(&tile), "n={len}: differs run to run");
        assert_eq!(jit_compile_count(), jit, "n={len}: a launch compiled");
    }
}

/// The object update reads and writes only the tiles with views: the other tiles' sums
/// poisoned (`num` NaN, `den` 1e30, above every lit `den`), those tiles keep `obj`, `num`
/// and `den` bit for bit, and every other pixel gets the SIMT kernels' bits, `max(den)`
/// included, with negative and NaN `den` among the lit pixels and stale `max_bits`, and
/// its sums zeroed.
#[test]
fn update_obj_from_sums_matches_simt_on_tiles_with_views() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&ctx).expect("load ptyrs-gpu kernels");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let inertia = 0.01f32;
    // Two 82x100 object modes and two frames, each viewing both modes at one position:
    // tiles rows 0-1 x columns 0-1 and rows 3-5 x columns 4-6 (the partial last row and
    // column) have views, the rest none.
    let d = ViewDims {
        nframe: 2,
        nmode: 2,
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
    let view_list = [
        view(0, 0, 0.0, 0.0, 0),
        view(0, 0, 0.0, 0.0, 1),
        view(57, 67, 0.6, 0.9, 0),
        view(57, 67, 0.6, 0.9, 1),
    ];
    let views = Views::new(&stream, d, &view_list).unwrap();
    let table = StackTable::new(&stream, &d, std::slice::from_ref(&(0..d.nframe))).unwrap();
    let tk = TileKernels::new(&stream, &views, &table).unwrap();
    let jit = jit_compile_count();
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
            tk.obj_update()
                .update_obj_from_sums(&ex, &views, &mut x, &mut sums, inertia)
                .unwrap();
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
    assert_eq!(
        bits(&flat(&tile.1)),
        bits(&flat(&simt.1)),
        "num differs from SIMT"
    );
    assert_eq!(bits(&tile.2), bits(&simt.2), "den differs from SIMT");
    for i in (0..len).filter(|&i| lit[i]) {
        assert_eq!(bits(&flat(&tile.1[i..=i])), [0, 0], "num {i} not consumed");
        assert_eq!(tile.2[i].to_bits(), 0, "den {i} not consumed");
    }
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
    assert_eq!(jit_compile_count(), jit, "a launch compiled");
}

/// The object update rejects views of other object sizes than it was compiled for.
#[test]
fn update_obj_from_sums_rejects_views_of_other_sizes() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ex = ExecutionContext::new(stream.clone());
    let (views, _) = probe_scan(&stream, 1, [32, 32]);
    let (wider, _) = probe_scan(&stream, 1, [32, 64]);
    let update = TileObjUpdate::new(&stream, &views).unwrap();
    let len = 32 * 64;
    let mut obj = up(&stream, &noise(len, 60, 1.0));
    let mut sums = UpdateSums::zeroed(&stream, len).unwrap();
    let e = update.update_obj_from_sums(&ex, &wider, &mut obj, &mut sums, 0.01);
    assert!(matches!(e, Err(ptyrs::tile::Error::Shape(_))), "{e:?}");
}
