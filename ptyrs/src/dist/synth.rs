//! Proof of B3 on the split: a synthetic accumulation is bit-identical on 1, 2 and 4 GPUs and
//! to its host twin. Each iteration every region pulls its halo, computes its frames' per-pixel
//! contributions from the object, then each owner adds every contribution touching its tiles
//! in frame order (reading neighbours' contributions from peer memory), updates its tiles, and
//! leaves a per-tile (dot, LLK-like) partial that every GPU sums through the tile tree.

use std::ffi::c_void;
use std::sync::Arc;

use crate::base::{AddrTable, sim};
use cuda_core::{Context, DeviceBuffer, Function, launch_kernel_on_stream};

use crate::dist::exchange::{
    Exchange, Layout, frame_sources, tile_sources, tree_scratch_len, tree_sum_host,
};
use crate::dist::partition::tests::addr;
use crate::dist::worker::tests::gpus;
use crate::dist::{DistError, Partition, Scheme, Workers};

const SEED: u32 = 0x2545_f491;
const ETA: f32 = 1.0 / 1024.0;
const ITERS: usize = 3;
const SYNTH_THREADS: u32 = 256;
const TWO_M24: f32 = 1.0 / 16_777_216.0;

fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Contribution of pixel `k` of frame `f` over object value `v`; the PTX twin is in
/// `synth_frames`.
fn contrib(f: u32, k: u32, v: f32) -> f32 {
    let key = f
        .wrapping_mul(0x9e37_79b1)
        .wrapping_add(SEED)
        .wrapping_add(k.wrapping_mul(0x85eb_ca77));
    let h1 = lowbias32(key);
    let h2 = lowbias32(h1 ^ 0x68e3_1da4);
    let w = (h1 >> 8) as f32 * TWO_M24;
    let b = (h2 >> 8) as f32 * TWO_M24 - 0.5;
    v * w + b
}

/// Starting object, tile-major like the device slots; zero outside the object.
fn initial(part: &Partition) -> Vec<f32> {
    let g = part.grid();
    let (t, [h, w]) = (g.tile(), g.obj());
    let mut obj = vec![0f32; g.len() * t * t];
    for (i, o) in obj.iter_mut().enumerate() {
        let [ty, tx] = g.coords(i / (t * t));
        let (y, x) = (ty * t + i % (t * t) / t, tx * t + i % t);
        if y < h && x < w {
            *o = (lowbias32((y * w + x) as u32 ^ 0xa511_e9b3) >> 8) as f32 * TWO_M24;
        }
    }
    obj
}

/// Final object (tile-major) and the per-iteration (dot, LLK) sums.
#[derive(Debug, PartialEq)]
struct Outcome {
    obj: Vec<u32>,
    scalars: Vec<[u64; 2]>,
}

/// Host twin of one run. `reverse_frames` adds contributions in descending frame order and
/// `sequential` sums the tile partials left to right; both exist to show the test sees order.
fn host(part: &Partition, reverse_frames: bool, sequential: bool) -> Outcome {
    let g = part.grid();
    let (t, tt) = (g.tile(), g.tile() * g.tile());
    let [fy, fx] = part.frame();
    let n = fy * fx;
    let mut obj = initial(part);
    let mut scalars = Vec::new();
    for _ in 0..ITERS {
        let mut c = vec![0f32; part.nframe() * n];
        let mut llk = vec![0f64; part.nframe()];
        for (f, &[cy, cx]) in part.corners().iter().enumerate() {
            let mut lanes = vec![0f64; SYNTH_THREADS as usize];
            for k in 0..n {
                let (y, x) = (cy + k / fx, cx + k % fx);
                let v = obj[g.of_pixel(y, x) * tt + y % t * t + x % t];
                let ck = contrib(f as u32, k as u32, v);
                c[f * n + k] = ck;
                lanes[k % SYNTH_THREADS as usize] += ck as f64 * ck as f64;
            }
            llk[f] = tree_sum_host(&lanes, 1)[0];
        }
        let mut leaves = vec![0f64; 2 * g.len()];
        for tile in 0..g.len() {
            let [ty, tx] = g.coords(tile);
            let mut frames = part.tile_frames().row(tile).to_vec();
            if reverse_frames {
                frames.reverse();
            }
            let mut dot = vec![0f64; tt];
            for (p, d) in dot.iter_mut().enumerate() {
                let (y, x) = (ty * t + p / t, tx * t + p % t);
                let mut acc = 0f32;
                for &f in &frames {
                    let [cy, cx] = part.corners()[f as usize];
                    let (dy, dx) = (y.wrapping_sub(cy), x.wrapping_sub(cx));
                    if dy < fy && dx < fx {
                        acc += c[f as usize * n + dy * fx + dx];
                    }
                }
                obj[tile * tt + p] += acc * ETA;
                *d = acc as f64 * acc as f64;
            }
            leaves[2 * tile] = tree_sum_host(&dot, 1)[0];
            leaves[2 * tile + 1] = part
                .home_frames()
                .row(tile)
                .iter()
                .fold(0.0, |s, &f| s + llk[f as usize]);
        }
        let sum = if sequential {
            leaves
                .chunks(2)
                .fold(vec![0.0, 0.0], |s, l| vec![s[0] + l[0], s[1] + l[1]])
        } else {
            tree_sum_host(&leaves, 2)
        };
        scalars.push([sum[0].to_bits(), sum[1].to_bits()]);
    }
    Outcome {
        obj: obj.iter().map(|v| v.to_bits()).collect(),
        scalars,
    }
}

struct Tables {
    halo: DeviceBuffer<u64>,
    frames: DeviceBuffer<u64>,
    leaves: DeviceBuffer<u64>,
}

/// One region's device state.
struct Dev {
    ex: Exchange,
    synth: Function,
    gather: Function,
    nown: usize,
    nlocal: usize,
    obj: DeviceBuffer<f32>,
    slot_of_tile: DeviceBuffer<i32>,
    own: DeviceBuffer<u32>,
    frames: DeviceBuffer<u32>,
    frame_local: DeviceBuffer<i32>,
    corners: DeviceBuffer<u32>,
    tf: [DeviceBuffer<u32>; 2],
    hf: [DeviceBuffer<u32>; 2],
    contrib: DeviceBuffer<f32>,
    llk: DeviceBuffer<f64>,
    partials: DeviceBuffer<f64>,
    scratch: DeviceBuffer<f64>,
    scalars: DeviceBuffer<f64>,
    tables: Option<Tables>,
}

impl Dev {
    fn new(g: &crate::dist::Gpu, part: &Partition, lay: &Layout) -> Result<Self, DistError> {
        let s = g.stream();
        let grid = part.grid();
        let tt = grid.tile() * grid.tile();
        let [fy, fx] = part.frame();
        let start = initial(part);
        let mut obj = vec![0f32; lay.slots().len() * tt];
        for (j, &tile) in lay.slots()[..lay.nown()].iter().enumerate() {
            let t = tile as usize;
            obj[j * tt..(j + 1) * tt].copy_from_slice(&start[t * tt..(t + 1) * tt]);
        }
        let corners: Vec<u32> = part
            .corners()
            .iter()
            .flat_map(|c| [c[0] as u32, c[1] as u32])
            .collect();
        let m = g.ctx().load_module_from_ptx_src(PTX)?;
        let (tf, hf) = (part.tile_frames(), part.home_frames());
        Ok(Self {
            ex: Exchange::load(g.ctx())?,
            synth: m.load_function("synth_frames")?,
            gather: m.load_function("gather")?,
            nown: lay.nown(),
            nlocal: lay.frames().len(),
            obj: DeviceBuffer::from_host(s, &obj)?,
            slot_of_tile: DeviceBuffer::from_host(s, lay.slot_of_tile())?,
            own: DeviceBuffer::from_host(s, &lay.slots()[..lay.nown()])?,
            frames: DeviceBuffer::from_host(s, lay.frames())?,
            frame_local: DeviceBuffer::from_host(s, lay.frame_local())?,
            corners: DeviceBuffer::from_host(s, &corners)?,
            tf: [
                DeviceBuffer::from_host(s, &tf.start)?,
                DeviceBuffer::from_host(s, &tf.items)?,
            ],
            hf: [
                DeviceBuffer::from_host(s, &hf.start)?,
                DeviceBuffer::from_host(s, &hf.items)?,
            ],
            contrib: DeviceBuffer::zeroed(s, lay.frames().len() * fy * fx)?,
            llk: DeviceBuffer::zeroed(s, lay.frames().len())?,
            partials: DeviceBuffer::zeroed(s, 2 * lay.nown())?,
            scratch: DeviceBuffer::zeroed(s, tree_scratch_len(grid.len(), 2))?,
            scalars: DeviceBuffer::zeroed(s, 2 * ITERS)?,
            tables: None,
        })
    }

    /// `ITERS` iterations; the halo is pulled in iteration `it` only when `pull(it)`.
    fn iterate(
        &self,
        g: &mut crate::dist::Gpu,
        part: &Partition,
        pull: impl Fn(usize) -> bool,
    ) -> Result<(Vec<f32>, Vec<f64>), DistError> {
        let s = g.stream().clone();
        let grid = part.grid();
        let tile = grid.tile() as u32;
        let tt = tile * tile;
        let [fy, fx] = part.frame().map(|v| v as u32);
        let ntx = grid.dims()[1] as u32;
        let tabs = self.tables.as_ref().ok_or(DistError::Aborted)?;
        for it in 0..ITERS {
            if pull(it) {
                let dst = self.obj.cu_deviceptr() + (self.nown as u64) * u64::from(tt) * 4;
                // SAFETY: slots `nown..` of `obj` are the halo, one source per halo slot; the
                // owners last wrote those tiles before the previous fence.
                unsafe { self.ex.halo_pull(&s, dst, &tabs.halo, tt)? };
            }
            if self.nlocal > 0 {
                let mut a = [
                    self.obj.cu_deviceptr(),
                    self.slot_of_tile.cu_deviceptr(),
                    self.frames.cu_deviceptr(),
                    self.corners.cu_deviceptr(),
                    self.contrib.cu_deviceptr(),
                    self.llk.cu_deviceptr(),
                ];
                let mut b = [fy, fx, tile, ntx, SEED];
                let mut params: Vec<*mut c_void> = a
                    .iter_mut()
                    .map(|p| (p as *mut u64).cast())
                    .chain(b.iter_mut().map(|p| (p as *mut u32).cast()))
                    .collect();
                // SAFETY: one block per local frame; every footprint lies in this region's
                // slots (`Layout::new`), and `contrib`/`llk` hold `nlocal` frames.
                unsafe {
                    launch_kernel_on_stream(
                        &self.synth,
                        (self.nlocal as u32, 1, 1),
                        (SYNTH_THREADS, 1, 1),
                        0,
                        &s,
                        &mut params,
                    )?
                };
            }
            g.fence()?;
            let mut a = [
                self.obj.cu_deviceptr(),
                self.own.cu_deviceptr(),
                self.tf[0].cu_deviceptr(),
                self.tf[1].cu_deviceptr(),
                self.hf[0].cu_deviceptr(),
                self.hf[1].cu_deviceptr(),
                self.corners.cu_deviceptr(),
                tabs.frames.cu_deviceptr(),
                self.llk.cu_deviceptr(),
                self.frame_local.cu_deviceptr(),
                self.partials.cu_deviceptr(),
            ];
            let mut b = [fy, fx, tile, ntx];
            let mut eta = ETA;
            let mut params: Vec<*mut c_void> = a
                .iter_mut()
                .map(|p| (p as *mut u64).cast())
                .chain(b.iter_mut().map(|p| (p as *mut u32).cast()))
                .chain([(&raw mut eta).cast()])
                .collect();
            // SAFETY: one block of `tile²` threads per own tile; every frame touching an own
            // tile has its contribution at `frames[f]`, complete since the fence above.
            unsafe {
                launch_kernel_on_stream(
                    &self.gather,
                    (self.nown as u32, 1, 1),
                    (tt, 1, 1),
                    0,
                    &s,
                    &mut params,
                )?
            };
            g.fence()?;
            let out = self.scalars.cu_deviceptr() + (it * 16) as u64;
            // SAFETY: `leaves` holds one partial address per tile, written before the fence.
            unsafe { self.ex.tree_sum(&s, &tabs.leaves, 2, &self.scratch, out)? };
        }
        let mut obj = self.obj.to_host_vec(&s)?;
        obj.truncate(self.nown * tt as usize);
        Ok((obj, self.scalars.to_host_vec(&s)?))
    }
}

/// Runs the synthetic accumulation of `part` on `ordinals` (region `r` on `ordinals[r]`).
fn device(
    part: &Arc<Partition>,
    ordinals: &[usize],
    pull: fn(usize) -> bool,
) -> Result<Outcome, DistError> {
    let lay = Arc::new(Layout::all(part)?);
    let w = {
        let (part, lay) = (part.clone(), lay.clone());
        Workers::spawn(ordinals, move |g| Dev::new(g, &part, &lay[g.region()]))?
    };
    let bases = w.run(|_, d| {
        Ok([
            d.obj.cu_deviceptr(),
            d.contrib.cu_deviceptr(),
            d.partials.cu_deviceptr(),
        ])
    })?;
    let base = |i: usize| bases.iter().map(|b| b[i]).collect::<Vec<u64>>();
    let tt = (part.grid().tile() * part.grid().tile()) as u64;
    let [fy, fx] = part.frame();
    let tiles = tile_sources(part, &lay, &base(0), tt * 4);
    let frames = Arc::new(frame_sources(part, &lay, &base(1), (fy * fx * 4) as u64));
    let leaves = Arc::new(tile_sources(part, &lay, &base(2), 16));
    let halo: Arc<Vec<Vec<u64>>> = Arc::new(lay.iter().map(|l| l.halo_sources(&tiles)).collect());
    w.run(move |g, d| {
        let s = g.stream();
        d.tables = Some(Tables {
            halo: DeviceBuffer::from_host(s, &halo[g.region()])?,
            frames: DeviceBuffer::from_host(s, &frames)?,
            leaves: DeviceBuffer::from_host(s, &leaves)?,
        });
        Ok(())
    })?;
    let p = part.clone();
    let out = w.run(move |g, d| d.iterate(g, &p, pull))?;
    let grid = part.grid();
    let tt = tt as usize;
    let mut obj = vec![0u32; grid.len() * tt];
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for ((o, scalars), l) in out.iter().zip(lay.iter()) {
        assert_eq!(
            bits(scalars),
            bits(&out[0].1),
            "every GPU's tree gives the same sums"
        );
        for (j, &t) in l.slots()[..l.nown()].iter().enumerate() {
            let t = t as usize;
            for (dst, v) in obj[t * tt..(t + 1) * tt].iter_mut().zip(&o[j * tt..]) {
                *dst = v.to_bits();
            }
        }
    }
    let scalars = out[0]
        .1
        .chunks(2)
        .map(|c| [c[0].to_bits(), c[1].to_bits()])
        .collect();
    Ok(Outcome { obj, scalars })
}

fn scans() -> Vec<(&'static str, AddrTable)> {
    vec![
        ("raster", addr(sim::raster(16, 24, 8e-8), 64)),
        ("fermat", addr(sim::fermat(300, 8e-8), 64)),
        // Frame not a tile multiple and partial edge tiles.
        ("odd", addr(sim::raster(9, 11, 7e-8), 40)),
    ]
}

const SCHEMES: [Scheme; 3] = [
    Scheme::Strips,
    Scheme::Sectors { centre: None },
    Scheme::Grid,
];

#[test]
fn synthetic_accumulation_is_bit_identical_on_1_2_and_4_gpus() {
    // Hold every primary context so each spawn reuses it.
    let _keep: Vec<_> = (0..4).map(|o| Context::new(o).unwrap()).collect();
    let numa = gpus(4);
    let mut runs: Vec<(Scheme, Vec<usize>)> = vec![
        (Scheme::Strips, vec![numa[0]]),
        (Scheme::Strips, vec![numa[3]]),
        (Scheme::Strips, vec![numa[1], numa[2]]),
        (Scheme::Strips, vec![numa[3], numa[1], numa[0], numa[2]]),
    ];
    for scheme in SCHEMES {
        runs.push((scheme, numa[..2].to_vec()));
        runs.push((scheme, numa.clone()));
    }
    for (name, a) in scans() {
        let one = Arc::new(
            Partition::from_addr(&a, crate::dist::DEFAULT_TILE, Scheme::Strips, 1).unwrap(),
        );
        let want = host(&one, false, false);
        for (scheme, ord) in &runs {
            let part = Arc::new(
                Partition::from_addr(&a, crate::dist::DEFAULT_TILE, *scheme, ord.len()).unwrap(),
            );
            let got = device(&part, ord, |_| true).unwrap();
            assert!(
                got == want,
                "{name} {scheme:?} on {ord:?}: obj {} of {} words differ, scalars {:?} vs {:?}",
                got.obj
                    .iter()
                    .zip(&want.obj)
                    .filter(|(a, b)| a != b)
                    .count(),
                want.obj.len(),
                got.scalars,
                want.scalars
            );
        }
    }
}

/// Regions sharing a GPU run the same kernels, pointer tables and fences without peer access,
/// so independence from the region count is checked on each device alone.
#[test]
fn synthetic_accumulation_is_bit_identical_for_1_2_and_4_regions_on_one_gpu() {
    for d in crate::dist::topo::devices().unwrap() {
        let _keep = Context::new(d.ordinal).unwrap();
        for (name, a) in scans() {
            let one = Arc::new(
                Partition::from_addr(&a, crate::dist::DEFAULT_TILE, Scheme::Strips, 1).unwrap(),
            );
            let want = host(&one, false, false);
            for scheme in SCHEMES {
                for n in [1, 2, 4] {
                    let part = Arc::new(
                        Partition::from_addr(&a, crate::dist::DEFAULT_TILE, scheme, n).unwrap(),
                    );
                    let got = device(&part, &vec![d.ordinal; n], |_| true).unwrap();
                    assert!(
                        got == want,
                        "{name} {scheme:?}, {n} regions on {}: obj {} of {} words differ, \
                         scalars {:?} vs {:?}",
                        d.pci_bus_id,
                        got.obj
                            .iter()
                            .zip(&want.obj)
                            .filter(|(a, b)| a != b)
                            .count(),
                        want.obj.len(),
                        got.scalars,
                        want.scalars
                    );
                }
            }
        }
    }
}

/// The equalities above are not vacuous: frame order, tree shape and the halo pull each
/// change the bits.
#[test]
fn order_and_halo_are_visible_in_the_bits() {
    let a = &scans()[0].1;
    let one = Partition::from_addr(a, crate::dist::DEFAULT_TILE, Scheme::Strips, 1).unwrap();
    let want = host(&one, false, false);
    assert_ne!(host(&one, true, false).obj, want.obj, "frame order");
    assert_ne!(host(&one, false, true).scalars, want.scalars, "tree shape");
    let four =
        Arc::new(Partition::from_addr(a, crate::dist::DEFAULT_TILE, Scheme::Strips, 4).unwrap());
    let stale = device(&four, &[0; 4], |it| it == 0).unwrap();
    assert_ne!(stale.obj, want.obj, "halo pulled only once");
    assert_eq!(
        stale.scalars[0], want.scalars[0],
        "first iteration is unaffected"
    );
}

const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

// Block b computes local frame f = frames[b]: thread i covers pixels k = i, i + ntid, ...
// c = v * w + b from the object slot holding pixel k and a hash of (f, k); contrib[b][k] = c;
// llk[b] = block tree over the threads' sequential f64 sums of c * c.
.visible .entry synth_frames(
    .param .u64 p_obj,
    .param .u64 p_slot,
    .param .u64 p_frames,
    .param .u64 p_corners,
    .param .u64 p_contrib,
    .param .u64 p_llk,
    .param .u32 p_fy,
    .param .u32 p_fx,
    .param .u32 p_tile,
    .param .u32 p_ntx,
    .param .u32 p_seed
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<6>;
    .reg .f64 %fd<6>;
    .shared .align 8 .f64 s_sh[1024];

    ld.param.u64 %rd1, [p_obj];
    ld.param.u64 %rd2, [p_slot];
    ld.param.u64 %rd3, [p_frames];
    ld.param.u64 %rd4, [p_corners];
    ld.param.u64 %rd5, [p_contrib];
    ld.param.u64 %rd6, [p_llk];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    cvta.to.global.u64 %rd5, %rd5;
    cvta.to.global.u64 %rd6, %rd6;
    ld.param.u32 %r1, [p_fy];
    ld.param.u32 %r2, [p_fx];
    ld.param.u32 %r3, [p_tile];
    ld.param.u32 %r4, [p_ntx];
    ld.param.u32 %r5, [p_seed];
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ntid.x;
    mul.wide.u32 %rd7, %r6, 4;
    add.s64 %rd7, %rd3, %rd7;
    ld.global.u32 %r9, [%rd7];
    mul.wide.u32 %rd8, %r9, 8;
    add.s64 %rd8, %rd4, %rd8;
    ld.global.u32 %r10, [%rd8];
    ld.global.u32 %r11, [%rd8+4];
    mul.lo.u32 %r12, %r1, %r2;
    mul.lo.u32 %r13, %r3, %r3;
    mul.wide.u32 %rd9, %r6, %r12;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd9, %rd5, %rd9;
    mul.lo.u32 %r14, %r9, 0x9E3779B1;
    add.u32 %r14, %r14, %r5;
    mov.f64 %fd1, 0d0000000000000000;
    mov.u32 %r15, %r7;
SF_LOOP:
    setp.ge.u32 %p1, %r15, %r12;
    @%p1 bra SF_DONE;
    div.u32 %r16, %r15, %r2;
    mul.lo.u32 %r17, %r16, %r2;
    sub.u32 %r17, %r15, %r17;
    add.u32 %r18, %r10, %r16;
    add.u32 %r19, %r11, %r17;
    div.u32 %r20, %r18, %r3;
    mul.lo.u32 %r21, %r20, %r3;
    sub.u32 %r21, %r18, %r21;
    div.u32 %r22, %r19, %r3;
    mul.lo.u32 %r23, %r22, %r3;
    sub.u32 %r23, %r19, %r23;
    mad.lo.u32 %r24, %r20, %r4, %r22;
    mul.wide.u32 %rd10, %r24, 4;
    add.s64 %rd10, %rd2, %rd10;
    ld.global.u32 %r25, [%rd10];
    mad.lo.u32 %r26, %r21, %r3, %r23;
    mul.wide.u32 %rd11, %r25, %r13;
    cvt.u64.u32 %rd12, %r26;
    add.s64 %rd11, %rd11, %rd12;
    shl.b64 %rd11, %rd11, 2;
    add.s64 %rd11, %rd1, %rd11;
    ld.global.f32 %f1, [%rd11];
    mad.lo.u32 %r27, %r15, 0x85EBCA77, %r14;
    shr.u32 %r29, %r27, 16;
    xor.b32 %r28, %r27, %r29;
    mul.lo.u32 %r28, %r28, 0x7FEB352D;
    shr.u32 %r29, %r28, 15;
    xor.b32 %r28, %r28, %r29;
    mul.lo.u32 %r28, %r28, 0x846CA68B;
    shr.u32 %r29, %r28, 16;
    xor.b32 %r28, %r28, %r29;
    xor.b32 %r30, %r28, 0x68E31DA4;
    shr.u32 %r29, %r30, 16;
    xor.b32 %r30, %r30, %r29;
    mul.lo.u32 %r30, %r30, 0x7FEB352D;
    shr.u32 %r29, %r30, 15;
    xor.b32 %r30, %r30, %r29;
    mul.lo.u32 %r30, %r30, 0x846CA68B;
    shr.u32 %r29, %r30, 16;
    xor.b32 %r30, %r30, %r29;
    shr.u32 %r31, %r28, 8;
    cvt.rn.f32.u32 %f2, %r31;
    mul.rn.f32 %f2, %f2, 0f33800000;
    shr.u32 %r32, %r30, 8;
    cvt.rn.f32.u32 %f3, %r32;
    mul.rn.f32 %f3, %f3, 0f33800000;
    sub.rn.f32 %f3, %f3, 0f3F000000;
    mul.rn.f32 %f4, %f1, %f2;
    add.rn.f32 %f4, %f4, %f3;
    mul.wide.u32 %rd13, %r15, 4;
    add.s64 %rd13, %rd9, %rd13;
    st.global.f32 [%rd13], %f4;
    cvt.f64.f32 %fd2, %f4;
    mul.rn.f64 %fd2, %fd2, %fd2;
    add.rn.f64 %fd1, %fd1, %fd2;
    add.u32 %r15, %r15, %r8;
    bra SF_LOOP;
SF_DONE:
    mov.u64 %rd14, s_sh;
    mul.wide.u32 %rd15, %r7, 8;
    add.s64 %rd15, %rd14, %rd15;
    st.shared.f64 [%rd15], %fd1;
    barrier.sync 0;
    shr.u32 %r33, %r8, 1;
SF_TREE:
    setp.eq.u32 %p2, %r33, 0;
    @%p2 bra SF_TREE_DONE;
    setp.lt.u32 %p3, %r7, %r33;
    mul.wide.u32 %rd16, %r33, 8;
    add.s64 %rd16, %rd15, %rd16;
    @%p3 ld.shared.f64 %fd3, [%rd15];
    @%p3 ld.shared.f64 %fd4, [%rd16];
    @%p3 add.rn.f64 %fd3, %fd3, %fd4;
    @%p3 st.shared.f64 [%rd15], %fd3;
    barrier.sync 0;
    shr.u32 %r33, %r33, 1;
    bra SF_TREE;
SF_TREE_DONE:
    setp.ne.u32 %p2, %r7, 0;
    @%p2 bra SF_END;
    ld.shared.f64 %fd5, [%rd14];
    mul.wide.u32 %rd17, %r6, 8;
    add.s64 %rd17, %rd6, %rd17;
    st.global.f64 [%rd17], %fd5;
SF_END:
    ret;
}

// Block j owns tile t = own[j], thread p its pixel (p / tile, p % tile); ntid = tile * tile.
// acc = sum over f in tile_frames[t] ascending of src[f][pixel - corner f] where inside the
// frame; obj[j][p] += acc * eta; part[j] = (block tree of acc * acc, sequential sum of
// llk[local f] over home_frames[t]).
.visible .entry gather(
    .param .u64 p_obj,
    .param .u64 p_own,
    .param .u64 p_tfs,
    .param .u64 p_tfi,
    .param .u64 p_hfs,
    .param .u64 p_hfi,
    .param .u64 p_corners,
    .param .u64 p_src,
    .param .u64 p_llk,
    .param .u64 p_flocal,
    .param .u64 p_part,
    .param .u32 p_fy,
    .param .u32 p_fx,
    .param .u32 p_tile,
    .param .u32 p_ntx,
    .param .f32 p_eta
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<48>;
    .reg .f32 %f<8>;
    .reg .f64 %fd<8>;
    .shared .align 8 .f64 g_sh[1024];

    ld.param.u64 %rd1, [p_obj];
    ld.param.u64 %rd2, [p_own];
    ld.param.u64 %rd3, [p_tfs];
    ld.param.u64 %rd4, [p_tfi];
    ld.param.u64 %rd5, [p_hfs];
    ld.param.u64 %rd6, [p_hfi];
    ld.param.u64 %rd7, [p_corners];
    ld.param.u64 %rd8, [p_src];
    ld.param.u64 %rd9, [p_llk];
    ld.param.u64 %rd10, [p_flocal];
    ld.param.u64 %rd11, [p_part];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    cvta.to.global.u64 %rd5, %rd5;
    cvta.to.global.u64 %rd6, %rd6;
    cvta.to.global.u64 %rd7, %rd7;
    cvta.to.global.u64 %rd8, %rd8;
    cvta.to.global.u64 %rd9, %rd9;
    cvta.to.global.u64 %rd10, %rd10;
    cvta.to.global.u64 %rd11, %rd11;
    ld.param.u32 %r1, [p_fy];
    ld.param.u32 %r2, [p_fx];
    ld.param.u32 %r3, [p_tile];
    ld.param.u32 %r4, [p_ntx];
    ld.param.f32 %f1, [p_eta];
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ntid.x;
    mul.wide.u32 %rd20, %r5, 4;
    add.s64 %rd20, %rd2, %rd20;
    ld.global.u32 %r8, [%rd20];
    div.u32 %r9, %r8, %r4;
    mul.lo.u32 %r10, %r9, %r4;
    sub.u32 %r10, %r8, %r10;
    div.u32 %r11, %r6, %r3;
    mul.lo.u32 %r12, %r11, %r3;
    sub.u32 %r12, %r6, %r12;
    mad.lo.u32 %r13, %r9, %r3, %r11;
    mad.lo.u32 %r14, %r10, %r3, %r12;
    mul.wide.u32 %rd21, %r8, 4;
    add.s64 %rd22, %rd3, %rd21;
    ld.global.u32 %r15, [%rd22];
    ld.global.u32 %r16, [%rd22+4];
    mov.f32 %f2, 0f00000000;
GA_LOOP:
    setp.ge.u32 %p1, %r15, %r16;
    @%p1 bra GA_DONE;
    mul.wide.u32 %rd23, %r15, 4;
    add.s64 %rd23, %rd4, %rd23;
    ld.global.u32 %r17, [%rd23];
    mul.wide.u32 %rd24, %r17, 8;
    add.s64 %rd25, %rd7, %rd24;
    ld.global.u32 %r18, [%rd25];
    ld.global.u32 %r19, [%rd25+4];
    sub.u32 %r20, %r13, %r18;
    sub.u32 %r21, %r14, %r19;
    setp.lt.u32 %p2, %r20, %r1;
    setp.lt.and.u32 %p2, %r21, %r2, %p2;
    @!%p2 bra GA_NEXT;
    add.s64 %rd26, %rd8, %rd24;
    ld.global.u64 %rd27, [%rd26];
    cvta.to.global.u64 %rd27, %rd27;
    mad.lo.u32 %r22, %r20, %r2, %r21;
    mul.wide.u32 %rd28, %r22, 4;
    add.s64 %rd28, %rd27, %rd28;
    ld.global.f32 %f3, [%rd28];
    add.rn.f32 %f2, %f2, %f3;
GA_NEXT:
    add.u32 %r15, %r15, 1;
    bra GA_LOOP;
GA_DONE:
    mul.lo.u32 %r23, %r3, %r3;
    mul.wide.u32 %rd29, %r5, %r23;
    cvt.u64.u32 %rd30, %r6;
    add.s64 %rd29, %rd29, %rd30;
    shl.b64 %rd29, %rd29, 2;
    add.s64 %rd29, %rd1, %rd29;
    ld.global.f32 %f4, [%rd29];
    mul.rn.f32 %f5, %f2, %f1;
    add.rn.f32 %f4, %f4, %f5;
    st.global.f32 [%rd29], %f4;
    cvt.f64.f32 %fd1, %f2;
    mul.rn.f64 %fd1, %fd1, %fd1;
    mov.u64 %rd31, g_sh;
    mul.wide.u32 %rd32, %r6, 8;
    add.s64 %rd32, %rd31, %rd32;
    st.shared.f64 [%rd32], %fd1;
    barrier.sync 0;
    shr.u32 %r24, %r7, 1;
GA_TREE:
    setp.eq.u32 %p3, %r24, 0;
    @%p3 bra GA_TREE_DONE;
    setp.lt.u32 %p4, %r6, %r24;
    mul.wide.u32 %rd33, %r24, 8;
    add.s64 %rd33, %rd32, %rd33;
    @%p4 ld.shared.f64 %fd2, [%rd32];
    @%p4 ld.shared.f64 %fd3, [%rd33];
    @%p4 add.rn.f64 %fd2, %fd2, %fd3;
    @%p4 st.shared.f64 [%rd32], %fd2;
    barrier.sync 0;
    shr.u32 %r24, %r24, 1;
    bra GA_TREE;
GA_TREE_DONE:
    setp.ne.u32 %p5, %r6, 0;
    @%p5 bra GA_END;
    add.s64 %rd34, %rd5, %rd21;
    ld.global.u32 %r25, [%rd34];
    ld.global.u32 %r26, [%rd34+4];
    mov.f64 %fd4, 0d0000000000000000;
GA_HOME:
    setp.ge.u32 %p6, %r25, %r26;
    @%p6 bra GA_HOME_DONE;
    mul.wide.u32 %rd35, %r25, 4;
    add.s64 %rd35, %rd6, %rd35;
    ld.global.u32 %r27, [%rd35];
    mul.wide.u32 %rd36, %r27, 4;
    add.s64 %rd36, %rd10, %rd36;
    ld.global.u32 %r28, [%rd36];
    mul.wide.u32 %rd37, %r28, 8;
    add.s64 %rd37, %rd9, %rd37;
    ld.global.f64 %fd5, [%rd37];
    add.rn.f64 %fd4, %fd4, %fd5;
    add.u32 %r25, %r25, 1;
    bra GA_HOME;
GA_HOME_DONE:
    ld.shared.f64 %fd6, [%rd31];
    mul.wide.u32 %rd38, %r5, 16;
    add.s64 %rd38, %rd11, %rd38;
    st.global.f64 [%rd38], %fd6;
    st.global.f64 [%rd38+8], %fd4;
GA_END:
    ret;
}
"#;
