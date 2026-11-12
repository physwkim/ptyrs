//! One AP epoch as one CUDA graph: the SIMT forward model, Fourier projection and LLK
//! partials of `ptyrs_gpu`, then the Tile object and probe updates and LLK totals. The
//! epoch is captured before any Tile kernel has launched, the graph replayed, and each
//! replay compared bit for bit with the same epoch run directly.

use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_gpu::{
    C32, Error, Kernels, Lines, LlkPartial, StackTable, UpdateSums, ViewAddr, ViewDims, Views,
    record,
};
use ptyrs_tile::{TileKernels, llk_sums_scratch_len};

const N: u32 = 128;
const EPOCHS: usize = 4;

/// Deterministic values in `[-1, 1)`.
fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Two probe modes on one 256x256 object, twelve frames on a 4x3 raster with sub-pixel
/// offsets.
fn dims() -> (ViewDims, Vec<ViewAddr>) {
    let d = ViewDims {
        nframe: 12,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [N, N],
        obj: [256, 256],
        quad_phase: 0.0,
    };
    let views = (0..d.nframe)
        .flat_map(|f| {
            let (oy, ox) = (8 + 38 * (f / 4), 4 + 38 * (f % 4));
            let (fy, fx) = (0.1 * (f % 3) as f32, 0.15 * (f % 5) as f32);
            (0..d.nmode).map(move |m| ViewAddr {
                oy,
                ox,
                fy,
                fx,
                probe: m,
                obj: 0,
            })
        })
        .collect();
    (d, views)
}

/// Probe mode `m`: a disc of radius 30 with a smooth phase, scaled by `1 / (m + 1)`.
fn probe(seed: u64) -> Vec<C32> {
    let jitter = noise(2 * (N * N) as usize, seed);
    (0..2 * (N * N) as usize)
        .map(|i| {
            let (m, y, x) = (
                i / (N * N) as usize,
                (i / N as usize) % N as usize,
                i % N as usize,
            );
            let (dy, dx) = (y as f32 - 64.0, x as f32 - 64.0);
            let a = if dy * dy + dx * dx < 900.0 { 1.0 } else { 0.0 } / (m + 1) as f32;
            let p = 0.002 * (dy * dy - dx * dx) + 0.05 * jitter[i];
            C32 {
                re: a * p.cos(),
                im: a * p.sin(),
            }
        })
        .collect()
}

/// Plane `v` of `psi` as a buffer of its own (same allocation), for the column FFTs, which
/// `Lines` can only address from element 0.
fn plane(psi: &DeviceBuffer<C32>, v: usize) -> DeviceBuffer<C32> {
    let n2 = (N * N) as usize;
    let bytes = n2 * size_of::<C32>();
    let raw = psi.storage().slice_bytes(v * bytes, bytes).unwrap();
    // SAFETY: the bytes are initialized C32 of `psi`, and every launch on them leases them.
    unsafe { DeviceBuffer::from_storage(raw, n2) }.unwrap()
}

/// Unnormalized 2D FFT of the `nk` planes of `psi`: rows in one launch, then the columns
/// of each plane.
fn fft2(
    k: &Kernels,
    ctx: &ExecutionContext,
    psi: &mut DeviceBuffer<C32>,
    nk: u32,
    inverse: bool,
) -> Result<(), Error> {
    let rows = Lines {
        n: N,
        count: nk * N,
        line_stride: N,
        elem_stride: 1,
    };
    k.fft_lines_inplace(ctx, psi, rows, inverse)?;
    let cols = Lines {
        n: N,
        count: N,
        line_stride: 1,
        elem_stride: N,
    };
    for v in 0..nk as usize {
        k.fft_lines_inplace(ctx, &mut plane(psi, v), cols, inverse)?;
    }
    Ok(())
}

struct Scan<'a> {
    k: &'a Kernels,
    /// The Tile kernels of `views` and `table`.
    tk: &'a TileKernels<'a>,
    views: &'a Views,
    /// One stack, the whole scan.
    table: &'a StackTable,
    counts: &'a DeviceBuffer<u16>,
}

/// What an epoch reads and writes; all of it allocated before capture.
struct State {
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    psi: DeviceBuffer<C32>,
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
    partials: DeviceBuffer<LlkPartial>,
    llk_scratch: DeviceBuffer<f64>,
    llk: DeviceBuffer<f64>,
}

/// One AP epoch over the whole scan as one stack: the object update, then the probe
/// update from the new object.
fn epoch(ctx: &ExecutionContext, scan: &Scan, s: &mut State) -> Result<(), Error> {
    let (k, tk, d, table) = (scan.k, scan.tk, scan.views.dims(), scan.table);
    let nk = d.nframe * d.nmode;
    k.obj_probe_to_psi(ctx, scan.views, table.at(0), &s.obj, &s.probe, &mut s.psi)?;
    fft2(k, ctx, &mut s.psi, nk, false)?;
    k.llk(ctx, d, table.at(0), scan.counts, &s.psi, &mut s.partials)?;
    k.apply_amplitude(ctx, d, table.at(0), scan.counts, &mut s.psi)?;
    fft2(k, ctx, &mut s.psi, nk, true)?;
    tk.psi_to_obj(ctx, table.at(0), &s.psi, &s.probe, &mut s.obj_sums)?;
    tk.obj_update()
        .update_obj_from_sums(ctx, scan.views, &mut s.obj, &mut s.obj_sums, 0.01)?;
    tk.psi_to_probe(ctx, table.at(0), &s.psi, &s.obj, &mut s.probe_sums)?;
    tk.probe_update()
        .update_probe_from_sums(ctx, &mut s.probe, &mut s.probe_sums, 0.01)?;
    tk.llk_sums(ctx, &s.partials, &mut s.llk_scratch, &mut s.llk)
}

/// The bits of the object, the probe and the LLK totals.
fn bits(stream: &CudaStream, s: &State) -> (Vec<u32>, Vec<u64>) {
    let c = |b: &DeviceBuffer<C32>| {
        b.to_host_vec(stream)
            .unwrap()
            .into_iter()
            .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
    };
    let llk = s.llk.to_host_vec(stream).unwrap();
    (
        c(&s.obj).chain(c(&s.probe)).collect(),
        llk.iter().map(|v| v.to_bits()).collect(),
    )
}

#[test]
fn ap_epoch_graph_replays_bit_for_bit() {
    let dev = CudaContext::new(0).expect("CUDA device 0");
    let k = Kernels::load(&dev).expect("load ptyrs-gpu kernels");
    // Capture needs a stream of its own; the legacy default stream cannot capture.
    let stream: Arc<CudaStream> = dev.new_stream().unwrap();
    let (d, view_list) = dims();
    let views = Views::new(&stream, d, &view_list).unwrap();
    let (n2, nk) = ((N * N) as usize, (d.nframe * d.nmode) as usize);
    let obj_len = (d.obj[0] * d.obj[1]) as usize;
    let up = |v: &[C32]| DeviceBuffer::from_host(&stream, v).unwrap();

    // Counts: the far-field intensity of a known object and probe, scaled to 2e4 at most.
    let truth: Vec<C32> = noise(obj_len, 5)
        .iter()
        .map(|&p| C32 {
            re: (0.8 * p).cos(),
            im: (0.8 * p).sin(),
        })
        .collect();
    let (truth, true_probe) = (up(&truth), up(&probe(1)));
    let mut psi = DeviceBuffer::<C32>::zeroed(&stream, nk * n2).unwrap();
    let mut icalc = DeviceBuffer::<f32>::zeroed(&stream, d.nframe as usize * n2).unwrap();
    let table = StackTable::new(&stream, &d, std::slice::from_ref(&(0..d.nframe))).unwrap();
    record(|ctx| {
        k.obj_probe_to_psi(ctx, &views, table.at(0), &truth, &true_probe, &mut psi)?;
        fft2(&k, ctx, &mut psi, nk as u32, false)?;
        k.mode_intensity(ctx, &d, table.at(0), &psi, &mut icalc)
    })
    .sync_on(&stream)
    .unwrap();
    let icalc = icalc.to_host_vec(&stream).unwrap();
    let scale = 2e4 / icalc.iter().fold(0f32, |m, &v| m.max(v));
    let counts: Vec<u16> = icalc.iter().map(|&v| (v * scale).round() as u16).collect();
    let counts = DeviceBuffer::from_host(&stream, &counts).unwrap();
    let tk = TileKernels::new(&stream, &views, &table).unwrap();
    let jit = cutile::tile_kernel::jit_compile_count();
    let scan = Scan {
        k: &k,
        tk: &tk,
        views: &views,
        table: &table,
        counts: &counts,
    };

    // Start: a flat object and a perturbed probe.
    let obj0 = vec![C32 { re: 1.0, im: 0.0 }; obj_len];
    let probe0 = probe(2);
    let mut s = State {
        obj: up(&obj0),
        probe: up(&probe0),
        psi,
        obj_sums: UpdateSums::zeroed(&stream, obj_len).unwrap(),
        probe_sums: UpdateSums::zeroed(&stream, probe0.len()).unwrap(),
        partials: DeviceBuffer::zeroed(&stream, d.nframe as usize).unwrap(),
        llk_scratch: DeviceBuffer::zeroed(&stream, llk_sums_scratch_len(d.nframe)).unwrap(),
        llk: DeviceBuffer::zeroed(&stream, 4).unwrap(),
    };

    // Capture before any Tile kernel has launched: building `tk` compiled them all.
    let start = bits(&stream, &s);
    let graph = CudaGraph::capture(stream.clone(), record(|ctx| epoch(ctx, &scan, &mut s)))
        .expect("capture the epoch");
    let jit_left = |what: &str| {
        assert_eq!(
            cutile::tile_kernel::jit_compile_count(),
            jit,
            "{what} compiled a Tile kernel"
        );
    };
    jit_left("capture");
    assert!(bits(&stream, &s).0 == start.0, "capture ran work");
    let mut replayed = Vec::with_capacity(EPOCHS);
    for _ in 0..EPOCHS {
        graph.launch().sync_on(&stream).unwrap();
        replayed.push(bits(&stream, &s));
    }
    jit_left("replay");
    // The updates consume every sum an epoch adds, so the next epoch starts from zero.
    for (what, sums) in [("object", &s.obj_sums), ("probe", &s.probe_sums)] {
        let num = sums.num.to_host_vec(&stream).unwrap();
        let den = sums.den.to_host_vec(&stream).unwrap();
        assert!(
            num.iter()
                .all(|c| c.re.to_bits() == 0 && c.im.to_bits() == 0)
                && den.iter().all(|v| v.to_bits() == 0),
            "{what} sums not consumed"
        );
    }

    s.obj.copy_from_host(&stream, &obj0).unwrap();
    s.probe.copy_from_host(&stream, &probe0).unwrap();
    assert!(bits(&stream, &s).0 == start.0, "reset to the start failed");
    for (e, want) in replayed.iter().enumerate() {
        record(|ctx| epoch(ctx, &scan, &mut s))
            .sync_on(&stream)
            .unwrap();
        let got = bits(&stream, &s);
        assert_eq!(
            got.1, want.1,
            "epoch {e}: direct LLK totals differ from replay"
        );
        assert!(
            got.0 == want.0,
            "epoch {e}: direct object or probe differs from replay"
        );
    }
    jit_left("a direct run");
    let poisson: Vec<f64> = replayed.iter().map(|e| f64::from_bits(e.1[0])).collect();
    println!("Poisson LLK per epoch: {poisson:?}");
    assert!(
        poisson.windows(2).all(|w| w[1] < w[0]),
        "AP does not descend"
    );
}
