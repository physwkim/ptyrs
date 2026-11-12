//! DESIGN §8: can a Tile program own one object tile and gather a run-time number of
//! overlapping views into it, from a tile→view CSR list, in ascending view order?
//!
//! `tile_gather_sum` is the smallest kernel with that shape: per object tile, a `for` over
//! the tile's list (bounds loaded from device memory), one masked pointer gather per view
//! from a wrapped, unaligned window, and one store of the owned tile.
//! `tile_gather_sum_view` does the same gather through a Tile IR 13.3 gather view.

use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

#[cutile::module]
mod feasibility {
    use cutile::core::*;
    use cutile::tileir::*;

    unsafe fn load_i32(p: *const i32, i: i32) -> i32 {
        let base: PointerTile<*const i32, { [] }> = pointer_to_tile(p);
        let at: PointerTile<*const i32, { [] }> = base.offset(i);
        let (v, _t): (Tile<i32, { [] }>, Token) = unsafe {
            load_ptr_tko(
                at,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        let s: i32 = tile_to_scalar(v);
        s
    }

    /// `out[y, x] = sum_k src_k[(y - oy_k) mod nyo, (x - ox_k) mod nxo]` over the views `k`
    /// of the tile's list, ascending, where that offset lies in the `[ny, nx]` frame.
    #[cutile::entry()]
    unsafe fn tile_gather_sum(
        out: &mut Tensor<f32, { [16, 16] }>,
        src: *const f32,
        corner: *const i32,
        tile_start: *const i32,
        tile_views: *const i32,
        ny: i32,
        nx: i32,
        nyo: i32,
        nxo: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        let tile: i32 = pid.0 * grid.1 + pid.1;
        let lo: i32 = unsafe { load_i32(tile_start, tile) };
        let hi: i32 = unsafe { load_i32(tile_start, tile + 1i32) };

        let lane: Tile<i32, { [16] }> = iota(shape![16]);
        let row: Tile<i32, { [16, 1] }> = lane.reshape(shape![16, 1]);
        let col: Tile<i32, { [1, 16] }> = lane.reshape(shape![1, 16]);
        let ys: Tile<i32, { [16, 16] }> =
            row.broadcast(shape![16, 16]) + broadcast_scalar(pid.0 * 16i32, shape![16, 16]);
        let xs: Tile<i32, { [16, 16] }> =
            col.broadcast(shape![16, 16]) + broadcast_scalar(pid.1 * 16i32, shape![16, 16]);
        let ny_t: Tile<i32, { [16, 16] }> = broadcast_scalar(ny, shape![16, 16]);
        let nx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nx, shape![16, 16]);
        let nyo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nyo, shape![16, 16]);
        let nxo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nxo, shape![16, 16]);

        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(src);
        let base: PointerTile<*const f32, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*const f32, { [16, 16] }> = base.broadcast(shape![16, 16]);

        let mut acc: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
        for j in lo..hi {
            let k: i32 = unsafe { load_i32(tile_views, j) };
            let oy: i32 = unsafe { load_i32(corner, 2i32 * k) };
            let ox: i32 = unsafe { load_i32(corner, 2i32 * k + 1i32) };
            let r: Tile<i32, { [16, 16] }> =
                (ys + broadcast_scalar(nyo - oy, shape![16, 16])) % nyo_t;
            let c: Tile<i32, { [16, 16] }> =
                (xs + broadcast_scalar(nxo - ox, shape![16, 16])) % nxo_t;
            let inside: Tile<bool, { [16, 16] }> = lt_tile(r, ny_t) & lt_tile(c, nx_t);
            let at: Tile<i32, { [16, 16] }> =
                broadcast_scalar(k * ny * nx, shape![16, 16]) + r * nx_t + c;
            let ptrs: PointerTile<*const f32, { [16, 16] }> = base.offset_tile(at);
            let (v, _t): (Tile<f32, { [16, 16] }>, Token) = unsafe {
                load_ptr_tko(
                    ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(inside),
                    None,
                    None,
                    Latency::<0>,
                )
            };
            acc = select(inside, acc + v, acc);
        }
        out.store(acc);
    }

    /// [`tile_gather_sum`] with each view's window read through a Tile IR 13.3 gather view.
    /// Its one sparse axis cannot express the 2D wrap, so it takes the 256 flat offsets of
    /// `src` seen as `[len, 1]` (a rank-1 view fails cutile's JIT); masked lanes index past
    /// the end and read the zero padding.
    #[cutile::entry()]
    unsafe fn tile_gather_sum_view(
        out: &mut Tensor<f32, { [16, 16] }>,
        src: &Tensor<f32, { [-1, -1] }>,
        corner: *const i32,
        tile_start: *const i32,
        tile_views: *const i32,
        ny: i32,
        nx: i32,
        nyo: i32,
        nxo: i32,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        let tile: i32 = pid.0 * grid.1 + pid.1;
        let lo: i32 = unsafe { load_i32(tile_start, tile) };
        let hi: i32 = unsafe { load_i32(tile_start, tile + 1i32) };

        let lane: Tile<i32, { [16] }> = iota(shape![16]);
        let row: Tile<i32, { [16, 1] }> = lane.reshape(shape![16, 1]);
        let col: Tile<i32, { [1, 16] }> = lane.reshape(shape![1, 16]);
        let ys: Tile<i32, { [16, 16] }> =
            row.broadcast(shape![16, 16]) + broadcast_scalar(pid.0 * 16i32, shape![16, 16]);
        let xs: Tile<i32, { [16, 16] }> =
            col.broadcast(shape![16, 16]) + broadcast_scalar(pid.1 * 16i32, shape![16, 16]);
        let ny_t: Tile<i32, { [16, 16] }> = broadcast_scalar(ny, shape![16, 16]);
        let nx_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nx, shape![16, 16]);
        let nyo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nyo, shape![16, 16]);
        let nxo_t: Tile<i32, { [16, 16] }> = broadcast_scalar(nxo, shape![16, 16]);
        let past_end: Tile<i32, { [16, 16] }> = broadcast_scalar(len, shape![16, 16]);
        let view: GatherScatterView<f32, { [256, 1] }, 0> =
            unsafe { make_gather_scatter_view(src, shape![256, 1], padding::Zero) };

        let mut acc: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
        for j in lo..hi {
            let k: i32 = unsafe { load_i32(tile_views, j) };
            let oy: i32 = unsafe { load_i32(corner, 2i32 * k) };
            let ox: i32 = unsafe { load_i32(corner, 2i32 * k + 1i32) };
            let r: Tile<i32, { [16, 16] }> =
                (ys + broadcast_scalar(nyo - oy, shape![16, 16])) % nyo_t;
            let c: Tile<i32, { [16, 16] }> =
                (xs + broadcast_scalar(nxo - ox, shape![16, 16])) % nxo_t;
            let inside: Tile<bool, { [16, 16] }> = lt_tile(r, ny_t) & lt_tile(c, nx_t);
            let at: Tile<i32, { [16, 16] }> =
                broadcast_scalar(k * ny * nx, shape![16, 16]) + r * nx_t + c;
            let at: Tile<i32, { [16, 16] }> = select(inside, at, past_end);
            let idx: Tile<i32, { [256] }> = at.reshape(shape![256]);
            let (v, _t): (Tile<f32, { [256, 1] }>, Token) = unsafe {
                load_view_raw(
                    &view,
                    (idx, 0i32),
                    None,
                    ordering::Weak,
                    scope::Device,
                    tma::Disabled,
                    None,
                    [false, false],
                    shape![256, 1],
                )
            };
            let v: Tile<f32, { [16, 16] }> = v.reshape(shape![16, 16]);
            acc = select(inside, acc + v, acc);
        }
        out.store(acc);
    }
}

use feasibility::{tile_gather_sum, tile_gather_sum_view};

const T: usize = 16;

/// Test-only tile lists: every view whose wrapped `[ny, nx]` window covers a pixel of the
/// tile, ascending (the library takes them from `ptyrs::gpu::Views`).
fn tile_lists(
    corners: &[(usize, usize)],
    frame: (usize, usize),
    obj: (usize, usize),
) -> (Vec<i32>, Vec<i32>) {
    let (ny, nx) = frame;
    let (nyo, nxo) = obj;
    let ntx = nxo.div_ceil(T);
    let mut lists = vec![Vec::new(); nyo.div_ceil(T) * ntx];
    for (k, &(oy, ox)) in corners.iter().enumerate() {
        for i in 0..ny {
            for j in 0..nx {
                let t = ((oy + i) % nyo / T) * ntx + (ox + j) % nxo / T;
                if lists[t].last() != Some(&(k as i32)) {
                    lists[t].push(k as i32);
                }
            }
        }
    }
    let mut start = vec![0i32];
    let mut flat = Vec::new();
    for l in lists {
        flat.extend(l);
        start.push(flat.len() as i32);
    }
    (start, flat)
}

fn up<T: cutile::DType>(v: Vec<T>) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(v)).sync().unwrap()
}

#[test]
fn tile_owned_gather_over_variable_view_lists() {
    // Object 53x61 (partial edge tiles), frames 12x20; views overlap, repeat, cross tile
    // edges and wrap around both object edges, so tiles see 0, 1 or several views.
    let (ny, nx, nyo, nxo) = (12usize, 20usize, 53usize, 61usize);
    let corners = [
        (0, 0),
        (3, 11),
        (45, 40),
        (17, 5),
        (17, 5),
        (25, 30),
        (50, 50),
    ];
    let src: Vec<f32> = (0..corners.len() * ny * nx)
        .map(|i| ((i * 7919) % 1013) as f32 / 97.0 - 5.0)
        .collect();
    let mut want = vec![0.0f32; nyo * nxo];
    let mut want64 = vec![0.0f64; nyo * nxo];
    for (k, &(oy, ox)) in corners.iter().enumerate() {
        for i in 0..ny {
            for j in 0..nx {
                let px = ((oy + i) % nyo) * nxo + (ox + j) % nxo;
                want[px] += src[(k * ny + i) * nx + j];
                want64[px] += src[(k * ny + i) * nx + j] as f64;
            }
        }
    }
    let (start, list) = tile_lists(&corners, (ny, nx), (nyo, nxo));
    assert!(
        start.windows(2).any(|w| w[1] - w[0] >= 3),
        "no tile gathers 3+ views"
    );
    assert!(
        start.windows(2).any(|w| w[1] == w[0]),
        "no tile gathers 0 views"
    );
    let flat_corners: Vec<i32> = corners
        .iter()
        .flat_map(|&(y, x)| [y as i32, x as i32])
        .collect();
    let src_len = src.len();
    let (src_d, corner_d) = (up(src), up(flat_corners));
    let (start_d, list_d) = (up(start), up(list));

    let run = || {
        let (out, ..) = unsafe {
            tile_gather_sum(
                api::zeros(&[nyo, nxo]).partition([T, T]),
                src_d.device_pointer(),
                corner_d.device_pointer(),
                start_d.device_pointer(),
                list_d.device_pointer(),
                ny as i32,
                nx as i32,
                nyo as i32,
                nxo as i32,
            )
        }
        .grid((nyo.div_ceil(T) as u32, nxo.div_ceil(T) as u32, 1))
        .sync()
        .expect("tile_gather_sum");
        out.unpartition().to_host_vec().sync().unwrap()
    };
    let src_col = src_d.view(&[src_len, 1]).unwrap();
    let run_view = || {
        let (out, ..) = unsafe {
            tile_gather_sum_view(
                api::zeros(&[nyo, nxo]).partition([T, T]),
                &src_col,
                corner_d.device_pointer(),
                start_d.device_pointer(),
                list_d.device_pointer(),
                ny as i32,
                nx as i32,
                nyo as i32,
                nxo as i32,
                src_len as i32,
            )
        }
        .grid((nyo.div_ceil(T) as u32, nxo.div_ceil(T) as u32, 1))
        .sync()
        .expect("tile_gather_sum_view");
        out.unpartition().to_host_vec().sync().unwrap()
    };
    let got = run();
    // Same adds in the same (ascending view) order as the f32 reference: bit for bit.
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&got),
        bits(&want),
        "Tile gather differs from the ordered f32 sum"
    );
    let err = got
        .iter()
        .zip(&want64)
        .map(|(&g, &w)| (g as f64 - w).abs())
        .fold(0.0f64, f64::max);
    println!("tile_gather_sum: max |Tile - f64| {err:.2e}");
    assert!(err < 1e-4);
    assert_eq!(bits(&run()), bits(&got), "run to run");
    assert_eq!(
        bits(&run_view()),
        bits(&got),
        "gather view differs from pointer gather"
    );
}
