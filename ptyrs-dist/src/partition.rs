//! Partition of a fixed object tile grid into GPU regions (DESIGN B4).
//!
//! The grid depends only on the object shape and the tile size, never on the GPU count, so
//! the tile → frames lists and therefore every addition order (invariant 8) are the same for
//! 1, 2 and 4 GPUs. Only tile ownership changes with the GPU count.
//!
//! A frame is computed by the owner of its home tile, the tile holding its footprint centre.
//! The halo of a region is every foreign tile within one probe width (in whole tiles) of an
//! owned tile, which covers every footprint the region computes.

use std::f64::consts::PI;

use ptyrs_core::AddrTable;

use crate::DistError;

/// Object tile edge in pixels; matches `OBJ_TILE` in the accumulation kernels.
pub const DEFAULT_TILE: usize = 16;

/// Square tiles over an object, row-major ids `t = ty * ntx + tx`. Edge tiles may be partial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGrid {
    obj: [usize; 2],
    tile: usize,
    dims: [usize; 2],
}

impl TileGrid {
    pub fn new(obj: [usize; 2], tile: usize) -> Result<Self, DistError> {
        if tile == 0 || obj[0] == 0 || obj[1] == 0 {
            return Err(DistError::Partition(format!(
                "object {obj:?} with tile {tile} has no tiles"
            )));
        }
        Ok(Self {
            obj,
            tile,
            dims: [obj[0].div_ceil(tile), obj[1].div_ceil(tile)],
        })
    }

    pub fn obj(&self) -> [usize; 2] {
        self.obj
    }

    pub fn tile(&self) -> usize {
        self.tile
    }

    /// `[nty, ntx]`.
    pub fn dims(&self) -> [usize; 2] {
        self.dims
    }

    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn id(&self, ty: usize, tx: usize) -> usize {
        ty * self.dims[1] + tx
    }

    /// `[ty, tx]` of tile `t`.
    pub fn coords(&self, t: usize) -> [usize; 2] {
        [t / self.dims[1], t % self.dims[1]]
    }

    pub fn of_pixel(&self, y: usize, x: usize) -> usize {
        self.id(y / self.tile, x / self.tile)
    }

    /// Centre of the tile's in-object pixels, in pixel units (pixel `i` spans `[i, i + 1)`).
    pub fn centre(&self, t: usize) -> [f64; 2] {
        let [ty, tx] = self.coords(t);
        let mid = |i: usize, n: usize| {
            let lo = i * self.tile;
            (lo + (lo + self.tile).min(n)) as f64 / 2.0
        };
        [mid(ty, self.obj[0]), mid(tx, self.obj[1])]
    }

    /// Tiles touched by the `frame`-shaped footprint with top-left pixel `corner`, row-major.
    pub fn footprint(
        &self,
        corner: [usize; 2],
        frame: [usize; 2],
    ) -> impl Iterator<Item = usize> + '_ {
        let ty = corner[0] / self.tile..=(corner[0] + frame[0] - 1) / self.tile;
        let tx = corner[1] / self.tile..=(corner[1] + frame[1] - 1) / self.tile;
        ty.flat_map(move |y| tx.clone().map(move |x| self.id(y, x)))
    }
}

/// How the tile grid is cut into regions; region `r` runs on the `r`-th GPU of a
/// NUMA-grouped device list, so regions `2k` and `2k + 1` share a socket.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scheme {
    /// Live raster: vertical strips along the fast axis x, left to right, so every scan row
    /// crosses every region. With 4 GPUs only the middle boundary (1–2) crosses sockets.
    Strips,
    /// Live Fermat spiral: equal-angle sectors around `centre` (object pixels, `[y, x]`),
    /// or around the mean frame centre when `None`, so early frames reach every region.
    Sectors { centre: Option<[f64; 2]> },
    /// Large area: one region, two halves across the long axis, or a 2×2 grid whose region
    /// pairs (0, 1) and (2, 3) share the boundary that runs along the long axis.
    Grid,
}

/// Compressed rows: row `i` is `items[start[i]..start[i + 1]]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Csr {
    pub start: Vec<u32>,
    pub items: Vec<u32>,
}

impl Csr {
    /// Rows from `(row, item)` pairs visited in ascending item order, so each row ascends.
    /// `pairs` is called twice: once to count, once to fill.
    fn build<I: Iterator<Item = (usize, u32)>>(rows: usize, pairs: impl Fn() -> I) -> Self {
        let mut start = vec![0u32; rows + 1];
        for (r, _) in pairs() {
            start[r + 1] += 1;
        }
        for r in 0..rows {
            start[r + 1] += start[r];
        }
        let mut fill: Vec<u32> = start[..rows].to_vec();
        let mut items = vec![0u32; start[rows] as usize];
        for (r, item) in pairs() {
            items[fill[r] as usize] = item;
            fill[r] += 1;
        }
        Self { start, items }
    }

    pub fn rows(&self) -> usize {
        self.start.len() - 1
    }

    pub fn row(&self, i: usize) -> &[u32] {
        &self.items[self.start[i] as usize..self.start[i + 1] as usize]
    }
}

/// One GPU's share: owned tiles, halo tiles and computed frames, each ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub own: Vec<u32>,
    pub halo: Vec<u32>,
    pub frames: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Partition {
    grid: TileGrid,
    frame: [usize; 2],
    corners: Vec<[usize; 2]>,
    owner: Vec<u32>,
    home: Vec<u32>,
    regions: Vec<Region>,
    tile_frames: Csr,
    home_frames: Csr,
}

impl Partition {
    /// Frames are `frame`-shaped footprints with top-left pixels `corners`, in frame order.
    pub fn new(
        grid: TileGrid,
        frame: [usize; 2],
        corners: Vec<[usize; 2]>,
        scheme: Scheme,
        nregion: usize,
    ) -> Result<Self, DistError> {
        let obj = grid.obj();
        if frame[0] == 0 || frame[1] == 0 {
            return Err(DistError::Partition(format!("empty frame {frame:?}")));
        }
        if corners.is_empty() {
            return Err(DistError::Partition("no frames".into()));
        }
        if corners.len() > u32::MAX as usize || grid.len() > u32::MAX as usize {
            return Err(DistError::Partition(
                "more than u32::MAX frames or tiles".into(),
            ));
        }
        if let Some((f, c)) = corners
            .iter()
            .enumerate()
            .find(|(_, c)| c[0] + frame[0] > obj[0] || c[1] + frame[1] > obj[1])
        {
            return Err(DistError::Partition(format!(
                "frame {f} at {c:?} leaves object {obj:?}"
            )));
        }
        let centres: Vec<[f64; 2]> = corners
            .iter()
            .map(|c| {
                [
                    (c[0] + frame[0] / 2) as f64 + 0.5,
                    (c[1] + frame[1] / 2) as f64 + 0.5,
                ]
            })
            .collect();
        let owner = assign(&grid, &centres, scheme, nregion)?;
        let home: Vec<u32> = corners
            .iter()
            .map(|c| grid.of_pixel(c[0] + frame[0] / 2, c[1] + frame[1] / 2) as u32)
            .collect();

        let regions = regions(&grid, frame, &owner, &home, nregion);
        if let Some(r) = regions.iter().position(|g| g.own.is_empty()) {
            return Err(DistError::Partition(format!(
                "{scheme:?} over {nregion} regions leaves region {r} without tiles"
            )));
        }

        let tile_frames = Csr::build(grid.len(), || {
            corners
                .iter()
                .enumerate()
                .flat_map(|(f, &c)| grid.footprint(c, frame).map(move |t| (t, f as u32)))
        });
        let home_frames = Csr::build(grid.len(), || {
            home.iter()
                .enumerate()
                .map(|(f, &t)| (t as usize, f as u32))
        });
        Ok(Self {
            grid,
            frame,
            corners,
            owner,
            home,
            regions,
            tile_frames,
            home_frames,
        })
    }

    /// Partition of the frames in `addr`; every mode of a frame shares one footprint.
    pub fn from_addr(
        addr: &AddrTable,
        tile: usize,
        scheme: Scheme,
        nregion: usize,
    ) -> Result<Self, DistError> {
        let corners = (0..addr.nframe())
            .map(|f| {
                let v = addr.frame_views(f)[0];
                [v.oy as usize, v.ox as usize]
            })
            .collect();
        Self::new(
            TileGrid::new(addr.obj_shape(), tile)?,
            addr.frame_shape(),
            corners,
            scheme,
            nregion,
        )
    }

    pub fn grid(&self) -> &TileGrid {
        &self.grid
    }

    pub fn frame(&self) -> [usize; 2] {
        self.frame
    }

    pub fn corners(&self) -> &[[usize; 2]] {
        &self.corners
    }

    pub fn nframe(&self) -> usize {
        self.corners.len()
    }

    pub fn nregion(&self) -> usize {
        self.regions.len()
    }

    /// Owning region of each tile.
    pub fn owner(&self) -> &[u32] {
        &self.owner
    }

    /// Home tile of each frame (the tile holding its footprint centre).
    pub fn home(&self) -> &[u32] {
        &self.home
    }

    /// Region that computes frame `f`: the owner of its home tile.
    pub fn computed_by(&self, f: usize) -> usize {
        self.owner[self.home[f] as usize] as usize
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// Frames whose footprint touches each tile, ascending: the fixed addition order of the
    /// object accumulation (invariant 8).
    pub fn tile_frames(&self) -> &Csr {
        &self.tile_frames
    }

    /// Frames homed in each tile, ascending: the order of per-tile scalar partials.
    pub fn home_frames(&self) -> &Csr {
        &self.home_frames
    }

    /// This partition with regions `r * k .. (r + 1) * k` merged into region `r`, where
    /// `k = nregion / n`. Grid, frames and addition orders are unchanged, so a run on `n` GPUs
    /// cut from fixed leaf regions keeps every leaf's tiles and frames whatever `n` is.
    pub fn merge(&self, n: usize) -> Result<Self, DistError> {
        if n == 0 || !self.nregion().is_multiple_of(n) {
            return Err(DistError::Partition(format!(
                "{} regions do not merge into {n}",
                self.nregion()
            )));
        }
        let k = (self.nregion() / n) as u32;
        let owner: Vec<u32> = self.owner.iter().map(|&o| o / k).collect();
        Ok(Self {
            regions: regions(&self.grid, self.frame, &owner, &self.home, n),
            owner,
            ..self.clone()
        })
    }
}

/// Owned tiles, halo tiles (foreign tiles within one frame width, in whole tiles, of an owned
/// tile) and computed frames of each of `n` regions.
fn regions(
    grid: &TileGrid,
    frame: [usize; 2],
    owner: &[u32],
    home: &[u32],
    n: usize,
) -> Vec<Region> {
    let reach = [
        frame[0].div_ceil(grid.tile()),
        frame[1].div_ceil(grid.tile()),
    ];
    let mut regions: Vec<Region> = (0..n)
        .map(|r| {
            let mine: Vec<bool> = owner.iter().map(|&o| o as usize == r).collect();
            let near = dilate(&mine, grid.dims(), reach);
            Region {
                own: (0..grid.len())
                    .filter(|&t| mine[t])
                    .map(|t| t as u32)
                    .collect(),
                halo: (0..grid.len())
                    .filter(|&t| near[t] && !mine[t])
                    .map(|t| t as u32)
                    .collect(),
                frames: Vec::new(),
            }
        })
        .collect();
    for (f, &h) in home.iter().enumerate() {
        regions[owner[h as usize] as usize].frames.push(f as u32);
    }
    regions
}

/// Region of every tile, decided by the tile centre against cuts over the frame-centre extent.
fn assign(
    grid: &TileGrid,
    centres: &[[f64; 2]],
    scheme: Scheme,
    n: usize,
) -> Result<Vec<u32>, DistError> {
    if n == 0 {
        return Err(DistError::Partition("zero regions".into()));
    }
    // Frame-centre extent per axis, at least one tile wide so a line scan still splits.
    let extent = |a: usize| {
        let lo = centres.iter().map(|c| c[a]).fold(f64::INFINITY, f64::min);
        let hi = centres
            .iter()
            .map(|c| c[a])
            .fold(f64::NEG_INFINITY, f64::max);
        let pad = ((grid.tile() as f64) - (hi - lo)).max(0.0) / 2.0;
        (lo - pad, hi + pad)
    };
    // Index of the equal slice of [lo, hi) holding v, clamped to 0..k.
    let slice = |v: f64, (lo, hi): (f64, f64), k: usize| {
        (((v - lo) / (hi - lo) * k as f64).floor().max(0.0) as usize).min(k - 1)
    };
    let tiles = 0..grid.len();
    let owner: Vec<usize> = match scheme {
        Scheme::Strips => {
            let ex = extent(1);
            tiles.map(|t| slice(grid.centre(t)[1], ex, n)).collect()
        }
        Scheme::Sectors { centre } => {
            let c = centre.unwrap_or_else(|| {
                let k = centres.len() as f64;
                [
                    centres.iter().map(|c| c[0]).sum::<f64>() / k,
                    centres.iter().map(|c| c[1]).sum::<f64>() / k,
                ]
            });
            tiles
                .map(|t| {
                    let p = grid.centre(t);
                    let a = (p[0] - c[0]).atan2(p[1] - c[1]);
                    slice(a, (-PI, PI), n)
                })
                .collect()
        }
        Scheme::Grid => {
            let (ey, ex) = (extent(0), extent(1));
            // The long axis follows the scanned extent; ties go to x.
            let wide = ex.1 - ex.0 >= ey.1 - ey.0;
            let (long, short, el, es) = if wide { (1, 0, ex, ey) } else { (0, 1, ey, ex) };
            match n {
                1 => vec![0; grid.len()],
                2 => tiles.map(|t| slice(grid.centre(t)[long], el, 2)).collect(),
                4 => tiles
                    .map(|t| {
                        let p = grid.centre(t);
                        2 * slice(p[long], el, 2) + slice(p[short], es, 2)
                    })
                    .collect(),
                _ => {
                    return Err(DistError::Partition(format!(
                        "Grid supports 1, 2 or 4 regions, not {n}"
                    )));
                }
            }
        }
    };
    Ok(owner.into_iter().map(|r| r as u32).collect())
}

/// Tiles within `reach` tiles (per axis, Chebyshev) of a set tile; separable max filter.
fn dilate(mask: &[bool], dims: [usize; 2], reach: [usize; 2]) -> Vec<bool> {
    let [ny, nx] = dims;
    let mut rows = vec![false; mask.len()];
    for y in 0..ny {
        for x in 0..nx {
            let (lo, hi) = (x.saturating_sub(reach[1]), (x + reach[1]).min(nx - 1));
            rows[y * nx + x] = (lo..=hi).any(|i| mask[y * nx + i]);
        }
    }
    let mut out = vec![false; mask.len()];
    for y in 0..ny {
        let (lo, hi) = (y.saturating_sub(reach[0]), (y + reach[0]).min(ny - 1));
        for x in 0..nx {
            out[y * nx + x] = (lo..=hi).any(|j| rows[j * nx + x]);
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ptyrs_core::{AddrTable, sim};

    const PIXEL: [f64; 2] = [1e-8, 1e-8];

    pub(crate) fn addr(pos: ptyrs_core::Positions, frame: usize) -> AddrTable {
        let pos = pos.pynx_preprocess(PIXEL);
        let obj = ptyrs_core::required_obj_shape(&pos, PIXEL, [frame, frame], 0);
        AddrTable::new(&pos, PIXEL, [frame, frame], obj, 1, 1).unwrap()
    }

    /// Raster with non-square extent so Grid has a long axis, a Fermat spiral, and a raster
    /// whose frame is not a tile multiple.
    fn cases() -> Vec<(&'static str, AddrTable, Scheme)> {
        vec![
            (
                "raster",
                addr(sim::raster(12, 20, 8e-8), 32),
                Scheme::Strips,
            ),
            (
                "fermat",
                addr(sim::fermat(300, 8e-8), 32),
                Scheme::Sectors { centre: None },
            ),
            ("wide", addr(sim::raster(10, 24, 8e-8), 32), Scheme::Grid),
            ("tall", addr(sim::raster(24, 10, 8e-8), 32), Scheme::Grid),
            ("odd", addr(sim::raster(9, 11, 7e-8), 21), Scheme::Strips),
        ]
    }

    fn parts() -> Vec<(String, Partition)> {
        let mut out = Vec::new();
        for (name, a, scheme) in cases() {
            for n in [1, 2, 4] {
                let p = Partition::from_addr(&a, DEFAULT_TILE, scheme, n).unwrap();
                out.push((format!("{name}/{n}"), p));
            }
        }
        out
    }

    #[test]
    fn every_tile_has_exactly_one_owner() {
        for (name, p) in parts() {
            let mut seen = vec![0usize; p.grid().len()];
            for (r, g) in p.regions().iter().enumerate() {
                assert!(!g.own.is_empty(), "{name}: region {r} empty");
                for &t in &g.own {
                    seen[t as usize] += 1;
                    assert_eq!(p.owner()[t as usize] as usize, r, "{name}: tile {t}");
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "{name}: {seen:?}");
        }
    }

    #[test]
    fn every_footprint_is_covered_by_its_region() {
        for (name, p) in parts() {
            for (r, g) in p.regions().iter().enumerate() {
                assert!(g.halo.iter().all(|&t| p.owner()[t as usize] as usize != r));
                let mut held = vec![false; p.grid().len()];
                for &t in g.own.iter().chain(&g.halo) {
                    held[t as usize] = true;
                }
                for &f in &g.frames {
                    let f = f as usize;
                    assert_eq!(p.computed_by(f), r);
                    for y in 0..p.frame()[0] {
                        for x in 0..p.frame()[1] {
                            let [cy, cx] = p.corners()[f];
                            let t = p.grid().of_pixel(cy + y, cx + x);
                            assert!(held[t], "{name}: frame {f} pixel ({y},{x}) tile {t}");
                        }
                    }
                }
            }
            let computed: usize = p.regions().iter().map(|g| g.frames.len()).sum();
            assert_eq!(computed, p.nframe(), "{name}: each frame computed once");
        }
    }

    #[test]
    fn halo_is_one_probe_wide() {
        for (name, p) in parts() {
            let g = p.grid();
            let reach = [
                p.frame()[0].div_ceil(g.tile()),
                p.frame()[1].div_ceil(g.tile()),
            ];
            for (r, reg) in p.regions().iter().enumerate() {
                for t in 0..g.len() {
                    let [ty, tx] = g.coords(t);
                    let near = reg.own.iter().any(|&u| {
                        let [uy, ux] = g.coords(u as usize);
                        ty.abs_diff(uy) <= reach[0] && tx.abs_diff(ux) <= reach[1]
                    });
                    let foreign = p.owner()[t] as usize != r;
                    assert_eq!(
                        reg.halo.contains(&(t as u32)),
                        near && foreign,
                        "{name}: region {r} tile {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn tile_frame_lists_are_ascending_and_exact() {
        for (name, p) in parts() {
            let g = p.grid();
            for t in 0..g.len() {
                let want: Vec<u32> = (0..p.nframe())
                    .filter(|&f| g.footprint(p.corners()[f], p.frame()).any(|u| u == t))
                    .map(|f| f as u32)
                    .collect();
                assert_eq!(p.tile_frames().row(t), want, "{name}: tile {t}");
                let homed: Vec<u32> = (0..p.nframe())
                    .filter(|&f| p.home()[f] as usize == t)
                    .map(|f| f as u32)
                    .collect();
                assert_eq!(p.home_frames().row(t), homed, "{name}: home {t}");
            }
        }
    }

    /// Addition orders come from the grid alone, so they are identical for every GPU count.
    #[test]
    fn orders_do_not_depend_on_region_count() {
        for (name, a, scheme) in cases() {
            let p1 = Partition::from_addr(&a, DEFAULT_TILE, scheme, 1).unwrap();
            for n in [2, 4] {
                let pn = Partition::from_addr(&a, DEFAULT_TILE, scheme, n).unwrap();
                assert_eq!(pn.grid(), p1.grid(), "{name}/{n}");
                assert_eq!(pn.tile_frames(), p1.tile_frames(), "{name}/{n}");
                assert_eq!(pn.home_frames(), p1.home_frames(), "{name}/{n}");
            }
        }
    }

    /// Every raster row puts frames in all four strips, left to right; a slow-axis cut would
    /// feed one GPU for the first quarter of the scan.
    #[test]
    fn strips_feed_every_gpu_on_each_row() {
        let (ny, nx) = (12, 20);
        let p = Partition::from_addr(
            &addr(sim::raster(ny, nx, 8e-8), 32),
            DEFAULT_TILE,
            Scheme::Strips,
            4,
        )
        .unwrap();
        for row in 0..ny {
            let regions: Vec<usize> = (0..nx).map(|i| p.computed_by(row * nx + i)).collect();
            assert!(
                regions.windows(2).all(|w| w[0] <= w[1]),
                "row {row}: {regions:?}"
            );
            for r in 0..4 {
                let k = regions.iter().filter(|&&g| g == r).count();
                assert!((4..=6).contains(&k), "row {row}: region {r} gets {k}");
            }
        }
    }

    /// The first frames of a Fermat spiral spread over all four sectors, and over the whole
    /// scan the sectors stay within 10 % of an equal share.
    #[test]
    fn sectors_feed_every_gpu_from_the_start() {
        let n = 400;
        let p = Partition::from_addr(
            &addr(sim::fermat(n, 8e-8), 32),
            DEFAULT_TILE,
            Scheme::Sectors { centre: None },
            4,
        )
        .unwrap();
        for prefix in [16, 50, n] {
            let mut count = [0usize; 4];
            for f in 0..prefix {
                count[p.computed_by(f)] += 1;
            }
            assert!(count.iter().all(|&c| c > 0), "first {prefix}: {count:?}");
            if prefix == n {
                let share = n / 4;
                assert!(
                    count.iter().all(|&c| c.abs_diff(share) * 10 <= share),
                    "{count:?}"
                );
            }
        }
    }

    /// 4-neighbour tile contacts between each pair of regions.
    fn contacts(p: &Partition) -> [[usize; 4]; 4] {
        let g = p.grid();
        let [ny, nx] = g.dims();
        let mut c = [[0; 4]; 4];
        for y in 0..ny {
            for x in 0..nx {
                let a = p.owner()[g.id(y, x)] as usize;
                for (yy, xx) in [(y + 1, x), (y, x + 1)] {
                    if yy < ny && xx < nx {
                        let b = p.owner()[g.id(yy, xx)] as usize;
                        if a != b {
                            c[a.min(b)][a.max(b)] += 1;
                        }
                    }
                }
            }
        }
        c
    }

    /// NUMA pairs (0, 1) and (2, 3) share the long boundary of the 2×2 grid, wide or tall.
    #[test]
    fn grid_pairs_share_the_long_boundary() {
        for (ny, nx) in [(10, 24), (24, 10)] {
            let p = Partition::from_addr(
                &addr(sim::raster(ny, nx, 8e-8), 32),
                DEFAULT_TILE,
                Scheme::Grid,
                4,
            )
            .unwrap();
            let c = contacts(&p);
            let pair = c[0][1] + c[2][3];
            let cross = c[0][2] + c[1][3];
            assert!(pair > cross, "{ny}x{nx}: pair {pair} cross {cross} {c:?}");
            assert!(c[0][1] > 0 && c[2][3] > 0 && c[0][2] > 0 && c[1][3] > 0);
        }
    }

    /// Strips run left to right, so of the three boundaries only 1–2 crosses sockets.
    #[test]
    fn strips_cross_sockets_only_in_the_middle() {
        let p = Partition::from_addr(
            &addr(sim::raster(12, 20, 8e-8), 32),
            DEFAULT_TILE,
            Scheme::Strips,
            4,
        )
        .unwrap();
        let c = contacts(&p);
        let touching: Vec<(usize, usize)> = (0..4)
            .flat_map(|a| (a + 1..4).map(move |b| (a, b)))
            .filter(|&(a, b)| c[a][b] > 0)
            .collect();
        assert_eq!(touching, [(0, 1), (1, 2), (2, 3)]);
    }

    /// Halving the cut count maps slice `i` of 4 to slice `i / 2` of 2 exactly (the scale is a
    /// power of two), so merged leaves equal a direct partition, and merging keeps every leaf
    /// whole.
    #[test]
    fn merged_leaves_equal_a_direct_partition() {
        for (name, a, scheme) in cases() {
            let leaves = Partition::from_addr(&a, DEFAULT_TILE, scheme, 4).unwrap();
            assert_eq!(leaves.merge(4).unwrap(), leaves, "{name}");
            for n in [1, 2] {
                let merged = leaves.merge(n).unwrap();
                let direct = Partition::from_addr(&a, DEFAULT_TILE, scheme, n).unwrap();
                assert_eq!(merged, direct, "{name}/{n}");
                for (l, leaf) in leaves.regions().iter().enumerate() {
                    let r = &merged.regions()[l / (4 / n)];
                    assert!(
                        leaf.own.iter().all(|t| r.own.contains(t)),
                        "{name}/{n} leaf {l}"
                    );
                    assert!(
                        leaf.frames.iter().all(|f| r.frames.contains(f)),
                        "{name}/{n} leaf {l}"
                    );
                }
            }
            assert!(
                leaves.merge(0).is_err() && leaves.merge(3).is_err() && leaves.merge(8).is_err()
            );
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let g = TileGrid::new([64, 64], 16).unwrap();
        assert!(TileGrid::new([64, 64], 0).is_err());
        assert!(Partition::new(g, [32, 32], vec![[40, 0]], Scheme::Strips, 1).is_err());
        assert!(Partition::new(g, [32, 32], vec![], Scheme::Strips, 1).is_err());
        assert!(Partition::new(g, [32, 32], vec![[0, 0]], Scheme::Grid, 3).is_err());
        // Eight strips over a four-tile-wide object leave regions without tiles.
        assert!(Partition::new(g, [32, 32], vec![[0, 0], [0, 32]], Scheme::Strips, 8).is_err());
    }
}
