//! The live split (DESIGN B3, B4, B6): a live scan's object cut into one region per GPU as
//! [`DistAp`](crate::dist::DistAp) cuts it, each frame landing, in arrival order, on every GPU
//! that owns a tile the frame touches. A GPU's own tiles then bin the frames one GPU bins
//! there, in the same order, so their object sums add in the same order (invariant 8); the
//! other regions' tiles its frames read come from their owners after each update
//! (invariant 5).

use crate::gpu::{ViewAddr, ViewDims, view_tiles};
use crate::live::scan::ScanShape;

use crate::dist::halo::{Halo, Window};
use crate::dist::{DEFAULT_TILE, DistError, NLEAF, Partition, Scheme, TileGrid};

/// One GPU's part of a live scan.
pub struct LiveRegion {
    /// The part of the object it holds.
    pub(crate) window: Window,
    /// Its scan: the planned frames routed to it, over its window.
    pub shape: ScanShape,
    /// Its own tiles, numbered in its window, ascending: the only ones it bins and updates.
    pub own: Vec<u32>,
    /// The other regions' tiles its planned frames read, ascending.
    pub(crate) reads: Vec<u32>,
    /// Per object tile, whether it is one of its own tiles or `reads`.
    held: Vec<bool>,
    /// The planned frames routed to it, ascending.
    pub frames: Vec<u32>,
    /// Their views, offsets into its window: what its scan's backer plans.
    pub planned: Vec<ViewAddr>,
}

/// A live scan's split over GPUs, fixed by its planned frames before the first one lands.
pub struct LivePlan {
    run: Partition,
    /// The one-GPU scan's.
    dims: ViewDims,
    regions: Vec<LiveRegion>,
    halo: Halo,
}

/// A batch's frames on one region.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Share {
    /// Their views, offsets into the region's window.
    pub views: Vec<ViewAddr>,
    /// Their index in the batch, ascending.
    pub frames: Vec<u32>,
}

impl LivePlan {
    /// The split over `ngpu` GPUs, cut by `scheme` from [`NLEAF`] leaves, of the one-GPU scan
    /// `shape` whose `shape.nframe` planned frames have the views `planned`, `nmode` each.
    pub fn new(
        shape: &ScanShape,
        planned: &[ViewAddr],
        scheme: Scheme,
        ngpu: usize,
    ) -> Result<Self, DistError> {
        let bad = |m: String| Err(DistError::Partition(m));
        let dims = shape.dims();
        let nmode = dims.nmode as usize;
        if nmode == 0 || planned.len() != dims.nframe as usize * nmode {
            return bad(format!(
                "{} planned views are not {} frames of {nmode}",
                planned.len(),
                dims.nframe
            ));
        }
        if let Some((k, e)) = planned
            .iter()
            .enumerate()
            .find_map(|(k, v)| dims.check_view(v).err().map(|e| (k, e)))
        {
            return bad(format!("planned view {k}: {e}"));
        }
        let to = |a: [u32; 2]| a.map(|n| n as usize);
        let grid = TileGrid::new(to(dims.obj), DEFAULT_TILE)?;
        if to(dims.tiles()) != grid.dims() {
            return bad(format!(
                "kernel tiles {:?} differ from partition tiles {:?}",
                dims.tiles(),
                grid.dims()
            ));
        }
        let frame = to(dims.frame);
        let corners = planned
            .chunks(nmode)
            .map(|f| [f[0].oy as usize, f[0].ox as usize])
            .collect();
        let run = Partition::new(grid, frame, corners, scheme, NLEAF)?.merge(ngpu)?;
        let owner = run.owner();
        let mut frames = vec![Vec::new(); ngpu];
        let mut held = vec![vec![false; grid.len()]; ngpu];
        for (r, g) in run.regions().iter().enumerate() {
            g.own.iter().for_each(|&t| held[r][t as usize] = true);
        }
        let mut read = vec![vec![false; grid.len()]; ngpu];
        let mut tiles = Vec::new();
        for (f, views) in planned.chunks(nmode).enumerate() {
            touched(&dims, &grid, views, &mut tiles);
            for r in owners(owner, &tiles, ngpu) {
                frames[r].push(f as u32);
                for &t in &tiles {
                    read[r][t] |= owner[t] as usize != r;
                }
            }
        }
        let mut regions = Vec::with_capacity(ngpu);
        for (r, g) in run.regions().iter().enumerate() {
            let reads: Vec<u32> = (0..grid.len())
                .filter(|&t| read[r][t])
                .map(|t| t as u32)
                .collect();
            reads.iter().for_each(|&t| held[r][t as usize] = true);
            let frames = std::mem::take(&mut frames[r]);
            if frames.is_empty() {
                return bad(format!("region {r} of {ngpu} has no frames"));
            }
            let window = Window::new(&grid, g.own.iter().chain(&reads).map(|&t| t as usize));
            let Some(window) = window.hold(&grid, frame) else {
                return bad(format!(
                    "region {r}'s window {window:?} ends past the object"
                ));
            };
            let planned: Vec<ViewAddr> = frames
                .iter()
                .flat_map(|&f| &planned[f as usize * nmode..][..nmode])
                .map(|&v| window.shift(v))
                .collect();
            regions.push(LiveRegion {
                shape: ScanShape {
                    nframe: frames.len() as u32,
                    reach: [0, 1].map(|a| (window.shape[a] - frame[a] - 1) as u32),
                    ..*shape
                },
                own: g
                    .own
                    .iter()
                    .map(|&t| window.tile(&grid, t as usize))
                    .collect(),
                window,
                reads,
                held: std::mem::take(&mut held[r]),
                frames,
                planned,
            });
        }
        let halo = Halo::new(
            &run,
            regions.iter().map(|g| g.reads.as_slice()),
            dims.nobj as usize,
        );
        Ok(Self {
            run,
            dims,
            regions,
            halo,
        })
    }

    pub fn nregion(&self) -> usize {
        self.regions.len()
    }

    pub fn regions(&self) -> &[LiveRegion] {
        &self.regions
    }

    pub fn partition(&self) -> &Partition {
        &self.run
    }

    /// Pixels region `r` sends to and receives from the others after each update.
    pub fn halo_words(&self, r: usize) -> [u64; 2] {
        self.halo.words(r)
    }

    /// Splits the arriving frames whose views are `views`, `nmode` per frame, in arrival
    /// order, into `shares`, one per region: each frame goes to the owner of every tile it
    /// touches. Every view must be inside the one-GPU scan's object, and every tile a frame
    /// touches one its regions hold; at the first frame that is not, `shares` hold the frames
    /// before it.
    pub fn split(&self, views: &[ViewAddr], shares: &mut [Share]) -> Result<(), DistError> {
        let bad = |m: String| Err(DistError::Partition(m));
        let nmode = self.dims.nmode as usize;
        if shares.len() != self.regions.len() || !views.len().is_multiple_of(nmode) {
            return bad(format!(
                "{} views of {nmode} per frame into {} shares of {} regions",
                views.len(),
                shares.len(),
                self.regions.len()
            ));
        }
        shares.iter_mut().for_each(|s| {
            s.views.clear();
            s.frames.clear();
        });
        let grid = self.run.grid();
        let mut tiles = Vec::new();
        for (i, frame) in views.chunks(nmode).enumerate() {
            if let Some(e) = frame.iter().find_map(|v| self.dims.check_view(v).err()) {
                return bad(format!("frame {i} of the batch: {e}"));
            }
            touched(&self.dims, grid, frame, &mut tiles);
            let to = owners(self.run.owner(), &tiles, self.regions.len());
            for &r in &to {
                if let Some(t) = tiles.iter().find(|&&t| !self.regions[r].held[t]) {
                    return bad(format!(
                        "frame {i} of the batch reads tile {t}, which region {r} does not hold"
                    ));
                }
            }
            for r in to {
                let win = self.regions[r].window;
                shares[r].views.extend(frame.iter().map(|&v| win.shift(v)));
                shares[r].frames.push(i as u32);
            }
        }
        Ok(())
    }
}

/// The tiles of `grid` the views of one frame touch, by the kernels' rule, into `out`,
/// ascending.
fn touched(dims: &ViewDims, grid: &TileGrid, views: &[ViewAddr], out: &mut Vec<usize>) {
    out.clear();
    out.extend(
        views
            .iter()
            .flat_map(|v| view_tiles(dims, v))
            .map(|[ty, tx]| grid.id(ty as usize, tx as usize)),
    );
    out.sort_unstable();
    out.dedup();
}

/// The owners of `tiles`, ascending.
fn owners(owner: &[u32], tiles: &[usize], n: usize) -> Vec<usize> {
    let mut hit = vec![false; n];
    tiles.iter().for_each(|&t| hit[owner[t] as usize] = true);
    (0..n).filter(|&r| hit[r]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{Positions, sim};
    use crate::dist::halo::HaloRows;
    use crate::live::recon::ViewPlacement;

    const PIXEL: [f64; 2] = [1e-8, 1e-8];

    /// The one-GPU scan of `frame`-pixel frames at `pos`, 2 probe modes, and its views.
    fn scan(pos: Positions, frame: u32) -> (ScanShape, Vec<ViewAddr>) {
        let xy: Vec<(f64, f64)> = pos.x.iter().copied().zip(pos.y.iter().copied()).collect();
        let (place, reach) = ViewPlacement::plan(PIXEL, &xy);
        let mut views = Vec::new();
        for &p in &xy {
            place.views(p, 2, 1, &mut views);
        }
        let shape = ScanShape {
            nframe: xy.len() as u32,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [frame; 2],
            reach,
            quad_phase: 0.0,
        };
        (shape, views)
    }

    /// A raster in strips, a Fermat spiral in sectors, and a raster whose frame is not a tile
    /// multiple.
    fn cases() -> Vec<(&'static str, ScanShape, Vec<ViewAddr>, Scheme)> {
        let mut out = Vec::new();
        for (name, pos, frame, scheme) in [
            ("raster", sim::raster(12, 20, 8e-8), 32, Scheme::Strips),
            (
                "fermat",
                sim::fermat(300, 8e-8),
                32,
                Scheme::Sectors { centre: None },
            ),
            ("odd", sim::raster(9, 11, 7e-8), 22, Scheme::Strips),
        ] {
            let (shape, views) = scan(pos, frame);
            out.push((name, shape, views, scheme));
        }
        out
    }

    /// Global ids of the tiles `views` touch on `dims`, numbered on `grid` from `at`.
    fn tiles_of(dims: &ViewDims, views: &[ViewAddr], ntx: usize, at: [usize; 2]) -> Vec<usize> {
        let mut t: Vec<usize> = views
            .iter()
            .flat_map(|v| view_tiles(dims, v))
            .map(|[ty, tx]| (ty as usize + at[0]) * ntx + tx as usize + at[1])
            .collect();
        t.sort_unstable();
        t.dedup();
        t
    }

    /// On one GPU the region is the one-GPU scan itself.
    #[test]
    fn one_gpu_holds_the_whole_scan() {
        for (name, shape, views, scheme) in cases() {
            let plan = LivePlan::new(&shape, &views, scheme, 1).unwrap();
            let g = &plan.regions()[0];
            assert_eq!(g.window.at, [0, 0], "{name}");
            assert_eq!(g.shape, shape, "{name}");
            assert_eq!(g.planned, views, "{name}");
            let ntiles = plan.partition().grid().len() as u32;
            assert_eq!(g.own, (0..ntiles).collect::<Vec<_>>(), "{name}");
            assert!(g.reads.is_empty(), "{name}");
            assert_eq!(plan.halo_words(0), [0, 0], "{name}");
        }
    }

    /// Split in arrival order, in batches of any length, every owned tile of every region
    /// bins the frames that touch it on one GPU, in their arrival order; each region gets
    /// its planned frames, and only frames that touch one of its tiles.
    #[test]
    fn each_owned_tile_bins_the_one_gpu_frames_in_arrival_order() {
        for (name, shape, views, scheme) in cases() {
            let dims = shape.dims();
            let nmode = dims.nmode as usize;
            for ngpu in [1, 2, 4] {
                let plan = LivePlan::new(&shape, &views, scheme, ngpu).unwrap();
                let (grid, owner) = (plan.partition().grid(), plan.partition().owner());
                let ntx = grid.dims()[1];
                let mut shares = vec![Share::default(); ngpu];
                let mut local: Vec<Vec<(u32, Vec<ViewAddr>)>> = vec![Vec::new(); ngpu];
                let (mut start, mut sizes) = (0, [1, 7, 64, 3].iter().cycle());
                while start < shape.nframe as usize {
                    let end = (start + sizes.next().unwrap()).min(shape.nframe as usize);
                    plan.split(&views[start * nmode..end * nmode], &mut shares)
                        .unwrap();
                    for (r, s) in shares.iter().enumerate() {
                        assert_eq!(s.views.len(), s.frames.len() * nmode);
                        for (k, &i) in s.frames.iter().enumerate() {
                            let v = s.views[k * nmode..][..nmode].to_vec();
                            local[r].push((start as u32 + i, v));
                        }
                    }
                    start = end;
                }
                let one: Vec<Vec<usize>> = (0..grid.len())
                    .map(|t| {
                        (0..shape.nframe as usize)
                            .filter(|&f| {
                                let v = &views[f * nmode..][..nmode];
                                tiles_of(&dims, v, ntx, [0, 0]).contains(&t)
                            })
                            .collect()
                    })
                    .collect();
                for (r, g) in plan.regions().iter().enumerate() {
                    let what = format!("{name} on {ngpu}: region {r}");
                    let got: Vec<u32> = local[r].iter().map(|(f, _)| *f).collect();
                    assert_eq!(got, g.frames, "{what}");
                    let (ld, at) = (g.shape.dims(), g.window.at.map(|p| p / grid.tile()));
                    let mut binned = vec![Vec::new(); grid.len()];
                    for (f, v) in &local[r] {
                        let global = &views[*f as usize * nmode..][..nmode];
                        let back: Vec<ViewAddr> = v
                            .iter()
                            .map(|w| ViewAddr {
                                oy: w.oy + g.window.at[0] as u32,
                                ox: w.ox + g.window.at[1] as u32,
                                ..*w
                            })
                            .collect();
                        assert_eq!(back, global, "{what}: frame {f}");
                        let t = tiles_of(&ld, v, ntx, at);
                        assert!(t.iter().any(|&t| owner[t] as usize == r), "{what}: {f}");
                        t.iter().for_each(|&t| binned[t].push(*f as usize));
                    }
                    for t in (0..grid.len()).filter(|&t| owner[t] as usize == r) {
                        assert_eq!(binned[t], one[t], "{what}: tile {t}");
                    }
                }
            }
        }
    }

    /// A region's window holds its own tiles and the tiles its frames read, as an object of
    /// its scan's shape whose tiles are the one-GPU object's; every planned view fits it.
    #[test]
    fn windows_hold_own_and_read_tiles_as_the_scan_object() {
        for (name, shape, views, scheme) in cases() {
            let dims = shape.dims();
            for ngpu in [2, 4] {
                let plan = LivePlan::new(&shape, &views, scheme, ngpu).unwrap();
                let (grid, owner) = (plan.partition().grid(), plan.partition().owner());
                let tile = grid.tile();
                for (r, g) in plan.regions().iter().enumerate() {
                    let what = format!("{name} on {ngpu}: region {r}");
                    let ld = g.shape.dims();
                    assert_eq!(ld.obj.map(|n| n as usize), g.window.shape, "{what}");
                    assert_eq!(g.window.at.map(|p| p % tile), [0, 0], "{what}");
                    let end = [0, 1].map(|a| g.window.at[a] + g.window.shape[a]);
                    let [ny, nx] = grid.obj();
                    assert!(end[0] % tile == 0 || end[0] == ny, "{what}: {end:?}");
                    assert!(end[1] % tile == 0 || end[1] == nx, "{what}: {end:?}");
                    let own: Vec<u32> = (0..grid.len())
                        .filter(|&t| owner[t] as usize == r)
                        .map(|t| g.window.tile(grid, t))
                        .collect();
                    assert_eq!(g.own, own, "{what}");
                    assert!(g.reads.iter().all(|&t| owner[t as usize] as usize != r));
                    let at = g.window.at.map(|p| p / tile);
                    let mut read = Vec::new();
                    for v in g.planned.chunks(dims.nmode as usize) {
                        assert!(v.iter().all(|v| ld.check_view(v).is_ok()), "{what}");
                        let t = tiles_of(&ld, v, grid.dims()[1], at);
                        read.extend(t.into_iter().filter(|&t| owner[t] as usize != r));
                    }
                    read.sort_unstable();
                    read.dedup();
                    let read: Vec<u32> = read.into_iter().map(|t| t as u32).collect();
                    assert_eq!(g.reads, read, "{what}");
                }
            }
        }
    }

    /// Each region's halo receive rows are exactly the pixels of the tiles it reads, every
    /// object mode; the send buffers hold what the receive buffers take.
    #[test]
    fn halo_rows_cover_the_read_tiles() {
        for (name, shape, views, scheme) in cases() {
            for ngpu in [2, 4] {
                let mut plan = LivePlan::new(&shape, &views, scheme, ngpu).unwrap();
                let mut rows = vec![HaloRows::default(); ngpu];
                for r in 0..ngpu {
                    let reads = plan.regions[r].reads.iter().map(|&t| t as usize);
                    plan.halo.read(r, reads, &mut rows);
                }
                let grid = plan.partition().grid();
                let [nyo, nxo] = grid.obj();
                let tile = grid.tile();
                let nobj = shape.nobj as usize;
                let (mut sent, mut taken) = (0, 0);
                for (r, g) in plan.regions().iter().enumerate() {
                    let what = format!("{name} on {ngpu}: region {r}");
                    let win = g.window;
                    let mut got = Vec::new();
                    for run in rows[r].unpack.iter().map(|w| w.unpack(win, 0, 0)) {
                        let at = run.dst as usize / size_of::<crate::gpu::C32>();
                        got.extend(at..at + run.len as usize);
                    }
                    got.sort_unstable();
                    let mut want = Vec::new();
                    for m in 0..nobj {
                        for &t in &g.reads {
                            let [ty, tx] = grid.coords(t as usize);
                            for y in ty * tile..((ty + 1) * tile).min(nyo) {
                                for x in tx * tile..((tx + 1) * tile).min(nxo) {
                                    want.push(win.index(m, y, x));
                                }
                            }
                        }
                    }
                    want.sort_unstable();
                    assert_eq!(got, want, "{what}");
                    let [send, recv] = plan.halo_words(r);
                    assert_eq!(recv as usize, want.len(), "{what}");
                    (sent, taken) = (sent + send, taken + recv);
                }
                assert_eq!(sent, taken, "{name} on {ngpu}");
                assert!(sent > 0, "{name} on {ngpu}: no halo");
            }
        }
    }

    /// A frame off the object, or one reading a tile its region's planned frames do not, is
    /// refused, with the frames before it split.
    #[test]
    fn split_refuses_frames_outside_the_plan() {
        let (shape, views) = scan(sim::raster(4, 2, 8e-8), 32);
        let nmode = shape.nmode as usize;
        // Columns at x = 0 and 8 px, then the same at x = 200 px, so each region's frames
        // read no tile of the other: a frame between them reads tiles of both.
        let far: Vec<ViewAddr> = views
            .iter()
            .map(|&v| ViewAddr {
                ox: v.ox + 200,
                ..v
            })
            .collect();
        let both = [views.clone(), far].concat();
        let shape = ScanShape {
            nframe: shape.nframe * 2,
            reach: [shape.reach[0], shape.reach[1] + 200],
            ..shape
        };
        let plan = LivePlan::new(&shape, &both, Scheme::Strips, 2).unwrap();
        assert!(plan.regions().iter().all(|g| g.reads.is_empty()));
        let mut shares = vec![Share::default(); 2];
        let mid: Vec<ViewAddr> = both[..nmode]
            .iter()
            .map(|&v| ViewAddr {
                ox: v.ox + 100,
                ..v
            })
            .collect();
        let batch = [&both[..nmode], &mid].concat();
        let err = plan.split(&batch, &mut shares).unwrap_err().to_string();
        assert!(err.contains("does not hold"), "{err}");
        let landed: Vec<&[u32]> = shares.iter().map(|s| s.frames.as_slice()).collect();
        assert_eq!(
            landed,
            [&[0][..], &[]],
            "only the frame before the refused one"
        );
        let off: Vec<ViewAddr> = both[..nmode]
            .iter()
            .map(|&v| ViewAddr { oy: u32::MAX, ..v })
            .collect();
        let err = plan.split(&off, &mut shares).unwrap_err().to_string();
        assert!(err.contains("outside"), "{err}");
        plan.split(&both, &mut shares).unwrap();
        assert_eq!(
            shares[0].frames.len() + shares[1].frames.len(),
            both.len() / nmode
        );
    }
}
