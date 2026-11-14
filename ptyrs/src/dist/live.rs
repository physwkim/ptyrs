//! The live split (DESIGN B3, B4, B6): a live scan's object cut into one region per GPU as
//! [`DistAp`](crate::dist::DistAp) cuts it, each frame landing, in arrival order, on every GPU
//! that owns a tile the frame touches. A GPU's own tiles then bin the frames one GPU bins
//! there, in the same order, so their object sums add in the same order (invariant 8); the
//! other regions' tiles its frames read come from their owners after each update
//! (invariant 5). A GPU's window holds every tile a frame touching its own tiles can read,
//! planned or not, and its halo grows as frames read new tiles, so a frame off the plan
//! lands wherever one GPU would take it. The split admits frames through the one-GPU scan's
//! gate on the whole object, so every GPU's batch ends at the frame one GPU's scan
//! refuses.

use std::sync::Arc;

use cuda_core::{Context, Stream};

use crate::gpu::{ViewAddr, ViewDims, view_tiles};
use crate::live::scan::{Gate, Refusal, ScanShape, off_view};

use crate::dist::exchange::Pull;
use crate::dist::halo::{Halo, HaloBufs, HaloRows, Window};
use crate::dist::{DEFAULT_TILE, DistError, NLEAF, Partition, Scheme, TileGrid};

/// One GPU's part of a live scan.
#[derive(Clone)]
pub struct LiveRegion {
    /// The part of the object it holds: its own tiles and their partition halo, every tile a
    /// frame inside the object that touches one of its own can read.
    pub(crate) window: Window,
    /// Its scan over its window, of the one-GPU scan's frame count, so whatever share of the
    /// frames it gets it closes past the one-GPU scan's end only.
    pub shape: ScanShape,
    /// Its own tiles, numbered in its window, ascending: the only ones it bins and updates.
    pub own: Vec<u32>,
    /// The planned frames routed to it, ascending.
    pub frames: Vec<u32>,
    /// Their views, offsets into its window: what its scan's backer plans.
    pub planned: Vec<ViewAddr>,
    /// Words of its halo send and receive buffers: every tile of its partition halo, so the
    /// buffers and pulls never change as frames read new tiles.
    halo_words: [u64; 2],
    /// The most halo rows it packs and unpacks over the whole scan, whatever frames land.
    halo_rows: [u64; 2],
}

impl LiveRegion {
    /// Its side of the halo exchange, for its window at `obj` on `stream`'s device, the row
    /// lists backed `chunk` bytes at a time.
    pub fn halo_bufs(
        &self,
        stream: &Arc<Stream>,
        obj: u64,
        chunk: usize,
    ) -> Result<HaloBufs, DistError> {
        HaloBufs::new(
            stream,
            self.halo_words,
            self.halo_rows,
            self.window,
            obj,
            chunk,
        )
    }
}

/// A live scan's split over GPUs, cut from its planned frames before the first one lands.
pub struct LivePlan {
    run: Partition,
    /// The one-GPU scan's.
    dims: ViewDims,
    /// The one-GPU scan's, on every tile.
    gate: Gate,
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
    /// The halo rows they add: those the region packs of its tiles the batch's frames read on
    /// other regions for the first time, and those it unpacks of others' tiles its own read
    /// for the first time.
    pub halo: HaloRows,
}

impl AsMut<HaloRows> for Share {
    fn as_mut(&mut self) -> &mut HaloRows {
        &mut self.halo
    }
}

impl LivePlan {
    /// The split over `ngpu` GPUs, cut by `scheme` from [`NLEAF`] leaves, of the one-GPU scan
    /// `shape` with `cap` view slots per tile, whose first frames are planned with the views
    /// `planned`, `nmode` each.
    pub fn new(
        shape: &ScanShape,
        planned: &[ViewAddr],
        scheme: Scheme,
        ngpu: usize,
        cap: u32,
    ) -> Result<Self, DistError> {
        let bad = |m: String| Err(DistError::Partition(m));
        let dims = shape.dims();
        let nmode = dims.nmode as usize;
        if nmode == 0
            || !planned.len().is_multiple_of(nmode)
            || planned.len() > dims.nframe as usize * nmode
        {
            return bad(format!(
                "{} planned views are not at most {} frames of {nmode}",
                planned.len(),
                dims.nframe
            ));
        }
        if let Some((f, v)) = (0..)
            .zip(planned.chunks(nmode))
            .find_map(|(f, views)| off_view(&dims, views).map(|v| (f, v)))
        {
            return bad(format!(
                "planned frame {f}: {v:?} is no view of it in {dims:?}"
            ));
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
        let mut frames = vec![Vec::new(); ngpu];
        let mut tiles = Vec::new();
        for (f, views) in planned.chunks(nmode).enumerate() {
            touched(&dims, &grid, &views[0], &mut tiles);
            for r in owners(run.owner(), &tiles, ngpu) {
                frames[r].push(f as u32);
            }
        }
        let halo = Halo::new(
            &run,
            run.regions().iter().map(|g| g.halo.as_slice()),
            dims.nobj as usize,
        );
        let mut regions = Vec::with_capacity(ngpu);
        for (r, g) in run.regions().iter().enumerate() {
            let frames = std::mem::take(&mut frames[r]);
            if frames.is_empty() {
                return bad(format!("region {r} of {ngpu} has no frames"));
            }
            let window = Window::new(&grid, g.own.iter().chain(&g.halo).map(|&t| t as usize));
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
                    reach: [0, 1].map(|a| (window.shape[a] - frame[a] - 1) as u32),
                    ..*shape
                },
                own: g
                    .own
                    .iter()
                    .map(|&t| window.tile(&grid, t as usize))
                    .collect(),
                window,
                frames,
                planned,
                halo_words: halo.words(r),
                halo_rows: halo.max_rows(r),
            });
        }
        let [nty, ntx] = dims.tiles();
        let all: Vec<u32> = (0..nty * ntx).collect();
        Ok(Self {
            run,
            dims,
            gate: Gate::new(dims, cap, &all),
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

    /// Words of region `r`'s halo send and receive buffers.
    pub fn halo_words(&self, r: usize) -> [u64; 2] {
        self.regions[r].halo_words
    }

    /// The copy-engine pulls into region `r`'s halo receive buffer at `recv` from the owners'
    /// send buffers, `send(o)` giving owner `o`'s context and send buffer
    /// ([`HaloBufs::ptrs`]): the same for the whole scan.
    pub fn halo_pulls(
        &self,
        r: usize,
        recv: u64,
        send: impl Fn(usize) -> (Arc<Context>, u64),
    ) -> Vec<Pull> {
        self.halo.pulls(r, self.regions.len(), recv, send)
    }

    /// Splits the arriving frames whose views are `views`, `nmode` per frame, in arrival
    /// order, into `shares`, one per region, up to the first one the one-GPU scan refuses,
    /// and returns how many it took: each frame goes to the owner of every tile it touches,
    /// and the halo rows of the tiles it reads on a region for the first time to that region
    /// and the tiles' owners. The refused frame closes the plan as it closes the one-GPU
    /// scan ([`Self::closed`]), and every frame after it is lost.
    pub fn split(&mut self, views: &[ViewAddr], shares: &mut [Share]) -> Result<u32, DistError> {
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
            s.halo.pack.clear();
            s.halo.unpack.clear();
        });
        let grid = self.run.grid();
        let (mut tiles, mut took) = (Vec::new(), 0);
        for frame in views.chunks(nmode) {
            if self.gate.admit(frame).is_err() {
                break;
            }
            touched(&self.dims, grid, &frame[0], &mut tiles);
            for r in owners(self.run.owner(), &tiles, self.regions.len()) {
                // Inside the object, the tiles a frame touches lie within its frame width of
                // each other, so within region `r`'s partition halo of the one it owns.
                self.halo.read(r, tiles.iter().copied(), shares);
                let win = self.regions[r].window;
                shares[r].views.extend(frame.iter().map(|&v| win.shift(v)));
                shares[r].frames.push(took);
            }
            took += 1;
        }
        Ok(took)
    }

    /// The frame that closed the plan, as it closed the one-GPU scan.
    pub fn closed(&self) -> Option<Refusal> {
        self.gate.closed()
    }
}

/// The tiles of `grid` a frame at `view`'s position touches, by the kernels' rule, into
/// `out`, ascending: every view of a frame is at its one position ([`off_view`]).
fn touched(dims: &ViewDims, grid: &TileGrid, view: &ViewAddr, out: &mut Vec<usize>) {
    out.clear();
    out.extend(view_tiles(dims, view).map(|[ty, tx]| grid.id(ty as usize, tx as usize)));
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
    use crate::dist::exchange::{self, Exchange, Run, Runs};
    use crate::dist::halo::{HaloBacker, HaloRows};
    use crate::dist::worker::{Workers, tests::gpus};
    use crate::live::graph::{LiveGraph, WarmUp};
    use crate::live::recon::ViewPlacement;
    use crate::live::scan::{LiveScan, ScanBacker};
    use cuda_core::DeviceBuffer;
    use std::num::NonZeroU32;

    const PIXEL: [f64; 2] = [1e-8, 1e-8];
    /// Slab slots no test here fills.
    const ROOM: u32 = u32::MAX;

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

    /// Every planned frame moved by `d` pixels, those still inside the object of `dims`.
    fn jittered(dims: &ViewDims, views: &[ViewAddr], d: [i64; 2]) -> Vec<ViewAddr> {
        let nmode = dims.nmode as usize;
        let mv = |o: u32, d: i64| u32::try_from(o as i64 + d).unwrap_or(u32::MAX);
        views
            .chunks(nmode)
            .map(|f| {
                f.iter()
                    .map(|&v| ViewAddr {
                        oy: mv(v.oy, d[0]),
                        ox: mv(v.ox, d[1]),
                        ..v
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|f| f.iter().all(|v| dims.check_view(v).is_ok()))
            .flatten()
            .collect()
    }

    /// Two columns of frames at x = 0 and 8 px, then the same at x = 200 px, planned as the
    /// first quarter of the scan and cut in two strips whose planned frames read no tile of
    /// the other, and a frame between them that reads tiles of both.
    fn far() -> (ScanShape, Vec<ViewAddr>, Vec<ViewAddr>) {
        let (shape, views) = scan(sim::raster(4, 2, 8e-8), 32);
        let nmode = shape.nmode as usize;
        let moved = |d: u32| -> Vec<ViewAddr> {
            views
                .iter()
                .map(|&v| ViewAddr { ox: v.ox + d, ..v })
                .collect()
        };
        let both = [views.clone(), moved(200)].concat();
        let shape = ScanShape {
            nframe: shape.nframe * 8,
            reach: [shape.reach[0], shape.reach[1] + 200],
            ..shape
        };
        (shape, both, moved(100)[..nmode].to_vec())
    }

    /// The pixel values each region's window holds after one exchange of the halo rows
    /// `rows`, from windows holding only their own tiles' pixels: every tile `read[r]` marks,
    /// and no other foreign one, must hold its owner's pixels.
    fn check_exchange(plan: &LivePlan, rows: &[HaloRows], read: &[Vec<bool>], what: &str) {
        let (grid, owner) = (plan.partition().grid(), plan.partition().owner());
        let [nyo, nxo] = grid.obj();
        let n = plan.nregion();
        let nobj = plan.dims.nobj as usize;
        let value = |m: usize, y: usize, x: usize| ((m * nyo + y) * nxo + x + 1) as u64;
        let pixels = |w: Window| {
            (0..nobj).flat_map(move |m| {
                (w.at[0]..w.at[0] + w.shape[0])
                    .flat_map(move |y| (w.at[1]..w.at[1] + w.shape[1]).map(move |x| [m, y, x]))
            })
        };
        let mut wins: Vec<Vec<u64>> = plan
            .regions()
            .iter()
            .enumerate()
            .map(|(r, g)| {
                let mut win = vec![0; g.window.len(nobj)];
                for [m, y, x] in pixels(g.window) {
                    if owner[grid.of_pixel(y, x)] as usize == r {
                        win[g.window.index(m, y, x)] = value(m, y, x);
                    }
                }
                win
            })
            .collect();
        let mut send: Vec<Vec<u64>> = (0..n)
            .map(|r| vec![0; plan.halo_words(r)[0] as usize])
            .collect();
        for (o, g) in plan.regions().iter().enumerate() {
            for run in rows[o].pack.iter().map(|w| w.pack(g.window, 0, 0)) {
                let (dst, src, len) = ((run.dst / 8) as usize, (run.src / 8) as usize, run.len);
                send[o][dst..][..len as usize].copy_from_slice(&wins[o][src..][..len as usize]);
            }
        }
        for (r, g) in plan.regions().iter().enumerate() {
            let mut recv = vec![0; plan.halo_words(r)[1] as usize];
            for [o, at, from, len] in plan.halo.links(r, n) {
                let [at, from, len] = [at, from, len].map(|w| w as usize);
                recv[at..][..len].copy_from_slice(&send[o as usize][from..][..len]);
            }
            for run in rows[r].unpack.iter().map(|w| w.unpack(g.window, 0, 0)) {
                let (dst, src, len) = ((run.dst / 8) as usize, (run.src / 8) as usize, run.len);
                wins[r][dst..][..len as usize].copy_from_slice(&recv[src..][..len as usize]);
            }
            for [m, y, x] in pixels(g.window) {
                let t = grid.of_pixel(y, x);
                let want = if owner[t] as usize == r || read[r][t] {
                    value(m, y, x)
                } else {
                    0
                };
                let got = wins[r][g.window.index(m, y, x)];
                assert_eq!(got, want, "{what}: region {r} mode {m} pixel {y} {x}");
            }
        }
    }

    /// Splits `batches` in turn, adding each share's halo rows to `rows` and marking in
    /// `read[r]` the foreign tiles its frames touch.
    fn split_all<'a>(
        plan: &mut LivePlan,
        batches: impl IntoIterator<Item = &'a [ViewAddr]>,
        rows: &mut [HaloRows],
        read: &mut [Vec<bool>],
    ) {
        let n = plan.nregion();
        let nmode = plan.dims.nmode as usize;
        let mut shares = vec![Share::default(); n];
        for batch in batches {
            plan.split(batch, &mut shares).unwrap();
            let (grid, owner) = (plan.partition().grid(), plan.partition().owner());
            for (r, (s, g)) in shares.iter().zip(plan.regions()).enumerate() {
                rows[r].pack.extend(&s.halo.pack);
                rows[r].unpack.extend(&s.halo.unpack);
                let (ld, at) = (g.shape.dims(), g.window.at.map(|p| p / grid.tile()));
                for v in s.views.chunks(nmode) {
                    for t in tiles_of(&ld, v, grid.dims()[1], at) {
                        read[r][t] |= owner[t] as usize != r;
                    }
                }
            }
        }
    }

    /// On one GPU the region is the one-GPU scan itself.
    #[test]
    fn one_gpu_holds_the_whole_scan() {
        for (name, shape, views, scheme) in cases() {
            let plan = LivePlan::new(&shape, &views, scheme, 1, ROOM).unwrap();
            let g = &plan.regions()[0];
            assert_eq!(g.window.at, [0, 0], "{name}");
            assert_eq!(g.shape, shape, "{name}");
            assert_eq!(g.planned, views, "{name}");
            let ntiles = plan.partition().grid().len() as u32;
            assert_eq!(g.own, (0..ntiles).collect::<Vec<_>>(), "{name}");
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
                let mut plan = LivePlan::new(&shape, &views, scheme, ngpu, ROOM).unwrap();
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
                let (grid, owner) = (plan.partition().grid(), plan.partition().owner());
                let ntx = grid.dims()[1];
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

    /// A region's window is an object of its scan's shape whose tiles are the one-GPU
    /// object's, and every view inside the one-GPU object that touches one of its own tiles,
    /// planned or not, whole-pixel or not, fits it and touches the same tiles there.
    #[test]
    fn any_frame_on_own_tiles_fits_the_window() {
        for (name, shape, views, scheme) in cases() {
            let dims = shape.dims();
            for ngpu in [2, 4] {
                let plan = LivePlan::new(&shape, &views, scheme, ngpu, ROOM).unwrap();
                let (grid, owner) = (plan.partition().grid(), plan.partition().owner());
                let (tile, ntx) = (grid.tile(), grid.dims()[1]);
                for (r, g) in plan.regions().iter().enumerate() {
                    let what = format!("{name} on {ngpu}: region {r}");
                    let ld = g.shape.dims();
                    assert_eq!(ld.nframe, dims.nframe, "{what}");
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
                    assert!(g.planned.iter().all(|v| ld.check_view(v).is_ok()), "{what}");
                }
                let mut fits = 0;
                for f in [0.0, 0.5] {
                    for oy in 0..dims.obj[0] {
                        for ox in 0..dims.obj[1] {
                            let v = ViewAddr {
                                oy,
                                ox,
                                fy: f,
                                fx: f,
                                probe: 0,
                                obj: 0,
                            };
                            if dims.check_view(&v).is_err() {
                                continue;
                            }
                            let t = tiles_of(&dims, &[v], ntx, [0, 0]);
                            for r in owners(owner, &t, ngpu) {
                                let g = &plan.regions()[r];
                                let (ld, w) = (g.shape.dims(), g.window.shift(v));
                                let what = format!("{name} on {ngpu}: region {r}, {v:?}");
                                assert!(ld.check_view(&w).is_ok(), "{what}");
                                let at = g.window.at.map(|p| p / tile);
                                assert_eq!(tiles_of(&ld, &[w], ntx, at), t, "{what}");
                                fits += 1;
                            }
                        }
                    }
                }
                assert!(fits > 0, "{name} on {ngpu}");
            }
        }
    }

    /// After the planned frames and frames moved off the plan, split in batches, each
    /// receiver's window holds its owners' pixels in exactly the foreign tiles its frames
    /// read, from one exchange of the rows the batches added. In some case the moved frames
    /// read tiles the planned ones do not.
    #[test]
    fn exchange_gives_receivers_the_owners_pixels_of_every_tile_they_read() {
        let mut grew = false;
        for (name, shape, views, scheme) in cases() {
            let dims = shape.dims();
            let nmode = dims.nmode as usize;
            let off = [[5, -9], [-13, 11], [0, 17]]
                .map(|d| jittered(&dims, &views, d))
                .concat();
            assert!(!off.is_empty(), "{name}");
            let shape = ScanShape {
                nframe: shape.nframe + (off.len() / nmode) as u32,
                ..shape
            };
            for ngpu in [2, 4] {
                let what = format!("{name} on {ngpu}");
                let mut plan = LivePlan::new(&shape, &views, scheme, ngpu, ROOM).unwrap();
                let ntiles = plan.partition().grid().len();
                let mut rows = vec![HaloRows::default(); ngpu];
                let mut read = vec![vec![false; ntiles]; ngpu];
                split_all(&mut plan, views.chunks(5 * nmode), &mut rows, &mut read);
                let planned: usize = rows.iter().map(|w| w.unpack.len()).sum();
                assert!(planned > 0, "{what}: no halo");
                check_exchange(&plan, &rows, &read, &format!("{what}, planned"));
                let words = |rows: &[HaloRows]| -> u64 {
                    let rows = rows.iter().flat_map(|w| &w.unpack);
                    rows.map(|w| u64::from(w.len)).sum()
                };
                let before = words(&rows);
                split_all(&mut plan, off.chunks(3 * nmode), &mut rows, &mut read);
                let max: u64 = (0..ngpu).map(|r| plan.halo_words(r)[1]).sum();
                assert!(words(&rows) <= max, "{what}");
                grew |= words(&rows) > before;
                check_exchange(&plan, &rows, &read, &format!("{what}, off the plan"));
                let packed: usize = rows.iter().map(|w| w.pack.len()).sum();
                let unpacked: usize = rows.iter().map(|w| w.unpack.len()).sum();
                assert_eq!(packed, unpacked, "{what}");
            }
        }
        assert!(grew, "no frame off the plan read a new tile");
    }

    /// A frame between two strips whose planned frames read nothing of each other lands on
    /// both, and adds the rows of the tiles it reads on each to both: the halo grows past the
    /// plan. Split again, it adds none.
    #[test]
    fn a_frame_off_the_plan_grows_the_halo() {
        let (shape, both, mid) = far();
        let mut plan = LivePlan::new(&shape, &both, Scheme::Strips, 2, ROOM).unwrap();
        let ntiles = plan.partition().grid().len();
        let mut rows = vec![HaloRows::default(); 2];
        let mut read = vec![vec![false; ntiles]; 2];
        split_all(&mut plan, [both.as_slice()], &mut rows, &mut read);
        assert_eq!(rows, vec![HaloRows::default(); 2], "the plan reads no halo");
        let mut shares = vec![Share::default(); 2];
        plan.split(&mid, &mut shares).unwrap();
        for s in &shares {
            assert_eq!(s.frames, [0]);
            assert!(!s.halo.pack.is_empty() && !s.halo.unpack.is_empty());
        }
        split_all(&mut plan, [mid.as_slice()], &mut rows, &mut read);
        assert_eq!(
            rows,
            vec![HaloRows::default(); 2],
            "a frame read again adds rows"
        );
        let mut rows: Vec<HaloRows> = shares.into_iter().map(|s| s.halo).collect();
        check_exchange(&plan, &rows, &read, "far");
        split_all(&mut plan, [both.as_slice()], &mut rows, &mut read);
        check_exchange(&plan, &rows, &read, "far, split again");
    }

    /// One region's GPU in [`gpus_exchange_the_halo_as_it_grows`]: its window, an epoch's
    /// values over the window, the copies of its own tiles' pixel rows from those into the
    /// window, its halo and the graphs packing and unpacking it.
    struct OnGpu {
        ex: Exchange,
        win: DeviceBuffer<u64>,
        stage: DeviceBuffer<u64>,
        own: Runs,
        halo: HaloBufs,
        graphs: [LiveGraph; 2],
        pulls: Vec<Pull>,
    }

    /// What a region's GPU needs from the plan: its region and the `[window index, length]`
    /// of each pixel row of its own tiles.
    struct Setup {
        region: LiveRegion,
        own: Vec<[u64; 2]>,
    }

    /// The planned frames, then frames off the plan, split in batches on the host while graphs
    /// captured once before the first frame exchange the halo on the GPUs at each epoch: each
    /// batch's rows are listed on the GPUs, owners write new pixels, pack, every receiver pulls
    /// its fixed segments and unpacks. After each epoch every GPU's window holds that epoch's
    /// owner pixels in exactly the foreign tiles its frames read so far, and no capture is
    /// redone; in some case frames off the plan read tiles the plan does not, from no halo at
    /// all for two strips that read nothing of each other.
    #[test]
    fn gpus_exchange_the_halo_as_it_grows() {
        let mut grew = 0;
        for ngpu in [2, 4] {
            let w = Workers::spawn(&gpus(ngpu), |_| Ok(None::<OnGpu>)).unwrap();
            let mut all: Vec<_> = cases()
                .into_iter()
                .map(|(name, shape, views, scheme)| {
                    let dims = shape.dims();
                    let off = [[5, -9], [-13, 11], [0, 17]].map(|d| jittered(&dims, &views, d));
                    let off = off.concat();
                    let shape = ScanShape {
                        nframe: shape.nframe + (off.len() / dims.nmode as usize) as u32,
                        ..shape
                    };
                    (name, shape, views, off, scheme)
                })
                .collect();
            if ngpu == 2 {
                let (shape, both, mid) = far();
                all.push(("far", shape, both, mid, Scheme::Strips));
            }
            for (name, shape, views, off, scheme) in all {
                let what = format!("{name} on {ngpu}");
                let dims = shape.dims();
                let (nmode, nobj) = (dims.nmode as usize, dims.nobj as usize);
                let mut plan = LivePlan::new(&shape, &views, scheme, ngpu, ROOM).unwrap();
                let (grid, owner) = (*plan.partition().grid(), plan.partition().owner().to_vec());
                let ([nyo, nxo], tile) = (grid.obj(), grid.tile());
                let setup: Vec<Setup> = (0..ngpu)
                    .map(|r| {
                        let window = plan.regions()[r].window;
                        let mut own = Vec::new();
                        for t in (0..grid.len()).filter(|&t| owner[t] as usize == r) {
                            let [y0, x0] = grid.coords(t).map(|c| c * tile);
                            let len = ((x0 + tile).min(nxo) - x0) as u64;
                            for m in 0..nobj {
                                for y in y0..(y0 + tile).min(nyo) {
                                    own.push([window.index(m, y, x0) as u64, len]);
                                }
                            }
                        }
                        Setup {
                            region: plan.regions()[r].clone(),
                            own,
                        }
                    })
                    .collect();
                let setup = Arc::new(setup);
                let sides = w
                    .run(move |g, side| {
                        let (st, at) = (g.stream().clone(), &setup[g.region()]);
                        let len = at.region.window.len(nobj);
                        let win = DeviceBuffer::<u64>::zeroed(&st, len)?;
                        let stage = DeviceBuffer::<u64>::zeroed(&st, len)?;
                        let own: Vec<Run> = at
                            .own
                            .iter()
                            .map(|&[i, len]| Run {
                                dst: win.cu_deviceptr() + 8 * i,
                                src: stage.cu_deviceptr() + 8 * i,
                                len,
                            })
                            .collect();
                        let (ex, own) = (Exchange::load(g.ctx())?, Runs::new(&st, &own)?);
                        let obj = win.cu_deviceptr();
                        let halo = at.region.halo_bufs(&st, obj, 4096)?;
                        // SAFETY: `win` holds the window at `obj` for as long as the graphs, leased
                        // in them; each epoch writes it before the pack and pulls before the
                        // unpack, on this stream.
                        let pack = LiveGraph::capture(&st, |c| unsafe {
                            crate::gpu::lease(c, &[win.storage()], &[])?;
                            halo.pack(&ex, c)
                        })?;
                        let unpack = LiveGraph::capture(&st, |c| unsafe {
                            crate::gpu::lease(c, &[], &[win.storage()])?;
                            halo.unpack(&ex, c)
                        })?;
                        let out = (g.ctx().clone(), halo.ptrs(), halo.backer());
                        *side = Some(OnGpu {
                            ex,
                            win,
                            stage,
                            own,
                            halo,
                            graphs: [pack, unpack],
                            pulls: Vec::new(),
                        });
                        Ok(out)
                    })
                    .unwrap();
                let warm = WarmUp::end();
                let pulls: Vec<Vec<Pull>> = (0..ngpu)
                    .map(|r| {
                        plan.halo_pulls(r, sides[r].1[1], |o| (sides[o].0.clone(), sides[o].1[0]))
                    })
                    .collect();
                let mut backers: Vec<HaloBacker> = sides.into_iter().map(|(_, _, b)| b).collect();
                let pulls = Arc::new(pulls);
                w.run(move |g, side| {
                    side.as_mut().unwrap().pulls = pulls[g.region()].clone();
                    Ok(())
                })
                .unwrap();

                let value = |k: usize, m: usize, y: usize, x: usize| {
                    (((k * nobj + m) * nyo + y) * nxo + x + 1) as u64
                };
                let mut read = vec![vec![false; grid.len()]; ngpu];
                let mut listed = vec![[0; 2]; ngpu];
                let mut shares = vec![Share::default(); ngpu];
                let planned = views.chunks(7 * nmode).count();
                for (k, batch) in views
                    .chunks(7 * nmode)
                    .chain(off.chunks(5 * nmode))
                    .enumerate()
                {
                    plan.split(batch, &mut shares).unwrap();
                    for (r, (s, g)) in shares.iter().zip(plan.regions()).enumerate() {
                        backers[r].back(&s.halo).unwrap();
                        listed[r][0] += s.halo.pack.len();
                        listed[r][1] += s.halo.unpack.len();
                        if k >= planned && !s.halo.unpack.is_empty() {
                            grew += 1;
                        }
                        let (ld, at) = (g.shape.dims(), g.window.at.map(|p| p / tile));
                        for v in s.views.chunks(nmode) {
                            for t in tiles_of(&ld, v, grid.dims()[1], at) {
                                read[r][t] |= owner[t] as usize != r;
                            }
                        }
                    }
                    let rows: Vec<HaloRows> = shares.iter().map(|s| s.halo.clone()).collect();
                    let stage: Vec<Vec<u64>> = plan
                        .regions()
                        .iter()
                        .map(|g| {
                            let mut v = vec![0; g.window.len(nobj)];
                            for m in 0..nobj {
                                for y in g.window.at[0]..g.window.at[0] + g.window.shape[0] {
                                    for x in g.window.at[1]..g.window.at[1] + g.window.shape[1] {
                                        v[g.window.index(m, y, x)] = value(k, m, y, x);
                                    }
                                }
                            }
                            v
                        })
                        .collect();
                    let (rows, stage) = (Arc::new(rows), Arc::new(stage));
                    let got = w
                        .run(move |g, side| {
                            let (st, r) = (g.stream().clone(), g.region());
                            let s = side.as_mut().unwrap();
                            s.halo.push(&rows[r])?;
                            s.stage.copy_from_host(&st, &stage[r])?;
                            // SAFETY: own pixel rows from the staged values into the window,
                            // which no peer reads before the fence below.
                            unsafe { s.ex.copy_runs(&st, &s.own) }?;
                            s.graphs[0].replay()?;
                            g.fence()?;
                            // SAFETY: the owners packed before the fence and pack again only
                            // after the next one; the receive buffer is this GPU's alone.
                            unsafe { exchange::pull(&st, &s.pulls) }?;
                            s.graphs[1].replay()?;
                            g.fence()?;
                            Ok((s.win.to_host_vec(&st)?, s.halo.rows()))
                        })
                        .unwrap();
                    for (r, ((win, rows), g)) in got.iter().zip(plan.regions()).enumerate() {
                        let what = format!("{what}, epoch {k}: region {r}");
                        assert_eq!(*rows, listed[r], "{what}");
                        let max = g.halo_rows;
                        assert!(
                            rows[0] as u64 <= max[0] && rows[1] as u64 <= max[1],
                            "{what}"
                        );
                        let w = g.window;
                        for m in 0..nobj {
                            for y in w.at[0]..w.at[0] + w.shape[0] {
                                for x in w.at[1]..w.at[1] + w.shape[1] {
                                    let t = grid.of_pixel(y, x);
                                    let want = if owner[t] as usize == r || read[r][t] {
                                        value(k, m, y, x)
                                    } else {
                                        0
                                    };
                                    let got = win[w.index(m, y, x)];
                                    assert_eq!(got, want, "{what}: mode {m} pixel {y} {x}");
                                }
                            }
                        }
                    }
                }
                assert_eq!(warm.recaptures(), 0, "{what}");
            }
        }
        assert!(grew > 0, "no frame off the plan read a new tile");
    }

    /// A frame off the object closes the plan: the frames before it and the halo rows they
    /// add are split, and no frame after it.
    #[test]
    fn split_refuses_frames_off_the_object() {
        let (shape, both, mid) = far();
        let mut plan = LivePlan::new(&shape, &both, Scheme::Strips, 2, ROOM).unwrap();
        let off: Vec<ViewAddr> = mid
            .iter()
            .map(|&v| ViewAddr { oy: u32::MAX, ..v })
            .collect();
        let mut shares = vec![Share::default(); 2];
        let took = plan.split(&[mid.as_slice(), &off].concat(), &mut shares);
        assert_eq!(took.unwrap(), 1);
        let refused = Some(Refusal::View {
            frame: 1,
            view: off[0],
        });
        assert_eq!(plan.closed(), refused);
        for s in &shares {
            assert_eq!(s.frames, [0], "only the frame before the refused one");
            assert!(!s.halo.unpack.is_empty());
        }
        assert_eq!(plan.split(&both, &mut shares).unwrap(), 0);
        assert_eq!(shares, vec![Share::default(); 2]);
        assert_eq!(plan.closed(), refused);
    }

    /// Split in the batches a one-GPU scan lands, the plan takes the frames it lands and
    /// closes at the frame it refuses: one off the object, one with no free slot left on a
    /// tile, one past the scan. Every region's scan, binning its own tiles with the one-GPU
    /// slab size, lands its share whole.
    #[test]
    fn split_closes_where_one_gpu_refuses() {
        let ctx = Context::new(0).expect("GPU 0");
        let s = ctx.new_stream().unwrap();
        let one = NonZeroU32::new(1).unwrap();
        let (_, shape, views, scheme) = cases().swap_remove(0);
        let (nmode, fill) = (shape.nmode as usize, [1.0, 0.0]);
        let k = shape.nframe as usize / 2;
        let cap = shape.tile_cap(&views);
        let mut off = views.clone();
        off[k * nmode..][..nmode]
            .iter_mut()
            .for_each(|v| v.oy = u32::MAX);
        let short = ScanShape {
            nframe: k as u32,
            ..shape
        };
        for (what, sh, planned, cap, pushed) in [
            ("off the object", shape, &views[..], cap, &off),
            ("a full slab", shape, &views[..], cap - 1, &views),
            ("past the scan", short, &views[..k * nmode], cap, &views),
        ] {
            let batches: Vec<&[ViewAddr]> = pushed.chunks(7 * nmode).collect();
            let mut scan = LiveScan::new(&s, &sh, one, cap, fill, 1).unwrap();
            let mut b = scan.backer();
            let landed: Vec<u32> = batches
                .iter()
                .map(|batch| {
                    b.back(batch).unwrap();
                    scan.push(batch).unwrap()
                })
                .collect();
            let closed = scan.closed();
            assert!(closed.is_some(), "{what}");
            for ngpu in [2, 4] {
                let what = format!("{what} on {ngpu}");
                let mut plan = LivePlan::new(&sh, planned, scheme, ngpu, cap).unwrap();
                let mut local: Vec<(LiveScan, ScanBacker)> = plan
                    .regions()
                    .iter()
                    .map(|g| {
                        let l = LiveScan::owned(&s, &g.shape, one, cap, fill, 1, &g.own).unwrap();
                        let b = l.backer();
                        (l, b)
                    })
                    .collect();
                let mut shares = vec![Share::default(); ngpu];
                for (batch, &want) in batches.iter().zip(&landed) {
                    assert_eq!(plan.split(batch, &mut shares).unwrap(), want, "{what}");
                    for ((l, b), share) in local.iter_mut().zip(&shares) {
                        assert!(share.frames.iter().all(|&f| f < want), "{what}");
                        b.back(&share.views).unwrap();
                        let n = l.push(&share.views).unwrap();
                        assert_eq!(n as usize, share.frames.len(), "{what}");
                    }
                }
                assert_eq!(plan.closed(), closed, "{what}");
                for (l, _) in &local {
                    assert_eq!((l.closed(), l.lost()), (None, 0), "{what}");
                }
            }
        }
    }
}
