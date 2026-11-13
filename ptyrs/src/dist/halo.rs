//! The halo of a spatial split (DESIGN B4, invariant 5): each region's GPU holds a window of
//! the object, its own tiles and the halo tiles it reads, and after each object update copies
//! those halo rows from the owners, which pack them into a send buffer, pulled by copy engine
//! into its receive buffer and unpacked into its window.

use std::sync::Arc;

use crate::base::View;
use crate::gpu::{C32, ViewAddr};
use crate::live::grow::{Backer, GrowError};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Context, DeviceBuffer, RawDeviceBuffer, Stream};

use crate::dist::exchange::{Exchange, Pull, Run, RunList, Runs};
use crate::dist::{DistError, Partition, TileGrid};

/// The part of the object a region's GPU holds: the bounding box of its own tiles and the
/// halo tiles it reads, in whole tiles, so its kernel tiles are the object's shifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Window {
    /// First object pixel `[y, x]`.
    pub(crate) at: [usize; 2],
    /// `[ny, nx]` pixels per object mode.
    pub(crate) shape: [usize; 2],
}

impl Window {
    pub(crate) fn new(grid: &TileGrid, tiles: impl IntoIterator<Item = usize>) -> Self {
        let (mut lo, mut hi) = ([usize::MAX; 2], [0; 2]);
        for t in tiles {
            let c = grid.coords(t);
            for a in 0..2 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a] + 1);
            }
        }
        let (obj, tile) = (grid.obj(), grid.tile());
        let at = lo.map(|l| l * tile);
        Self {
            at,
            shape: [0, 1].map(|a| (hi[a] * tile).min(obj[a]) - at[a]),
        }
    }

    pub(crate) fn len(&self, nobj: usize) -> usize {
        nobj * self.shape[0] * self.shape[1]
    }

    /// Index in the window's `[nobj, ny, nx]` of object pixel `(y, x)` of mode `m`.
    pub(crate) fn index(&self, m: usize, y: usize, x: usize) -> usize {
        let [dy, dx] = [(y, 0), (x, 1)].map(|(p, a)| {
            p.checked_sub(self.at[a])
                .filter(|&d| d < self.shape[a])
                .expect("pixel inside the window")
        });
        (m * self.shape[0] + dy) * self.shape[1] + dx
    }

    /// The window of the `[nobj, nyo, nxo]` object `obj`.
    pub(crate) fn crop(&self, obj: &[C32], [nyo, nxo]: [usize; 2]) -> Vec<C32> {
        let ([y0, x0], [ny, nx]) = (self.at, self.shape);
        let mut out = Vec::with_capacity(self.len(obj.len() / (nyo * nxo)));
        for plane in obj.chunks_exact(nyo * nxo) {
            for y in y0..y0 + ny {
                out.extend_from_slice(&plane[y * nxo + x0..][..nx]);
            }
        }
        out
    }

    /// Index of `grid` tile `t` among the window's tiles, numbered as the kernels number the
    /// tiles of an object of the window's shape.
    pub(crate) fn tile(&self, grid: &TileGrid, t: usize) -> u32 {
        let tile = grid.tile();
        let ([ty, tx], [y0, x0]) = (grid.coords(t), self.at.map(|p| p / tile));
        ((ty - y0) * self.shape[1].div_ceil(tile) + tx - x0) as u32
    }

    /// `v` with offsets into the window.
    pub(crate) fn view(&self, v: &View) -> ViewAddr {
        self.shift(view_addr(v))
    }

    /// `v` with offsets into the window.
    pub(crate) fn shift(&self, v: ViewAddr) -> ViewAddr {
        ViewAddr {
            oy: v.oy - self.at[0] as u32,
            ox: v.ox - self.at[1] as u32,
            ..v
        }
    }

    /// This window grown at its far edges by whole tiles of `grid`, inside the object, to at
    /// least `len + 1` pixels per axis: the object a live scan of `len`-pixel frames sizes
    /// ([`ScanShape::dims`](crate::live::scan::ScanShape::dims)). `None` when the object ends
    /// first.
    pub(crate) fn hold(self, grid: &TileGrid, len: [usize; 2]) -> Option<Self> {
        let (obj, tile) = (grid.obj(), grid.tile());
        let mut shape = self.shape;
        for a in 0..2 {
            let need = self.at[a] + len[a] + 1;
            if need > obj[a] {
                return None;
            }
            let end = (self.at[a] + shape[a]).max(need.next_multiple_of(tile).min(obj[a]));
            shape[a] = end - self.at[a];
        }
        Some(Self { shape, ..self })
    }
}

/// A view as the kernels take it; integer offsets only.
pub(crate) fn view_addr(v: &View) -> ViewAddr {
    ViewAddr {
        oy: v.oy,
        ox: v.ox,
        fy: 0.0,
        fx: 0.0,
        probe: v.probe,
        obj: v.obj,
    }
}

/// The regions other than `r` of `n`, from `r + 1` on, wrapping: at each step of pull lists
/// in this order every region reads a different source, where in index order all would read
/// region 0 first and queue on its link.
pub(crate) fn peers(r: usize, n: usize) -> impl Iterator<Item = usize> {
    (1..n).map(move |i| (r + i) % n)
}

/// `len` object pixels of mode `m` from `(y, x)` on, at word `word` of a halo buffer: a row an
/// owner packs into its send buffer, or a receiver unpacks from its receive buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HaloRow {
    pub m: u32,
    pub y: u32,
    pub x: u32,
    pub len: u32,
    pub word: u64,
}

impl HaloRow {
    /// The copy of this row from the window `win` at `obj` into the send buffer at `send`.
    pub(crate) fn pack(&self, win: Window, obj: u64, send: u64) -> Run {
        Run {
            dst: send + self.word * 8,
            src: obj + self.offset(win),
            len: self.len as u64,
        }
    }

    /// The copy of this row from the receive buffer at `recv` into the window `win` at `obj`.
    pub(crate) fn unpack(&self, win: Window, obj: u64, recv: u64) -> Run {
        Run {
            dst: obj + self.offset(win),
            src: recv + self.word * 8,
            len: self.len as u64,
        }
    }

    fn offset(&self, win: Window) -> u64 {
        let [m, y, x] = [self.m, self.y, self.x].map(|i| i as usize);
        (win.index(m, y, x) * size_of::<C32>()) as u64
    }
}

/// The halo rows a region copies at each exchange, as owner and as receiver.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HaloRows {
    pub pack: Vec<HaloRow>,
    pub unpack: Vec<HaloRow>,
}

impl AsMut<HaloRows> for HaloRows {
    fn as_mut(&mut self) -> &mut HaloRows {
        self
    }
}

impl HaloRows {
    /// The pack and unpack copies of these rows for the window `win` at `obj` and the halo
    /// buffers at `send` and `recv`.
    fn runs(&self, win: Window, obj: u64, [send, recv]: [u64; 2]) -> [Vec<Run>; 2] {
        [
            self.pack.iter().map(|w| w.pack(win, obj, send)).collect(),
            self.unpack
                .iter()
                .map(|w| w.unpack(win, obj, recv))
                .collect(),
        ]
    }
}

/// Consecutive tiles of tile row `ty` from `tx0` on, one owner's, that a receiver may read; per
/// object mode, their pixel rows sit from word `at` of the mode's part of the segment, `width`
/// words each.
struct TileRun {
    ty: usize,
    tx0: usize,
    width: u64,
    at: u64,
}

/// The tiles region `recv` may read from region `owner`, laid out for good: `owner` packs them
/// from word `send_at` of its send buffer on; `recv` pulls them to word `recv_at` of its
/// receive buffer and unpacks them into its window.
struct HaloSeg {
    recv: usize,
    owner: usize,
    runs: Vec<TileRun>,
    /// Words per object mode.
    mode_words: u64,
    words: u64,
    /// Rows [`Halo::read`] adds at most: one per tile, object mode and pixel row.
    max_rows: u64,
    send_at: u64,
    recv_at: u64,
}

/// A tile a receiver may read: in run `run` of segment `seg`, and whether it reads it yet.
#[derive(Clone, Copy)]
struct MayRead {
    tile: u32,
    seg: u32,
    run: u32,
    read: bool,
}

/// The halo of a partition: the buffers of every tile each region may read from the others,
/// laid out before the first exchange, and the rows of those it reads so far
/// (DESIGN B6: the halo grows as frames read new tiles, its buffers never move).
pub(crate) struct Halo {
    grid: TileGrid,
    owner: Vec<u32>,
    nobj: usize,
    /// By receiver, then owner.
    segs: Vec<HaloSeg>,
    /// Per receiver, ascending.
    may: Vec<Vec<MayRead>>,
}

impl Halo {
    /// The halo of `run` whose region `r` may read the foreign tiles `may[r]` (ascending): per
    /// object mode, each pixel row of consecutive tiles with one owner; receive buffers hold
    /// the owners in order, send buffers the receivers. No region reads any yet.
    pub(crate) fn new<'a>(
        run: &Partition,
        may: impl IntoIterator<Item = &'a [u32]>,
        nobj: usize,
    ) -> Self {
        let grid = *run.grid();
        let [nyo, nxo] = grid.obj();
        let tile = grid.tile();
        let owner = run.owner().to_vec();
        let n = run.regions().len();
        let mut send_at = vec![0; n];
        let (mut segs, mut mays) = (Vec::new(), Vec::new());
        for (r, tiles) in may.into_iter().enumerate() {
            let mut runs: Vec<Vec<TileRun>> = (0..n).map(|_| Vec::new()).collect();
            let mut at = vec![0; n];
            let mut rows = vec![0; n];
            let mut members = Vec::with_capacity(tiles.len());
            let mut i = 0;
            while i < tiles.len() {
                let t0 = tiles[i] as usize;
                let [ty, tx0] = grid.coords(t0);
                let o = owner[t0] as usize;
                let mut tx1 = tx0;
                while let Some(&t) = tiles.get(i + 1) {
                    let t = t as usize;
                    if grid.coords(t) != [ty, tx1 + 1] || owner[t] as usize != o {
                        break;
                    }
                    tx1 += 1;
                    i += 1;
                }
                i += 1;
                let width = (((tx1 + 1) * tile).min(nxo) - tx0 * tile) as u64;
                let height = (((ty + 1) * tile).min(nyo) - ty * tile) as u64;
                members.extend((tx0..=tx1).map(|tx| (grid.id(ty, tx) as u32, o, runs[o].len())));
                runs[o].push(TileRun {
                    ty,
                    tx0,
                    width,
                    at: at[o],
                });
                at[o] += height * width;
                rows[o] += height * (tx1 - tx0 + 1) as u64;
            }
            let mut seg_of = vec![u32::MAX; n];
            let mut recv_at = 0;
            for (o, runs) in runs.into_iter().enumerate() {
                if runs.is_empty() {
                    continue;
                }
                let words = nobj as u64 * at[o];
                seg_of[o] = segs.len() as u32;
                segs.push(HaloSeg {
                    recv: r,
                    owner: o,
                    runs,
                    mode_words: at[o],
                    words,
                    max_rows: nobj as u64 * rows[o],
                    send_at: send_at[o],
                    recv_at,
                });
                send_at[o] += words;
                recv_at += words;
            }
            mays.push(
                members
                    .into_iter()
                    .map(|(tile, o, run)| MayRead {
                        tile,
                        seg: seg_of[o],
                        run: run as u32,
                        read: false,
                    })
                    .collect(),
            );
        }
        Self {
            grid,
            owner,
            nobj,
            segs,
            may: mays,
        }
    }

    /// Words of region `r`'s halo send and receive buffers.
    pub(crate) fn words(&self, r: usize) -> [u64; 2] {
        self.sides(r, |s| s.words)
    }

    /// The most pack and unpack rows [`Self::read`] adds for region `r` over a whole scan,
    /// reached when each call reads a single new tile of a segment.
    pub(crate) fn max_rows(&self, r: usize) -> [u64; 2] {
        self.sides(r, |s| s.max_rows)
    }

    /// `f` summed over the segments region `r` owns and over those it receives.
    fn sides(&self, r: usize, f: impl Fn(&HaloSeg) -> u64) -> [u64; 2] {
        let sum = |side: fn(&HaloSeg) -> usize| -> u64 {
            self.segs.iter().filter(|s| side(s) == r).map(&f).sum()
        };
        [sum(|s| s.owner), sum(|s| s.recv)]
    }

    /// Region `r` reads the tiles `tiles` from now on: for each foreign one it did
    /// not read yet, appends to `rows[o].pack` the rows its owner `o` packs and to
    /// `rows[r].unpack` those `r` unpacks, per segment, object mode, run of consecutive new
    /// tiles, pixel row. Its own tiles are skipped.
    ///
    /// # Panics
    /// If a foreign tile is not one `r` may read: the layout fixed at [`Self::new`] has no
    /// room for it.
    pub(crate) fn read(
        &mut self,
        r: usize,
        tiles: impl IntoIterator<Item = usize>,
        rows: &mut [impl AsMut<HaloRows>],
    ) {
        let may = &mut self.may[r];
        let mut new = Vec::new();
        for t in tiles {
            if self.owner[t] as usize == r {
                continue;
            }
            let k = may
                .binary_search_by_key(&(t as u32), |m| m.tile)
                .unwrap_or_else(|_| panic!("region {r} may not read tile {t}"));
            if !std::mem::replace(&mut may[k].read, true) {
                new.push(may[k]);
            }
        }
        // Tile order is each segment's run order, and each run's tile order.
        new.sort_by_key(|m| (m.seg, m.tile));
        let tile = self.grid.tile();
        let [nyo, nxo] = self.grid.obj();
        for group in new.chunk_by(|a, b| a.seg == b.seg) {
            let seg = &self.segs[group[0].seg as usize];
            let mut spans: Vec<(usize, usize, usize)> = Vec::new();
            for m in group {
                let tx = self.grid.coords(m.tile as usize)[1];
                match spans.last_mut() {
                    Some((run, _, end)) if *run == m.run as usize && *end + 1 == tx => *end = tx,
                    _ => spans.push((m.run as usize, tx, tx)),
                }
            }
            for mode in 0..self.nobj {
                for &(k, txa, txb) in &spans {
                    let run = &seg.runs[k];
                    let (y0, x0) = (run.ty * tile, run.tx0 * tile);
                    let (x, x1) = (txa * tile, ((txb + 1) * tile).min(nxo));
                    for y in y0..(y0 + tile).min(nyo) {
                        let word = mode as u64 * seg.mode_words
                            + run.at
                            + (y - y0) as u64 * run.width
                            + (x - x0) as u64;
                        let row = |at: u64| HaloRow {
                            m: mode as u32,
                            y: y as u32,
                            x: x as u32,
                            len: (x1 - x) as u32,
                            word: at + word,
                        };
                        rows[seg.owner].as_mut().pack.push(row(seg.send_at));
                        rows[r].as_mut().unpack.push(row(seg.recv_at));
                    }
                }
            }
        }
    }

    /// `[owner, receive word, send word, words]` of each pull into region `r`, in [`peers`]
    /// order over `n` regions: every tile `r` may read, read yet or not, so they never change.
    pub(crate) fn links(&self, r: usize, n: usize) -> impl Iterator<Item = [u64; 4]> {
        peers(r, n).flat_map(move |o| {
            self.segs
                .iter()
                .filter(move |s| s.recv == r && s.owner == o)
                .map(|s| [s.owner as u64, s.recv_at, s.send_at, s.words])
        })
    }

    /// Copy-engine [`Self::links`] into region `r`'s receive buffer at `recv` from the owners'
    /// send buffers, `send(o)` giving owner `o`'s context and send buffer.
    pub(crate) fn pulls(
        &self,
        r: usize,
        n: usize,
        recv: u64,
        send: impl Fn(usize) -> (Arc<Context>, u64),
    ) -> Vec<Pull> {
        self.links(r, n)
            .map(|[o, recv_at, send_at, words]| {
                let (from, at) = send(o as usize);
                Pull {
                    dst: recv + recv_at * 8,
                    src: at + send_at * 8,
                    bytes: (words * 8) as usize,
                    from,
                }
            })
            .collect()
    }
}

/// A region's side of the halo exchange: it packs what others read of its tiles into `send`
/// and unpacks what it pulled from the owners into `recv`.
pub(crate) struct HaloBufs {
    pub(crate) send: DeviceBuffer<u64>,
    pub(crate) recv: DeviceBuffer<u64>,
    pub(crate) pack: Runs,
    pub(crate) unpack: Runs,
}

impl HaloBufs {
    /// Region `r`'s buffers of `halo` and its runs copying `rows`, for its window `win` at `obj`.
    pub(crate) fn new(
        stream: &Stream,
        halo: &Halo,
        r: usize,
        rows: &HaloRows,
        win: Window,
        obj: u64,
    ) -> Result<Self, DistError> {
        let [send, recv] = halo
            .words(r)
            .map(|w| DeviceBuffer::<u64>::zeroed(stream, (w as usize).max(1)));
        let (send, recv) = (send?, recv?);
        let [pack, unpack] = rows.runs(win, obj, [send.cu_deviceptr(), recv.cu_deviceptr()]);
        Ok(Self {
            pack: Runs::new(stream, &pack)?,
            unpack: Runs::new(stream, &unpack)?,
            send,
            recv,
        })
    }
}

/// A region's side of the halo exchange as the halo grows (DESIGN B6, invariant 9): its send
/// and receive buffers hold every tile it may read or be read of, so the pulls between them
/// never change, and its pack and unpack rows are listed in device memory, so graphs captured
/// once over [`Self::pack`] and [`Self::unpack`] copy, at each replay, every row pushed before.
pub struct LiveHaloBufs {
    send: DeviceBuffer<u64>,
    recv: DeviceBuffer<u64>,
    pack: RunList,
    unpack: RunList,
    win: Window,
    obj: u64,
}

impl LiveHaloBufs {
    /// Buffers of `words` send and receive words and lists of `max_rows` pack and unpack rows,
    /// a region's [`Halo::words`] and [`Halo::max_rows`], backed `chunk` bytes at a time, for
    /// its window `win` at `obj` on `stream`'s device; rows are pushed on `stream`.
    pub(crate) fn new(
        stream: &Arc<Stream>,
        words: [u64; 2],
        max_rows: [u64; 2],
        win: Window,
        obj: u64,
        chunk: usize,
    ) -> Result<Self, DistError> {
        let [send, recv] = words.map(|w| DeviceBuffer::zeroed(stream, (w as usize).max(1)));
        let [pack, unpack] = max_rows.map(|n| RunList::new(stream, (n as usize).max(1), chunk));
        Ok(Self {
            send: send?,
            recv: recv?,
            pack: pack?,
            unpack: unpack?,
            win,
            obj,
        })
    }

    /// Backs the rows pushed from now on, from any thread.
    pub fn backer(&self) -> HaloBacker {
        HaloBacker {
            lists: [self.pack.backer(), self.unpack.backer()],
            next: [self.pack.len(), self.unpack.len()],
        }
    }

    /// Device addresses of the send and receive buffers, for
    /// [`LivePlan::halo_pulls`](crate::dist::LivePlan::halo_pulls).
    pub fn ptrs(&self) -> [u64; 2] {
        [self.send.cu_deviceptr(), self.recv.cu_deviceptr()]
    }

    /// Rows listed so far: `[pack, unpack]`.
    pub fn rows(&self) -> [usize; 2] {
        [self.pack.len(), self.unpack.len()]
    }

    /// Lists `rows`, which a [`HaloBacker`] backed: launches queued on the stream afterwards
    /// copy them too.
    pub fn push(&mut self, rows: &HaloRows) -> Result<(), DistError> {
        let [pack, unpack] = rows.runs(self.win, self.obj, self.ptrs());
        self.pack.push(&pack)?;
        self.unpack.push(&unpack)
    }

    /// Copies every row listed when it runs from the window into the send buffer, on `ctx`'s
    /// stream.
    ///
    /// # Safety
    /// `obj` must be the storage holding the window at the address [`Self::new`] took, on
    /// `ctx`'s device, and no peer may pull the send buffer meanwhile
    /// ([`crate::dist::Gpu::fence`]).
    pub unsafe fn pack(
        &self,
        ex: &Exchange,
        ctx: &ExecutionContext,
        obj: &RawDeviceBuffer,
    ) -> Result<(), crate::gpu::Error> {
        let [runs, count] = self.pack.storages();
        crate::gpu::lease(ctx, &[runs, count, obj], &[self.send.storage()])?;
        // SAFETY: each row reads its own pixels of the window and writes its own words of the
        // send buffer, both laid out by the halo; the caller keeps the window.
        unsafe { ex.copy_listed(ctx.get_cuda_stream(), &self.pack) }?;
        Ok(())
    }

    /// Copies every row listed when it runs from the receive buffer into the window, on
    /// `ctx`'s stream.
    ///
    /// # Safety
    /// As [`Self::pack`]; the pulls into the receive buffer must be queued before it on the
    /// stream.
    pub unsafe fn unpack(
        &self,
        ex: &Exchange,
        ctx: &ExecutionContext,
        obj: &RawDeviceBuffer,
    ) -> Result<(), crate::gpu::Error> {
        let [runs, count] = self.unpack.storages();
        crate::gpu::lease(ctx, &[runs, count, self.recv.storage()], &[obj])?;
        // SAFETY: each row reads its own words of the receive buffer and writes its own
        // foreign pixels of the window, both laid out by the halo; the caller keeps the window.
        unsafe { ex.copy_listed(ctx.get_cuda_stream(), &self.unpack) }?;
        Ok(())
    }
}

/// Backs a region's halo row lists on any thread, ahead of [`LiveHaloBufs::push`]: rows are
/// listed in push order, so backing each batch's rows here before pushing them backs every
/// row listed.
pub struct HaloBacker {
    lists: [Backer; 2],
    /// Pack and unpack rows backed so far.
    next: [usize; 2],
}

impl HaloBacker {
    /// Backs what pushing `rows` after the rows backed so far needs.
    pub fn back(&mut self, rows: &HaloRows) -> Result<(), GrowError> {
        let sz = size_of::<Run>();
        let end = [
            self.next[0] + rows.pack.len(),
            self.next[1] + rows.unpack.len(),
        ];
        for ((list, next), end) in self.lists.iter().zip(self.next).zip(end) {
            list.back(next * sz..end * sz)?;
        }
        self.next = end;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::sim;
    use crate::dist::partition::tests::addr;
    use crate::dist::{DEFAULT_TILE, NLEAF, Scheme};

    fn runs() -> Vec<Partition> {
        [
            (sim::raster(12, 20, 8e-8), 32, Scheme::Strips),
            (sim::fermat(300, 8e-8), 30, Scheme::Sectors { centre: None }),
        ]
        .into_iter()
        .flat_map(|(pos, frame, scheme)| {
            let leaves = Partition::from_addr(&addr(pos, frame), DEFAULT_TILE, scheme, NLEAF);
            let leaves = leaves.unwrap();
            [2, 4].map(|n| leaves.merge(n).unwrap())
        })
        .collect()
    }

    fn may(run: &Partition) -> impl Iterator<Item = &[u32]> {
        run.regions().iter().map(|g| g.halo.as_slice())
    }

    /// Tiles read in steps, in any order, own ones among them and some twice, put each pixel
    /// at the buffer word reading them all at once does; with every tile read, each word of a
    /// buffer is exactly one pixel's.
    #[test]
    fn reading_in_steps_lays_pixels_where_reading_at_once_does() {
        let pixels = |rows: Vec<HaloRow>| {
            let mut p: Vec<[u64; 4]> = rows
                .into_iter()
                .flat_map(|w| {
                    let at = [w.m, w.y, w.x].map(u64::from);
                    (0..w.len as u64).map(move |i| [w.word + i, at[0], at[1], at[2] + i])
                })
                .collect();
            p.sort_unstable();
            p
        };
        for run in runs() {
            let n = run.nregion();
            let mut once = Halo::new(&run, may(&run), 2);
            let mut want = vec![HaloRows::default(); n];
            for (r, g) in run.regions().iter().enumerate() {
                once.read(r, g.halo.iter().map(|&t| t as usize), &mut want);
            }
            let mut steps = Halo::new(&run, may(&run), 2);
            let mut got = vec![HaloRows::default(); n];
            let count = |rows: &[HaloRows]| rows.iter().map(|w| w.pack.len()).sum::<usize>();
            for (r, g) in run.regions().iter().enumerate().rev() {
                let tiles = g.halo.iter().chain(&g.own).rev().map(|&t| t as usize);
                let tiles: Vec<usize> = tiles.collect();
                for step in tiles.chunks(3) {
                    steps.read(r, step.iter().copied(), &mut got);
                    let before = count(&got);
                    steps.read(r, step.iter().copied(), &mut got);
                    assert_eq!(count(&got), before, "a tile read again adds rows");
                }
            }
            for (r, (got, want)) in got.into_iter().zip(want).enumerate() {
                let [send, recv] = steps.words(r);
                for (got, want, words) in
                    [(got.pack, want.pack, send), (got.unpack, want.unpack, recv)]
                {
                    let (got, want) = (pixels(got), pixels(want));
                    assert!(got == want, "region {r}");
                    assert!(got.iter().map(|p| p[0]).eq(0..words), "region {r}");
                }
            }
        }
    }

    /// Read one tile per call, every tile of every region's halo adds rows up to exactly
    /// [`Halo::max_rows`]; read all at once, fewer.
    #[test]
    fn a_tile_per_read_reaches_max_rows() {
        let count = |w: &HaloRows| [w.pack.len() as u64, w.unpack.len() as u64];
        let mut fewer = false;
        for run in runs() {
            let n = run.nregion();
            let (mut apart, mut once) =
                (Halo::new(&run, may(&run), 2), Halo::new(&run, may(&run), 2));
            let mut by_tile = vec![HaloRows::default(); n];
            let mut at_once = vec![HaloRows::default(); n];
            for (r, g) in run.regions().iter().enumerate() {
                for &t in &g.halo {
                    apart.read(r, [t as usize], &mut by_tile);
                }
                once.read(r, g.halo.iter().map(|&t| t as usize), &mut at_once);
            }
            for r in 0..n {
                assert_eq!(count(&by_tile[r]), apart.max_rows(r), "region {r}");
                let [got, max] = [count(&at_once[r]), once.max_rows(r)];
                assert!(got[0] <= max[0] && got[1] <= max[1], "region {r}");
                fewer |= got != max;
            }
        }
        assert!(fewer, "no read merged tiles into one row");
    }

    /// Rows backed by a [`HaloBacker`] batch by batch push, across several chunks of the row
    /// lists (7.2 MB each); a row no backer reached is refused.
    #[test]
    fn a_backer_backs_each_batch_before_its_push() {
        let ctx = Context::new(0).unwrap();
        let s = ctx.new_stream().unwrap();
        let win = Window {
            at: [32, 16],
            shape: [64, 48],
        };
        let obj = DeviceBuffer::<u64>::zeroed(&s, win.len(1)).unwrap();
        let total = 300_000;
        let words = [total as u64; 2];
        let mut bufs = LiveHaloBufs::new(&s, words, words, win, obj.cu_deviceptr(), 4096).unwrap();
        let row = |i: usize| HaloRow {
            m: 0,
            y: (32 + i % 64) as u32,
            x: (16 + i / 64 % 48) as u32,
            len: 1,
            word: i as u64,
        };
        let batch = |r: std::ops::Range<usize>| {
            let rows: Vec<HaloRow> = r.map(row).collect();
            HaloRows {
                pack: rows.clone(),
                unpack: rows,
            }
        };
        let err = bufs.push(&batch(0..1)).unwrap_err();
        assert!(
            matches!(err, DistError::Grow(GrowError::Unbacked { .. })),
            "{err}"
        );
        let mut backer = bufs.backer();
        let (mut n, mut sizes) = (0, [1, 7, 5000, 33, 90_000, 2, 100_000, 999].iter().cycle());
        while n < total {
            let rows = batch(n..(n + sizes.next().unwrap()).min(total));
            backer.back(&rows).unwrap();
            bufs.push(&rows).unwrap();
            n += rows.pack.len();
            assert_eq!(bufs.rows(), [n; 2]);
        }
    }

    /// A foreign tile outside a region's halo has no room in its buffers.
    #[test]
    #[should_panic(expected = "may not read tile")]
    fn reading_outside_the_halo_panics() {
        let run = &runs()[1];
        let mut halo = Halo::new(run, may(run), 1);
        let g = &run.regions()[0];
        let far = (0..run.grid().len() as u32)
            .find(|t| !g.own.contains(t) && !g.halo.contains(t))
            .unwrap();
        halo.read(
            0,
            [far as usize],
            &mut vec![HaloRows::default(); run.nregion()],
        );
    }

    #[test]
    fn peers_read_a_different_source_at_each_step() {
        for n in [1, 2, 4] {
            for r in 0..n {
                let mut all: Vec<usize> = peers(r, n).collect();
                all.sort();
                assert_eq!(all, (0..n).filter(|&o| o != r).collect::<Vec<_>>());
            }
            for step in 0..n - 1 {
                let mut src: Vec<usize> = (0..n).map(|r| peers(r, n).nth(step).unwrap()).collect();
                src.sort();
                assert_eq!(src, (0..n).collect::<Vec<_>>(), "n {n} step {step}");
            }
        }
    }
}
