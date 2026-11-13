//! The halo of a spatial split (DESIGN B4, invariant 5): each region's GPU holds a window of
//! the object, its own tiles and the halo tiles it reads, and after each object update copies
//! those halo rows from the owners, which pack them into a send buffer, pulled by copy engine
//! into its receive buffer and unpacked into its window.

use std::sync::Arc;

use crate::base::View;
use crate::gpu::{C32, ViewAddr};
use cuda_core::{Context, DeviceBuffer, Stream};

use crate::dist::exchange::{Pull, Run, Runs};
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

/// `len` object pixels of mode `m` from `(y, x)` on.
#[derive(Clone)]
struct HaloRow {
    m: usize,
    y: usize,
    x: usize,
    len: usize,
}

/// The object rows region `recv` reads from region `owner`'s tiles. `owner` packs them from
/// word `send_at` of its send buffer on; `recv` pulls them to word `recv_at` of its receive
/// buffer and unpacks them into its window.
struct HaloSeg {
    recv: usize,
    owner: usize,
    rows: Vec<HaloRow>,
    words: u64,
    send_at: u64,
    recv_at: u64,
}

/// The halo rows every region of a partition reads from the others, fixed at setup.
pub(crate) struct Halo {
    /// By receiver, then owner.
    segs: Vec<HaloSeg>,
}

impl Halo {
    /// The halo of `run` whose region `r` reads the halo tiles `reads[r]` (ascending): each
    /// pixel row of consecutive read tiles with one owner, every object mode; receive buffers
    /// hold the owners in order, send buffers the receivers.
    pub(crate) fn new<'a>(
        run: &Partition,
        reads: impl IntoIterator<Item = &'a [u32]>,
        nobj: usize,
    ) -> Self {
        let grid = run.grid();
        let [nyo, nxo] = grid.obj();
        let tile = grid.tile();
        let owner = run.owner();
        let n = run.regions().len();
        let mut send_at = vec![0; n];
        let mut segs = Vec::new();
        for (r, read) in reads.into_iter().enumerate() {
            let mut rows = vec![Vec::new(); n];
            for m in 0..nobj {
                let mut i = 0;
                while i < read.len() {
                    let t0 = read[i] as usize;
                    let [ty, tx0] = grid.coords(t0);
                    let o = owner[t0] as usize;
                    let mut tx1 = tx0;
                    while let Some(&t) = read.get(i + 1) {
                        let t = t as usize;
                        if grid.coords(t) != [ty, tx1 + 1] || owner[t] as usize != o {
                            break;
                        }
                        tx1 += 1;
                        i += 1;
                    }
                    i += 1;
                    let (x, x1) = (tx0 * tile, ((tx1 + 1) * tile).min(nxo));
                    for y in ty * tile..((ty + 1) * tile).min(nyo) {
                        rows[o].push(HaloRow {
                            m,
                            y,
                            x,
                            len: x1 - x,
                        });
                    }
                }
            }
            let mut recv_at = 0;
            for (o, rows) in rows.into_iter().enumerate() {
                let words: u64 = rows.iter().map(|w| w.len as u64).sum();
                if words == 0 {
                    continue;
                }
                segs.push(HaloSeg {
                    recv: r,
                    owner: o,
                    rows,
                    words,
                    send_at: send_at[o],
                    recv_at,
                });
                send_at[o] += words;
                recv_at += words;
            }
        }
        Self { segs }
    }

    /// Words of region `r`'s halo send and receive buffers.
    pub(crate) fn words(&self, r: usize) -> [u64; 2] {
        let sum = |f: &dyn Fn(&HaloSeg) -> bool| -> u64 {
            self.segs.iter().filter(|s| f(s)).map(|s| s.words).sum()
        };
        [sum(&|s| s.owner == r), sum(&|s| s.recv == r)]
    }

    /// `[byte offset in window win, buffer word, words]` of every row of the segments `pick`
    /// selects, laid out in the buffer from word `at(segment)` on.
    fn rows(
        &self,
        win: Window,
        pick: impl Fn(&HaloSeg) -> bool,
        at: fn(&HaloSeg) -> u64,
    ) -> impl Iterator<Item = [u64; 3]> {
        self.segs
            .iter()
            .filter(move |s| pick(s))
            .flat_map(move |s| {
                let mut w = at(s);
                s.rows.iter().map(move |row| {
                    let (off, len) = (win.index(row.m, row.y, row.x), row.len as u64);
                    w += len;
                    [(off * size_of::<C32>()) as u64, w - len, len]
                })
            })
    }

    /// Runs copying region `r`'s object rows that others read, from its window `win` at
    /// `obj`, into its send buffer.
    pub(crate) fn pack(&self, r: usize, win: Window, obj: u64, send: u64) -> Vec<Run> {
        self.rows(win, |s| s.owner == r, |s| s.send_at)
            .map(|[off, w, len]| Run {
                dst: send + w * 8,
                src: obj + off,
                len,
            })
            .collect()
    }

    /// Runs copying region `r`'s receive buffer into the halo rows of its window `win` at
    /// `obj`.
    pub(crate) fn unpack(&self, r: usize, win: Window, obj: u64, recv: u64) -> Vec<Run> {
        self.rows(win, |s| s.recv == r, |s| s.recv_at)
            .map(|[off, w, len]| Run {
                dst: obj + off,
                src: recv + w * 8,
                len,
            })
            .collect()
    }

    /// Copy-engine pulls into region `r`'s receive buffer at `recv` from the owners' send
    /// buffers, `send(o)` giving owner `o`'s context and send buffer, in [`peers`] order over
    /// `n` regions.
    pub(crate) fn pulls(
        &self,
        r: usize,
        n: usize,
        recv: u64,
        send: impl Fn(usize) -> (Arc<Context>, u64),
    ) -> Vec<Pull> {
        peers(r, n)
            .flat_map(|o| {
                self.segs
                    .iter()
                    .filter(move |s| s.recv == r && s.owner == o)
            })
            .map(|s| {
                let (from, at) = send(s.owner);
                Pull {
                    dst: recv + s.recv_at * 8,
                    src: at + s.send_at * 8,
                    bytes: (s.words * 8) as usize,
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
    /// Region `r`'s buffers and runs of `halo`, for its window `win` at `obj`.
    pub(crate) fn new(
        stream: &Stream,
        halo: &Halo,
        r: usize,
        win: Window,
        obj: u64,
    ) -> Result<Self, DistError> {
        let [send, recv] = halo
            .words(r)
            .map(|w| DeviceBuffer::<u64>::zeroed(stream, (w as usize).max(1)));
        let (send, recv) = (send?, recv?);
        Ok(Self {
            pack: Runs::new(stream, &halo.pack(r, win, obj, send.cu_deviceptr()))?,
            unpack: Runs::new(stream, &halo.unpack(r, win, obj, recv.cu_deviceptr()))?,
            send,
            recv,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
