//! The two payload layouts areaDetector `NDPluginCodec` publishes in an
//! NTNDArray `value` (`codec.name` "lz4" and "bslz4").
//!
//! `lz4` is one LZ4 block of the whole raw image with no header; the decoder
//! takes the size from `uncompressedSize`. `bslz4` is what the bitshuffle
//! library's `bshuf_compress_lz4` emits with the default block size: per
//! block `[u32 BE compressed length][LZ4 block of the bit-transposed block]`,
//! the last block cut down to a multiple of 8 elements, then the final
//! `n % 8` elements raw. There is no 12-byte `[u64 total][u32 block]` header;
//! that one belongs to the HDF5 chunk and the Eiger stream, not to the
//! NDArray. Layout taken from epics-rs `ad-plugins-rs` `codec.rs`, the port
//! of ADCore `NDPluginCodec.cpp` (ADCore C is not checked out on this host).

use std::fmt;

/// `BSHUF_TARGET_BLOCK_SIZE_B`.
const BSHUF_TARGET_BLOCK_BYTES: usize = 8192;
/// `BSHUF_BLOCKED_MULT`.
const BSHUF_BLOCKED_MULT: usize = 8;
/// `BSHUF_MIN_RECOMMEND_BLOCK`.
const BSHUF_MIN_RECOMMEND_BLOCK: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// Uncompressed `ushortValue`.
    None,
    Lz4,
    Bslz4,
}

impl Codec {
    pub fn name(self) -> &'static str {
        match self {
            Codec::None => "",
            Codec::Lz4 => "lz4",
            Codec::Bslz4 => "bslz4",
        }
    }

    pub fn from_name(name: &[u8]) -> Option<Self> {
        match name {
            b"" => Some(Codec::None),
            b"lz4" => Some(Codec::Lz4),
            b"bslz4" => Some(Codec::Bslz4),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// The LZ4 stream is malformed or does not decode to exactly the image.
    Lz4,
    /// A bslz4 block prefix points past the payload, or bytes are left over.
    Framing,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::Lz4 => f.write_str("lz4 block does not decode to the image size"),
            CodecError::Framing => f.write_str("bslz4 block framing does not match the image"),
        }
    }
}

impl std::error::Error for CodecError {}

/// `bshuf_default_block_size`: elements per bitshuffle block.
pub fn bshuf_block_elems(elem_size: usize) -> usize {
    let bs = BSHUF_TARGET_BLOCK_BYTES / elem_size.max(1);
    (bs / BSHUF_BLOCKED_MULT * BSHUF_BLOCKED_MULT).max(BSHUF_MIN_RECOMMEND_BLOCK)
}

pub fn lz4_compress(raw: &[u8]) -> Vec<u8> {
    lz4_flex::block::compress(raw)
}

/// Decode an `lz4` payload into `dst`, which must be exactly the image.
pub fn lz4_decompress_into(src: &[u8], dst: &mut [u8]) -> Result<(), CodecError> {
    match lz4_flex::block::decompress_into(src, dst) {
        Ok(n) if n == dst.len() => Ok(()),
        _ => Err(CodecError::Lz4),
    }
}

/// 8x8 bit-matrix transpose of a quadword, bitshuffle `TRANS_BIT_8X8`. Its
/// own inverse.
#[inline(always)]
fn trans_bit_8x8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^ t ^ (t << 28)
}

/// Bit-transpose `n` elements (a multiple of 8) of `elem_size` bytes: row
/// `8 * byte + bit` of the output holds that bit of every element, element
/// `i` at bit `i % 8` of byte `i / 8`.
fn bitshuffle_block(src: &[u8], dst: &mut [u8], n: usize, elem_size: usize) {
    let row = n / 8;
    for m in 0..row {
        for j in 0..elem_size {
            let mut x = 0u64;
            for e in 0..8 {
                x |= (src[(8 * m + e) * elem_size + j] as u64) << (8 * e);
            }
            let y = trans_bit_8x8(x);
            for k in 0..8 {
                dst[(8 * j + k) * row + m] = (y >> (8 * k)) as u8;
            }
        }
    }
}

/// Inverse of [`bitshuffle_block`] for u16, written straight as `u16`.
/// A zero quadword transposes to zero, so the high-byte planes of sparse
/// photon-counting data cost a load and a compare.
fn bitunshuffle_block_u16(src: &[u8], dst: &mut [u16]) {
    let n = dst.len();
    let row = n / 8;
    debug_assert_eq!(src.len(), 2 * n);
    let plane = |j: usize, m: usize| -> u64 {
        let mut y = 0u64;
        for k in 0..8 {
            y |= (src[(8 * j + k) * row + m] as u64) << (8 * k);
        }
        y
    };
    for m in 0..row {
        let lo = plane(0, m);
        let hi = plane(1, m);
        let lo = if lo == 0 { 0 } else { trans_bit_8x8(lo) };
        let hi = if hi == 0 { 0 } else { trans_bit_8x8(hi) };
        let out = &mut dst[8 * m..8 * m + 8];
        for (e, o) in out.iter_mut().enumerate() {
            *o = ((lo >> (8 * e)) as u8 as u16) | (((hi >> (8 * e)) as u8 as u16) << 8);
        }
    }
}

/// `bshuf_compress_lz4` of a u16 image with the default block size.
pub fn bslz4_compress_u16(raw: &[u16]) -> Vec<u8> {
    let bytes: &[u8] = bytemuck_u16(raw);
    let n = raw.len();
    let block = bshuf_block_elems(2);
    let blocked = n - n % BSHUF_BLOCKED_MULT;
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut shuffled = vec![0u8; block * 2];
    let mut start = 0;
    while start < blocked {
        let len = block.min(blocked - start);
        let sh = &mut shuffled[..len * 2];
        bitshuffle_block(&bytes[start * 2..(start + len) * 2], sh, len, 2);
        let c = lz4_flex::block::compress(sh);
        out.extend_from_slice(&(c.len() as u32).to_be_bytes());
        out.extend_from_slice(&c);
        start += len;
    }
    out.extend_from_slice(&bytes[blocked * 2..]);
    out
}

/// Scratch a worker keeps for [`bslz4_decompress_into_u16`].
pub struct Bslz4Scratch {
    shuffled: Vec<u8>,
}

impl Default for Bslz4Scratch {
    fn default() -> Self {
        Self {
            shuffled: vec![0u8; bshuf_block_elems(2) * 2],
        }
    }
}

/// Decode a `bslz4` u16 payload into `dst`, which must be exactly the image.
pub fn bslz4_decompress_into_u16(
    src: &[u8],
    dst: &mut [u16],
    scratch: &mut Bslz4Scratch,
) -> Result<(), CodecError> {
    let n = dst.len();
    bslz4_decompress_range_u16(src, dst, scratch, 0..n)
}

/// As [`bslz4_decompress_into_u16`], but only the blocks that overlap
/// element range `want` are decompressed; the rest of `dst` is left as it
/// was. Blocks are independent, so a crop of a few hundred rows skips most
/// of the frame. The framing of every block is still checked.
pub fn bslz4_decompress_range_u16(
    src: &[u8],
    dst: &mut [u16],
    scratch: &mut Bslz4Scratch,
    want: std::ops::Range<usize>,
) -> Result<(), CodecError> {
    let n = dst.len();
    let block = bshuf_block_elems(2);
    let blocked = n - n % BSHUF_BLOCKED_MULT;
    let mut pos = 0usize;
    let mut start = 0usize;
    while start < blocked {
        let len = block.min(blocked - start);
        let clen = src
            .get(pos..pos + 4)
            .map(|b| u32::from_be_bytes(b.try_into().unwrap()) as usize)
            .ok_or(CodecError::Framing)?;
        let frame = src
            .get(pos + 4..pos + 4 + clen)
            .ok_or(CodecError::Framing)?;
        if start < want.end && want.start < start + len {
            let sh = &mut scratch.shuffled[..len * 2];
            lz4_decompress_into(frame, sh)?;
            bitunshuffle_block_u16(sh, &mut dst[start..start + len]);
        }
        pos += 4 + clen;
        start += len;
    }
    let tail = &src[pos.min(src.len())..];
    if tail.len() != (n - blocked) * 2 {
        return Err(CodecError::Framing);
    }
    for (d, b) in dst[blocked..].iter_mut().zip(tail.as_chunks::<2>().0) {
        *d = u16::from_le_bytes(*b);
    }
    Ok(())
}

/// The little-endian bytes of a u16 slice (this crate runs on x86_64 only).
pub fn bytemuck_u16(v: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding and any byte pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Mutable little-endian bytes of a u16 slice.
pub fn bytemuck_u16_mut(v: &mut [u16]) -> &mut [u8] {
    // SAFETY: u16 has no padding and every byte pattern is a valid u16.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<u16> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if i % 5 == 0 {
                    (s % 70_000).min(65_535) as u16
                } else {
                    (s % 3) as u16
                }
            })
            .collect()
    }

    #[test]
    fn bitshuffle_round_trips_one_block() {
        let raw = sample(4096);
        let mut sh = vec![0u8; 8192];
        bitshuffle_block(bytemuck_u16(&raw), &mut sh, 4096, 2);
        let mut back = vec![0u16; 4096];
        bitunshuffle_block_u16(&sh, &mut back);
        assert_eq!(raw, back);
    }

    #[test]
    fn bslz4_round_trips_partial_last_block_and_raw_tail() {
        // 1028 x 1062 leaves a 2200-element block; +3 adds a 3-element raw tail.
        for n in [1028 * 1062, 1028 * 1062 + 3, 7, 4096] {
            let raw = sample(n);
            let c = bslz4_compress_u16(&raw);
            let mut out = vec![0u16; n];
            bslz4_decompress_into_u16(&c, &mut out, &mut Bslz4Scratch::default()).unwrap();
            assert_eq!(raw, out, "n = {n}");
        }
    }

    #[test]
    fn bslz4_range_decode_matches_the_full_decode_on_the_range() {
        let raw = sample(1028 * 1062);
        let c = bslz4_compress_u16(&raw);
        let want = 403 * 1028..659 * 1028;
        let mut out = vec![0xFFFFu16; raw.len()];
        bslz4_decompress_range_u16(&c, &mut out, &mut Bslz4Scratch::default(), want.clone())
            .unwrap();
        assert_eq!(&out[want.clone()], &raw[want]);
        // Block 0 lies outside the range and is never written.
        assert!(out[..4096].iter().all(|&v| v == 0xFFFF));
    }

    #[test]
    fn bslz4_rejects_a_truncated_payload() {
        let raw = sample(10_000);
        let c = bslz4_compress_u16(&raw);
        let mut out = vec![0u16; raw.len()];
        let r =
            bslz4_decompress_into_u16(&c[..c.len() - 1], &mut out, &mut Bslz4Scratch::default());
        assert!(r.is_err());
    }

    #[test]
    fn lz4_rejects_a_wrong_image_size() {
        let raw = sample(1000);
        let c = lz4_compress(bytemuck_u16(&raw));
        let mut out = vec![0u8; 1999];
        assert!(lz4_decompress_into(&c, &mut out).is_err());
        let mut out = vec![0u8; 2000];
        lz4_decompress_into(&c, &mut out).unwrap();
        assert_eq!(out, bytemuck_u16(&raw));
    }
}
