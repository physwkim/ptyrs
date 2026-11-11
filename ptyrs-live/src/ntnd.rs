//! Allocation-free parse of the head of a raw MONITOR body: the changed
//! BitSet and the NTNDArray members the receiver needs, read straight from
//! the wire bytes. The payload is returned as a byte range so the caller can
//! slice the `Bytes` without copying. Members after `dataTimeStamp`
//! (alarm, timeStamp, dimension, attribute) are never read.

use std::ops::Range;

use epics_pva_rs::proto::ByteOrder;
use epics_pva_rs::pvdata::{FieldDesc, ScalarType};

use crate::codec::Codec;

/// NTNDArray head in pvxs `nt.cpp` order, with the BitSet index each member
/// sits at when the head is laid out as this table says.
const HEAD: [(&str, usize); 6] = [
    ("value", 1),
    ("codec", 2),
    ("compressedSize", 5),
    ("uncompressedSize", 6),
    ("uniqueId", 7),
    ("dataTimeStamp", 8),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Layout(&'static str),
    Truncated,
    /// A size or type byte this parser does not handle.
    Unsupported(&'static str),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Layout(s) => write!(f, "unexpected NTNDArray layout: {s}"),
            ParseError::Truncated => f.write_str("monitor body truncated"),
            ParseError::Unsupported(s) => write!(f, "unsupported wire encoding: {s}"),
        }
    }
}

impl std::error::Error for ParseError {}

fn members(desc: &FieldDesc) -> Option<&[(String, FieldDesc)]> {
    match desc {
        FieldDesc::Structure { fields, .. } => Some(fields),
        _ => None,
    }
}

fn is_scalar(desc: &FieldDesc, t: ScalarType) -> bool {
    matches!(desc, FieldDesc::Scalar(s) if *s == t)
}

fn elem_size(t: ScalarType) -> Option<u8> {
    Some(match t {
        ScalarType::Boolean | ScalarType::Byte | ScalarType::UByte => 1,
        ScalarType::Short | ScalarType::UShort => 2,
        ScalarType::Int | ScalarType::UInt | ScalarType::Float => 4,
        ScalarType::Long | ScalarType::ULong | ScalarType::Double => 8,
        ScalarType::String => return None,
    })
}

/// Variant indices of the `value` union, checked once per introspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameLayout {
    /// Element size per union variant; 0 for a non-numeric-array variant.
    elem: [u8; 16],
    nvariants: usize,
    pub ubyte_sel: usize,
    pub ushort_sel: usize,
}

impl FrameLayout {
    pub fn from_desc(desc: &FieldDesc) -> Result<Self, ParseError> {
        let m = members(desc).ok_or(ParseError::Layout("root is not a structure"))?;
        if m.len() < HEAD.len() {
            return Err(ParseError::Layout("fewer members than the NTNDArray head"));
        }
        for ((name, _), (want, _)) in m.iter().zip(HEAD) {
            if name != want {
                return Err(ParseError::Layout("head member order"));
            }
        }
        let FieldDesc::Union { variants, .. } = &m[0].1 else {
            return Err(ParseError::Layout("value is not a union"));
        };
        if variants.len() > 16 {
            return Err(ParseError::Layout("value union has more than 16 variants"));
        }
        let mut elem = [0u8; 16];
        let (mut ubyte_sel, mut ushort_sel) = (None, None);
        for (i, (_, d)) in variants.iter().enumerate() {
            if let FieldDesc::ScalarArray(t) = d {
                elem[i] = elem_size(*t).unwrap_or(0);
                match t {
                    ScalarType::UByte => ubyte_sel = Some(i),
                    ScalarType::UShort => ushort_sel = Some(i),
                    _ => {}
                }
            }
        }
        let codec = members(&m[1].1).ok_or(ParseError::Layout("codec is not a structure"))?;
        let codec_ok = codec.len() == 2
            && codec[0].0 == "name"
            && is_scalar(&codec[0].1, ScalarType::String)
            && codec[1].0 == "parameters"
            && matches!(codec[1].1, FieldDesc::Variant);
        let ts = members(&m[5].1).ok_or(ParseError::Layout("dataTimeStamp is not a structure"))?;
        let ts_ok = ts.len() == 3
            && is_scalar(&ts[0].1, ScalarType::Long)
            && is_scalar(&ts[1].1, ScalarType::Int)
            && is_scalar(&ts[2].1, ScalarType::Int);
        if !(codec_ok
            && ts_ok
            && is_scalar(&m[2].1, ScalarType::Long)
            && is_scalar(&m[3].1, ScalarType::Long)
            && is_scalar(&m[4].1, ScalarType::Int))
        {
            return Err(ParseError::Layout("head member types"));
        }
        Ok(Self {
            elem,
            nvariants: variants.len(),
            ubyte_sel: ubyte_sel.ok_or(ParseError::Layout("no ubyteValue"))?,
            ushort_sel: ushort_sel.ok_or(ParseError::Layout("no ushortValue"))?,
        })
    }
}

/// What one update carried. A member absent from the changed set is `None`
/// and keeps its previous value (pvAccess delta semantics).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameHead {
    /// `(union selector, element count, byte range in the body)`; a null
    /// union selector is `Some((usize::MAX, 0, empty))`.
    pub value: Option<(usize, usize, Range<usize>)>,
    pub codec: Option<Option<Codec>>,
    /// `codec.parameters` when it is an int scalar (the original ScalarType).
    pub codec_param: Option<Option<i32>>,
    pub compressed_size: Option<i64>,
    pub uncompressed_size: Option<i64>,
    pub unique_id: Option<i32>,
    pub data_time_stamp: Option<(Option<i64>, Option<i32>)>,
}

struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
    le: bool,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        let s = self
            .b
            .get(self.pos..self.pos + n)
            .ok_or(ParseError::Truncated)?;
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, ParseError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, ParseError> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn i32(&mut self) -> Result<i32, ParseError> {
        Ok(self.u32()? as i32)
    }

    fn i64(&mut self) -> Result<i64, ParseError> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.le {
            i64::from_le_bytes(b)
        } else {
            i64::from_be_bytes(b)
        })
    }

    fn f64(&mut self) -> Result<f64, ParseError> {
        Ok(f64::from_bits(self.i64()? as u64))
    }

    /// pvData size: `< 254` inline, `254` then a u32, `255` null.
    fn size(&mut self) -> Result<Option<usize>, ParseError> {
        match self.u8()? {
            0xFF => Ok(None),
            0xFE => {
                let n = self.u32()?;
                if n >= 0x7FFF_FFFF {
                    return Err(ParseError::Unsupported("64-bit size"));
                }
                Ok(Some(n as usize))
            }
            n => Ok(Some(n as usize)),
        }
    }

    /// The changed BitSet's bytes; bit `i` is `bytes[i / 8] >> (i % 8)`.
    fn bitset(&mut self) -> Result<&'a [u8], ParseError> {
        let n = self.size()?.ok_or(ParseError::Unsupported("null bitset"))?;
        self.take(n)
    }

    fn string(&mut self) -> Result<&'a [u8], ParseError> {
        let n = self.size()?.unwrap_or(0);
        self.take(n)
    }

    /// A `variant` value; returns the int when it is an int scalar.
    fn variant(&mut self) -> Result<Option<i32>, ParseError> {
        match self.u8()? {
            0xFF => Ok(None),
            0x22 => Ok(Some(self.i32()?)),
            0x20 | 0x24 => self.take(1).map(|_| None),
            0x21 | 0x25 => self.take(2).map(|_| None),
            0x26 | 0x42 => self.take(4).map(|_| None),
            0x23 | 0x27 | 0x43 => self.take(8).map(|_| None),
            0x60 => self.string().map(|_| None),
            _ => Err(ParseError::Unsupported("codec.parameters type")),
        }
    }
}

fn bit(bits: &[u8], i: usize) -> bool {
    bits.get(i / 8).is_some_and(|b| b >> (i % 8) & 1 == 1)
}

pub fn parse_frame_head(
    layout: &FrameLayout,
    body: &[u8],
    order: ByteOrder,
) -> Result<FrameHead, ParseError> {
    let mut c = Cur {
        b: body,
        pos: 0,
        le: matches!(order, ByteOrder::Little),
    };
    let bits = c.bitset()?;
    let all = bit(bits, 0);
    let has = |i: usize| all || bit(bits, i);
    let mut h = FrameHead::default();
    if has(1) {
        h.value = Some(match c.size()? {
            None => (usize::MAX, 0, c.pos..c.pos),
            Some(sel) => {
                if sel >= layout.nvariants || layout.elem[sel] == 0 {
                    return Err(ParseError::Unsupported("value union variant"));
                }
                let n = c.size()?.unwrap_or(0);
                let start = c.pos;
                c.take(n * layout.elem[sel] as usize)?;
                (sel, n, start..c.pos)
            }
        });
    }
    if has(2) || has(3) {
        h.codec = Some(Codec::from_name(c.string()?));
    }
    if has(2) || has(4) {
        h.codec_param = Some(c.variant()?);
    }
    if has(5) {
        h.compressed_size = Some(c.i64()?);
    }
    if has(6) {
        h.uncompressed_size = Some(c.i64()?);
    }
    if has(7) {
        h.unique_id = Some(c.i32()?);
    }
    let ts_all = has(8);
    if ts_all || has(9) || has(10) || has(11) {
        let s = if ts_all || has(9) {
            Some(c.i64()?)
        } else {
            None
        };
        let ns = if ts_all || has(10) {
            Some(c.i32()?)
        } else {
            None
        };
        h.data_time_stamp = Some((s, ns));
    }
    Ok(h)
}

/// The position PV: `{uniqueId int, x double, y double}`.
pub fn position_desc() -> FieldDesc {
    FieldDesc::Structure {
        struct_id: "ptyrs:position:1.0".into(),
        fields: vec![
            ("uniqueId".into(), FieldDesc::Scalar(ScalarType::Int)),
            ("x".into(), FieldDesc::Scalar(ScalarType::Double)),
            ("y".into(), FieldDesc::Scalar(ScalarType::Double)),
        ]
        .into(),
    }
}

pub fn check_position_layout(desc: &FieldDesc) -> Result<(), ParseError> {
    let m = members(desc).ok_or(ParseError::Layout("position root is not a structure"))?;
    let ok = m.len() == 3
        && m[0].0 == "uniqueId"
        && is_scalar(&m[0].1, ScalarType::Int)
        && m[1].0 == "x"
        && is_scalar(&m[1].1, ScalarType::Double)
        && m[2].0 == "y"
        && is_scalar(&m[2].1, ScalarType::Double);
    if ok {
        Ok(())
    } else {
        Err(ParseError::Layout("position members"))
    }
}

/// `(uniqueId, x, y)`, each `None` when unchanged.
pub type PositionUpdate = (Option<i32>, Option<f64>, Option<f64>);

pub fn parse_position(body: &[u8], order: ByteOrder) -> Result<PositionUpdate, ParseError> {
    let mut c = Cur {
        b: body,
        pos: 0,
        le: matches!(order, ByteOrder::Little),
    };
    let bits = c.bitset()?;
    let all = bit(bits, 0);
    let id = if all || bit(bits, 1) {
        Some(c.i32()?)
    } else {
        None
    };
    let x = if all || bit(bits, 2) {
        Some(c.f64()?)
    } else {
        None
    };
    let y = if all || bit(bits, 3) {
        Some(c.f64()?)
    } else {
        None
    };
    Ok((id, x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use epics_pva_rs::nt::nd_array::{
        NdArrayBuffer, NdCodec, NdDimension, NdTimeStamp, NtNdArray, nt_nd_array_desc,
        nt_nd_array_value,
    };
    use epics_pva_rs::proto::BitSet;
    use epics_pva_rs::pvdata::encode::encode_pv_field_with_bitset;
    use epics_pva_rs::pvdata::{PvField, PvStructure, ScalarValue, VariantValue};
    use std::sync::Arc;

    fn frame(bytes: Vec<u8>) -> NtNdArray {
        NtNdArray {
            value: NdArrayBuffer::UByte(Arc::<[u8]>::from(bytes).into()),
            codec: NdCodec {
                name: "bslz4".into(),
                parameters: Some(VariantValue::scalar(ScalarValue::Int(6))),
            },
            compressed_size: 300,
            uncompressed_size: 2 * 1028 * 1062,
            unique_id: 4242,
            data_time_stamp: NdTimeStamp {
                seconds_past_epoch: 1_700_000_000,
                nanoseconds: 123_456_789,
                user_tag: 0,
            },
            alarm: Default::default(),
            time_stamp: Default::default(),
            dimension: vec![
                NdDimension {
                    size: 1028,
                    full_size: 1028,
                    binning: 1,
                    ..Default::default()
                },
                NdDimension {
                    size: 1062,
                    full_size: 1062,
                    binning: 1,
                    ..Default::default()
                },
            ],
            attribute: vec![],
        }
    }

    fn body(value: &PvField, desc: &FieldDesc, bits: &BitSet, order: ByteOrder) -> Vec<u8> {
        let mut out = Vec::new();
        bits.write_into(order, &mut out);
        encode_pv_field_with_bitset(value, desc, bits, 0, order, &mut out);
        out
    }

    #[test]
    fn head_parses_a_library_encoded_full_update_in_both_orders() {
        let desc = nt_nd_array_desc();
        let layout = FrameLayout::from_desc(&desc).unwrap();
        let payload: Vec<u8> = (0..300u32).map(|i| (i * 7) as u8).collect();
        let value = nt_nd_array_value(&frame(payload.clone()));
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let b = body(&value, &desc, &BitSet::all_set(desc.total_bits()), order);
            let h = parse_frame_head(&layout, &b, order).unwrap();
            let (sel, n, r) = h.value.unwrap();
            assert_eq!((sel, n), (layout.ubyte_sel, 300));
            assert_eq!(&b[r], &payload[..]);
            assert_eq!(h.codec, Some(Some(Codec::Bslz4)));
            assert_eq!(h.codec_param, Some(Some(6)));
            assert_eq!(h.compressed_size, Some(300));
            assert_eq!(h.uncompressed_size, Some(2 * 1028 * 1062));
            assert_eq!(h.unique_id, Some(4242));
            assert_eq!(
                h.data_time_stamp,
                Some((Some(1_700_000_000), Some(123_456_789)))
            );
        }
    }

    #[test]
    fn head_parses_a_delta_with_only_value_and_unique_id() {
        let desc = nt_nd_array_desc();
        let layout = FrameLayout::from_desc(&desc).unwrap();
        let value = nt_nd_array_value(&frame(vec![9; 17]));
        let mut bits = BitSet::new();
        bits.set(1);
        bits.set(7);
        bits.set(10);
        let b = body(&value, &desc, &bits, ByteOrder::Little);
        let h = parse_frame_head(&layout, &b, ByteOrder::Little).unwrap();
        assert_eq!(h.value.as_ref().map(|v| v.1), Some(17));
        assert_eq!(h.codec, None);
        assert_eq!(h.uncompressed_size, None);
        assert_eq!(h.unique_id, Some(4242));
        assert_eq!(h.data_time_stamp, Some((None, Some(123_456_789))));
    }

    #[test]
    fn head_reads_an_uncompressed_ushort_frame() {
        let desc = nt_nd_array_desc();
        let layout = FrameLayout::from_desc(&desc).unwrap();
        let mut nt = frame(vec![]);
        nt.value = NdArrayBuffer::UShort(vec![1u16, 2, 3, 0xABCD].into());
        nt.codec = NdCodec::default();
        let value = nt_nd_array_value(&nt);
        let b = body(
            &value,
            &desc,
            &BitSet::all_set(desc.total_bits()),
            ByteOrder::Little,
        );
        let h = parse_frame_head(&layout, &b, ByteOrder::Little).unwrap();
        let (sel, n, r) = h.value.unwrap();
        assert_eq!((sel, n), (layout.ushort_sel, 4));
        assert_eq!(&b[r], &[1, 0, 2, 0, 3, 0, 0xCD, 0xAB]);
        assert_eq!(h.codec, Some(Some(Codec::None)));
        assert_eq!(h.codec_param, Some(None));
    }

    #[test]
    fn position_parses_a_library_encoded_update() {
        let desc = position_desc();
        check_position_layout(&desc).unwrap();
        let value = PvField::Structure(PvStructure {
            struct_id: "ptyrs:position:1.0".into(),
            fields: vec![
                ("uniqueId".into(), PvField::Scalar(ScalarValue::Int(77))),
                ("x".into(), PvField::Scalar(ScalarValue::Double(1.5e-6))),
                ("y".into(), PvField::Scalar(ScalarValue::Double(-2.0e-7))),
            ],
        });
        let b = body(
            &value,
            &desc,
            &BitSet::all_set(desc.total_bits()),
            ByteOrder::Big,
        );
        assert_eq!(
            parse_position(&b, ByteOrder::Big).unwrap(),
            (Some(77), Some(1.5e-6), Some(-2.0e-7))
        );
    }
}
