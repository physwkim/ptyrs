use std::path::Path;

use ptyrs_core::{Counts, Geometry, MASKED, Positions, Scan};
use rust_hdf5::{DatatypeMessage, H5Dataset, H5File, H5Group, Hdf5Error};

use crate::IoError;

const TRANSLATION: &str = "/entry_1/sample_1/geometry_1/translation";
const DETECTOR_DATA: &str = "/entry_1/instrument_1/detector_1/data";
const DATA: &str = "entry_1/data_1/data";
const ENERGY: &str = "entry_1/instrument_1/source_1/energy";
const DISTANCE: &str = "entry_1/instrument_1/detector_1/distance";
const X_PIXEL: &str = "entry_1/instrument_1/detector_1/x_pixel_size";
const Y_PIXEL: &str = "entry_1/instrument_1/detector_1/y_pixel_size";
const MASK: &str = "entry_1/instrument_1/detector_1/mask";

/// Writes `scan` in the layout of PyNX `ptycho.save_ptycho_data_cxi`.
///
/// Counts go to `entry_1/instrument_1/detector_1/data` as `u16 [nframe, ny, nx]` with
/// `(1, ny, nx)` chunks, shuffle and deflate 4; `entry_1/data_1/data` links to it. Positions
/// go to `entry_1/sample_1/geometry_1/translation` as `f32 [3, nframe]` (x, y, 0), so a scan
/// read back has its positions rounded to f32 exactly as PyNX stores them. The energy is
/// in joules, distance and pixel sizes in metres.
///
/// A masked pixel is written as 0 in `data` and as 1 in the 2-D `detector_1/mask`, which
/// is written only when some pixel is masked. That mask applies to every frame, so a pixel
/// masked in some frames but not all is refused with [`IoError::Layout`].
pub fn write_cxi(path: impl AsRef<Path>, scan: &Scan) -> Result<(), IoError> {
    let counts = &scan.counts;
    let [nframe, ny, nx] = counts.dims();
    let plane = ny * nx;
    let mask = counts.detector_mask();
    if let Some(i) = counts
        .as_slice()
        .iter()
        .enumerate()
        .position(|(i, &v)| v == MASKED && !mask[i % plane])
    {
        return Err(IoError::Layout(format!(
            "frame {} pixel ({}, {}) is masked in some frames only; the PyNX CXI mask covers \
             every frame",
            i / plane,
            i % plane / nx,
            i % nx
        )));
    }
    let data: Vec<u16> = counts
        .as_slice()
        .iter()
        .map(|&v| if v == MASKED { 0 } else { v })
        .collect();
    let g = &scan.geometry;

    let f = H5File::create(path)?;
    f.set_attr_string("default", "entry_1")?;
    f.new_dataset::<i64>()
        .scalar()
        .create("cxi_version")?
        .write_raw(&[140i64])?;

    let entry = nx_group(&f.root_group(), "entry_1", "NXentry")?;
    entry.set_attr_string("default", "data_1")?;
    write_str(
        &entry,
        "program_name",
        concat!("ptyrs ", env!("CARGO_PKG_VERSION")),
    )?;

    let sample = nx_group(&entry, "sample_1", "NXsample")?;
    let geometry_1 = sample.create_group("geometry_1")?;
    let mut xyz = vec![0f32; 3 * nframe];
    for (i, (x, y)) in scan.positions.x.iter().zip(&scan.positions.y).enumerate() {
        xyz[i] = *x as f32;
        xyz[nframe + i] = *y as f32;
    }
    geometry_1
        .new_dataset::<f32>()
        .shape([3, nframe])
        .create("translation")?
        .write_raw(&xyz)?;

    let data_1 = nx_group(&entry, "data_1", "NXdata")?;
    data_1.set_attr_string("signal", "data")?;
    data_1.set_attr_string("interpretation", "image")?;
    data_1.create_soft_link("translation", TRANSLATION)?;

    let instrument = nx_group(&entry, "instrument_1", "NXinstrument")?;
    write_str(&instrument, "name", "ptyrs")?;
    let source = nx_group(&instrument, "source_1", "NXsource")?;
    write_f64(&source, "energy", g.energy_joule(), None)?;

    let detector = nx_group(&instrument, "detector_1", "NX_detector")?;
    detector
        .new_dataset::<u16>()
        .shape([nframe, ny, nx])
        .chunk(&[1, ny, nx])
        .shuffle_deflate(4)
        .create("data")?
        .write_raw(&data)?;
    write_f64(&detector, "distance", g.distance, Some("m"))?;
    write_f64(&detector, "x_pixel_size", g.detector_pixel, Some("m"))?;
    write_f64(&detector, "y_pixel_size", g.detector_pixel, Some("m"))?;
    if mask.iter().any(|&m| m) {
        let m: Vec<u8> = mask.iter().map(|&m| u8::from(m)).collect();
        detector
            .new_dataset::<u8>()
            .shape([ny, nx])
            .chunk(&[ny, nx])
            .shuffle_deflate(4)
            .create("mask")?
            .write_raw(&m)?;
    }
    // CXI default 'top, left' origin convention, as PyNX writes it.
    let ps = g.detector_pixel as f32;
    detector
        .new_dataset::<f32>()
        .shape([2, 3])
        .create("basis_vectors")?
        .write_raw(&[0.0, -ps, 0.0, -ps, 0.0, 0.0])?;
    detector.create_soft_link("translation", TRANSLATION)?;
    data_1.create_soft_link("data", DETECTOR_DATA)?;
    f.close()?;
    Ok(())
}

/// Reads a scan from a CXI file in the layout `pynx-ptycho-cxi` reads.
///
/// Uses `entry_1/data_1/data`, rows 0 and 1 (x, y) of `entry_1/sample_1/geometry_1/translation`,
/// `source_1/energy` (J), `detector_1/distance` and `x_pixel_size`/`y_pixel_size` (m, must be
/// equal), and `detector_1/mask` when present (non-zero = masked in every frame).
///
/// Data stored as `u16` is taken as raw counts, with 65535 ([`MASKED`]) a masked pixel. Any
/// other numeric type goes through [`Counts::from_f64`]: a negative value is masked (PyNX
/// `iobs < 0`), and a value that is non-integer or above `MAX_COUNT` is an error.
pub fn read_cxi(path: impl AsRef<Path>) -> Result<Scan, IoError> {
    let f = H5File::open(path)?;
    let energy = read_scalar(&f, ENERGY)?;
    let distance = read_scalar(&f, DISTANCE)?;
    let (px, py) = (read_scalar(&f, X_PIXEL)?, read_scalar(&f, Y_PIXEL)?);
    if px != py {
        return Err(IoError::Layout(format!(
            "x_pixel_size {px} != y_pixel_size {py}; PyNX assumes square pixels"
        )));
    }

    let data = f.dataset(DATA)?;
    let [nframe, ny, nx]: [usize; 3] = data.shape().try_into().map_err(|s: Vec<usize>| {
        IoError::Layout(format!("{DATA} has shape {s:?}, expected [nframe, ny, nx]"))
    })?;
    let geometry = Geometry::new(
        Geometry::wavelength_from_energy_joule(energy),
        distance,
        px,
        [ny, nx],
    )?;

    let translation = f.dataset(&TRANSLATION[1..])?;
    let shape = translation.shape();
    if shape.len() != 2 || shape[0] < 2 || shape[1] != nframe {
        return Err(IoError::Layout(format!(
            "{TRANSLATION} has shape {shape:?}, expected [3, {nframe}]"
        )));
    }
    let t = read_f64(&translation)?;
    let positions = Positions::new(t[nframe..2 * nframe].to_vec(), t[..nframe].to_vec())?;

    let mask = match f.dataset(MASK) {
        Ok(ds) => {
            if ds.shape() != [ny, nx] {
                return Err(IoError::Layout(format!(
                    "{MASK} has shape {:?}, expected [{ny}, {nx}]",
                    ds.shape()
                )));
            }
            Some(
                read_f64(&ds)?
                    .into_iter()
                    .map(|v| v != 0.0)
                    .collect::<Vec<bool>>(),
            )
        }
        Err(Hdf5Error::NotFound(_)) => None,
        Err(e) => return Err(e.into()),
    };

    let dims = [nframe, ny, nx];
    let counts = if is_u16(&data)? {
        let mut raw = data.read_numeric_as::<u16>()?;
        if let Some(m) = &mask {
            for (i, v) in raw.iter_mut().enumerate() {
                if m[i % (ny * nx)] {
                    *v = MASKED;
                }
            }
        }
        Counts::from_raw(dims, raw)?
    } else {
        Counts::from_f64(dims, &read_f64(&data)?, mask.as_deref())?
    };
    Ok(Scan::new(geometry, positions, counts)?)
}

fn is_u16(ds: &H5Dataset) -> Result<bool, IoError> {
    Ok(matches!(
        ds.datatype()?,
        DatatypeMessage::FixedPoint {
            size: 2,
            signed: false,
            ..
        }
    ))
}

fn nx_group(parent: &H5Group, name: &str, class: &str) -> Result<H5Group, IoError> {
    let g = parent.create_group(name)?;
    g.set_attr_string("NX_class", class)?;
    Ok(g)
}

/// Scalar NUL-terminated UTF-8 string dataset; h5py reads it back as `bytes`, as it does the
/// scalar strings PyNX writes.
fn write_str(g: &H5Group, name: &str, value: &str) -> Result<(), IoError> {
    let mut bytes = value.as_bytes().to_vec();
    bytes.push(0);
    g.new_dataset::<u8>()
        .datatype(DatatypeMessage::fixed_string_utf8(bytes.len() as u32))
        .scalar()
        .create(name)?
        .write_raw_bytes(&bytes)?;
    Ok(())
}

fn write_f64(g: &H5Group, name: &str, value: f64, units: Option<&str>) -> Result<(), IoError> {
    let ds = g.new_dataset::<f64>().scalar().create(name)?;
    ds.write_raw(&[value])?;
    if let Some(u) = units {
        ds.new_attr::<rust_hdf5::VarLenUnicode>()
            .shape(())
            .create("units")?
            .write_string(u)?;
    }
    Ok(())
}

/// Every element as f64 whatever the stored type: integers (exact up to 2^53, beyond any
/// count or flag this reads), floats, and h5py `bool` (a 1-byte enum) as 0/1.
fn read_f64(ds: &H5Dataset) -> Result<Vec<f64>, IoError> {
    Ok(match ds.datatype()? {
        DatatypeMessage::FixedPoint { signed: true, .. } => ds
            .read_numeric_as::<i64>()?
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        DatatypeMessage::FixedPoint { .. } | DatatypeMessage::BitField { .. } => ds
            .read_numeric_as::<u64>()?
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        DatatypeMessage::Enum { base, .. }
            if matches!(*base, DatatypeMessage::FixedPoint { size: 1, .. }) =>
        {
            ds.read_raw_bytes()?.into_iter().map(f64::from).collect()
        }
        _ => ds.read_numeric_as::<f64>()?,
    })
}

fn read_scalar(f: &H5File, path: &str) -> Result<f64, IoError> {
    match read_f64(&f.dataset(path)?)?[..] {
        [v] => Ok(v),
        ref v => Err(IoError::Layout(format!(
            "{path} holds {} values, expected one",
            v.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(values: Vec<u16>, nframe: usize) -> Scan {
        let g = Geometry::new(1.5e-10, 1.2, 75e-6, [4, 6]).unwrap();
        let x: Vec<f64> = (0..nframe)
            .map(|i| i as f64 * 1.1e-7 + 1e-9 / 3.0)
            .collect();
        let y: Vec<f64> = (0..nframe).map(|i| -(i as f64) * 0.7e-7).collect();
        let c = Counts::from_raw([nframe, 4, 6], values).unwrap();
        Scan::new(g, Positions::new(y, x).unwrap(), c).unwrap()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ptyrs-io-{}-{name}", std::process::id()))
    }

    #[test]
    fn round_trip_keeps_counts_mask_and_f32_positions() {
        let n = 3;
        let mut v: Vec<u16> = (0..n * 24).map(|i| (i * 977 % 65534) as u16).collect();
        for f in 0..n {
            v[f * 24 + 5] = MASKED;
        }
        let s = scan(v, n);
        let p = tmp("rt.cxi");
        write_cxi(&p, &s).unwrap();
        let r = read_cxi(&p).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(r.counts, s.counts);
        assert_eq!(r.geometry.frame, s.geometry.frame);
        assert_eq!(r.geometry.distance, s.geometry.distance);
        assert_eq!(r.geometry.detector_pixel, s.geometry.detector_pixel);
        assert!((r.geometry.wavelength / s.geometry.wavelength - 1.0).abs() < 1e-15);
        for (a, b) in r.positions.x.iter().zip(&s.positions.x) {
            assert_eq!(*a, *b as f32 as f64);
        }
        for (a, b) in r.positions.y.iter().zip(&s.positions.y) {
            assert_eq!(*a, *b as f32 as f64);
        }
    }

    #[test]
    fn refuses_mask_that_is_not_in_every_frame() {
        let mut v = vec![1u16; 2 * 24];
        v[24 + 3] = MASKED;
        let p = tmp("partial.cxi");
        let e = write_cxi(&p, &scan(v, 2)).unwrap_err();
        std::fs::remove_file(&p).ok();
        assert!(matches!(e, IoError::Layout(_)), "{e}");
    }
}
