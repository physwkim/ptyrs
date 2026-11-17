//! `ptyrs::base::frc`: the FRC of two complex images over a region, a raster's lattice
//! frequencies, the half-bit thresholds and the sub-pixel shift between two images. Arrays are
//! anything numpy converts; frequencies are in FFT order.

use ::ptyrs::base::C64;
use ::ptyrs::base::frc::{self, Frc, LATTICE_RADIUS, Registration};
use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods,
    PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::Error;

type Mask<'py> = PyArrayLike2<'py, bool, AllowTypeChange>;
type Pair<'py> = PyArrayLike1<'py, f64, AllowTypeChange>;

pub(crate) fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute, m)?)?;
    m.add_function(wrap_pyfunction!(lattice, m)?)?;
    m.add_function(wrap_pyfunction!(threshold, m)?)?;
    m.add_function(wrap_pyfunction!(shift, m)?)?;
    Ok(())
}

/// `a`, 2-D and cast to complex128 (pyo3 has no `Vec<C64>`, which `PyArrayLike` needs), as
/// pixels row-major and its `[ny, nx]`.
fn image(a: &Bound<'_, PyAny>) -> PyResult<(Vec<C64>, [usize; 2])> {
    let py = a.py();
    let a = py
        .import("numpy")?
        .call_method1("asarray", (a, numpy::dtype::<C64>(py)))?;
    let a = a.extract::<PyReadonlyArray2<'_, C64>>()?;
    let a = a.as_array();
    Ok((a.iter().copied().collect(), a.dim().into()))
}

/// `[a, b]` of the same shape, as row-major pixels and that shape.
fn images(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<(Vec<C64>, Vec<C64>, [usize; 2])> {
    let ((a, sa), (b, sb)) = (image(a)?, image(b)?);
    if sa != sb {
        return Err(PyValueError::new_err(format!(
            "images of {sa:?} and {sb:?} pixels"
        )));
    }
    Ok((a, b, sa))
}

/// Two values, as `[dy, dx]` or `[sy, sx]`.
fn pair(name: &str, v: &Pair<'_>) -> PyResult<[f64; 2]> {
    let v = v.as_array();
    match v.as_slice() {
        Some(&[y, x]) => Ok([y, x]),
        _ => Err(PyValueError::new_err(format!(
            "{name} must be [y, x], got {v}"
        ))),
    }
}

/// `exclude` as flags of `shape`, row-major.
fn mask(exclude: Option<&Mask<'_>>, shape: [usize; 2]) -> PyResult<Option<Vec<bool>>> {
    exclude
        .map(|e| {
            let e = e.as_array();
            if <[usize; 2]>::from(e.dim()) != shape {
                return Err(PyValueError::new_err(format!(
                    "exclude of {:?} for a region of {shape:?}",
                    e.shape()
                )));
            }
            Ok(e.iter().copied().collect())
        })
        .transpose()
}

/// FRC of `b` aligned to `a` (2-D, one shape) over `region` `((y0, y1), (x0, x1))`.
///
/// Without `shift` the registration is blind: the regions' cross-correlation. With `shift`
/// (`[dy, dx]`, `a(y) ~ b(y - shift)` expected) each region first loses its own phase ramp and
/// phase; without `radius` the shift is `shift`, with it the shift within `radius` pixels of
/// `shift` that maximizes the FRC below the crossing. `exclude` (bool, the region's shape)
/// leaves frequencies out of every ring and its pixel count.
///
/// Returns a dict: `frc`, `half_bit` (NaN for an empty ring) and `npix` per ring, `crossing`
/// (the first ring from 1 with pixels under its threshold, or None), `side` (`min(ny, nx)` of
/// the region), `shift`, `ramp` (cycles per pixel), `phase`, and `trace`, the search's passes
/// as `(crossing, [dy, dx])`.
#[pyfunction]
#[pyo3(name = "frc", signature = (a, b, region, shift=None, radius=None, exclude=None))]
fn compute<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    region: PyArrayLike2<'py, usize, AllowTypeChange>,
    shift: Option<Pair<'py>>,
    radius: Option<f64>,
    exclude: Option<Mask<'py>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (a, b, shape) = images(a, b)?;
    let region = region.as_array();
    let [[y0, y1], [x0, x1]] = match region.as_slice() {
        Some(&[y0, y1, x0, x1]) if region.dim() == (2, 2) => [[y0, y1], [x0, x1]],
        _ => {
            return Err(PyValueError::new_err(format!(
                "region must be ((y0, y1), (x0, x1)), got {region}"
            )));
        }
    };
    let registration = match (shift, radius) {
        (Some(s), radius) => Registration::Near {
            shift: pair("shift", &s)?,
            radius,
        },
        (None, None) => Registration::Blind,
        (None, Some(_)) => return Err(PyValueError::new_err("a radius needs a shift")),
    };
    let side = [y1.saturating_sub(y0), x1.saturating_sub(x0)];
    let exclude = mask(exclude.as_ref(), side)?;
    let f = Frc::compute_with(
        &a,
        &b,
        shape,
        &[y0..y1, x0..x1],
        registration,
        exclude.as_deref(),
    )
    .map_err(Error::from)?;
    let d = PyDict::new(py);
    let per_ring = |v: fn(&frc::Ring) -> f64| PyArray1::from_iter(py, f.rings.iter().map(v));
    d.set_item("frc", per_ring(|r| r.frc))?;
    d.set_item("half_bit", per_ring(|r| r.half_bit))?;
    d.set_item(
        "npix",
        PyArray1::from_iter(py, f.rings.iter().map(|r| r.npix)),
    )?;
    d.set_item("crossing", f.crossing())?;
    d.set_item("side", side[0].min(side[1]))?;
    d.set_item("shift", PyArray1::from_slice(py, &f.shift))?;
    d.set_item("ramp", PyArray1::from_slice(py, &f.ramp))?;
    d.set_item("phase", f.phase)?;
    let trace = f.passes.iter().map(|p| (p.crossing, p.shift));
    d.set_item("trace", PyList::new(py, trace)?)?;
    Ok(d)
}

/// The frequencies (bool, `[ny, nx]`) within `radius` bins of the reciprocal lattice, origin
/// aside, of a checkerboard half of a raster with steps `step` `[sy, sx]` pixels, where a
/// raster's reconstruction carries its periodic artifact.
#[pyfunction]
#[pyo3(name = "frc_lattice", signature = (ny, nx, step, radius=LATTICE_RADIUS))]
fn lattice<'py>(
    py: Python<'py>,
    ny: usize,
    nx: usize,
    step: Pair<'py>,
    radius: f64,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let m = frc::lattice([ny, nx], pair("step", &step)?, radius).map_err(Error::from)?;
    PyArray1::from_vec(py, m).reshape([ny, nx])
}

/// The half-bit threshold of each ring of an `[ny, nx]` region without the `exclude`
/// frequencies, as `frc` counts them.
#[pyfunction]
#[pyo3(name = "frc_threshold", signature = (ny, nx, exclude=None))]
fn threshold<'py>(
    py: Python<'py>,
    ny: usize,
    nx: usize,
    exclude: Option<Mask<'py>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let exclude = mask(exclude.as_ref(), [ny, nx])?;
    let t = frc::threshold([ny, nx], exclude.as_deref()).map_err(Error::from)?;
    Ok(PyArray1::from_vec(py, t))
}

/// The shift `s` `[dy, dx]`, to 0.01 pixel, with `a(y) ~ b(y - s)` for two 2-D images of one
/// shape: the peak of their periodic cross-correlation.
#[pyfunction]
#[pyo3(name = "frc_shift")]
fn shift<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (a, b, shape) = images(a, b)?;
    let s = frc::shift(&a, &b, shape).map_err(Error::from)?;
    Ok(PyArray1::from_slice(py, &s))
}
