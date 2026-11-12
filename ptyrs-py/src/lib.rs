//! Python module `ptyrs`: the engine's scan, CPU f64 and GPU reconstructions, forward model and
//! LLK, numpy in and out. For validation against pty-chi, PyNX and PtyPy only; nothing on the
//! reconstruction path goes through it.
//!
//! Arrays are row-major as in the engine: counts `[nframe, ny, nx]` in detector order,
//! object `[nmode, nyo, nxo]` and probe `[nmode, ny, nx]` complex128 in PyNX convention,
//! positions as the raw sample translations in metres (CXI `translation`).

mod cpu;
mod gpu;

use ::ptyrs::base::{C64, CoreError, Counts, Geometry, Positions, Stack};
use ::ptyrs::engine::{EngineError, Llk, LlkRecord};
use numpy::{Element, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Engine errors as Python exceptions: bad input is a `ValueError`, a GPU failure a
/// `RuntimeError`.
struct Error(EngineError);

impl From<EngineError> for Error {
    fn from(e: EngineError) -> Self {
        Self(e)
    }
}

impl From<CoreError> for Error {
    fn from(e: CoreError) -> Self {
        Self(e.into())
    }
}

impl From<Error> for PyErr {
    fn from(Error(e): Error) -> Self {
        match e {
            EngineError::Core(_) | EngineError::Invalid(_) => PyValueError::new_err(e.to_string()),
            _ => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

/// Keyword arguments over an engine parameter struct: each keyword replaces the engine
/// default of the field of the same name, and an unknown keyword is an error.
struct Kwargs<'py>(Option<Bound<'py, PyDict>>);

impl<'py> Kwargs<'py> {
    fn new(kw: Option<&Bound<'py, PyDict>>) -> PyResult<Self> {
        Ok(Self(kw.map(|d| d.copy()).transpose()?))
    }

    fn set<T>(&self, key: &str, field: &mut T) -> PyResult<()>
    where
        T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
    {
        if let Some(d) = &self.0
            && let Some(v) = d.get_item(key)?
        {
            *field = v.extract()?;
            d.del_item(key)?;
        }
        Ok(())
    }

    fn finish(self) -> PyResult<()> {
        match self.0 {
            Some(d) if !d.is_empty() => Err(PyTypeError::new_err(format!(
                "unknown parameters {}",
                d.keys()
            ))),
            _ => Ok(()),
        }
    }
}

/// `$ty::default()` with the listed fields taken from the keywords `$kw`.
macro_rules! params {
    ($ty:ty, $kw:expr, $($field:ident),+ $(,)?) => {{
        let mut p = <$ty>::default();
        let k = $crate::Kwargs::new($kw)?;
        $(k.set(stringify!($field), &mut p.$field)?;)+
        k.finish()?;
        p
    }};
}
use params;

fn ap_params(kw: Option<&Bound<'_, PyDict>>) -> PyResult<::ptyrs::engine::ApParams> {
    Ok(params!(
        ::ptyrs::engine::ApParams,
        kw,
        obj_inertia,
        probe_inertia,
        calc_llk,
        zero_phase_ramp
    ))
}

fn dm_params(kw: Option<&Bound<'_, PyDict>>) -> PyResult<::ptyrs::engine::DmParams> {
    Ok(params!(
        ::ptyrs::engine::DmParams,
        kw,
        obj_inertia,
        probe_inertia,
        alpha,
        loop_obj_probe,
        calc_llk,
        zero_phase_ramp
    ))
}

fn ml_params(kw: Option<&Bound<'_, PyDict>>) -> PyResult<::ptyrs::engine::MlParams> {
    Ok(params!(
        ::ptyrs::engine::MlParams,
        kw,
        update_object,
        update_probe,
        calc_llk,
        zero_phase_ramp
    ))
}

fn rpie_params(kw: Option<&Bound<'_, PyDict>>) -> PyResult<::ptyrs::engine::RpieParams> {
    Ok(params!(
        ::ptyrs::engine::RpieParams,
        kw,
        obj_step,
        probe_step,
        obj_alpha,
        probe_alpha,
        batch_size,
        calc_llk
    ))
}

fn stack_in(a: PyReadonlyArray3<'_, C64>) -> Result<Stack<C64>, Error> {
    let a = a.as_array();
    let (n, ny, nx) = a.dim();
    Ok(Stack::from_vec([n, ny, nx], a.iter().copied().collect())?)
}

fn stack_out<'py, T: Element>(py: Python<'py>, s: Stack<T>) -> PyResult<Bound<'py, PyArray3<T>>> {
    let dims = s.dims();
    PyArray1::from_vec(py, s.into_vec()).reshape(dims)
}

/// The engine's [`Llk`] as a dict of its fields.
fn llk_dict<'py>(py: Python<'py>, l: &Llk) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("poisson", l.poisson)?;
    d.set_item("gaussian", l.gaussian)?;
    d.set_item("euclidian", l.euclidian)?;
    d.set_item("nb_photons_calc", l.nb_photons_calc)?;
    Ok(d)
}

/// An LLK history as columns: `cycle`, `algorithm` (lower-case names) and the [`Llk`] fields.
fn history<'py>(py: Python<'py>, h: &[LlkRecord]) -> PyResult<Bound<'py, PyDict>> {
    let col = |f: fn(&Llk) -> f64| PyArray1::from_iter(py, h.iter().map(|r| f(&r.llk)));
    let d = PyDict::new(py);
    d.set_item("cycle", PyArray1::from_iter(py, h.iter().map(|r| r.cycle)))?;
    let names = h
        .iter()
        .map(|r| format!("{:?}", r.algorithm).to_lowercase());
    d.set_item("algorithm", PyList::new(py, names)?)?;
    d.set_item("poisson", col(|l| l.poisson))?;
    d.set_item("gaussian", col(|l| l.gaussian))?;
    d.set_item("euclidian", col(|l| l.euclidian))?;
    d.set_item("nb_photons_calc", col(|l| l.nb_photons_calc))?;
    Ok(d)
}

/// A far-field scan: counts `[nframe, ny, nx]` (uint16 with 65535 masked, or float64 integers
/// with negative values masked), positions `y`, `x` in metres, and the geometry in metres.
#[pyclass(frozen)]
struct Scan(::ptyrs::base::Scan);

#[pymethods]
impl Scan {
    #[new]
    fn new(
        counts: &Bound<'_, PyAny>,
        y: PyReadonlyArray1<'_, f64>,
        x: PyReadonlyArray1<'_, f64>,
        wavelength: f64,
        distance: f64,
        detector_pixel: f64,
    ) -> PyResult<Self> {
        let counts = if let Ok(c) = counts.extract::<PyReadonlyArray3<'_, u16>>() {
            let c = c.as_array();
            let (n, ny, nx) = c.dim();
            Counts::from_raw([n, ny, nx], c.iter().copied().collect()).map_err(Error::from)?
        } else if let Ok(c) = counts.extract::<PyReadonlyArray3<'_, f64>>() {
            let c = c.as_array();
            let (n, ny, nx) = c.dim();
            let values: Vec<f64> = c.iter().copied().collect();
            Counts::from_f64([n, ny, nx], &values, None).map_err(Error::from)?
        } else {
            return Err(PyTypeError::new_err(
                "counts must be a 3-D uint16 or float64 array",
            ));
        };
        let geometry = Geometry::new(wavelength, distance, detector_pixel, counts.frame_shape())
            .map_err(Error::from)?;
        let positions =
            Positions::new(y.as_array().to_vec(), x.as_array().to_vec()).map_err(Error::from)?;
        Ok(Self(
            ::ptyrs::base::Scan::new(geometry, positions, counts).map_err(Error::from)?,
        ))
    }
}

#[pymodule]
fn ptyrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scan>()?;
    m.add_class::<cpu::Ptycho>()?;
    m.add_class::<cpu::Lsqml>()?;
    m.add_class::<cpu::Bh>()?;
    m.add_class::<gpu::GpuPtycho>()?;
    Ok(())
}
