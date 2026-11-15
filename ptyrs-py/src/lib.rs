//! Python module `ptyrs`: the engine's scan, CPU f64 and GPU reconstructions, forward model and
//! LLK, numpy in and out. For validation against pty-chi, PyNX and PtyPy only; nothing on the
//! reconstruction path goes through it.
//!
//! Arrays are row-major as in the engine: counts `[nframe, ny, nx]` in detector order,
//! object `[nmode, nyo, nxo]` (`[nslice, nmode, nyo, nxo]` in epochs over `Views` and for a
//! multislice `GpuPtycho`) and probe
//! `[nmode, ny, nx]` complex128 in PyNX convention, a `Scan`'s positions as the raw sample
//! translations in metres (CXI `translation`).

mod cpu;
mod gpu;

use ::ptyrs::base::{C64, CoreError, Counts, Geometry, Object, Positions, Stack};
use ::ptyrs::engine::{
    BhStep, Cost, EngineError, Feature, Features, Llk, LlkRecord, LsqmlStep, Sampling,
};
use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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

fn raar_params(kw: Option<&Bound<'_, PyDict>>) -> PyResult<::ptyrs::engine::RaarParams> {
    Ok(params!(
        ::ptyrs::engine::RaarParams,
        kw,
        obj_inertia,
        probe_inertia,
        beta,
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

/// `a` `[nslice, nobj, nyo, nxo]` as an object's slices.
fn objects_in(a: PyReadonlyArray4<'_, C64>) -> Result<Vec<Object>, Error> {
    let a = a.as_array();
    let (_, nobj, nyo, nxo) = a.dim();
    a.outer_iter()
        .map(|s| {
            let s = Stack::from_vec([nobj, nyo, nxo], s.iter().copied().collect())?;
            Ok(Object::new(s))
        })
        .collect()
}

/// `objects` as `[nslice, nobj, nyo, nxo]`.
fn objects_out<'py>(py: Python<'py>, objects: Vec<Object>) -> PyResult<Bound<'py, PyArray4<C64>>> {
    let [nobj, nyo, nxo] = objects[0].dims();
    let nslice = objects.len();
    let data = objects.into_iter().flat_map(|o| o.0.into_vec()).collect();
    PyArray1::from_vec(py, data).reshape([nslice, nobj, nyo, nxo])
}

/// BH's default `position_rho`, pty-chi's `BHProbePositionOptions.rho`.
const POSITION_RHO: f64 = 0.1;

/// `"amplitude"` or `"poisson"` as the engine's [`Cost`].
fn cost(name: &str) -> PyResult<Cost> {
    match name {
        "amplitude" => Ok(Cost::Amplitude),
        "poisson" => Ok(Cost::Poisson),
        _ => Err(PyValueError::new_err(format!(
            "cost must be \"amplitude\" or \"poisson\", got {name:?}"
        ))),
    }
}

/// pty-chi's `"patch"` or `"full_object"` (`ObjectPosOriginCoordsMethods`) as the engine's
/// [`Sampling`].
fn sampling(name: &str) -> PyResult<Sampling> {
    match name {
        "patch" => Ok(Sampling::Patch),
        "full_object" => Ok(Sampling::FullObject),
        _ => Err(PyValueError::new_err(format!(
            "sampling must be \"patch\" or \"full_object\", got {name:?}"
        ))),
    }
}

/// [`Sampling`] as [`sampling`] takes it.
fn sampling_name(s: Sampling) -> &'static str {
    match s {
        Sampling::Patch => "patch",
        Sampling::FullObject => "full_object",
    }
}

/// LSQML steps as dicts of `frames`, `a_far`, `a_obj` (`[nslice, frames]`) and `a_probe`.
fn lsqml_steps<'py>(py: Python<'py>, steps: &[LsqmlStep]) -> PyResult<Bound<'py, PyList>> {
    let steps = steps.iter().map(|s| {
        let d = PyDict::new(py);
        d.set_item("frames", PyArray1::from_slice(py, &s.frames))?;
        d.set_item("a_far", PyArray1::from_slice(py, &s.a_far))?;
        let a_obj = PyArray1::from_vec(py, s.a_obj.concat());
        d.set_item("a_obj", a_obj.reshape([s.a_obj.len(), s.frames.len()])?)?;
        d.set_item("a_probe", PyArray1::from_slice(py, &s.a_probe))?;
        Ok(d)
    });
    PyList::new(py, steps.collect::<PyResult<Vec<_>>>()?)
}

/// BH steps as dicts of `frames`, `beta`, `alpha`, `top` and `bottom`.
fn bh_steps<'py>(py: Python<'py>, steps: &[BhStep]) -> PyResult<Bound<'py, PyList>> {
    let steps = steps.iter().map(|s| {
        let d = PyDict::new(py);
        d.set_item("frames", PyArray1::from_slice(py, &s.frames))?;
        d.set_item("beta", s.beta)?;
        d.set_item("alpha", s.alpha)?;
        d.set_item("top", s.top)?;
        d.set_item("bottom", s.bottom)?;
        Ok(d)
    });
    PyList::new(py, steps.collect::<PyResult<Vec<_>>>()?)
}

/// Which of `names` a `hold` keyword holds: each held name is one of `names` and `given`.
fn held<const N: usize>(
    hold: &[String],
    names: [&str; N],
    given: [bool; N],
) -> PyResult<[bool; N]> {
    let mut held = [false; N];
    for h in hold {
        match names.iter().position(|n| n == h) {
            Some(i) if given[i] => held[i] = true,
            Some(_) => {
                return Err(PyValueError::new_err(format!(
                    "hold {h:?} but no {h} given"
                )));
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "hold {h:?} is not one of {names:?}"
                )));
            }
        }
    }
    Ok(held)
}

/// The names of `features`' held quantities, as `hold` takes them.
fn hold_out<'py>(
    py: Python<'py>,
    features: impl IntoIterator<Item = (&'static str, Option<bool>)>,
) -> PyResult<Bound<'py, PyTuple>> {
    let held = features
        .into_iter()
        .filter(|(_, refine)| *refine == Some(false));
    PyTuple::new(py, held.map(|(name, _)| name).collect::<Vec<_>>())
}

/// `values`, refined unless `held`.
fn feature<T>(values: T, held: bool) -> Feature<T> {
    Feature {
        values,
        refine: !held,
    }
}

/// The names of [`Features`]' quantities, as `hold` takes them.
const FEATURES: [&str; 3] = ["scale", "background", "position"];

/// [`Features`] of `set_features`' keywords on `frame` (`[ny, nx]`) detector frames: `scale`
/// `[nframe]`, `background` `[ny, nx]` in the counts' detector order and `position`
/// `[nframe, 2]` offsets `[dy, dx]` in object pixels, each held if `hold` names it; each left
/// out is not modelled. The engine checks the frame count and the values.
fn features_in(
    scale: Option<PyReadonlyArray1<'_, f64>>,
    background: Option<PyReadonlyArray2<'_, f64>>,
    position: Option<PyReadonlyArray2<'_, f64>>,
    hold: &[String],
    frame: [usize; 2],
) -> PyResult<Features> {
    let given = [scale.is_some(), background.is_some(), position.is_some()];
    let [hs, hb, hp] = held(hold, FEATURES, given)?;
    let background = background
        .map(|b| {
            let b = b.as_array();
            if b.dim() != frame.into() {
                return Err(PyValueError::new_err(format!(
                    "a background of {:?} for {frame:?} frames",
                    b.shape()
                )));
            }
            Ok(feature(b.iter().copied().collect(), hb))
        })
        .transpose()?;
    Ok(Features {
        scale: scale.map(|s| feature(s.as_array().to_vec(), hs)),
        background,
        position: position
            .map(|d| Ok::<_, PyErr>(feature(position_in(d)?, hp)))
            .transpose()?,
    })
}

/// `[nframe, 2]` offsets `[dy, dx]` as the values of the engine's [`Features::position`].
fn position_in(d: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 2]>> {
    let d = d.as_array();
    if d.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "position offsets must be [nframe, 2], got {:?}",
            d.shape()
        )));
    }
    Ok(d.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

/// The values of [`Features::position`] as [`position_in`] takes them.
fn position_out<'py>(py: Python<'py>, d: Vec<[f64; 2]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray1::from_vec(py, d.concat()).reshape([d.len(), 2])
}

/// `features` as a dict of `scale`, `background` and `position` in [`features_in`]'s shapes,
/// `None` for each not modelled, and `hold`, the names of those held.
fn features_out<'py>(
    py: Python<'py>,
    f: Features,
    frame: [usize; 2],
) -> PyResult<Bound<'py, PyDict>> {
    let refine = [
        f.scale.as_ref().map(|s| s.refine),
        f.background.as_ref().map(|b| b.refine),
        f.position.as_ref().map(|p| p.refine),
    ];
    let d = PyDict::new(py);
    d.set_item("scale", f.scale.map(|s| PyArray1::from_vec(py, s.values)))?;
    let background = f
        .background
        .map(|b| PyArray1::from_vec(py, b.values).reshape(frame));
    d.set_item("background", background.transpose()?)?;
    let position = f.position.map(|p| position_out(py, p.values));
    d.set_item("position", position.transpose()?)?;
    d.set_item("hold", hold_out(py, FEATURES.into_iter().zip(refine))?)?;
    Ok(d)
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

/// Counts `[nframe, ny, nx]` in detector order: uint16 with 65535 masked, or float64 integers
/// with negative values masked.
fn counts_in(counts: &Bound<'_, PyAny>) -> PyResult<Counts> {
    if let Ok(c) = counts.extract::<PyReadonlyArray3<'_, u16>>() {
        let c = c.as_array();
        let (n, ny, nx) = c.dim();
        Ok(Counts::from_raw([n, ny, nx], c.iter().copied().collect()).map_err(Error::from)?)
    } else if let Ok(c) = counts.extract::<PyReadonlyArray3<'_, f64>>() {
        let c = c.as_array();
        let (n, ny, nx) = c.dim();
        let values: Vec<f64> = c.iter().copied().collect();
        Ok(Counts::from_f64([n, ny, nx], &values, None).map_err(Error::from)?)
    } else {
        Err(PyTypeError::new_err(
            "counts must be a 3-D uint16 or float64 array",
        ))
    }
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
        let counts = counts_in(counts)?;
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
    m.add_class::<cpu::Wavefield>()?;
    m.add_class::<cpu::Optics>()?;
    m.add_class::<cpu::Views>()?;
    m.add_class::<cpu::Ptycho>()?;
    m.add_class::<cpu::Lsqml>()?;
    m.add_class::<cpu::Bh>()?;
    m.add_class::<gpu::GpuPtycho>()?;
    Ok(())
}
