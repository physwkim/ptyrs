//! The CPU f64 engine: [`ptyrs::engine::Ptycho`] (AP, DM, RAAR, ML, rPIE) and the forward model
//! and LLK of given arrays over its scan, under far-field or near-field optics, and LSQML and
//! BH epochs over the views of a scan through far-field, near-field or multislice optics.

use numpy::{
    PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
};
use ptyrs::base::{AddrTable, C64, Geometry, Object, Positions, Probe};
use ptyrs::engine::{Features, Observed, ProbeVariation, obj_probe_to_psi};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{Error, Scan, history, llk_dict, objects_in, objects_out, stack_in, stack_out};

/// The features of `position`'s offsets alone, as LSQML and BH refine.
fn position_only(position: Option<PyReadonlyArray2<'_, f64>>) -> PyResult<Features> {
    Ok(Features {
        position: position.map(crate::position_in).transpose()?,
        ..Features::default()
    })
}

/// An error unless `features` hold one offset per frame of `nframe`, if any.
fn offsets_fit(features: &Features, nframe: usize) -> PyResult<()> {
    match &features.position {
        Some(d) if d.len() != nframe => Err(PyValueError::new_err(format!(
            "{} position offsets for {nframe} frames",
            d.len()
        ))),
        _ => Ok(()),
    }
}

/// An epoch's new object slices and probe and its steps.
type Epoch<'py> = (
    Bound<'py, PyArray4<C64>>,
    Bound<'py, PyArray3<C64>>,
    Bound<'py, PyList>,
);

/// One free-space propagation (`ptyrs::engine::Wavefield`): `[ny, nx]` planes (both even),
/// `wavelength`, source-plane `pixel` `[py, px]` and signed `distance` in metres.
#[pyclass(frozen)]
pub(crate) struct Wavefield(ptyrs::engine::Wavefield);

#[pymethods]
impl Wavefield {
    #[new]
    fn new(shape: [usize; 2], wavelength: f64, pixel: [f64; 2], distance: f64) -> PyResult<Self> {
        let w = ptyrs::engine::Wavefield::new(shape, wavelength, pixel, distance);
        Ok(Self(w.map_err(Error::from)?))
    }
}

/// The exit model an [`ptyrs::engine::Optics`] was built on; `Optics.with_slices` and
/// [`Optics::engine`] rebuild from it, as the engine consumes the optics it slices or runs.
#[derive(Clone, Copy)]
enum Exit {
    FarField(Geometry),
    FreeSpace(ptyrs::engine::Wavefield),
}

impl Exit {
    fn optics(self) -> ptyrs::engine::Optics {
        match self {
            Self::FarField(g) => ptyrs::engine::Optics::far_field(&g),
            Self::FreeSpace(w) => ptyrs::engine::Optics::free_space(&w),
        }
    }
}

/// The engine's exit-wave model from object to detector (`ptyrs::engine::Optics`): PyNX's
/// far field (`far_field`), pty-chi's planar near field (`free_space`), and either through
/// several object slices (`with_slices`).
#[pyclass(frozen)]
pub(crate) struct Optics {
    pub(crate) inner: ptyrs::engine::Optics,
    exit: Exit,
    /// The slice gaps `inner` was built with.
    gaps: Vec<ptyrs::engine::Wavefield>,
}

impl Optics {
    fn new(exit: Exit) -> Self {
        Self {
            inner: exit.optics(),
            exit,
            gaps: Vec::new(),
        }
    }

    /// A model equal to `inner`, for an engine state that takes its optics by value.
    fn engine(&self) -> Result<ptyrs::engine::Optics, Error> {
        Ok(self.exit.optics().with_slices(&self.gaps)?)
    }
}

#[pymethods]
impl Optics {
    /// PyNX's far field of `frame` `[ny, nx]` detector frames, as a `Scan`'s geometry.
    #[staticmethod]
    fn far_field(
        wavelength: f64,
        distance: f64,
        detector_pixel: f64,
        frame: [usize; 2],
    ) -> PyResult<Self> {
        let g = Geometry::new(wavelength, distance, detector_pixel, frame).map_err(Error::from)?;
        Ok(Self::new(Exit::FarField(g)))
    }

    /// pty-chi's planar model of one free-space propagation from the object plane.
    #[staticmethod]
    fn free_space(w: &Wavefield) -> Self {
        Self::new(Exit::FreeSpace(w.0))
    }

    /// This exit model through `len(gaps) + 1` object slices, gap `s` from slice `s` to
    /// `s + 1` (pty-chi `slice_spacings_m[s]` at the object pixel), in place of any slices
    /// this one has.
    fn with_slices(&self, gaps: Vec<Bound<'_, Wavefield>>) -> PyResult<Self> {
        let gaps: Vec<_> = gaps.iter().map(|w| w.get().0).collect();
        let inner = self.exit.optics().with_slices(&gaps);
        Ok(Self {
            inner: inner.map_err(Error::from)?,
            exit: self.exit,
            gaps,
        })
    }

    /// The object slices the wave crosses.
    #[getter]
    fn nslice(&self) -> usize {
        self.inner.nslice()
    }
}

/// A scan as the engine's epochs read it: counts `[nframe, ny, nx]` as `Scan` takes them,
/// `positions` `[nframe, 2]` `[y, x]` already preprocessed (as `Ptycho.views` has them) in the
/// units of `pixel` `[py, px]`, the views of `nprobe` probe and `nobj` object modes over an
/// `object_shape` `[nyo, nxo]` object, and the `optics` to the counts' plane.
#[pyclass(frozen)]
pub(crate) struct Views {
    addr: AddrTable,
    obs: Observed,
    optics: Py<Optics>,
}

impl Views {
    fn optics(&self) -> &ptyrs::engine::Optics {
        &self.optics.get().inner
    }

    /// `object` `[nslice, nobj, nyo, nxo]` and `probe` `[nprobe, ny, nx]` as engine arrays,
    /// of the shapes these views and their optics read.
    fn modes(
        &self,
        object: PyReadonlyArray4<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<(Vec<Object>, Probe)> {
        let a = &self.addr;
        let ([nyo, nxo], [ny, nx]) = (a.obj_shape(), a.frame_shape());
        let want = (
            [self.optics().nslice(), a.nobj(), nyo, nxo],
            [a.nprobe(), ny, nx],
        );
        let (o, p) = (object.as_array(), probe.as_array());
        if (o.shape(), p.shape()) != (&want.0[..], &want.1[..]) {
            return Err(PyValueError::new_err(format!(
                "object {:?} and probe {:?} differ from the views' {:?} and {:?}",
                o.shape(),
                p.shape(),
                want.0,
                want.1
            )));
        }
        Ok((objects_in(object)?, Probe::new(stack_in(probe)?)))
    }
}

#[pymethods]
impl Views {
    #[new]
    fn new(
        counts: &Bound<'_, PyAny>,
        positions: PyReadonlyArray2<'_, f64>,
        pixel: [f64; 2],
        object_shape: [usize; 2],
        nprobe: usize,
        nobj: usize,
        optics: Bound<'_, Optics>,
    ) -> PyResult<Self> {
        let counts = crate::counts_in(counts)?;
        let [nframe, ny, nx] = counts.dims();
        let plane = optics.get().inner.exit().shape();
        if [ny, nx] != plane {
            return Err(PyValueError::new_err(format!(
                "{:?} frames on the optics' {plane:?} plane",
                [ny, nx]
            )));
        }
        let positions = positions.as_array();
        if positions.ncols() != 2 {
            return Err(PyValueError::new_err(format!(
                "positions must be [nframe, 2], got {:?}",
                positions.shape()
            )));
        }
        let [y, x] = [0, 1].map(|axis| positions.column(axis).to_vec());
        let positions = Positions::new(y, x).map_err(Error::from)?;
        if positions.len() != nframe {
            return Err(PyValueError::new_err(format!(
                "{} positions for {nframe} frames",
                positions.len()
            )));
        }
        let addr = AddrTable::new(&positions, pixel, [ny, nx], object_shape, nprobe, nobj);
        Ok(Self {
            addr: addr.map_err(Error::from)?,
            obs: Observed::from_counts(&counts),
            optics: optics.unbind(),
        })
    }
}

/// The waves `E(P·O·q)` of every view of `p`'s scan at `object` and `probe` under `optics`,
/// frame major and mode minor, in the FFT-shifted layout of the engine's observed intensities
/// (the far field's zero frequency at `[0, 0]`, PyNX's `psi` layout); `p`'s features are not
/// modelled.
fn detector_waves(
    p: &ptyrs::engine::Ptycho,
    optics: &ptyrs::engine::Optics,
    object: &Object,
    probe: &Probe,
) -> Vec<C64> {
    let views = p.addr().views();
    let mut psi = vec![C64::default(); views.len() * probe.plane_len()];
    obj_probe_to_psi(views, object, probe, optics.quad(), &mut psi);
    optics.exit().forward(&mut psi);
    psi
}

/// The CPU f64 reconstruction state of a scan from a starting object and probe, under
/// `optics` in place of the far field of the scan's geometry if given
/// (`ptyrs::engine::Ptycho::with_optics`: one object slice, a unitary propagator). Its scan
/// also carries the forward model of arrays of the same shapes, and its `views` the LSQML and
/// BH epochs.
#[pyclass]
pub(crate) struct Ptycho {
    inner: ptyrs::engine::Ptycho,
    /// `inner`'s views and optics.
    views: Py<Views>,
    /// The scan `inner` was built on, for the engine's LLK of other arrays.
    scan: ptyrs::base::Scan,
}

impl Ptycho {
    /// `object` and `probe` as engine arrays, of the shapes this state was built with.
    fn modes(
        &self,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<(Object, Probe)> {
        let (object, probe) = (Object::new(stack_in(object)?), Probe::new(stack_in(probe)?));
        let want = (self.inner.object().dims(), self.inner.probe().dims());
        if (object.dims(), probe.dims()) != want {
            return Err(PyValueError::new_err(format!(
                "object {:?} and probe {:?} differ from this state's {:?} and {:?}",
                object.dims(),
                probe.dims(),
                want.0,
                want.1
            )));
        }
        Ok((object, probe))
    }
}

#[pymethods]
impl Ptycho {
    #[new]
    #[pyo3(signature = (scan, object, probe, optics = None))]
    fn new(
        py: Python<'_>,
        scan: &Scan,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
        optics: Option<Bound<'_, Optics>>,
    ) -> PyResult<Self> {
        let optics = match optics {
            Some(o) => o.unbind(),
            None => Py::new(py, Optics::new(Exit::FarField(scan.0.geometry)))?,
        };
        let (object, probe) = (Object::new(stack_in(object)?), Probe::new(stack_in(probe)?));
        let engine = optics.get().engine()?;
        let inner = ptyrs::engine::Ptycho::with_optics(&scan.0, engine, object, probe);
        let inner = inner.map_err(Error::from)?;
        let views = Views {
            addr: inner.addr().clone(),
            obs: inner.observed().clone(),
            optics,
        };
        Ok(Self {
            views: Py::new(py, views)?,
            inner,
            scan: scan.0.clone(),
        })
    }

    /// The views of the scan through its optics, which the LSQML and BH epochs take.
    #[getter]
    fn views(&self, py: Python<'_>) -> Py<Views> {
        self.views.clone_ref(py)
    }

    /// `ncycle` AP cycles; keywords are `ApParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn ap(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        self.inner.ap(&crate::ap_params(params)?, ncycle);
        Ok(())
    }

    /// `ncycle` DM cycles; keywords are `DmParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn dm(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        self.inner.dm(&crate::dm_params(params)?, ncycle);
        Ok(())
    }

    /// `ncycle` RAAR cycles; keywords are `RaarParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn raar(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        self.inner.raar(&crate::raar_params(params)?, ncycle);
        Ok(())
    }

    /// `ncycle` ML cycles; keywords are `MlParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn ml(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        self.inner.ml(&crate::ml_params(params)?, ncycle);
        Ok(())
    }

    /// `nepoch` rPIE epochs; keywords are `RpieParams` fields.
    #[pyo3(signature = (nepoch, **params))]
    fn rpie(&mut self, nepoch: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Ok(self
            .inner
            .rpie(&crate::rpie_params(params)?, nepoch)
            .map_err(Error::from)?)
    }

    /// Models and refines these features from here on, in place of the current ones:
    /// `scale` `[nframe]`, `background` `[ny, nx]` in the counts' detector order and
    /// `position` `[nframe, 2]` offsets `[dy, dx]` in object pixels from each view's corner;
    /// each left out is not modelled. An error leaves the current ones.
    #[pyo3(signature = (*, scale = None, background = None, position = None))]
    fn set_features(
        &mut self,
        scale: Option<PyReadonlyArray1<'_, f64>>,
        background: Option<PyReadonlyArray2<'_, f64>>,
        position: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<()> {
        let frame = self.inner.addr().frame_shape();
        let features = crate::features_in(scale, background, position, frame)?;
        Ok(self.inner.set_features(features).map_err(Error::from)?)
    }

    /// The features modelled as `set_features` takes them, `None` for each not modelled.
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let frame = self.inner.addr().frame_shape();
        crate::features_out(py, self.inner.features().clone(), frame)
    }

    /// How the `position` feature's offsets enter the exit waves, in pty-chi's names:
    /// `"patch"` moves the probe by them on a window of whole object pixels, `"full_object"`
    /// samples the object at each view's corner plus offset by Fourier interpolation.
    /// `Sampling::auto` (pty-chi's `AUTO`) of the optics, object and frame this state was
    /// built with, unless set.
    #[getter]
    fn sampling(&self) -> &'static str {
        crate::sampling_name(self.inner.sampling())
    }

    /// Enters the `position` offsets as `sampling`, `"patch"` or `"full_object"`, from here
    /// on.
    fn set_sampling(&mut self, sampling: &str) -> PyResult<()> {
        self.inner.set_sampling(crate::sampling(sampling)?);
        Ok(())
    }

    #[getter]
    fn object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<C64>>> {
        stack_out(py, self.inner.object().0.clone())
    }

    #[getter]
    fn probe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<C64>>> {
        stack_out(py, self.inner.probe().0.clone())
    }

    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        history(py, self.inner.history())
    }

    #[getter]
    fn cycle(&self) -> usize {
        self.inner.cycle()
    }

    /// The waves at the counts' plane of every view at `object` and `probe` through this
    /// state's optics, `[nframe, nmode, ny, nx]` in the FFT-shifted layout (the far field's
    /// zero frequency at `[0, 0]`), without the features.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Bound<'py, PyArray4<C64>>> {
        let (object, probe) = self.modes(object, probe)?;
        let psi = detector_waves(&self.inner, self.views.get().optics(), &object, &probe);
        let addr = self.inner.addr();
        let [ny, nx] = addr.frame_shape();
        PyArray1::from_vec(py, psi).reshape([addr.nframe(), addr.nmode(), ny, nx])
    }

    /// Normalised PyNX LLK of `object` and `probe` over the scan, modelling this state's
    /// features as it samples them: the engine's LLK of a state of these arrays, features,
    /// sampling and optics.
    fn llk<'py>(
        &self,
        py: Python<'py>,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (object, probe) = self.modes(object, probe)?;
        let optics = self.views.get().optics.get().engine()?;
        let at = ptyrs::engine::Ptycho::with_optics(&self.scan, optics, object, probe);
        let mut at = at.map_err(Error::from)?;
        at.set_features(self.inner.features().clone())
            .map_err(Error::from)?;
        at.set_sampling(self.inner.sampling());
        llk_dict(py, &at.llk())
    }
}

/// `eigenmodes` `[k, ny, nx]` (k ≥ 1), `weights` `[nframe, k]` and `intensity` `[nframe]` as
/// the engine's [`ProbeVariation`], `None` without eigenmodes and weights.
fn variation(
    eigenmodes: Option<PyReadonlyArray3<'_, C64>>,
    weights: Option<PyReadonlyArray2<'_, f64>>,
    intensity: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Option<ProbeVariation>> {
    let (eigenmodes, weights) = match (eigenmodes, weights) {
        (Some(e), Some(w)) => (stack_in(e)?, w),
        (None, None) if intensity.is_none() => return Ok(None),
        (None, None) => return Err(PyValueError::new_err("intensity without eigenmodes")),
        _ => {
            return Err(PyValueError::new_err("eigenmodes and weights go together"));
        }
    };
    let k = eigenmodes.len();
    let w = weights.as_array();
    if k == 0 || w.ncols() != k {
        return Err(PyValueError::new_err(format!(
            "weights must be [nframe, k] for k ≥ 1 eigenmodes, got {:?} for {k}",
            w.shape()
        )));
    }
    let var = ProbeVariation::new(eigenmodes, w.iter().copied().collect());
    let Some(c) = intensity else {
        return Ok(Some(var));
    };
    let c = c.as_array().to_vec();
    if c.len() != w.nrows() {
        return Err(PyValueError::new_err(format!(
            "{} main-mode weights for {} weight rows",
            c.len(),
            w.nrows()
        )));
    }
    Ok(Some(var.with_intensity(c)))
}

/// An error unless `var` holds weights for every frame of `nframe` and eigenmodes of `plane`.
fn variation_fits(var: &ProbeVariation, nframe: usize, plane: [usize; 2]) -> PyResult<()> {
    let e = var.eigenmodes();
    let rows = var.weights().len() / e.len();
    if rows != nframe || e.plane_shape() != plane {
        return Err(PyValueError::new_err(format!(
            "eigenmodes {:?} and {rows} weight rows for {nframe} frames of {plane:?} probe planes",
            e.dims()
        )));
    }
    Ok(())
}

/// LSQML across epochs (`ptyrs::engine::Lsqml`); `cost` is `"amplitude"` or `"poisson"`.
/// With `position`, `[nframe, 2]` offsets as `Ptycho.set_features` takes them, every epoch
/// models and refines them. With `eigenmodes` `U_i` `[k, ny, nx]` and `weights` `w_fi`
/// `[nframe, k]`, it models and refines OPR of probe mode 0, frame `f` seeing
/// `c_f·P_0 + Σ_i w_fi·U_i`, and with `intensity` `[nframe]` the main-mode weights `c_f` too
/// (1 without); `Ptycho.llk` does not model it.
#[pyclass]
pub(crate) struct Lsqml {
    inner: ptyrs::engine::Lsqml,
    var: Option<ProbeVariation>,
}

#[pymethods]
impl Lsqml {
    #[new]
    #[pyo3(signature = (batch_size, cost, seed, *, position = None, eigenmodes = None,
                        weights = None, intensity = None))]
    fn new(
        batch_size: usize,
        cost: &str,
        seed: u64,
        position: Option<PyReadonlyArray2<'_, f64>>,
        eigenmodes: Option<PyReadonlyArray3<'_, C64>>,
        weights: Option<PyReadonlyArray2<'_, f64>>,
        intensity: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let lsqml = ptyrs::engine::Lsqml::new(ptyrs::engine::LsqmlParams {
            batch_size,
            cost: crate::cost(cost)?,
            seed,
        })
        .map_err(Error::from)?;
        let features = position_only(position)?;
        Ok(Self {
            inner: lsqml.with_features(features).map_err(Error::from)?,
            var: variation(eigenmodes, weights, intensity)?,
        })
    }

    /// The offsets as the last epoch left them, `None` without a position.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let d = self.inner.features().position.clone();
        d.map(|d| crate::position_out(py, d)).transpose()
    }

    /// The eigenmodes as the last epoch left them, `None` without OPR.
    #[getter]
    fn eigenmodes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray3<C64>>>> {
        let e = self.var.as_ref().map(|v| v.eigenmodes().clone());
        e.map(|e| stack_out(py, e)).transpose()
    }

    /// The weights as the last epoch left them, `None` without OPR.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let w = self.var.as_ref().map(|v| {
            let (w, k) = (v.weights(), v.eigenmodes().len());
            PyArray1::from_slice(py, w).reshape([w.len() / k, k])
        });
        w.transpose()
    }

    /// The main-mode weights as the last epoch left them, `None` unless refined.
    #[getter]
    fn intensity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        let c = self.var.as_ref().and_then(|v| v.intensity());
        c.map(|c| PyArray1::from_slice(py, c))
    }

    /// One epoch over `views` from `object` `[nslice, nobj, nyo, nxo]` and `probe`; returns
    /// the new object and probe and each batch's step sizes.
    fn epoch<'py>(
        &mut self,
        py: Python<'py>,
        views: &Views,
        object: PyReadonlyArray4<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Epoch<'py>> {
        let (mut objects, mut probe) = views.modes(object, probe)?;
        let (nframe, plane) = (views.addr.nframe(), probe.plane_shape());
        offsets_fit(self.inner.features(), nframe)?;
        if let Some(v) = &self.var {
            variation_fits(v, nframe, plane)?;
        }
        let mut none = ProbeVariation::none(plane);
        let steps = self.inner.epoch(
            &views.addr,
            &views.obs,
            views.optics(),
            &mut objects,
            &mut probe,
            self.var.as_mut().unwrap_or(&mut none),
        );
        let steps = crate::lsqml_steps(py, &steps)?;
        Ok((objects_out(py, objects)?, stack_out(py, probe.0)?, steps))
    }
}

/// BH across epochs (`ptyrs::engine::Bh`). With `position`, offsets as `Lsqml` takes them,
/// every epoch models and refines them, their gradient weighted by `position_rho`².
#[pyclass]
pub(crate) struct Bh(ptyrs::engine::Bh);

#[pymethods]
impl Bh {
    #[new]
    #[pyo3(signature = (batch_size, probe_rho, seed, *, position = None,
                        position_rho = crate::POSITION_RHO))]
    fn new(
        batch_size: usize,
        probe_rho: f64,
        seed: u64,
        position: Option<PyReadonlyArray2<'_, f64>>,
        position_rho: f64,
    ) -> PyResult<Self> {
        let bh = ptyrs::engine::Bh::new(ptyrs::engine::BhParams {
            batch_size,
            probe_rho,
            position_rho,
            seed,
        })
        .map_err(Error::from)?;
        let features = position_only(position)?;
        Ok(Self(bh.with_features(features).map_err(Error::from)?))
    }

    /// The offsets as the last epoch left them, `None` without a position.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let d = self.0.features().position.clone();
        d.map(|d| crate::position_out(py, d)).transpose()
    }

    /// One epoch over `views` from `object` `[nslice, nobj, nyo, nxo]` and `probe`; returns
    /// the new object and probe and each batch's step.
    fn epoch<'py>(
        &mut self,
        py: Python<'py>,
        views: &Views,
        object: PyReadonlyArray4<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Epoch<'py>> {
        let (mut objects, mut probe) = views.modes(object, probe)?;
        offsets_fit(self.0.features(), views.addr.nframe())?;
        let (addr, obs) = (&views.addr, &views.obs);
        let steps = self
            .0
            .epoch(addr, obs, views.optics(), &mut objects, &mut probe);
        let steps = crate::bh_steps(py, &steps)?;
        Ok((objects_out(py, objects)?, stack_out(py, probe.0)?, steps))
    }
}

#[cfg(test)]
mod tests {
    use ptyrs::base::{Geometry, required_obj_shape};
    use ptyrs::engine::llk;

    use super::*;

    /// `forward`'s waves of a featureless state's own arrays, under the far field and a
    /// free-space propagation alike, are the engine's model: their LLK is the engine's.
    #[test]
    fn detector_waves_of_the_state_are_the_engine_model() {
        let g = Geometry::new(1.5e-10, 1.0, 75e-6, [16, 16]).unwrap();
        let pixel = g.pixel_size_object();
        let positions = ptyrs::base::sim::fermat(5, 3.0 * pixel[1]);
        let shape = required_obj_shape(&positions.pynx_preprocess(pixel), pixel, g.frame, 0);
        let object = ptyrs::base::sim::object(shape);
        let probe = ptyrs::base::sim::probe(g.frame, 2).unwrap();
        let scan =
            ptyrs::engine::sim::simulate(g, positions, &object, &probe, 2e3, None, 3).unwrap();
        let w = ptyrs::engine::Wavefield::new(g.frame, g.wavelength, pixel, 1e-4).unwrap();
        for exit in [Exit::FarField(g), Exit::FreeSpace(w)] {
            let (o, pr) = (object.clone(), probe.clone());
            let p = ptyrs::engine::Ptycho::with_optics(&scan, exit.optics(), o, pr).unwrap();
            let psi = detector_waves(&p, &exit.optics(), p.object(), p.probe());
            let (addr, obs) = (p.addr(), p.observed());
            let got = llk(obs, 0..addr.nframe(), addr.nmode(), &psi).normalised(obs.nb_obs());
            assert_eq!(got, p.llk());
        }
    }
}
