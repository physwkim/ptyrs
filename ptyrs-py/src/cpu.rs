//! The CPU f64 engine: [`ptyrs::engine::Ptycho`] (AP, DM, ML, rPIE), LSQML and BH epochs over
//! its scan, and the forward model and LLK of given arrays.

use std::slice;

use numpy::{PyArray1, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray3};
use ptyrs::base::{C64, Object, Probe};
use ptyrs::engine::{Cost, Llk, Optics, ProbeVariation, llk, obj_probe_to_psi};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{Error, Scan, history, llk_dict, stack_in, stack_out};

/// An epoch's new object and probe and its steps.
type Epoch<'py> = (
    Bound<'py, PyArray3<C64>>,
    Bound<'py, PyArray3<C64>>,
    Bound<'py, PyList>,
);

/// Far field `FT(fftshift(P·O·q))` of every view of `p`'s scan at `object` and `probe`, frame
/// major and mode minor, zero frequency at `[0, 0]` (PyNX's `psi` layout).
fn far_field(
    p: &ptyrs::engine::Ptycho,
    optics: &Optics,
    object: &Object,
    probe: &Probe,
) -> Vec<C64> {
    let views = p.addr().views();
    let mut psi = vec![C64::default(); views.len() * probe.plane_len()];
    obj_probe_to_psi(views, object, probe, optics.quad(), &mut psi);
    optics.propagator().forward(&mut psi);
    psi
}

/// [`ptyrs::engine::Ptycho::llk`] of `object` and `probe` in place of `p`'s own.
fn llk_of(p: &ptyrs::engine::Ptycho, optics: &Optics, object: &Object, probe: &Probe) -> Llk {
    let psi = far_field(p, optics, object, probe);
    let (addr, obs) = (p.addr(), p.observed());
    llk(obs, 0..addr.nframe(), addr.nmode(), &psi).normalised(obs.nb_obs())
}

/// The CPU f64 reconstruction state of a scan from a starting object and probe
/// (`ptyrs::engine::Ptycho`). Its scan also carries the LSQML and BH epochs and the forward model
/// of arrays of the same shapes.
#[pyclass]
pub(crate) struct Ptycho {
    inner: ptyrs::engine::Ptycho,
    /// [`Optics::far_field`] of the scan: the engine's own quadratic phase and FFT.
    optics: Optics,
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
    fn new(
        scan: &Scan,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Self> {
        let (object, probe) = (Object::new(stack_in(object)?), Probe::new(stack_in(probe)?));
        let inner = ptyrs::engine::Ptycho::new(&scan.0, object, probe).map_err(Error::from)?;
        Ok(Self {
            optics: Optics::far_field(&scan.0.geometry),
            inner,
        })
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

    /// Far field of every view at `object` and `probe`, `[nframe, nmode, ny, nx]` with zero
    /// frequency at `[0, 0]`.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Bound<'py, PyArray4<C64>>> {
        let (object, probe) = self.modes(object, probe)?;
        let psi = far_field(&self.inner, &self.optics, &object, &probe);
        let addr = self.inner.addr();
        let [ny, nx] = addr.frame_shape();
        PyArray1::from_vec(py, psi).reshape([addr.nframe(), addr.nmode(), ny, nx])
    }

    /// Normalised PyNX LLK of `object` and `probe` over the scan.
    fn llk<'py>(
        &self,
        py: Python<'py>,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (object, probe) = self.modes(object, probe)?;
        llk_dict(py, &llk_of(&self.inner, &self.optics, &object, &probe))
    }
}

/// LSQML across epochs (`ptyrs::engine::Lsqml`); `cost` is `"amplitude"` or `"poisson"`.
#[pyclass]
pub(crate) struct Lsqml(ptyrs::engine::Lsqml);

#[pymethods]
impl Lsqml {
    #[new]
    fn new(batch_size: usize, cost: &str, seed: u64) -> PyResult<Self> {
        let cost = match cost {
            "amplitude" => Cost::Amplitude,
            "poisson" => Cost::Poisson,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "cost must be \"amplitude\" or \"poisson\", got {cost:?}"
                )));
            }
        };
        Ok(Self(ptyrs::engine::Lsqml::new(
            ptyrs::engine::LsqmlParams {
                batch_size,
                cost,
                seed,
            },
        )))
    }

    /// One epoch over `ptycho`'s scan from `object` and `probe`, without probe variation;
    /// returns the new object and probe and each batch's step sizes.
    fn epoch<'py>(
        &mut self,
        py: Python<'py>,
        ptycho: &Ptycho,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Epoch<'py>> {
        let (mut object, mut probe) = ptycho.modes(object, probe)?;
        let p = &ptycho.inner;
        let mut var = ProbeVariation::none(probe.plane_shape());
        let steps = self.0.epoch(
            p.addr(),
            p.observed(),
            &ptycho.optics,
            slice::from_mut(&mut object),
            &mut probe,
            &mut var,
        );
        let steps = steps.iter().map(|s| {
            let d = PyDict::new(py);
            d.set_item("frames", PyArray1::from_slice(py, &s.frames))?;
            d.set_item("a_far", PyArray1::from_slice(py, &s.a_far))?;
            // the far-field model has one object slice
            d.set_item("a_obj", PyArray1::from_slice(py, &s.a_obj[0]))?;
            d.set_item("a_probe", PyArray1::from_slice(py, &s.a_probe))?;
            Ok(d)
        });
        let steps = PyList::new(py, steps.collect::<PyResult<Vec<_>>>()?)?;
        Ok((stack_out(py, object.0)?, stack_out(py, probe.0)?, steps))
    }
}

/// BH across epochs (`ptyrs::engine::Bh`).
#[pyclass]
pub(crate) struct Bh(ptyrs::engine::Bh);

#[pymethods]
impl Bh {
    #[new]
    fn new(batch_size: usize, probe_rho: f64, seed: u64) -> Self {
        Self(ptyrs::engine::Bh::new(ptyrs::engine::BhParams {
            batch_size,
            probe_rho,
            position_rho: 0.1,
            seed,
        }))
    }

    /// One epoch over `ptycho`'s scan from `object` and `probe`; returns the new object and
    /// probe and each batch's step.
    fn epoch<'py>(
        &mut self,
        py: Python<'py>,
        ptycho: &Ptycho,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
    ) -> PyResult<Epoch<'py>> {
        let (mut object, mut probe) = ptycho.modes(object, probe)?;
        let p = &ptycho.inner;
        let objects = slice::from_mut(&mut object);
        let steps = self
            .0
            .epoch(p.addr(), p.observed(), &ptycho.optics, objects, &mut probe);
        let steps = steps.iter().map(|s| {
            let d = PyDict::new(py);
            d.set_item("frames", PyArray1::from_slice(py, &s.frames))?;
            d.set_item("beta", s.beta)?;
            d.set_item("alpha", s.alpha)?;
            d.set_item("top", s.top)?;
            d.set_item("bottom", s.bottom)?;
            Ok(d)
        });
        let steps = PyList::new(py, steps.collect::<PyResult<Vec<_>>>()?)?;
        Ok((stack_out(py, object.0)?, stack_out(py, probe.0)?, steps))
    }
}

#[cfg(test)]
mod tests {
    use ptyrs::base::{Geometry, required_obj_shape};

    use super::*;

    /// The binding's LLK of given arrays is the engine's own LLK when they are its state.
    #[test]
    fn llk_of_the_state_is_the_engine_llk() {
        let g = Geometry::new(1.5e-10, 1.0, 75e-6, [16, 16]).unwrap();
        let pixel = g.pixel_size_object();
        let positions = ptyrs::base::sim::fermat(5, 3.0 * pixel[1]);
        let shape = required_obj_shape(&positions.pynx_preprocess(pixel), pixel, g.frame, 0);
        let object = ptyrs::base::sim::object(shape);
        let probe = ptyrs::base::sim::probe(g.frame, 2).unwrap();
        let scan =
            ptyrs::engine::sim::simulate(g, positions, &object, &probe, 2e3, None, 3).unwrap();
        let p = ptyrs::engine::Ptycho::new(&scan, object, probe).unwrap();
        let optics = Optics::far_field(&g);
        assert_eq!(llk_of(&p, &optics, p.object(), p.probe()), p.llk());
    }
}
