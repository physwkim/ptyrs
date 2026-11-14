//! The GPU engine ([`ptyrs::engine::GpuPtycho`]): AP, DM, RAAR, ML, rPIE, BH and LSQML in f32.

use cuda_core::CudaContext;
use numpy::{
    PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use ptyrs::base::{C64, Object, Probe};
use ptyrs::engine::{BhParams, GpuOptions, LsqmlParams};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::cpu::Optics;
use crate::{Error, Scan, history, llk_dict, objects_in, objects_out, params, stack_in, stack_out};

/// The GPU reconstruction state of a scan on CUDA device `device`, under `optics` in place of
/// the far field of the scan's geometry if given; keywords are `GpuOptions` fields. `object`
/// is `[nmode, nyo, nxo]`, or `[nslice, nmode, nyo, nxo]` with a slice per slice of `optics`
/// (`GpuPtycho::with_slices`: more than one runs LSQML alone, of one mode).
#[pyclass(unsendable)]
pub(crate) struct GpuPtycho(
    ptyrs::engine::GpuPtycho,
    /// The scan's frame shape `[ny, nx]`, which the engine keeps on the device.
    [usize; 2],
);

#[pymethods]
impl GpuPtycho {
    #[new]
    #[pyo3(signature = (scan, object, probe, device = 0, optics = None, **options))]
    fn new(
        scan: &Scan,
        object: &Bound<'_, PyAny>,
        probe: PyReadonlyArray3<'_, C64>,
        device: usize,
        optics: Option<&Bound<'_, Optics>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let options = params!(
            GpuOptions, options, stack_size, graph, llk_slots, fused, psi_f16
        );
        let objects = match object.extract::<PyReadonlyArray4<'_, C64>>() {
            Ok(a) => objects_in(a)?,
            Err(_) => {
                let a = object.extract::<PyReadonlyArray3<'_, C64>>().map_err(|_| {
                    PyTypeError::new_err(
                        "object must be complex128 [nmode, nyo, nxo] or [nslice, nmode, nyo, nxo]",
                    )
                })?;
                vec![Object::new(stack_in(a)?)]
            }
        };
        let probe = Probe::new(stack_in(probe)?);
        let far;
        let optics = match optics {
            Some(o) => &o.get().inner,
            None => {
                far = ptyrs::engine::Optics::far_field(&scan.0.geometry);
                &far
            }
        };
        let ctx = CudaContext::new(device).map_err(|e| Error(e.into()))?;
        let engine =
            ptyrs::engine::GpuPtycho::with_slices(&ctx, &scan.0, optics, &objects, probe, options);
        Ok(Self(
            engine.map_err(Error::from)?,
            scan.0.counts.frame_shape(),
        ))
    }

    /// `ncycle` AP cycles; keywords are `ApParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn ap(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Ok(self
            .0
            .ap(&crate::ap_params(params)?, ncycle)
            .map_err(Error::from)?)
    }

    /// `ncycle` DM cycles; keywords are `DmParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn dm(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Ok(self
            .0
            .dm(&crate::dm_params(params)?, ncycle)
            .map_err(Error::from)?)
    }

    /// `ncycle` RAAR cycles; keywords are `RaarParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn raar(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Ok(self
            .0
            .raar(&crate::raar_params(params)?, ncycle)
            .map_err(Error::from)?)
    }

    /// `ncycle` ML cycles; keywords are `MlParams` fields.
    #[pyo3(signature = (ncycle, **params))]
    fn ml(&mut self, ncycle: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Ok(self
            .0
            .ml(&crate::ml_params(params)?, ncycle)
            .map_err(Error::from)?)
    }

    /// `nepoch` rPIE epochs; keywords are `RpieParams` fields.
    #[pyo3(signature = (nepoch, **params))]
    fn rpie(&mut self, nepoch: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Ok(self
            .0
            .rpie(&crate::rpie_params(params)?, nepoch)
            .map_err(Error::from)?)
    }

    /// `nepoch` BH epochs from a fresh BH, an LLK record every `calc_llk` epochs and after
    /// the last; `position_rho` as `Bh`'s, for the modelled position.
    #[pyo3(signature = (nepoch, calc_llk, batch_size, probe_rho, seed, *,
                        position_rho = crate::POSITION_RHO))]
    fn bh(
        &mut self,
        nepoch: usize,
        calc_llk: usize,
        batch_size: usize,
        probe_rho: f64,
        seed: u64,
        position_rho: f64,
    ) -> PyResult<()> {
        let params = BhParams {
            batch_size,
            probe_rho,
            position_rho,
            seed,
        };
        Ok(self.0.bh(&params, nepoch, calc_llk).map_err(Error::from)?)
    }

    /// The last BH epoch's steps as `Bh.epoch` returns them; none before a BH call.
    fn bh_steps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        crate::bh_steps(py, &self.0.bh_steps().map_err(Error::from)?)
    }

    /// `nepoch` LSQML epochs from a fresh LSQML (`cost` `"amplitude"` or `"poisson"`), an LLK
    /// record every `calc_llk` epochs and after the last.
    fn lsqml(
        &mut self,
        nepoch: usize,
        calc_llk: usize,
        batch_size: usize,
        cost: &str,
        seed: u64,
    ) -> PyResult<()> {
        let params = LsqmlParams {
            batch_size,
            cost: crate::cost(cost)?,
            seed,
        };
        Ok(self
            .0
            .lsqml(&params, nepoch, calc_llk)
            .map_err(Error::from)?)
    }

    /// The last LSQML epoch's steps as `Lsqml.epoch` returns them; none before an LSQML call.
    fn lsqml_steps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        crate::lsqml_steps(py, &self.0.lsqml_steps().map_err(Error::from)?)
    }

    /// As `Ptycho.set_features`, in `f32` on the device; with `fused=True` only. Refused
    /// features (a `ValueError`) leave the current ones; after a device error (a
    /// `RuntimeError`), the new ones may stay modelled, with the LLK pass run without its
    /// graph, or be partly written.
    #[pyo3(signature = (*, scale = None, background = None, position = None))]
    fn set_features(
        &mut self,
        scale: Option<PyReadonlyArray1<'_, f64>>,
        background: Option<PyReadonlyArray2<'_, f64>>,
        position: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<()> {
        let features = crate::features_in(scale, background, position, self.1)?;
        Ok(self.0.set_features(&features).map_err(Error::from)?)
    }

    /// As `Ptycho.features`, read back from the device.
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        crate::features_out(py, self.0.features().map_err(Error::from)?, self.1)
    }

    /// As `Ptycho.sampling`.
    #[getter]
    fn sampling(&self) -> &'static str {
        crate::sampling_name(self.0.sampling())
    }

    /// As `Ptycho.set_sampling`. After a device error (a `RuntimeError`) the current sampling
    /// stays if the new one's buffers could not be built; else the new one is entered, and
    /// the LLK pass runs without its graph until a later `set_features` or `set_sampling`
    /// captures it again.
    fn set_sampling(&mut self, sampling: &str) -> PyResult<()> {
        let sampling = crate::sampling(sampling)?;
        Ok(self.0.set_sampling(sampling).map_err(Error::from)?)
    }

    /// The object, of a multislice object its first slice.
    #[getter]
    fn object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<C64>>> {
        stack_out(py, self.0.object().map_err(Error::from)?.0)
    }

    /// The object's slices `[nslice, nmode, nyo, nxo]` in the order the wave crosses them.
    #[getter]
    fn objects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<C64>>> {
        objects_out(py, self.0.objects().map_err(Error::from)?)
    }

    #[getter]
    fn probe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<C64>>> {
        stack_out(py, self.0.probe().map_err(Error::from)?.0)
    }

    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        history(py, self.0.history())
    }

    #[getter]
    fn cycle(&self) -> usize {
        self.0.cycle()
    }

    /// Normalised PyNX LLK of the current object and probe.
    fn llk<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        llk_dict(py, &self.0.llk().map_err(Error::from)?)
    }
}
