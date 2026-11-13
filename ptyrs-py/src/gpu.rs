//! The GPU engine ([`ptyrs::engine::GpuPtycho`]): AP, DM, ML, rPIE and BH in f32.

use cuda_core::CudaContext;
use numpy::{PyArray3, PyReadonlyArray3};
use ptyrs::base::{C64, Object, Probe};
use ptyrs::engine::{BhParams, GpuOptions};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{Error, Scan, history, llk_dict, params, stack_in, stack_out};

/// The GPU reconstruction state of a scan on CUDA device `device`; keywords are
/// `GpuOptions` fields.
#[pyclass(unsendable)]
pub(crate) struct GpuPtycho(ptyrs::engine::GpuPtycho);

#[pymethods]
impl GpuPtycho {
    #[new]
    #[pyo3(signature = (scan, object, probe, device = 0, **options))]
    fn new(
        scan: &Scan,
        object: PyReadonlyArray3<'_, C64>,
        probe: PyReadonlyArray3<'_, C64>,
        device: usize,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let options = params!(GpuOptions, options, stack_size, graph, llk_slots, fused);
        let (object, probe) = (Object::new(stack_in(object)?), Probe::new(stack_in(probe)?));
        let ctx = CudaContext::new(device).map_err(|e| Error(e.into()))?;
        Ok(Self(
            ptyrs::engine::GpuPtycho::new(&ctx, &scan.0, object, probe, options)
                .map_err(Error::from)?,
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
    /// the last.
    fn bh(
        &mut self,
        nepoch: usize,
        calc_llk: usize,
        batch_size: usize,
        probe_rho: f64,
        seed: u64,
    ) -> PyResult<()> {
        let params = BhParams {
            batch_size,
            probe_rho,
            position_rho: 0.1,
            seed,
        };
        Ok(self.0.bh(&params, nepoch, calc_llk).map_err(Error::from)?)
    }

    #[getter]
    fn object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<C64>>> {
        stack_out(py, self.0.object().map_err(Error::from)?.0)
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
