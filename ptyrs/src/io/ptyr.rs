use std::path::Path;

use rust_hdf5::format::messages::datatype::DatatypeMessage;
use rust_hdf5::{Complex32, Complex64, H5Dataset, H5File};

use crate::base::{C64, Object, Probe, Stack};
use crate::io::IoError;

/// The object and probe of a PtyPy `.ptyr` file.
#[derive(Debug, Clone, PartialEq)]
pub struct Ptyr {
    pub object: Object,
    pub probe: Probe,
    /// Object pixel size `[y, x]` (m), PtyPy `Storage.psize`.
    pub psize: [f64; 2],
    /// Position `[y, x]` (m) of object pixel `[0, 0]`, PtyPy `Storage.origin`.
    pub origin: [f64; 2],
}

/// Reads the object and probe storages of a `.ptyr` file PtyPy `Ptycho.save_run` writes
/// (any `kind`): `content/obj/<ID>` and `content/probe/<ID>`, each with a `data` stack
/// (`[layers, ny, nx]`, complex64 or complex128 as PtyPy `data_type` sets) and the `_psize`
/// and `_origin` of its pixels. A run of several scans keeps one storage per scan; that is
/// a layout error here, since ptyrs reconstructs one object and one probe.
pub fn read_ptyr(path: impl AsRef<Path>) -> Result<Ptyr, IoError> {
    let f = H5File::open(path)?;
    let object = only_storage(&f, "obj")?;
    let probe = only_storage(&f, "probe")?;
    Ok(Ptyr {
        object: Object::new(read_stack(&f.dataset(&format!("{object}/data"))?)?),
        probe: Probe::new(read_stack(&f.dataset(&format!("{probe}/data"))?)?),
        psize: read_pair(&f, &format!("{object}/_psize"))?,
        origin: read_pair(&f, &format!("{object}/_origin"))?,
    })
}

/// The path of the one storage of container `content/<name>`.
fn only_storage(f: &H5File, name: &str) -> Result<String, IoError> {
    let container = format!("content/{name}");
    match &f.root_group().group(&container)?.group_names()?[..] {
        [id] => Ok(format!("{container}/{id}")),
        ids => Err(IoError::Layout(format!(
            "{container} holds the storages {ids:?}, expected one"
        ))),
    }
}

fn read_stack(ds: &H5Dataset) -> Result<Stack<C64>, IoError> {
    let dims: [usize; 3] = ds.shape().try_into().map_err(|s: Vec<usize>| {
        IoError::Layout(format!(
            "storage data has shape {s:?}, expected 3 dimensions"
        ))
    })?;
    let data = match ds.datatype()? {
        DatatypeMessage::Compound { size: 8, .. } => ds
            .read_raw::<Complex32>()?
            .into_iter()
            .map(|c| C64::new(c.re.into(), c.im.into()))
            .collect(),
        DatatypeMessage::Compound { size: 16, .. } => ds
            .read_raw::<Complex64>()?
            .into_iter()
            .map(|c| C64::new(c.re, c.im))
            .collect(),
        t => {
            return Err(IoError::Layout(format!(
                "storage data has type {t:?}, expected complex64 or complex128"
            )));
        }
    };
    Ok(Stack::from_vec(dims, data)?)
}

fn read_pair(f: &H5File, path: &str) -> Result<[f64; 2], IoError> {
    f.dataset(path)?
        .read_raw::<f64>()?
        .try_into()
        .map_err(|v: Vec<f64>| IoError::Layout(format!("{path} holds {v:?}, expected [y, x]")))
}
