use std::path::Path;

use ptyrs_core::{C64, Object, Probe, Stack};
use rust_hdf5::{Complex64, H5File};

use crate::IoError;

/// Writes `object` to `/object` (`[nobj, nyo, nxo]`) and `probe` to `/probe`
/// (`[nprobe, ny, nx]`), both complex128 (compound `{r, i}`, h5py's complex layout), in the
/// in-memory orders of [`Object`] and [`Probe`].
pub fn write_modes(path: impl AsRef<Path>, object: &Object, probe: &Probe) -> Result<(), IoError> {
    let f = H5File::create(path)?;
    for (name, stack) in [("object", &object.0), ("probe", &probe.0)] {
        let data: Vec<Complex64> = stack
            .as_slice()
            .iter()
            .map(|c| Complex64 { re: c.re, im: c.im })
            .collect();
        f.new_dataset::<Complex64>()
            .shape(stack.dims())
            .create(name)?
            .write_raw(&data)?;
    }
    f.close()?;
    Ok(())
}

/// Reads the pair [`write_modes`] writes.
pub fn read_modes(path: impl AsRef<Path>) -> Result<(Object, Probe), IoError> {
    let f = H5File::open(path)?;
    let read = |name: &str| -> Result<Stack<C64>, IoError> {
        let ds = f.dataset(name)?;
        let dims: [usize; 3] = ds.shape().try_into().map_err(|s: Vec<usize>| {
            IoError::Layout(format!("/{name} has shape {s:?}, expected 3 dimensions"))
        })?;
        let data = ds
            .read_raw::<Complex64>()?
            .into_iter()
            .map(|c| C64::new(c.re, c.im))
            .collect();
        Ok(Stack::from_vec(dims, data)?)
    };
    Ok((Object::new(read("object")?), Probe::new(read("probe")?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_exact() {
        let o = ptyrs_core::sim::object([12, 10]);
        let p = ptyrs_core::sim::probe([4, 6], 2).unwrap();
        let path = std::env::temp_dir().join(format!("ptyrs-io-{}-modes.h5", std::process::id()));
        write_modes(&path, &o, &p).unwrap();
        let (o2, p2) = read_modes(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(o2, o);
        assert_eq!(p2, p);
    }
}
