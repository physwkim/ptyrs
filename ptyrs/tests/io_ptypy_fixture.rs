//! Reads `.ptyr` files PtyPy `Ptycho.save_run` wrote through h5py (gzip storages, complex64
//! and complex128); see `data/make_ptypy_small.py`.

use ptyrs::base::C64;

#[test]
fn reads_ptypy_written_ptyr() {
    for kind in ["single", "double"] {
        let path = format!(
            "{}/tests/io/data/ptypy_small_{kind}.ptyr",
            env!("CARGO_MANIFEST_DIR")
        );
        let r = ptyrs::io::read_ptyr(&path).unwrap();
        assert_eq!(r.object.0.dims(), [1, 10, 11], "{kind}");
        assert_eq!(r.probe.0.dims(), [2, 8, 8], "{kind}");
        for (i, &v) in r.object.0.as_slice().iter().enumerate() {
            let (y, x) = ((i / 11) as f64, (i % 11) as f64);
            assert_eq!(v, C64::new(y - 0.25 * x, 0.5 * y + x), "{kind} object {i}");
        }
        for (i, &v) in r.probe.0.as_slice().iter().enumerate() {
            let (m, y, x) = ((i / 64) as f64, (i / 8 % 8) as f64, (i % 8) as f64);
            assert_eq!(v, C64::new(m + 0.5 * y, 0.25 * m - x), "{kind} probe {i}");
        }
        assert_eq!(
            (r.psize, r.origin),
            ([1.5e-8, 2.5e-8], [-3e-7, 4e-7]),
            "{kind}"
        );
    }
}
