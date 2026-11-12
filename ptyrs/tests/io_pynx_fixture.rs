//! Reads a CXI file PyNX `save_ptycho_data_cxi` wrote through libhdf5 (float32 data, gzip +
//! shuffle chunks, h5py `bool` mask, f32 translation); see `data/make_pynx_small.py`.

use ptyrs::base::MASKED;

#[test]
fn reads_pynx_written_cxi() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/io/data/pynx_small.cxi");
    let scan = ptyrs::io::read_cxi(path).unwrap();
    let (nz, ny, nx) = (4, 4, 6);
    assert_eq!(scan.counts.dims(), [nz, ny, nx]);
    for (i, &v) in scan.counts.as_slice().iter().enumerate() {
        let (f, p) = (i / (ny * nx), i % (ny * nx));
        let want = if p == nx + 2 || (f, p) == (2, 3 * nx + 1) {
            MASKED
        } else {
            (i * 37 % 500) as u16
        };
        assert_eq!(v, want, "flat index {i}");
    }
    let g = scan.geometry;
    assert_eq!(
        (g.distance, g.detector_pixel, g.frame),
        (1.2, 75e-6, [4, 6])
    );
    assert!(
        (g.wavelength / 1.5e-10 - 1.0).abs() < 1e-12,
        "{}",
        g.wavelength
    );
    for i in 0..nz {
        let x = (i as f64 * 1.1e-7 + 1e-9 / 3.0) as f32 as f64;
        let y = (-(i as f64) * 0.7e-7) as f32 as f64;
        assert_eq!((scan.positions.x[i], scan.positions.y[i]), (x, y));
    }
}
