//! The FRC between the first object modes of two `write_modes` files over one region, as
//! `ptypy_compare`'s compare step prints it; `refs/py/compare_cxidb.py` runs it on its final
//! objects.
//!
//! `cargo run --release -p ptyrs --example frc_pair -- <a.h5> <b.h5> <y0> <y1> <x0> <x1>`
//! compares rows `y0..y1` and columns `x0..x1` of the two objects, which must be the same shape.

use std::ops::Range;

use ptyrs::base::frc::Frc;

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [a, b, y0, y1, x0, x1] = <[String; 6]>::try_from(args)
        .map_err(|_| "usage: frc_pair <a.h5> <b.h5> <y0> <y1> <x0> <x1>")?;
    let at = |s: &str| -> Result<usize, Error> { Ok(s.parse()?) };
    let region: [Range<usize>; 2] = [at(&y0)?..at(&y1)?, at(&x0)?..at(&x1)?];
    let (oa, _) = ptyrs::io::read_modes(&a)?;
    let (ob, _) = ptyrs::io::read_modes(&b)?;
    let shape = oa.0.plane_shape();
    if ob.0.plane_shape() != shape {
        return Err(format!("{a}: {shape:?} object, {b}: {:?}", ob.0.plane_shape()).into());
    }
    let f = Frc::compute(oa.0.plane(0), ob.0.plane(0), shape, &region)?;
    let rings = &f.rings[1..];
    let min = rings.iter().map(|r| r.frc).fold(f64::INFINITY, f64::min);
    let mean = rings.iter().map(|r| r.frc).sum::<f64>() / rings.len() as f64;
    println!(
        "FRC over rows {:?}, columns {:?}: min ring {min:.6}, mean {mean:.6}, first ring under \
         half-bit {:?} of {}; shift {:?}",
        region[0],
        region[1],
        f.crossing(),
        rings.len(),
        f.shift
    );
    Ok(())
}
