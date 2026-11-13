//! The CPU engine's results the Python tests compare the bindings against, run here on the
//! engine itself: `engine_reference <dir>` writes `<dir>/opr.h5`, h5py-readable (complex as
//! `{r, i}`); `scripts/build.sh` runs it into the module's directory.
//!
//! `opr.h5` holds a simulated far-field scan (`counts`, raw `y` and `x`, the geometry as root
//! attributes), its `positions` `[nframe, 2]` as the views read them at the object `pixel`,
//! its start `object` and `probe` (`Ptycho::scale_obj_probe` of a flat object and the
//! simulation's probe) and one group per LSQML run with probe variation, its `cost`,
//! `batch_size` and `seed` as attributes: the start `eigenmodes`, `weights` and (`intensity`
//! run) `intensity`, then for each epoch `e` of [`NEPOCH`] a group `epoch<e>` of the arrays it
//! left (`object` `[nslice, nobj, nyo, nxo]`) and its steps, every batch's `frames`, `a_far`, `a_obj` (`[nslice, nframe]`) and
//! `a_probe` concatenated and `batch_len`.

use std::error::Error;
use std::path::Path;
use std::slice;

use ptyrs::base::sim::{fermat, object, probe};
use ptyrs::base::{C64, Geometry, Object, Stack, required_obj_shape};
use ptyrs::engine::sim::simulate;
use ptyrs::engine::{Cost, Lsqml, LsqmlParams, Optics, ProbeVariation, Ptycho};
use rust_hdf5::{Complex64, H5File, H5Group, H5Type};

const NEPOCH: usize = 3;

fn write<T: H5Type>(
    g: &H5Group,
    name: &str,
    dims: &[usize],
    data: &[T],
) -> Result<(), Box<dyn Error>> {
    g.new_dataset::<T>()
        .shape(dims)
        .create(name)?
        .write_raw(data)?;
    Ok(())
}

fn write_stack(g: &H5Group, name: &str, s: &Stack<C64>) -> Result<(), Box<dyn Error>> {
    let data: Vec<Complex64> = s
        .as_slice()
        .iter()
        .map(|c| Complex64 { re: c.re, im: c.im })
        .collect();
    write(g, name, &s.dims(), &data)
}

fn write_variation(g: &H5Group, var: &ProbeVariation) -> Result<(), Box<dyn Error>> {
    let (e, w) = (var.eigenmodes(), var.weights());
    write_stack(g, "eigenmodes", e)?;
    write(g, "weights", &[w.len() / e.len(), e.len()], w)?;
    if let Some(c) = var.intensity() {
        write(g, "intensity", &[c.len()], c)?;
    }
    Ok(())
}

/// `objects` as `[nslice, nobj, nyo, nxo]`.
fn write_objects(g: &H5Group, name: &str, objects: &[Object]) -> Result<(), Box<dyn Error>> {
    let data: Vec<Complex64> = objects
        .iter()
        .flat_map(|o| o.0.as_slice())
        .map(|c| Complex64 { re: c.re, im: c.im })
        .collect();
    let [nobj, nyo, nxo] = objects[0].dims();
    write(g, name, &[objects.len(), nobj, nyo, nxo], &data)
}

/// `√(mean |x|²)`.
fn mnorm(x: &[C64]) -> f64 {
    (x.iter().map(|c| c.norm_sqr()).sum::<f64>() / x.len() as f64).sqrt()
}

/// Two eigenmodes, Hermite-Gauss modes 1 and 2 of [`probe`] at the engine's eigenmode norm
/// (`mnorm` √N), weighted up to 5% of `p0` at every one of `nframe` frames, and, if
/// `intensity`, main-mode weights within 2% of 1.
fn variation(p0: &[C64], frame: [usize; 2], nframe: usize, intensity: bool) -> ProbeVariation {
    let k = 2;
    let modes = probe(frame, k + 1).unwrap();
    let n = p0.len() as f64;
    let data = (1..=k)
        .flat_map(|i| {
            let u = modes.plane(i);
            let s = n.sqrt() / mnorm(u);
            u.iter().map(move |u| s * u)
        })
        .collect();
    let eigenmodes = Stack::from_vec([k, frame[0], frame[1]], data).unwrap();
    let w0 = 0.05 * mnorm(p0) / n.sqrt();
    let weights = (0..nframe * k)
        .map(|j| w0 * (0.7 * (j / k + 1) as f64 * (j % k + 1) as f64).sin())
        .collect();
    let var = ProbeVariation::new(eigenmodes, weights);
    if !intensity {
        return var;
    }
    var.with_intensity(
        (0..nframe)
            .map(|f| 1.0 + 0.02 * (1.3 * f as f64).cos())
            .collect(),
    )
}

fn opr(path: &Path) -> Result<(), Box<dyn Error>> {
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [32, 32])?;
    let pixel = g.pixel_size_object();
    let positions = fermat(20, 4.0 * pixel[1]);
    let shape = required_obj_shape(&positions.pynx_preprocess(pixel), pixel, g.frame, 0);
    let scan = simulate(
        g,
        positions,
        &object(shape),
        &probe(g.frame, 2)?,
        2e5,
        None,
        3,
    )?;
    let flat = Object::new(Stack::from_vec(
        [1, shape[0], shape[1]],
        vec![C64::new(1.0, 0.0); shape[0] * shape[1]],
    )?);
    let mut p = Ptycho::new(&scan, flat, probe(g.frame, 2)?)?;
    p.scale_obj_probe();
    let optics = Optics::far_field(&g);
    let nframe = p.addr().nframe();

    let f = H5File::create(path)?;
    let root = f.root_group();
    for (name, v) in [
        ("wavelength", g.wavelength),
        ("distance", g.distance),
        ("detector_pixel", g.detector_pixel),
    ] {
        root.set_attr_numeric(name, &v)?;
    }
    write(&root, "counts", &scan.counts.dims(), scan.counts.as_slice())?;
    write(&root, "y", &[nframe], &scan.positions.y)?;
    write(&root, "x", &[nframe], &scan.positions.x)?;
    let at = scan.positions.pynx_preprocess(pixel);
    let positions: Vec<f64> = at.y.iter().zip(&at.x).flat_map(|(y, x)| [*y, *x]).collect();
    write(&root, "positions", &[nframe, 2], &positions)?;
    write(&root, "pixel", &[2], &pixel)?;
    write_stack(&root, "object", &p.object().0)?;
    write_stack(&root, "probe", &p.probe().0)?;

    for (run, cost, intensity) in [
        ("opr", Cost::Amplitude, false),
        ("intensity", Cost::Poisson, true),
    ] {
        let params = LsqmlParams {
            batch_size: 4,
            cost,
            seed: 7,
        };
        let group = root.create_group(run)?;
        group.set_attr_string("cost", &format!("{cost:?}").to_lowercase())?;
        group.set_attr_numeric("batch_size", &(params.batch_size as u64))?;
        group.set_attr_numeric("seed", &params.seed)?;
        let (mut object, mut probe) = (p.object().clone(), p.probe().clone());
        let mut var = variation(probe.plane(0), g.frame, nframe, intensity);
        write_variation(&group, &var)?;
        let mut lsqml = Lsqml::new(params);
        for e in 1..=NEPOCH {
            let objects = slice::from_mut(&mut object);
            let steps = lsqml.epoch(
                p.addr(),
                p.observed(),
                &optics,
                objects,
                &mut probe,
                &mut var,
            );
            let at = group.create_group(&format!("epoch{e}"))?;
            write_objects(&at, "object", slice::from_ref(&object))?;
            write_stack(&at, "probe", &probe.0)?;
            write_variation(&at, &var)?;
            let cat = |f: fn(&ptyrs::engine::LsqmlStep) -> &[f64]| -> Vec<f64> {
                steps.iter().flat_map(|s| f(s).iter().copied()).collect()
            };
            let frames: Vec<u64> = steps
                .iter()
                .flat_map(|s| s.frames.iter().map(|&f| f as u64))
                .collect();
            write(&at, "frames", &[nframe], &frames)?;
            write(&at, "a_far", &[nframe], &cat(|s| &s.a_far))?;
            let a_obj: Vec<f64> = (0..optics.nslice())
                .flat_map(|j| steps.iter().flat_map(move |s| s.a_obj[j].iter().copied()))
                .collect();
            write(&at, "a_obj", &[optics.nslice(), nframe], &a_obj)?;
            write(&at, "a_probe", &[nframe], &cat(|s| &s.a_probe))?;
            let len: Vec<u64> = steps.iter().map(|s| s.frames.len() as u64).collect();
            write(&at, "batch_len", &[len.len()], &len)?;
        }
    }
    f.close()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: engine_reference <dir>")?;
    opr(&Path::new(&dir).join("opr.h5"))
}
