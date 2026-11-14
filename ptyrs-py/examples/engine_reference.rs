//! The engine's results the Python tests compare the bindings against, run here on the
//! engine itself: `engine_reference <dir>` writes `<dir>/opr.h5`, `<dir>/near.h5`,
//! `<dir>/multislice.h5` (the CPU engine's) and `<dir>/gpu_optics.h5` (the GPU engine's, on
//! CUDA device `PTYRS_DEVICE`, default 0, and the CPU engine's on the same scans), h5py-readable (complex as `{r, i}`); `scripts/build.sh` runs it into the module's directory.
//!
//! `opr.h5` holds a simulated far-field scan (`counts`, raw `y` and `x`, the geometry as root
//! attributes), its `positions` `[nframe, 2]` as the views read them at the object `pixel`,
//! its start `object` and `probe` (`Ptycho::scale_obj_probe` of a flat object and the
//! simulation's probe) and one group per LSQML run with probe variation, its `cost`,
//! `batch_size` and `seed` as attributes: the start `eigenmodes`, `weights` and (`intensity`
//! run) `intensity`, then for each epoch `e` of `NEPOCH` a group `epoch<e>` of the arrays it
//! left (`object` `[nslice, nobj, nyo, nxo]`) and its steps, every batch's `frames`, `a_far`, `a_obj` (`[nslice, nframe]`) and
//! `a_probe` concatenated and `batch_len`.
//!
//! `near.h5` holds a scan simulated through pty-chi's planar near field ([`Optics::free_space`]
//! of the `wavelength`, `wave_pixel` and `distance` it names, `exit` `"free_space"`): its
//! `counts`, `positions` at the object `pixel` and start `object` (`[nslice, nobj, nyo, nxo]`)
//! and `probe`, then an `lsqml` and a `bh` group of `NEPOCH` epochs from that start, their
//! parameters as attributes and each epoch's groups as in `opr.h5`, BH's steps as `frames`,
//! `beta`, `alpha`, `top`, `bottom` and `batch_len`.
//!
//! `multislice.h5` is laid out as `near.h5` for a scan through three object slices and PyNX's
//! far field ([`Optics::with_slices`] of [`Optics::far_field`], `exit` `"far_field"` with its
//! `detector_pixel`), the slice `gaps` in metres at the object pixel.
//!
//! `gpu_optics.h5` holds one group per near-field propagator the GPU engine runs
//! (`angular_spectrum`, `fresnel`, `fresnel_back`): the `wavelength`, `pixel` and `distance`
//! of its [`Optics::free_space`] as attributes, a scan drawn through it (`counts`, raw `y` and
//! `x`), the start `object` and `probe`, and for each algorithm a fresh [`GpuPtycho`] runs
//! `NEPOCH` times from that start a group of its name: the `object` and `probe` it left and
//! the `poisson` LLK after each. LSQML and BH run as in `near.h5`, the others at their
//! defaults with `calc_llk` 1; `dm_f16` and `raar_f16` are DM's and RAAR's under
//! [`GpuOptions::psi_f16`]. `lsqml_sliced` is LSQML's run through the propagator after a
//! 20 µm gap behind a first object slice, both slices the start object: the `objects`
//! `[2, nobj, nyo, nxo]`, `probe` and `poisson` LLK it left. Its group `cpu` holds the CPU
//! engine's [`Ptycho::with_optics`] from that start: the start's `llk` (`poisson`,
//! `gaussian`, `euclidian`, `nb_photons_calc`), a group per algorithm of AP, DM, RAAR, ML and
//! rPIE laid out as the GPU's, and `lsqml`, the `objects` and `probe` LSQML's `NEPOCH` epochs
//! over the state's own views and optics left.

use std::error::Error;
use std::path::Path;
use std::slice;

use cuda_core::CudaContext;
use ptyrs::base::required_obj_shape;
use ptyrs::base::sim::{fermat, object, poisson, probe};
use ptyrs::base::{AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack};
use ptyrs::engine::sim::simulate;
use ptyrs::engine::{
    ApParams, Bh, BhParams, Cost, DmParams, EngineError, GpuOptions, GpuPtycho, Lsqml, LsqmlParams,
    MlParams, Observed, Optics, ProbeVariation, Ptycho, RaarParams, RpieParams, SliceWaves,
    Wavefield, shift,
};
use ptyrs::gpu::FUSED_SIDES;
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
    write(&root, "y", &[nframe], scan.positions.y())?;
    write(&root, "x", &[nframe], scan.positions.x())?;
    let at = scan.positions.pynx_preprocess(pixel);
    let positions: Vec<f64> = at
        .y()
        .iter()
        .zip(at.x())
        .flat_map(|(y, x)| [*y, *x])
        .collect();
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
        let mut lsqml = Lsqml::new(params)?;
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

/// Counts of `objects` and `probe` through `optics` at `addr`'s views, in detector order,
/// Poisson-drawn with `seed` from intensities scaled to `photons` per frame on average, and
/// that scale.
fn draw(
    addr: &AddrTable,
    optics: &Optics,
    objects: &[Object],
    probe: &Probe,
    photons: f64,
    seed: u64,
) -> Result<(Counts, f64), Box<dyn Error>> {
    let mut psi = SliceWaves::new(optics, addr.views(), objects, probe, None)
        .psi()
        .to_vec();
    optics.exit().forward(&mut psi);
    let ([ny, nx], nframe, nmode) = (addr.frame_shape(), addr.nframe(), addr.nmode());
    let n = ny * nx;
    let mut intensity: Vec<f64> = psi
        .chunks_exact(nmode * n)
        .flat_map(|block| {
            (0..n).map(move |i| {
                let j = shift(i, ny, nx);
                (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum::<f64>()
            })
        })
        .collect();
    let scale = photons / (intensity.iter().sum::<f64>() / nframe as f64);
    intensity.iter_mut().for_each(|v| *v *= scale);
    Ok((poisson([nframe, ny, nx], &intensity, None, seed)?, scale))
}

/// The scan `truth` draws through `optics` at `positions` (at the object `pixel`), and LSQML
/// and BH runs from flat object slices and the truth's probe at the counts' scale, into
/// `root` as `near.h5` lays them out.
fn optics_runs(
    root: &H5Group,
    optics: &Optics,
    positions: &Positions,
    pixel: [f64; 2],
    truth: (&[Object], &Probe),
) -> Result<(), Box<dyn Error>> {
    let (objects, probe) = truth;
    let shape = objects[0].plane_shape();
    let frame = probe.plane_shape();
    let addr = AddrTable::new(positions, pixel, frame, shape, probe.nmode(), 1)?;
    let (counts, scale) = draw(&addr, optics, objects, probe, 2e5, 3)?;
    let obs = Observed::from_counts(&counts);
    let nframe = addr.nframe();
    write(root, "counts", &counts.dims(), counts.as_slice())?;
    let yx: Vec<f64> = positions
        .y()
        .iter()
        .zip(positions.x())
        .flat_map(|(y, x)| [*y, *x])
        .collect();
    write(root, "positions", &[nframe, 2], &yx)?;
    write(root, "pixel", &[2], &pixel)?;
    let flat = Object::new(Stack::from_vec(
        [1, shape[0], shape[1]],
        vec![C64::new(1.0, 0.0); shape[0] * shape[1]],
    )?);
    let start = vec![flat; optics.nslice()];
    let r = scale.sqrt();
    let p0 = Probe::new(Stack::from_vec(
        probe.dims(),
        probe.as_slice().iter().map(|c| c * r).collect(),
    )?);
    write_objects(root, "object", &start)?;
    write_stack(root, "probe", &p0.0)?;

    let params = LsqmlParams {
        batch_size: 5,
        cost: Cost::Poisson,
        seed: 7,
    };
    let group = root.create_group("lsqml")?;
    group.set_attr_string("cost", "poisson")?;
    group.set_attr_numeric("batch_size", &(params.batch_size as u64))?;
    group.set_attr_numeric("seed", &params.seed)?;
    let (mut objects, mut probe) = (start.clone(), p0.clone());
    let mut lsqml = Lsqml::new(params)?;
    let mut none = ProbeVariation::none(frame);
    for e in 1..=NEPOCH {
        let steps = lsqml.epoch(&addr, &obs, optics, &mut objects, &mut probe, &mut none);
        let at = group.create_group(&format!("epoch{e}"))?;
        write_objects(&at, "object", &objects)?;
        write_stack(&at, "probe", &probe.0)?;
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

    let params = BhParams {
        batch_size: 5,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    };
    let group = root.create_group("bh")?;
    group.set_attr_numeric("batch_size", &(params.batch_size as u64))?;
    group.set_attr_numeric("probe_rho", &params.probe_rho)?;
    group.set_attr_numeric("seed", &params.seed)?;
    let (mut objects, mut probe) = (start, p0);
    let mut bh = Bh::new(params)?;
    for e in 1..=NEPOCH {
        let steps = bh.epoch(&addr, &obs, optics, &mut objects, &mut probe);
        let at = group.create_group(&format!("epoch{e}"))?;
        write_objects(&at, "object", &objects)?;
        write_stack(&at, "probe", &probe.0)?;
        let frames: Vec<u64> = steps
            .iter()
            .flat_map(|s| s.frames.iter().map(|&f| f as u64))
            .collect();
        write(&at, "frames", &[nframe], &frames)?;
        let nbatch = steps.len();
        write(
            &at,
            "beta",
            &[nbatch],
            &steps.iter().map(|s| s.beta).collect::<Vec<_>>(),
        )?;
        write(
            &at,
            "alpha",
            &[nbatch],
            &steps.iter().map(|s| s.alpha).collect::<Vec<_>>(),
        )?;
        write(
            &at,
            "top",
            &[nbatch],
            &steps.iter().map(|s| s.top).collect::<Vec<_>>(),
        )?;
        write(
            &at,
            "bottom",
            &[nbatch],
            &steps.iter().map(|s| s.bottom).collect::<Vec<_>>(),
        )?;
        let len: Vec<u64> = steps.iter().map(|s| s.frames.len() as u64).collect();
        write(&at, "batch_len", &[nbatch], &len)?;
    }
    Ok(())
}

/// pty-chi's planar near field at an angular-spectrum distance.
fn near(path: &Path) -> Result<(), Box<dyn Error>> {
    let frame = [32, 32];
    let (wavelength, pixel, distance) = (1e-10, [5e-8, 5e-8], 5e-4);
    let w = Wavefield::new(frame, wavelength, pixel, distance)?;
    let positions = fermat(20, 3.0 * pixel[1]);
    let shape = required_obj_shape(&positions, pixel, frame, 0);
    let f = H5File::create(path)?;
    let root = f.root_group();
    root.set_attr_string("exit", "free_space")?;
    root.set_attr_numeric("wavelength", &wavelength)?;
    root.set_attr_numeric("distance", &distance)?;
    write(&root, "wave_pixel", &[2], &pixel)?;
    let truth = (&[object(shape)][..], &probe(frame, 2)?);
    optics_runs(&root, &Optics::free_space(&w), &positions, pixel, truth)?;
    f.close()?;
    Ok(())
}

/// Three object slices with unequal gaps before PyNX's far field.
fn multislice(path: &Path) -> Result<(), Box<dyn Error>> {
    let frame = [32, 32];
    let (wavelength, distance) = (1e-10, 5e-3);
    let g = Geometry::new(wavelength, distance, wavelength * distance / 1.6e-6, frame)?;
    let pixel = g.pixel_size_object();
    let gaps = [4e-5, 6e-5];
    let waves = gaps
        .iter()
        .map(|&d| Wavefield::new(frame, wavelength, pixel, d))
        .collect::<Result<Vec<_>, _>>()?;
    let optics = Optics::far_field(&g).with_slices(&waves)?;
    let positions = fermat(20, 3.0 * pixel[1]);
    let shape = required_obj_shape(&positions, pixel, frame, 0);
    let o = object(shape);
    let third = Stack::from_vec(
        o.0.dims(),
        o.0.as_slice().iter().map(|c| c.powf(1.0 / 3.0)).collect(),
    )?;
    let f = H5File::create(path)?;
    let root = f.root_group();
    root.set_attr_string("exit", "far_field")?;
    for (name, v) in [
        ("wavelength", g.wavelength),
        ("distance", g.distance),
        ("detector_pixel", g.detector_pixel),
    ] {
        root.set_attr_numeric(name, &v)?;
    }
    write(&root, "gaps", &[gaps.len()], &gaps)?;
    let objects = vec![Object::new(third); optics.nslice()];
    optics_runs(
        &root,
        &optics,
        &positions,
        pixel,
        (&objects, &probe(frame, 2)?),
    )?;
    f.close()?;
    Ok(())
}

/// One GPU algorithm call of `NEPOCH` cycles or epochs.
type GpuRun = fn(&mut GpuPtycho) -> Result<(), EngineError>;

const GPU_RUNS: [(&str, GpuRun); 7] = [
    ("ap", |g| {
        let params = ApParams {
            calc_llk: 1,
            ..Default::default()
        };
        g.ap(&params, NEPOCH)
    }),
    ("dm", |g| {
        let params = DmParams {
            calc_llk: 1,
            ..Default::default()
        };
        g.dm(&params, NEPOCH)
    }),
    ("raar", |g| {
        let params = RaarParams {
            calc_llk: 1,
            ..Default::default()
        };
        g.raar(&params, NEPOCH)
    }),
    ("ml", |g| {
        let params = MlParams {
            calc_llk: 1,
            ..Default::default()
        };
        g.ml(&params, NEPOCH)
    }),
    ("rpie", |g| {
        let params = RpieParams {
            calc_llk: 1,
            ..Default::default()
        };
        g.rpie(&params, NEPOCH)
    }),
    ("lsqml", |g| {
        let params = LsqmlParams {
            batch_size: 5,
            cost: Cost::Poisson,
            seed: 7,
        };
        g.lsqml(&params, NEPOCH, 1)
    }),
    ("bh", |g| {
        let params = BhParams {
            batch_size: 5,
            probe_rho: 0.01,
            position_rho: 0.1,
            seed: 7,
        };
        g.bh(&params, NEPOCH, 1)
    }),
];

/// One CPU algorithm call of `NEPOCH` cycles or epochs.
type CpuRun = fn(&mut Ptycho) -> Result<(), EngineError>;

const CPU_RUNS: [(&str, CpuRun); 5] = [
    ("ap", |p| {
        let params = ApParams {
            calc_llk: 1,
            ..Default::default()
        };
        p.ap(&params, NEPOCH);
        Ok(())
    }),
    ("dm", |p| {
        let params = DmParams {
            calc_llk: 1,
            ..Default::default()
        };
        p.dm(&params, NEPOCH);
        Ok(())
    }),
    ("raar", |p| {
        let params = RaarParams {
            calc_llk: 1,
            ..Default::default()
        };
        p.raar(&params, NEPOCH);
        Ok(())
    }),
    ("ml", |p| {
        let params = MlParams {
            calc_llk: 1,
            ..Default::default()
        };
        p.ml(&params, NEPOCH);
        Ok(())
    }),
    ("rpie", |p| {
        let params = RpieParams {
            calc_llk: 1,
            ..Default::default()
        };
        p.rpie(&params, NEPOCH)
    }),
];

/// The CPU engine's runs of `gpu_optics.h5`'s group `cpu` under the free-space propagation
/// `w`, from `start`.
fn cpu_optics(
    root: &H5Group,
    scan: &Scan,
    w: &Wavefield,
    start: (&Object, &Probe),
) -> Result<(), Box<dyn Error>> {
    let state = || {
        Ptycho::with_optics(
            scan,
            Optics::free_space(w),
            start.0.clone(),
            start.1.clone(),
        )
    };
    let cpu = root.create_group("cpu")?;
    let p = state()?;
    let l = p.llk();
    write(
        &cpu,
        "llk",
        &[4],
        &[l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc],
    )?;
    for (algorithm, run) in CPU_RUNS {
        let mut p = state()?;
        run(&mut p)?;
        let at = cpu.create_group(algorithm)?;
        write_stack(&at, "object", &p.object().0)?;
        write_stack(&at, "probe", &p.probe().0)?;
        let llk: Vec<f64> = p.history().iter().map(|r| r.llk.poisson).collect();
        write(&at, "poisson", &[llk.len()], &llk)?;
    }
    let mut lsqml = Lsqml::new(LsqmlParams {
        batch_size: 5,
        cost: Cost::Poisson,
        seed: 7,
    })?;
    let (mut objects, mut probe) = (vec![start.0.clone()], start.1.clone());
    let mut none = ProbeVariation::none(probe.plane_shape());
    for _ in 0..NEPOCH {
        let (addr, obs) = (p.addr(), p.observed());
        lsqml.epoch(addr, obs, p.optics(), &mut objects, &mut probe, &mut none);
    }
    let at = cpu.create_group("lsqml")?;
    write_objects(&at, "objects", &objects)?;
    write_stack(&at, "probe", &probe.0)?;
    Ok(())
}

/// The GPU engine under an angular-spectrum propagator and a Fresnel transform forward and
/// back, every algorithm from a fresh state, and the CPU engine's [`cpu_optics`].
fn gpu_optics(path: &Path) -> Result<(), Box<dyn Error>> {
    let device = match std::env::var("PTYRS_DEVICE") {
        Ok(d) => d.parse()?,
        Err(_) => 0,
    };
    let ctx = CudaContext::new(device)?;
    let side = FUSED_SIDES[0] as usize;
    let frame = [side, side];
    let (wavelength, pixel) = (1e-10, [5e-8, 5e-8]);
    let f = H5File::create(path)?;
    for (name, distance) in [
        ("angular_spectrum", 1e-4),
        ("fresnel", 5e-3),
        ("fresnel_back", -5e-3),
    ] {
        let w = Wavefield::new(frame, wavelength, pixel, distance)?;
        let optics = Optics::free_space(&w);
        let raw = fermat(20, side as f64 / 8.0 * pixel[1]);
        let positions = raw.pynx_preprocess(pixel);
        let shape = required_obj_shape(&positions, pixel, frame, 16);
        let truth = probe(frame, 2)?;
        let addr = AddrTable::new(&positions, pixel, frame, shape, truth.nmode(), 1)?;
        let (counts, scale) = draw(&addr, &optics, &[object(shape)], &truth, 2e5, 3)?;
        let flat = Object::new(Stack::from_vec(
            [1, shape[0], shape[1]],
            vec![C64::new(1.0, 0.0); shape[0] * shape[1]],
        )?);
        let r = scale.sqrt();
        let p0 = Probe::new(Stack::from_vec(
            truth.dims(),
            truth.as_slice().iter().map(|c| c * r).collect(),
        )?);

        let root = f.root_group().create_group(name)?;
        root.set_attr_numeric("wavelength", &wavelength)?;
        root.set_attr_numeric("distance", &distance)?;
        write(&root, "pixel", &[2], &pixel)?;
        write(&root, "counts", &counts.dims(), counts.as_slice())?;
        write(&root, "y", &[raw.y().len()], raw.y())?;
        write(&root, "x", &[raw.x().len()], raw.x())?;
        write_stack(&root, "object", &flat.0)?;
        write_stack(&root, "probe", &p0.0)?;
        // Of the geometry the engine reads the frame alone.
        let scan = Scan::new(
            Geometry::new(wavelength, 1.0, pixel[1], frame)?,
            raw,
            counts,
        )?;
        let f16 = GpuOptions {
            psi_f16: true,
            ..GpuOptions::default()
        };
        let runs = GPU_RUNS
            .iter()
            .map(|&(a, run)| (a.to_string(), run, GpuOptions::default()))
            .chain(
                GPU_RUNS
                    .iter()
                    .filter(|(a, _)| matches!(*a, "dm" | "raar"))
                    .map(|&(a, run)| (format!("{a}_f16"), run, f16)),
            );
        for (algorithm, run, options) in runs {
            let mut gpu =
                GpuPtycho::with_optics(&ctx, &scan, &optics, flat.clone(), p0.clone(), options)?;
            run(&mut gpu)?;
            let at = root.create_group(&algorithm)?;
            write_stack(&at, "object", &gpu.object()?.0)?;
            write_stack(&at, "probe", &gpu.probe()?.0)?;
            let llk: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
            write(&at, "poisson", &[llk.len()], &llk)?;
        }
        let gap = Wavefield::new(frame, wavelength, pixel, 2e-5)?;
        let sliced = optics.with_slices(&[gap])?;
        let objects = [flat.clone(), flat.clone()];
        let options = GpuOptions::default();
        let mut gpu = GpuPtycho::with_slices(&ctx, &scan, &sliced, &objects, p0.clone(), options)?;
        let params = LsqmlParams {
            batch_size: 5,
            cost: Cost::Poisson,
            seed: 7,
        };
        gpu.lsqml(&params, NEPOCH, 1)?;
        let at = root.create_group("lsqml_sliced")?;
        write_objects(&at, "objects", &gpu.objects()?)?;
        write_stack(&at, "probe", &gpu.probe()?.0)?;
        let llk: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
        write(&at, "poisson", &[llk.len()], &llk)?;
        cpu_optics(&root, &scan, &w, (&flat, &p0))?;
    }
    f.close()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: engine_reference <dir>")?;
    let dir = Path::new(&dir);
    opr(&dir.join("opr.h5"))?;
    near(&dir.join("near.h5"))?;
    multislice(&dir.join("multislice.h5"))?;
    gpu_optics(&dir.join("gpu_optics.h5"))
}
