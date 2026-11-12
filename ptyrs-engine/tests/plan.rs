//! A parsed plan run on [`Ptycho`] does exactly what the same engine calls made by hand do.

use ptyrs_core::{C64, Object, Stack};
use ptyrs_engine::{ApParams, DmParams, Plan, Ptycho};

mod common;
mod ptychi;

#[test]
fn plan_runs_its_steps_in_order() {
    let (scan, object, probe) = common::dm_scenario(32, 30, 2, 2e5, 4.0);
    let mut start = Ptycho::new(&scan, object, probe).unwrap();
    start.scale_obj_probe();
    let (object, probe) = start.into_modes();

    let plan: Plan = "dm(calc_llk=1):3 -> ap(probe_inertia=0.001, calc_llk=2):3 -> \
                      rpie(obj_alpha=0.5, probe_alpha=0.5, batch_size=4, calc_llk=1):2"
        .parse()
        .unwrap();
    let mut by_plan = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    plan.run(&mut by_plan).unwrap();

    let mut by_hand = Ptycho::new(&scan, object, probe).unwrap();
    let dm = DmParams {
        calc_llk: 1,
        ..DmParams::default()
    };
    let ap = ApParams {
        probe_inertia: Some(0.001),
        calc_llk: 2,
        ..ApParams::default()
    };
    by_hand.dm(&dm, 3);
    by_hand.ap(&ap, 3);
    by_hand.rpie(&ptychi::RPIE, 2).unwrap();

    assert_eq!(by_plan.cycle(), 8);
    assert_eq!(by_plan.history(), by_hand.history());
    assert_eq!(
        by_plan.object().as_slice(),
        by_hand.object().as_slice(),
        "object"
    );
    assert_eq!(
        by_plan.probe().as_slice(),
        by_hand.probe().as_slice(),
        "probe"
    );
}

#[test]
fn plan_stops_at_a_failing_step() {
    let (scan, object, probe) = common::dm_scenario(32, 30, 1, 2e5, 4.0);
    let two = Object::new(Stack::filled(
        [2, object.dims()[1], object.dims()[2]],
        C64::new(1.0, 0.0),
    ));
    let mut p = Ptycho::new(&scan, two, probe).unwrap();
    let plan: Plan = "dm:1 -> rpie:1 -> dm:1".parse().unwrap();
    assert!(plan.run(&mut p).is_err(), "rPIE needs one object mode");
    assert_eq!(p.cycle(), 1);
}
