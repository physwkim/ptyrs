//! The plan language (DESIGN B7): PyNX `ML**40 * DM**100` written as typed steps run left to
//! right, `"dm:100 -> ap(probe_inertia=0.001, calc_llk=5):40"`.
//!
//! ```text
//! plan   := step ("->" step)*
//! step   := engine ["(" [option ("," option)*] ")"] ":" count
//! engine := "ap" | "dm" | "rpie"
//! option := name "=" value          value: number, "none", "true" or "false"
//! ```
//!
//! Each step starts from its engine's `Default` parameters and sets only the options it
//! names. Options are checked against the engine's schema when the plan is parsed: an unknown
//! or repeated name, a value of the wrong kind or out of range, and a count of 0 are errors, so
//! a parsed [`Plan`] runs without further validation. A TOML array of step strings maps onto
//! [`Plan::from_steps`].

use std::str::FromStr;

use crate::engine::{ApParams, DmParams, EngineError, GpuPtycho, Ptycho, RpieParams};

/// An engine and its parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Params {
    Ap(ApParams),
    Dm(DmParams),
    Rpie(RpieParams),
}

/// `count` cycles (epochs for rPIE) of one engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub params: Params,
    pub count: usize,
}

/// Steps run in order on one [`Backend`].
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    steps: Vec<Step>,
}

/// What a [`Plan`] runs on.
pub trait Backend {
    fn run(&mut self, step: &Step) -> Result<(), EngineError>;
}

impl Backend for Ptycho {
    fn run(&mut self, step: &Step) -> Result<(), EngineError> {
        match &step.params {
            Params::Ap(p) => self.ap(p, step.count),
            Params::Dm(p) => self.dm(p, step.count),
            Params::Rpie(p) => self.rpie(p, step.count)?,
        }
        Ok(())
    }
}

impl Backend for GpuPtycho {
    fn run(&mut self, step: &Step) -> Result<(), EngineError> {
        match &step.params {
            Params::Ap(p) => self.ap(p, step.count),
            Params::Dm(p) => self.dm(p, step.count),
            Params::Rpie(p) => self.rpie(p, step.count),
        }
    }
}

impl Plan {
    /// One step per string, each in the `step` grammar.
    pub fn from_steps<S: AsRef<str>>(
        steps: impl IntoIterator<Item = S>,
    ) -> Result<Self, EngineError> {
        let steps = steps
            .into_iter()
            .map(|s| s.as_ref().parse())
            .collect::<Result<Vec<Step>, _>>()?;
        if steps.is_empty() {
            return Err(EngineError::Invalid("plan has no steps".into()));
        }
        Ok(Self { steps })
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Every step in order; stops at the first error.
    pub fn run(&self, backend: &mut impl Backend) -> Result<(), EngineError> {
        self.steps.iter().try_for_each(|s| backend.run(s))
    }
}

impl FromStr for Plan {
    type Err = EngineError;

    fn from_str(s: &str) -> Result<Self, EngineError> {
        Self::from_steps(s.split("->"))
    }
}

impl FromStr for Step {
    type Err = EngineError;

    fn from_str(text: &str) -> Result<Self, EngineError> {
        parse_step(text.trim())
            .map_err(|e| EngineError::Invalid(format!("plan step `{}`: {e}", text.trim())))
    }
}

fn parse_step(s: &str) -> Result<Step, String> {
    let (head, count) = s.rsplit_once(':').ok_or("expected `engine:count`")?;
    let count = parse_count(count.trim()).map_err(|e| format!("count: {e}"))?;
    let head = head.trim();
    let (engine, options) = match head.split_once('(') {
        Some((e, rest)) => (
            e.trim(),
            rest.strip_suffix(')').ok_or("options must end with `)`")?,
        ),
        None => (head, ""),
    };
    let params = match engine {
        "ap" => Params::Ap(parse_options(options)?),
        "dm" => Params::Dm(parse_options(options)?),
        "rpie" => Params::Rpie(parse_options(options)?),
        _ => return Err(format!("unknown engine `{engine}`; engines: ap, dm, rpie")),
    };
    Ok(Step { params, count })
}

type Setter<P> = fn(&mut P, &str) -> Result<(), String>;

/// An engine's option schema: every name a step may set, with the setter that parses and
/// range-checks its value.
trait Schema: Default + 'static {
    const ENGINE: &'static str;
    const OPTIONS: &'static [(&'static str, Setter<Self>)];
}

fn parse_options<P: Schema>(options: &str) -> Result<P, String> {
    let mut p = P::default();
    if options.trim().is_empty() {
        return Ok(p);
    }
    let mut seen = Vec::new();
    for option in options.split(',') {
        let (name, value) = option
            .split_once('=')
            .ok_or_else(|| format!("expected `name=value`, got `{}`", option.trim()))?;
        let (name, value) = (name.trim(), value.trim());
        let (_, set) = P::OPTIONS.iter().find(|(n, _)| *n == name).ok_or_else(|| {
            let names: Vec<_> = P::OPTIONS.iter().map(|(n, _)| *n).collect();
            format!(
                "unknown {} option `{name}`; options: {}",
                P::ENGINE,
                names.join(", ")
            )
        })?;
        if seen.contains(&name) {
            return Err(format!("option `{name}` given twice"));
        }
        seen.push(name);
        set(&mut p, value).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(p)
}

fn parse_count(v: &str) -> Result<usize, String> {
    match v.parse() {
        Ok(0) => Err("must be at least 1".into()),
        Ok(n) => Ok(n),
        Err(_) => Err(format!("expected a positive integer, got `{v}`")),
    }
}

fn parse_usize(v: &str) -> Result<usize, String> {
    v.parse()
        .map_err(|_| format!("expected a non-negative integer, got `{v}`"))
}

/// A finite real `>= 0`.
fn parse_real(v: &str) -> Result<f64, String> {
    match v.parse::<f64>() {
        Ok(x) if x.is_finite() && x >= 0.0 => Ok(x),
        _ => Err(format!("expected a finite number >= 0, got `{v}`")),
    }
}

/// [`parse_real`], or `none` for an array the engine leaves fixed.
fn parse_update(v: &str) -> Result<Option<f64>, String> {
    if v == "none" {
        Ok(None)
    } else {
        parse_real(v).map(Some)
    }
}

fn parse_bool(v: &str) -> Result<bool, String> {
    v.parse()
        .map_err(|_| format!("expected `true` or `false`, got `{v}`"))
}

impl Schema for ApParams {
    const ENGINE: &'static str = "ap";
    const OPTIONS: &'static [(&'static str, Setter<Self>)] = &[
        ("obj_inertia", |p, v| {
            p.obj_inertia = parse_update(v)?;
            Ok(())
        }),
        ("probe_inertia", |p, v| {
            p.probe_inertia = parse_update(v)?;
            Ok(())
        }),
        ("calc_llk", |p, v| {
            p.calc_llk = parse_usize(v)?;
            Ok(())
        }),
        ("zero_phase_ramp", |p, v| {
            p.zero_phase_ramp = parse_bool(v)?;
            Ok(())
        }),
    ];
}

impl Schema for DmParams {
    const ENGINE: &'static str = "dm";
    const OPTIONS: &'static [(&'static str, Setter<Self>)] = &[
        ("obj_inertia", |p, v| {
            p.obj_inertia = parse_update(v)?;
            Ok(())
        }),
        ("probe_inertia", |p, v| {
            p.probe_inertia = parse_update(v)?;
            Ok(())
        }),
        // PyNX: alpha·AP + (1 − alpha)·DM
        ("alpha", |p, v| match parse_real(v)? {
            a if a <= 1.0 => {
                p.alpha = a;
                Ok(())
            }
            _ => Err(format!("expected a number in [0, 1], got `{v}`")),
        }),
        ("loop_obj_probe", |p, v| {
            p.loop_obj_probe = parse_count(v)?;
            Ok(())
        }),
        ("calc_llk", |p, v| {
            p.calc_llk = parse_usize(v)?;
            Ok(())
        }),
        ("zero_phase_ramp", |p, v| {
            p.zero_phase_ramp = parse_bool(v)?;
            Ok(())
        }),
    ];
}

impl Schema for RpieParams {
    const ENGINE: &'static str = "rpie";
    const OPTIONS: &'static [(&'static str, Setter<Self>)] = &[
        ("obj_step", |p, v| {
            p.obj_step = parse_update(v)?;
            Ok(())
        }),
        ("probe_step", |p, v| {
            p.probe_step = parse_update(v)?;
            Ok(())
        }),
        ("obj_alpha", |p, v| {
            p.obj_alpha = parse_real(v)?;
            Ok(())
        }),
        ("probe_alpha", |p, v| {
            p.probe_alpha = parse_real(v)?;
            Ok(())
        }),
        ("batch_size", |p, v| {
            p.batch_size = parse_count(v)?;
            Ok(())
        }),
        ("calc_llk", |p, v| {
            p.calc_llk = parse_usize(v)?;
            Ok(())
        }),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(s: &str) -> String {
        s.parse::<Plan>().unwrap_err().to_string()
    }

    #[test]
    fn parses_steps_over_engine_defaults() {
        let plan: Plan = " dm:100 -> ap( probe_inertia = 0.001 , calc_llk=5 ):40\n->rpie(obj_step=none, batch_size=4):3 -> dm():1"
            .parse()
            .unwrap();
        let ap = ApParams {
            probe_inertia: Some(0.001),
            calc_llk: 5,
            ..ApParams::default()
        };
        let rpie = RpieParams {
            obj_step: None,
            batch_size: 4,
            ..RpieParams::default()
        };
        let want = [
            (Params::Dm(DmParams::default()), 100),
            (Params::Ap(ap), 40),
            (Params::Rpie(rpie), 3),
            (Params::Dm(DmParams::default()), 1),
        ]
        .map(|(params, count)| Step { params, count });
        assert_eq!(plan.steps(), want);
        assert_eq!(
            Plan::from_steps(["dm:100", "ap(probe_inertia=0.001,calc_llk=5):40"])
                .unwrap()
                .steps(),
            &want[..2]
        );
    }

    /// Each engine's options, all named once with distinct non-default values, so a setter
    /// writing the wrong field or a schema missing a field fails.
    #[test]
    fn every_option_sets_its_own_field() {
        let cases = [
            (
                "ap(obj_inertia=0.5, probe_inertia=0.25, calc_llk=7, zero_phase_ramp=false):1",
                Params::Ap(ApParams {
                    obj_inertia: Some(0.5),
                    probe_inertia: Some(0.25),
                    calc_llk: 7,
                    zero_phase_ramp: false,
                }),
                ApParams::OPTIONS.len(),
            ),
            (
                "dm(obj_inertia=0.5, probe_inertia=0.25, alpha=0.75, loop_obj_probe=2, calc_llk=7, zero_phase_ramp=false):1",
                Params::Dm(DmParams {
                    obj_inertia: Some(0.5),
                    probe_inertia: Some(0.25),
                    alpha: 0.75,
                    loop_obj_probe: 2,
                    calc_llk: 7,
                    zero_phase_ramp: false,
                }),
                DmParams::OPTIONS.len(),
            ),
            (
                "rpie(obj_step=0.5, probe_step=none, obj_alpha=0.25, probe_alpha=0.75, batch_size=4, calc_llk=7):1",
                Params::Rpie(RpieParams {
                    obj_step: Some(0.5),
                    probe_step: None,
                    obj_alpha: 0.25,
                    probe_alpha: 0.75,
                    batch_size: 4,
                    calc_llk: 7,
                }),
                RpieParams::OPTIONS.len(),
            ),
        ];
        for (text, want, noptions) in cases {
            assert_eq!(text.matches('=').count(), noptions, "{text}");
            assert_eq!(text.parse::<Step>().unwrap().params, want, "{text}");
        }
    }

    #[test]
    fn rejects_invalid_plans() {
        for (plan, msg) in [
            ("", "expected `engine:count`"),
            ("dm:10 ->", "plan step ``: expected `engine:count`"),
            ("dm", "expected `engine:count`"),
            ("dm:0", "plan step `dm:0`: count: must be at least 1"),
            ("dm:-1", "count: expected a positive integer, got `-1`"),
            ("ml:10", "unknown engine `ml`; engines: ap, dm, rpie"),
            ("dm(alpha=0.1:10", "must end with `)`"),
            (
                "dm(beta=1):10",
                "unknown dm option `beta`; options: obj_inertia, probe_inertia, alpha, loop_obj_probe, calc_llk, zero_phase_ramp",
            ),
            ("ap(alpha=0.1):10", "unknown ap option `alpha`"),
            ("dm(alpha=0.1, alpha=0.2):10", "option `alpha` given twice"),
            ("dm(alpha):10", "expected `name=value`, got `alpha`"),
            (
                "dm(alpha=1.5):10",
                "alpha: expected a number in [0, 1], got `1.5`",
            ),
            (
                "dm(obj_inertia=-0.1):10",
                "obj_inertia: expected a finite number >= 0",
            ),
            (
                "dm(probe_inertia=nan):10",
                "probe_inertia: expected a finite number >= 0",
            ),
            (
                "dm(loop_obj_probe=0):10",
                "loop_obj_probe: must be at least 1",
            ),
            ("rpie(batch_size=0):10", "batch_size: must be at least 1"),
            (
                "rpie(calc_llk=1.5):10",
                "calc_llk: expected a non-negative integer",
            ),
            (
                "ap(zero_phase_ramp=1):10",
                "zero_phase_ramp: expected `true` or `false`",
            ),
        ] {
            let e = err(plan);
            assert!(e.contains(msg), "`{plan}`: {e}");
        }
        assert!(
            Plan::from_steps(Vec::<String>::new())
                .unwrap_err()
                .to_string()
                .contains("no steps")
        );
    }
}
