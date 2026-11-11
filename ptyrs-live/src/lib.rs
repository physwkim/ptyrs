//! ptyrs-live: host input path from a compressed detector stream (PVA
//! NTNDArray or Eiger stream v1 over ZMQ) to a GPU ring (DESIGN.md §3 B6,
//! §7 M0, M5).

pub mod codec;
pub mod consumer;
pub mod eiger;
pub mod farfield;
pub mod grow;
pub mod handoff;
pub mod model;
pub mod ntnd;
pub mod object;
pub mod pipeline;
pub mod pos;
pub mod pv_out;
pub mod pva_source;
pub mod ring;
pub mod sched;
pub mod stats;
pub mod swmr;
pub mod synth;

/// Small `--key value` command-line reader shared by the bins.
pub mod args {
    use std::collections::HashMap;
    use std::str::FromStr;

    pub struct Args(HashMap<String, String>);

    impl Args {
        pub fn parse() -> Self {
            let mut m = HashMap::new();
            let mut it = std::env::args().skip(1);
            while let Some(k) = it.next() {
                let Some(k) = k.strip_prefix("--") else {
                    panic!("unexpected argument {k}");
                };
                let v = it.next().unwrap_or_else(|| panic!("--{k} needs a value"));
                m.insert(k.to_string(), v);
            }
            Self(m)
        }

        pub fn get<T: FromStr>(&self, k: &str, default: T) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.0.get(k).map_or(default, |v| {
                v.parse().unwrap_or_else(|e| panic!("--{k} {v}: {e:?}"))
            })
        }

        pub fn str(&self, k: &str, default: &str) -> String {
            self.0
                .get(k)
                .cloned()
                .unwrap_or_else(|| default.to_string())
        }
    }
}
