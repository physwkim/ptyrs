//! ptyrs::live: host input path from a compressed detector stream (PVA
//! NTNDArray or Eiger stream v1 over ZMQ) to a GPU ring (DESIGN.md §3 B6,
//! §7 M0, M5).

#![warn(clippy::disallowed_methods)] // clippy.toml: no capture outside LiveGraph::capture

pub mod codec;
pub mod consumer;
pub mod converge;
pub mod eiger;
pub mod farfield;
pub mod fill;
pub mod follow;
pub mod graph;
pub mod grow;
pub mod handoff;
pub mod model;
pub mod ntnd;
pub mod object;
pub mod pipeline;
pub mod pos;
pub mod pv_out;
pub mod pva_source;
pub mod recon;
pub mod recv;
pub mod ring;
pub mod scan;
pub mod sched;
#[cfg(test)]
pub(crate) mod sent;
pub mod stack;
pub mod stats;
pub mod swmr;
pub mod synth;

/// Small `--key value` command-line reader shared by the bins: [`read`](args::read) runs a binary's
/// reader over the flags given and refuses every flag it did not look up, so a misspelt
/// flag, or one the other flags make moot, cannot pass unseen.
pub mod args {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fmt;
    use std::str::FromStr;

    /// sysexits' EX_USAGE: apart from the receiver's report codes.
    pub const EX_USAGE: i32 = 64;

    /// The flags given, and those a reader looked up, each with its default as shown.
    pub struct Args {
        given: HashMap<String, String>,
        read: RefCell<Vec<(String, String)>>,
    }

    /// A flag refused, as its message.
    #[derive(Debug)]
    pub struct ArgError(String);

    impl ArgError {
        pub fn new(msg: String) -> Self {
            Self(msg)
        }
    }

    impl fmt::Display for ArgError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for ArgError {}

    /// What the flags ask for: a run's settings, or with `--help` the usage.
    #[derive(Debug)]
    pub enum Parsed<T> {
        Run(T),
        Help(String),
    }

    impl Args {
        /// `--k`, or `default`.
        pub fn get<T: FromStr + fmt::Display>(&self, k: &str, default: T) -> Result<T, ArgError>
        where
            T::Err: fmt::Display,
        {
            self.note(k, default.to_string());
            Ok(self.parse(k)?.unwrap_or(default))
        }

        /// `--k` if given.
        pub fn opt<T: FromStr>(&self, k: &str) -> Result<Option<T>, ArgError>
        where
            T::Err: fmt::Display,
        {
            self.note(k, "unset".into());
            self.parse(k)
        }

        pub fn str(&self, k: &str, default: &str) -> String {
            self.note(k, default.into());
            self.given
                .get(k)
                .cloned()
                .unwrap_or_else(|| default.to_string())
        }

        /// The value `of` pairs with `--k`'s name; the first's when not given.
        pub fn choice<T: Copy>(&self, k: &str, of: &[(&str, T)]) -> Result<T, ArgError> {
            let names: Vec<_> = of.iter().map(|&(n, _)| n).collect();
            self.note(k, names.join("|"));
            let Some(v) = self.given.get(k) else {
                return Ok(of[0].1);
            };
            of.iter()
                .find(|&&(n, _)| n == v)
                .map(|&(_, t)| t)
                .ok_or_else(|| ArgError(format!("--{k} {v}: expected one of {}", names.join(", "))))
        }

        fn parse<T: FromStr>(&self, k: &str) -> Result<Option<T>, ArgError>
        where
            T::Err: fmt::Display,
        {
            self.given
                .get(k)
                .map(|v| v.parse().map_err(|e| ArgError(format!("--{k} {v}: {e}"))))
                .transpose()
        }

        fn note(&self, k: &str, shown: String) {
            let mut read = self.read.borrow_mut();
            if !read.iter().any(|(r, _)| r == k) {
                read.push((k.into(), shown));
            }
        }
    }

    /// `argv`'s `--key value` pairs and `--help` through `reader`. A flag given twice or
    /// with no value, a bare argument, a value `reader` refuses and a flag it never looks
    /// up are errors. With `--help`, the usage lists the flags `reader` looked up with
    /// the others given, so a choice's other values show theirs.
    pub fn read<T>(
        argv: impl IntoIterator<Item = String>,
        reader: impl FnOnce(&Args) -> Result<T, ArgError>,
    ) -> Result<Parsed<T>, ArgError> {
        let (mut given, mut help) = (HashMap::new(), false);
        let mut it = argv.into_iter();
        while let Some(tok) = it.next() {
            if tok == "--help" {
                help = true;
                continue;
            }
            let Some(k) = tok.strip_prefix("--") else {
                return Err(ArgError(format!("unexpected argument {tok}")));
            };
            let Some(v) = it.next() else {
                return Err(ArgError(format!("--{k} needs a value")));
            };
            if given.insert(k.to_string(), v).is_some() {
                return Err(ArgError(format!("--{k} given twice")));
            }
        }
        let a = Args {
            given,
            read: RefCell::default(),
        };
        let r = reader(&a);
        let read = a.read.into_inner();
        if help {
            let mut u = String::from(
                "flags read with those given, at their defaults; another value of a \
                 choice (a|b) reads others:\n",
            );
            for (k, shown) in &read {
                u += &format!("  --{k} {shown}\n");
            }
            if let Err(e) = r {
                u += &format!("stopped at: {e}\n");
            }
            return Ok(Parsed::Help(u));
        }
        let t = r?;
        let mut unread: Vec<_> = a
            .given
            .keys()
            .filter(|k| !read.iter().any(|(r, _)| r == *k))
            .map(|k| format!("--{k}"))
            .collect();
        if !unread.is_empty() {
            unread.sort();
            return Err(ArgError(format!(
                "not read with the flags given: {} (--help lists those read)",
                unread.join(", ")
            )));
        }
        Ok(Parsed::Run(t))
    }

    /// [`read`] over the process's arguments for binary `name`: the usage to stdout and
    /// exit 0 on `--help`, the error to stderr and exit [`EX_USAGE`] on a refused flag.
    pub fn cli<T>(name: &str, reader: impl FnOnce(&Args) -> Result<T, ArgError>) -> T {
        match read(std::env::args().skip(1), reader) {
            Ok(Parsed::Run(t)) => t,
            Ok(Parsed::Help(u)) => {
                print!("usage: {name} [--help] [--FLAG VALUE]...\n{u}");
                std::process::exit(0)
            }
            Err(e) => {
                eprintln!("{name}: {e}");
                std::process::exit(EX_USAGE)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn refused(argv: &[&str]) -> String {
            let r = read(argv.iter().map(|s| s.to_string()), |a| a.get("k", 0u32));
            r.unwrap_err().to_string()
        }

        #[test]
        fn a_flag_with_no_value_twice_bare_or_unparsed_is_refused() {
            assert_eq!(refused(&["--k"]), "--k needs a value");
            assert_eq!(refused(&["--k", "1", "--k", "2"]), "--k given twice");
            assert_eq!(refused(&["k", "1"]), "unexpected argument k");
            assert_eq!(
                refused(&["--k", "-1"]),
                "--k -1: invalid digit found in string"
            );
        }
    }
}
