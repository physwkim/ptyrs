# ptyrs

Ptychography reconstruction engine in Rust on cuda-rust; [DESIGN.md](DESIGN.md) is the design.

## Build

`cargo build` fetches cuda-rust and epics-rs from GitHub at the commits pinned in `Cargo.toml`. To build against local clones instead, copy `.cargo/config.local.example.toml` to `.cargo/config.local.toml` in every worktree that should use them (the file is untracked, so each worktree needs its own copy) and fill in the paths; it can also set the `PYO3_PYTHON` that `ptyrs-py/scripts/build.sh` uses. Cargo then builds each local clone at the pinned commit, not its working tree, and Cargo.lock stays as committed; move the file aside while adding or bumping a git dependency so the lock can update.
