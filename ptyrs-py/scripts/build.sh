#!/usr/bin/env bash
# Builds the `ptyrs` Python module without maturin and installs it as OUT/ptyrs<EXT_SUFFIX>.
#   build.sh [OUT]    (OUT defaults to ~/work/ptyrs-scratch/py)
# Import it with OUT on sys.path, under the interpreter in PYO3_PYTHON: the environment's, else
# .cargo/config.local.toml's, else python3. It is exported so pyo3 builds against the same one.
# PYO3_BUILD_EXTENSION_MODULE leaves libpython unlinked: the interpreter provides it.
set -euo pipefail
out=${1:-$HOME/work/ptyrs-scratch/py}
root=$(cd "$(dirname "$0")/../.." && pwd)
local_config=$root/.cargo/config.local.toml
if [ -z "${PYO3_PYTHON:-}" ] && [ -f "$local_config" ]; then
    PYO3_PYTHON=$(sed -n -E 's/^PYO3_PYTHON *= *(\{ *value *= *)?"([^"]*)".*/\2/p' "$local_config")
fi
export PYO3_PYTHON=${PYO3_PYTHON:-python3}
suffix=$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
cd "$root"
PYO3_BUILD_EXTENSION_MODULE=1 cargo build --release -p ptyrs-py
mkdir -p "$out"
cp target/release/libptyrs_py.so "$out/ptyrs$suffix"
echo "$out/ptyrs$suffix"
