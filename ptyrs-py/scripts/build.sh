#!/usr/bin/env bash
# Builds the `ptyrs` Python module without maturin and installs it as OUT/ptyrs<EXT_SUFFIX>.
#   build.sh [OUT]    (OUT defaults to ~/work/ptyrs-scratch/py)
# Import it with OUT on sys.path, under the interpreter .cargo/config.toml names in PYO3_PYTHON.
# PYO3_BUILD_EXTENSION_MODULE leaves libpython unlinked: the interpreter provides it.
set -euo pipefail
out=${1:-$HOME/work/ptyrs-scratch/py}
root=$(cd "$(dirname "$0")/../.." && pwd)
py=$(sed -n 's/^PYO3_PYTHON = { value = "\([^"]*\)".*/\1/p' "$root/.cargo/config.toml")
suffix=$("$py" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
cd "$root"
PYO3_BUILD_EXTENSION_MODULE=1 cargo build --release -p ptyrs-py
mkdir -p "$out"
cp target/release/libptyrs_py.so "$out/ptyrs$suffix"
echo "$out/ptyrs$suffix"
