#!/usr/bin/env bash
# Python baseline (py_recv.py) against pty-sim, pinned as m0_input_bench.sh
# pins pty-recv: simulator on NUMA node 1, receiver on CPUs 2-23 of node 0.
#   py_input_bench.sh MODE CODEC RATE FRAMES [extra py_recv.py args...]
# OMP_NUM_THREADS defaults to 1: bitshuffle's OpenMP team per decode call
# oversubscribes the decode pool otherwise.
# UTIL_MIN=1024 runs the receiver under uclampset, the frequency floor
# pty-recv sets with its default --util-min 1024.
# Build first: cargo build --release -p ptyrs-live --bins
set -uo pipefail
mode=${1:?mode}
codec=${2:?codec}
rate=${3:?rate}
frames=${4:?frames}
shift 4
here=$(cd "$(dirname "$0")" && pwd)
bin=$(cd "$here/../.." && pwd)/target/release
port=${PORT:-15075}

taskset -c "${SIM_CPUS:-24-47}" "$bin/pty-sim" --codec "$codec" --rate "$rate" \
    --frames "$frames" --port "$port" &
sim=$!
trap 'kill $sim 2>/dev/null' EXIT
clamp=()
[[ -n ${UTIL_MIN:-} ]] && clamp=(uclampset -m "$UTIL_MIN")
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} taskset -c "${RECV_CPUS:-2-23}" "${clamp[@]}" \
    ${PY:-micromamba run -n epicsrs-pva python} "$here/py_recv.py" --mode "$mode" \
    --addr "127.0.0.1:$port" --frames "$frames" "$@"
rc=$?
wait $sim
exit $rc
