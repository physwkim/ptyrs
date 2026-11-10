#!/usr/bin/env bash
# M0 host input path (DESIGN.md §0 fold condition 1): simulator pinned to
# NUMA node 1, receiver pinned to node 0 next to GPU 0, separate processes.
# CPUs 0-1 are left out: the runtime thread on CPU 0 received ~6% slower.
#   m0_input_bench.sh CODEC RATE FRAMES [extra pty-recv args...]
# Build first: cargo build --release -p ptyrs-live --bins
set -uo pipefail
codec=${1:?codec}
rate=${2:?rate}
frames=${3:?frames}
shift 3
bin=$(cd "$(dirname "$0")/../.." && pwd)/target/release
port=${PORT:-15075}

taskset -c "${SIM_CPUS:-24-47}" "$bin/pty-sim" --codec "$codec" --rate "$rate" \
    --frames "$frames" --port "$port" &
sim=$!
trap 'kill $sim 2>/dev/null' EXIT
"$bin/pty-recv" --addr "127.0.0.1:$port" --frames "$frames" --cpus "${RECV_CPUS:-2-23}" "$@"
rc=$?
wait $sim
exit $rc
