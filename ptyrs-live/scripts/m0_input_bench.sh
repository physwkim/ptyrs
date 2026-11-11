#!/usr/bin/env bash
# M0 host input path (DESIGN.md §0 fold condition 1): simulator pinned to
# NUMA node 1, receiver pinned to node 0 next to GPU 0, separate processes.
# CPUs 0-1 are left out: the runtime thread on CPU 0 received ~6% slower.
#   m0_input_bench.sh CODEC RATE FRAMES [extra pty-recv args...]
# SOURCE=zmq sends frames as an Eiger stream v1 series over ZMQ instead of
# PVA (positions stay on PVA); SOURCE=h5 H5=PATH has the simulator write
# frames and positions to an HDF5 SWMR file that the receiver follows,
# removed afterwards. MODEL_ARGS go to both processes, e.g.
# MODEL_ARGS="--model ptycho --path fermat --points 30000" for the M5 scan.
# Build first: cargo build --release -p ptyrs-live --bins
set -uo pipefail
codec=${1:?codec}
rate=${2:?rate}
frames=${3:?frames}
shift 3
bin=$(cd "$(dirname "$0")/../.." && pwd)/target/release
port=${PORT:-15075}
zport=${ZMQ_PORT:-9999}
sim_src=() recv_src=()
if [ "${SOURCE:-pva}" = zmq ]; then
    sim_src=(--out zmq --zmq-port "$zport")
    recv_src=(--source zmq --zmq "tcp://127.0.0.1:$zport")
elif [ "${SOURCE:-pva}" = h5 ]; then
    h5=${H5:?SOURCE=h5 needs H5=PATH}
    # A stale file would be followed before the simulator replaces it.
    rm -f "$h5"
    sim_src=(--out h5 --h5 "$h5")
    recv_src=(--source h5 --h5 "$h5")
fi

taskset -c "${SIM_CPUS:-24-47}" "$bin/pty-sim" --codec "$codec" --rate "$rate" \
    --frames "$frames" --port "$port" "${sim_src[@]}" ${MODEL_ARGS:-} &
sim=$!
trap 'kill $sim 2>/dev/null; [ -n "${h5:-}" ] && rm -f "$h5"' EXIT
"$bin/pty-recv" --addr "127.0.0.1:$port" --frames "$frames" --cpus "${RECV_CPUS:-2-23}" \
    "${recv_src[@]}" ${MODEL_ARGS:-} "$@"
rc=$?
wait $sim
exit $rc
