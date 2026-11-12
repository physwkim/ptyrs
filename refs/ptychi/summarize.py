"""Tables over the run JSONs in out/ -> out/summary.json, and a short print.

    python refs/ptychi/summarize.py      (plain Python 3, no GPU)

Targets are factors of the truth LLK (PyNX definition, m3ref.py) in the LLK of the run's own
noise model: Poisson for LSQML-Poisson, Gaussian for LSQML-Gaussian and BH (amplitude model only).
A run fitted in one model does not reach the other model's truth (a Gaussian-model run ends
~1.3x the Poisson truth), so runs of different models are compared only on a common LLK.
wall_s is a timing figure only where the run saw no other process on its GPU throughout
(gpu_watch_period_s); runs from before that watch were checked at start and end only and, like
runs that saw one, carry "indicative": true. Epoch counts are the main result.
"""

import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LLKS = ("poisson", "gaussian", "euclidian")
BH_CHOSEN = "bh_sweep_cg_rho0.01"  # the engine's BH-CG: one batch per epoch


def load():
    runs = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name not in ("check_llk", "summary"):
            with open(path) as f:
                runs[name] = json.load(f)
    return runs


def summary_row(d):
    last = d["curve"][-1]
    noise = d["settings"].get("noise", "gaussian")
    row = {
        "dataset": d["dataset"],
        "algorithm": d["algorithm"],
        "batch_size": d["batch_size"],
        "noise_model": noise,
        "epochs_run": last["epoch"],
        "gpu_uuid": d["environment"]["gpu_uuid"],
        "indicative": bool(d["gpu_shared_with"]) or "gpu_watch_period_s" not in d,
        "gpu_shared_with": d["gpu_shared_with"],
        "wall_per_epoch_s": last["wall_s"] / last["epoch"],
        "peak_memory_reserved_gib": d["peak_memory_reserved_gib"],
        "final_over_truth": {k: last[k] / d["llk_truth"][k] for k in LLKS},
        "to_target": d["time_to_target"][noise],
    }
    for col in ("position_rms_px", "intensity_rms_rel"):
        if col in last:
            row[col] = {"initial": d["curve"][0][col], "final": last[col]}
    return row


def best(rows, factor):
    """Run reaching `factor` in the fewest epochs (ties to a timing run, then to wall time), in
    the least wall time among timing runs, and in the least wall time of all runs."""
    hit = {n: r["to_target"][factor] for n, r in rows.items() if r["to_target"][factor]}
    if not hit:
        return None
    timed = {n: h for n, h in hit.items() if not rows[n]["indicative"]}
    return {
        "by_epochs": min(hit, key=lambda n: (hit[n]["epoch"], rows[n]["indicative"],
                                             hit[n]["wall_s"])),
        "by_wall": min(timed, key=lambda n: timed[n]["wall_s"]) if timed else None,
        "by_wall_indicative": min(hit, key=lambda n: hit[n]["wall_s"]),
    }


def optimum(rows):
    """Per target factor, the best run; "optimal" is the best at the tightest factor any run
    reaches."""
    factors = list(next(iter(rows.values()))["to_target"])
    per = {f: best(rows, f) for f in factors}
    tight = next((f for f in factors if per[f]), None)
    return {"per_factor": per, "tightest_reached": tight,
            "optimal": None if tight is None else per[tight]}


def epoch_ratio(num, den):
    """num epochs / den epochs per factor, where both reach it."""
    out = {}
    for f, h in den["to_target"].items():
        g = num["to_target"][f]
        out[f] = None if g is None or h is None else g["epoch"] / h["epoch"]
    return out


def main():
    raw = load()
    runs = {n: summary_row(d) for n, d in raw.items()}
    out = {"runs": runs}

    with open(os.path.join(OUT, "check_llk.json")) as f:
        c = json.load(f)
    out["check_llk"] = {"worst_rel": c["worst_rel"], "tol": c["tol"]}

    sweeps = {}
    for noise in ("poisson", "gaussian"):
        rows = {n: r for n, r in runs.items() if n.startswith(f"sweep_lsqml_{noise}_b")}
        if rows:
            sweeps[noise] = optimum(rows)
    out["lsqml_batch"] = sweeps

    bh = {n: r for n, r in runs.items() if n.startswith("bh_sweep_")}
    if bh:
        out["bh"] = {"chosen": BH_CHOSEN, **optimum(bh)}

    # DESIGN B7: BH-CG against first-order LSQML in iterations, same (Gaussian) model and LLK.
    lsq = sweeps.get("gaussian", {}).get("optimal")
    if BH_CHOSEN in runs and lsq:
        b7 = {}
        for label, name in (("lsqml_best_batch", lsq["by_epochs"]),
                            ("lsqml_full_batch", "sweep_lsqml_gaussian_b1024")):
            if name in runs:
                b7[label] = {"run": name, "over_bh_cg": epoch_ratio(runs[name], runs[BH_CHOSEN])}
        bh_best = out["bh"]["optimal"]["by_epochs"] if out["bh"]["optimal"] else None
        if bh_best and bh_best != BH_CHOSEN and lsq["by_epochs"] in runs:
            b7["lsqml_best_batch_over_bh_best"] = {
                "bh_run": bh_best,
                "ratio": epoch_ratio(runs[lsq["by_epochs"]], runs[bh_best])}
        out["b7_epoch_ratio"] = {"bh_cg": BH_CHOSEN, **b7}

    # Each feature run against the same data without the feature model, same algorithm.
    feats = {}
    for n, r in runs.items():
        if n.startswith(("sweep_", "bh_sweep_", "timing_")):
            continue
        feature, sep, tail = n.partition(f"_{r['algorithm']}")
        off = f"{feature.split('_')[0]}_off{sep}{tail}"
        entry = {"run": n, "final_over_truth": r["final_over_truth"], "to_target": r["to_target"]}
        for col in ("position_rms_px", "intensity_rms_rel"):
            if col in r:
                entry[col] = r[col]
        if off in runs and off != n:
            m = r["noise_model"]
            entry["vs_off"] = {"run": off, f"final_{m}_over_off": r["final_over_truth"][m]
                               / runs[off]["final_over_truth"][m]}
        feats.setdefault(feature, {})[r["algorithm"]] = entry
    out["features"] = feats

    # Timing pass of a run's settings on the timing GPU (run_m3.py --prefix timing_): its wall
    # times, and how far its LLK curve is from the base run's (same seed, other GPU).
    timing = {}
    for n, r in runs.items():
        base = n[len("timing_"):]
        if n.startswith("timing_") and base in raw:
            m = r["noise_model"]
            diff = max(abs(a[m] - b[m]) / b[m] for a, b in zip(raw[n]["curve"][1:],
                                                               raw[base]["curve"][1:]))
            timing[n] = {k: r[k] for k in ("gpu_uuid", "indicative", "gpu_shared_with",
                                           "wall_per_epoch_s", "to_target")}
            timing[n].update(base_run=base, max_rel_llk_diff_vs_base=diff)
    out["timing"] = timing

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)

    for n, r in runs.items():
        t = r["to_target"]
        hits = " ".join(f"{f}:{t[f]['epoch'] if t[f] else '-'}" for f in ("1.01", "1.05", "1.2"))
        print(f"{n:34s} {r['noise_model']:8s} b{r['batch_size']:<5d} {hits:22s} "
              f"final {r['final_over_truth'][r['noise_model']]:.4f}x "
              f"{r['wall_per_epoch_s']:.3f}s/ep{' indicative' if r['indicative'] else ''}")
    for noise, s in sweeps.items():
        print(f"LSQML {noise} optimal at {s['tightest_reached']}x: {s['optimal']}")
    if "bh" in out:
        print(f"BH optimal at {out['bh']['tightest_reached']}x: {out['bh']['optimal']}")
    if "b7_epoch_ratio" in out:
        print("B7 epoch ratio:", json.dumps(out["b7_epoch_ratio"]))


if __name__ == "__main__":
    main()
