"""Summarize a downloaded loading benchmark without treating profiled warmups as serving results."""

import argparse
import json
import pstats
import statistics
from collections import defaultdict
from pathlib import Path


def load_window_gaps(traffic: dict, loads: list[dict]) -> dict:
    samples: dict[str, list[float]] = {"during_load": [], "outside_load": []}
    for request in traffic["requests"]:
        cursor = request["start"] + request["ttft_s"]
        for gap in request["gaps_s"]:
            end = cursor + gap
            loading = any(
                cursor < load["end"] and end > load["start"] for load in loads
            )
            samples["during_load" if loading else "outside_load"].append(gap)
            cursor = end
    return {
        label: {
            "samples": len(values),
            "median_s": statistics.median(values),
            "p99_s": sorted(values)[min(int(len(values) * 0.99), len(values) - 1)],
            "max_s": max(values),
        }
        for label, values in samples.items()
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    anchor_path = args.directory / "clock-anchor.json"
    anchor = json.loads(anchor_path.read_text()) if anchor_path.exists() else None
    events = [
        json.loads(line)
        for path in args.directory.glob("events-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    names = {row["uid"]: row["name"] for row in events if row["phase"] == "prepare"}
    by_name: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        if event["phase"] == "activate" and event["uid"] in names:
            by_name[names[event["uid"]]].append(event)
    rows = []
    for trial in report["trials"]:
        row = {"mode": trial["mode"]}
        for case in (
            "resident",
            "registration_traffic",
            "churn",
            "resident_after_churn",
            "eviction_traffic",
        ):
            if case in trial:
                value = trial[case]
                row[case] = {
                    key: value[key]
                    for key in (
                        "duration_s",
                        "output_tps",
                        "median_ttft_s",
                        "p99_stream_gap_s",
                        "max_stream_gap_s",
                    )
                }
                if anchor is not None:
                    submissions: dict[int, list[float]] = defaultdict(list)
                    for event in events:
                        if (
                            event["phase"] == "activate"
                            and value["start"] + anchor["offset_s"]
                            <= event["at"]
                            <= value["end"] + anchor["offset_s"]
                        ):
                            submissions[event["pid"]].append(event["submit_s"])
                    row[case]["uploads_per_rank"] = [
                        len(values) for values in submissions.values()
                    ]
                    row[case]["max_rank_host_submit_s"] = max(
                        (sum(values) for values in submissions.values()), default=0
                    )
        registered = {
            item["name"]
            for key in ("registrations", "background_loads", "evicting_loads")
            for item in trial.get(key, [])
        }
        preparations = [
            event
            for event in events
            if event["phase"] == "prepare" and event["name"] in registered
        ]
        if preparations:
            row["preparation"] = {
                key: statistics.median(event[key] for event in preparations)
                for key in ("wall_s", "normal_prepare_s", "pack_s", "thread_cpu_s")
            }
        if "first_activation_after_registration" in trial:
            activation = trial["first_activation_after_registration"]
            values = by_name[activation["name"]]
            first_per_pid = {}
            for event in sorted(values, key=lambda item: item["at"]):
                first_per_pid.setdefault(event["pid"], event)
            row["first_activation"] = {
                "request_s": activation["duration_s"],
                "max_rank_submit_s": max(
                    (v["submit_s"] for v in first_per_pid.values()), default=None
                ),
                "median_rank_submit_s": statistics.median(
                    v["submit_s"] for v in first_per_pid.values()
                )
                if first_per_pid
                else None,
                "copies_by_rank": [v["copies"] for v in first_per_pid.values()],
                "bytes_by_rank": [v["bytes"] for v in first_per_pid.values()],
            }
        if "background_loads" in trial:
            row["all_registrations_inside_traffic"] = all(
                load["end"] <= trial["registration_traffic"]["end"]
                for load in trial["background_loads"]
            )
            row["registration_gaps"] = load_window_gaps(
                trial["registration_traffic"], trial["background_loads"]
            )
        if "evicting_loads" in trial:
            row["eviction_gaps"] = load_window_gaps(
                trial["eviction_traffic"], trial["evicting_loads"]
            )
        rows.append(row)
    profiles = []
    for label in ("prepare", "activate"):
        paths = list(args.directory.glob(f"{label}-*.pstats"))
        if not paths:
            continue
        selected = max(paths, key=lambda path: pstats.Stats(str(path)).total_tt)
        stats = pstats.Stats(str(selected))
        profiles.append(
            {
                "profile": selected.name,
                "scope": "Mixed preparation and scheduler calls; use phase timers"
                if label == "prepare"
                else "Scheduler activation diagnostic",
                "total_s": stats.total_tt,
                "largest_cumulative": [
                    {
                        "file": Path(key[0]).name,
                        "line": key[1],
                        "function": key[2],
                        "calls": values[1],
                        "self_s": values[2],
                        "cumulative_s": values[3],
                    }
                    for key, values in sorted(
                        stats.stats.items(), key=lambda item: item[1][3], reverse=True
                    )[:20]
                ],
            }
        )
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "status": report.get("status"),
                "source": report["source"],
                "workload_version": report.get("workload_version", 1),
                "probe_sha256": report["probe_sha256"],
                "packing_sha256": report.get("packing_sha256"),
                "benchmark_sha256": report.get("benchmark_sha256"),
                "adapter_bytes": report["adapter_bytes"],
                "adapter_tensors": report["adapter_tensors"],
                "warm_volume_verification": report.get("warm_volume_verification"),
                "trials": rows,
                "diagnostic_profiles": profiles,
                "packed_verifications": sum(
                    e["phase"] == "packed_verified" for e in events
                ),
                "activation_output_variants": len(
                    {
                        tuple(activation["result"]["output_ids"])
                        for trial in report["trials"]
                        for activation in trial["activations"]
                    }
                ),
                "warning": "First-use warmups and diagnostic profiles are not steady-state serving measurements.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
