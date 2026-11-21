"""Draw comparison charts from recorded reference runs. Needs no network.

uv run python reference_runs/plot.py RUN_ID_OR_JSONL [RUN_ID_OR_JSONL ...]
"""

import argparse
from math import nan
import tomllib
from pathlib import Path
from statistics import mean

from matplotlib import pyplot
from catalog import ReferenceRunIndex, read_output


PANELS = [
    ("evaluation", "reward", "Held-out reward"),
    ("update", "reward", "Training reward"),
    ("update", "response_tokens_mean", "Training response tokens (mean)"),
    ("update", "seconds_since_previous_update", "Seconds between accepted updates"),
]


def plot_initial_distributions(
    runs: list[tuple[str, list[dict]]], output: Path
) -> None:
    figure, axes = pyplot.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = []
    for index, (label, records) in enumerate(runs):
        evaluations = [
            r for r in records if r["event"] == "evaluation" and r["step"] == 0
        ]
        if not evaluations:
            raise ValueError(f"No step-0 evaluations for {label}")
        labels.append(f"{label}\n(n={len(evaluations)})")
        for axis, field, title in zip(
            axes,
            ("reward", "success_fraction"),
            ("Held-out reward", "Held-out success fraction"),
            strict=True,
        ):
            values = [r[field] for r in evaluations]
            color = f"C{index % 10}"
            boxes = axis.boxplot(
                values,
                positions=[index],
                widths=0.35,
                patch_artist=True,
                showfliers=False,
            )
            boxes["boxes"][0].set(facecolor=color, alpha=0.25)
            offsets = [
                index + 0.2 * (j / max(1, len(values) - 1) - 0.5)
                for j in range(len(values))
            ]
            axis.scatter(offsets, values, color=color, s=25, alpha=0.8, zorder=3)
            axis.scatter(
                [index], [mean(values)], color="black", marker="D", s=25, zorder=4
            )
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.3)
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels)
    figure.suptitle(
        "Step-0 evaluation distributions (dots: repetitions; diamonds: means)"
    )
    figure.savefig(output, dpi=150)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="index IDs or JSONL paths")
    parser.add_argument("--labels", nargs="+", help="one display label per run")
    parser.add_argument(
        "--index", type=Path, default=Path(__file__).with_name("index.toml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("reference-runs.png")
    )
    parser.add_argument(
        "--initial-distribution",
        action="store_true",
        help="show distributions of step-0 evaluation records",
    )
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.runs):
        parser.error("--labels requires one label per run")
    catalog = ReferenceRunIndex.model_validate(tomllib.loads(args.index.read_text()))
    index = {entry.id: entry for entry in catalog.runs}
    runs = []
    for run_number, run in enumerate(args.runs):
        entry = index.get(run)
        path = args.index.parent / entry.output if entry else Path(run)
        records = read_output(path, published=bool(entry and entry.source))
        label = (
            args.labels[run_number]
            if args.labels
            else run
            if run in index
            else path.stem
        )
        runs.append((label, records))
    if args.initial_distribution:
        plot_initial_distributions(runs, args.output)
        return
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 7), layout="constrained")
    update_durations = all(
        "update_seconds" in r
        for _, records in runs
        for r in records
        if r["event"] == "update"
    )
    for run_number, (label, records) in enumerate(runs):
        if len(args.runs) == 1:
            figure.suptitle(label, fontsize=10)
            label = "Retained"
        color = f"C{run_number % 10}"
        for axis, (event, field, title) in zip(axes.flat, PANELS):
            if field == "seconds_since_previous_update" and update_durations:
                field = "update_seconds"
                title = "Seconds per update (sampling + training)"
            measurements = [r for r in records if r["event"] == event]
            points = [
                (r["step"], r.get(field, nan))
                for r in measurements
                if not r.get("discarded", False)
            ]
            axis.plot(
                *zip(*points),
                marker="o",
                markersize=3,
                label=label,
                color=color,
            )
            discarded = [r for r in measurements if r.get("discarded", False)]
            if discarded:
                first = measurements.index(discarded[0])
                branch = measurements[max(0, first - 1) : first] + discarded
                axis.plot(
                    [r["step"] for r in branch],
                    [r.get(field, nan) for r in branch],
                    linestyle="--",
                    marker="x",
                    markersize=5,
                    color="tab:orange" if len(args.runs) == 1 else color,
                    label="Discarded"
                    if len(args.runs) == 1
                    else f"{label} (discarded)",
                )
                axis.legend(fontsize=8)
            axis.set(title=title, xlabel="Logical optimizer step")
            axis.grid(alpha=0.3)
    axes.flat[0].legend()
    figure.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
