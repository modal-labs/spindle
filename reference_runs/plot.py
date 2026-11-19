"""Draw comparison charts from recorded reference runs. Needs no network.

uv run python reference_runs/plot.py RUN_ID_OR_JSONL [RUN_ID_OR_JSONL ...]
"""

import argparse
from math import nan
import tomllib
from pathlib import Path

from matplotlib import pyplot
from catalog import ReferenceRunIndex, read_output


PANELS = [
    ("evaluation", "reward", "Held-out reward"),
    ("update", "reward", "Training reward"),
    ("update", "response_tokens_mean", "Training response tokens (mean)"),
    ("update", "seconds_since_previous_update", "Seconds between accepted updates"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="index IDs or JSONL paths")
    parser.add_argument(
        "--index", type=Path, default=Path(__file__).with_name("index.toml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("reference-runs.png")
    )
    args = parser.parse_args()
    catalog = ReferenceRunIndex.model_validate(tomllib.loads(args.index.read_text()))
    index = {entry.id: entry for entry in catalog.runs}
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for run_number, run in enumerate(args.runs):
        entry = index.get(run)
        path = args.index.parent / entry.output if entry else Path(run)
        records = read_output(path, published=bool(entry and entry.source))
        label = run if run in index else path.stem
        if len(args.runs) == 1:
            figure.suptitle(label, fontsize=10)
            label = "Retained"
        color = f"C{run_number % 10}"
        for axis, (event, field, title) in zip(axes.flat, PANELS):
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
