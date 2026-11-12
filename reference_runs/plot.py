"""Draw comparison charts from recorded reference runs. Needs no network.

uv run --with matplotlib python reference_runs/plot.py RUN_DIR [RUN_DIR ...]
"""

import argparse
from pathlib import Path

from dapo_math import read_records, surviving
from matplotlib import pyplot

PANELS = [
    ("evaluation", "accuracy", "Held-out accuracy"),
    ("update", "reward", "Training reward"),
    ("update", "response_tokens_mean", "Training response tokens (mean)"),
    ("update", "seconds_since_previous_update", "Seconds between accepted updates"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--output", type=Path, help="default: charts.png in the first run"
    )
    args = parser.parse_args()
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for run in args.runs:
        records = surviving(read_records(run / "metrics.jsonl"))
        for axis, (event, field, title) in zip(axes.flat, PANELS):
            points = [
                (r["step"], r[field])
                for r in records
                if r["event"] == event and field in r
            ]
            axis.plot(*zip(*points), marker="o", markersize=3, label=run.name)
            axis.set(title=title, xlabel="Accepted optimizer updates")
            axis.grid(alpha=0.3)
    axes.flat[0].legend()
    output = args.output or args.runs[0] / "charts.png"
    figure.savefig(output, dpi=150)
    print(output)


if __name__ == "__main__":
    main()
