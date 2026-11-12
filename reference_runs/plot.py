"""Draw comparison charts from recorded reference runs. Needs no network.

uv run --with matplotlib python reference_runs/plot.py RUN_DIR [RUN_DIR ...]
"""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot

PANELS = [
    ("evaluation", "reward", "Held-out accuracy"),
    ("update", "reward", "Training reward"),
    ("update", "response_tokens_mean", "Training response tokens (mean)"),
    ("update", "seconds_since_previous_update", "Seconds between accepted updates"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("reference-runs.png")
    )
    args = parser.parse_args()
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for run in args.runs:
        # A run that was interrupted mid-write ends in a partial line.
        lines = (run / "metrics.jsonl").read_text().splitlines(keepends=True)
        records = [json.loads(line) for line in lines if line.endswith("\n")]
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
    figure.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
