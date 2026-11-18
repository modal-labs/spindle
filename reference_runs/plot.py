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
    ("evaluation", "reward", "Held-out accuracy"),
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
    for run in args.runs:
        entry = index.get(run)
        path = args.index.parent / entry.output if entry else Path(run)
        records = read_output(path, published=bool(entry and entry.source))
        for axis, (event, field, title) in zip(axes.flat, PANELS):
            points = [
                (r["step"], r.get(field, nan)) for r in records if r["event"] == event
            ]
            axis.plot(
                *zip(*points),
                marker="o",
                markersize=3,
                label=run if run in index else path.stem,
            )
            axis.set(title=title, xlabel="Accepted optimizer updates")
            axis.grid(alpha=0.3)
    axes.flat[0].legend()
    figure.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
