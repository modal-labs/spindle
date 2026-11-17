"""Draw comparison charts from recorded reference runs. Needs no network.

uv run python reference_runs/plot.py RUN_ID_OR_JSONL [RUN_ID_OR_JSONL ...]
"""

import argparse
import json
import tomllib
from pathlib import Path
from typing import Self

from matplotlib import pyplot
from pydantic import BaseModel, ConfigDict, model_validator


class ReferenceRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    validation: str
    output: str
    spindle_commit: str
    trainer_image: str
    miles_commit: str
    tinker_sdk: str
    hardware: str
    region: str | None = None
    notes: str = ""


class ReferenceRunIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[ReferenceRun]

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({run.id for run in self.runs}) != len(self.runs):
            raise ValueError("duplicate run IDs in index")
        return self


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
        path = args.index.parent / index[run].output if run in index else Path(run)
        # A run that was interrupted mid-write ends in a partial line.
        lines = path.read_text().splitlines(keepends=True)
        records = [json.loads(line) for line in lines if line.endswith("\n")]
        for axis, (event, field, title) in zip(axes.flat, PANELS):
            points = [
                (r["step"], r[field])
                for r in records
                if r["event"] == event and field in r
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
