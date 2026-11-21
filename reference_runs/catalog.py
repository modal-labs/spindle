"""Execution provenance and collision-safe reference index updates."""

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import tomllib
from collections.abc import Generator
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    validation: str
    output: str
    spindle_commit: str | None = None
    spindle_version: str | None = None
    trainer_image: str | None = None
    inference_image: str | None = None
    tinker_sdk: str | None = None
    hardware: str | None = None
    source: str | None = None
    region: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def spindle_or_external(self) -> Self:
        """A Spindle execution records its runtime; an external one only its source.

        spindle_commit and spindle_version stay unset when the server does not report them.
        """
        runtime = {
            "trainer_image": self.trainer_image,
            "tinker_sdk": self.tinker_sdk,
            "hardware": self.hardware,
        }
        if self.source is None:
            missing = [name for name, value in runtime.items() if value is None]
            if missing:
                raise ValueError(f"Spindle execution is missing {', '.join(missing)}")
        elif any(
            value is not None
            for value in (
                self.spindle_commit,
                self.spindle_version,
                self.inference_image,
                *runtime.values(),
            )
        ):
            raise ValueError(
                "external execution records source, not Spindle provenance"
            )
        return self


class ReferenceRunIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[ReferenceRun]

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({run.id for run in self.runs}) != len(self.runs):
            raise ValueError("duplicate run IDs in index")
        return self


class Record(BaseModel):
    """One line of a Spindle execution's output. Scripts may add any other fields."""

    model_config = ConfigDict(extra="allow")

    event: str
    time: float


class Measurement(BaseModel):
    """Fields every evaluation and update line carries."""

    model_config = ConfigDict(extra="allow")

    time: float
    step: int = Field(ge=0)
    responses: int = Field(gt=0)
    reward: float
    response_tokens_mean: float
    truncated_fraction: float = Field(ge=0, le=1)


class Evaluation(Measurement):
    """Held-out sampling with the policy after ``step`` accepted updates."""

    event: Literal["evaluation"]


class Update(Measurement):
    """The batch that produced accepted optimizer update ``step``."""

    event: Literal["update"]
    seconds_since_previous_update: float | None = None


class PublishedEvaluation(BaseModel):
    """A point read from an external execution's published chart."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["evaluation"]
    step: int = Field(ge=0)
    reward: float | None = None
    response_tokens_mean: float | None = None


SCHEMAS: dict[str, type[BaseModel]] = {"evaluation": Evaluation, "update": Update}


def read_output(path: Path, *, published: bool = False) -> list[dict]:
    """Validate an output's complete lines; a run interrupted mid-write ends in a partial one."""
    lines = path.read_text().splitlines(keepends=True)
    records = [json.loads(line) for line in lines if line.endswith("\n")]
    for record in records:
        if published:
            PublishedEvaluation.model_validate(record)
        else:
            SCHEMAS.get(record.get("event"), Record).model_validate(record)
    return records


@contextlib.contextmanager
def _lock(path: Path, *, blocking: bool = True) -> Generator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ValueError(f"Another run holds {path}") from exc
        yield


def _check_available(index: Path, entry: ReferenceRun) -> str:
    content = index.read_text() if index.exists() else ""
    catalog = ReferenceRunIndex.model_validate(tomllib.loads(content) or {"runs": []})
    output = (index.parent / entry.output).resolve()
    for run in catalog.runs:
        if run.id == entry.id or (index.parent / run.output).resolve() == output:
            raise ValueError(f"Run ID or output already indexed: {entry.id}")
    if not catalog.runs:
        content = (
            "\n".join(
                line for line in content.splitlines() if line.lstrip().startswith("#")
            )
            + "\n"
        )
    return content


@contextlib.contextmanager
def record_run(
    index: Path, entry: ReferenceRun, *, indexed: bool = True
) -> Generator[Path]:
    index = index.resolve()
    output = (index.parent / entry.output).resolve()
    run_key = hashlib.sha256(entry.id.encode()).hexdigest()
    index_lock = index.with_name(f".{index.name}.lock")
    with (
        (
            _lock(index.with_name(f".{index.name}.{run_key}.lock"), blocking=False)
            if indexed
            else contextlib.nullcontext()
        ),
        _lock(output.with_name(f".{output.name}.lock"), blocking=False),
    ):
        if indexed:
            with _lock(index_lock):
                _check_available(index, entry)
        if output.exists():
            raise FileExistsError(output)
        yield output
        if not output.is_file() or not output.stat().st_size:
            raise ValueError(f"Workload produced no metrics: {output}")
        if not indexed:
            return
        with _lock(index_lock):
            content = _check_available(index, entry)
            content += "\n[[runs]]\n" + "".join(
                f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
                for key, value in entry.model_dump(exclude_none=True).items()
            )
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=index.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(index)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
