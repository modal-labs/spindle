"""Choose bundled configs to move to the newest trainer and sampler images.

Configs whose pins are furthest behind come first; ties are broken at random.
The script takes ceil(fraction * number of bundled configs) of the configs that
are behind, prints them as JSON, and lists the reference-run validation
definitions that already deploy each one.

    uv run --no-sync python scripts/select_image_upgrades.py --fraction 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from spindle.deployments import config_path, load
from spindle.providers.modal.deployment_apps import SAMPLER_IMAGES, TRAINER_IMAGES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = config_path("qwen3-8b-lora-32k").parent
IMAGE_NAME = re.compile(
    r"(?P<family>[a-z]+)-(?:v[\d.]+-)?[0-9a-f]{7,40}(?P<variant>-[a-z0-9-]+)?"
)


@dataclass
class Upgrade:
    config: str
    path: str
    images_behind: int
    trainer_image: str
    inference_image: str
    target_trainer_image: str
    target_inference_image: str
    validations: list[str]


def family(name: str) -> tuple[str, bool]:
    """The image's runtime family, and whether it is a model-specific variant."""
    match = IMAGE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"image name {name!r} is not <runtime>-[v<release>-]<commit>")
    return match["family"], match["variant"] is not None


def newer_base_images(registry: dict[str, object]) -> dict[str, list[str]]:
    """For each image, the base images of its family registered after it."""
    names = list(registry)
    return {
        name: [
            later
            for later in names[index + 1 :]
            if family(later) == (family(name)[0], False)
        ]
        for index, name in enumerate(names)
    }


def validations(config: str) -> list[str]:
    """Reference-run validation definitions that deploy this config."""
    return sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "reference_runs" / "configs").glob("*.toml")
        if tomllib.loads(path.read_text())["config"] == config
    )


def candidates(config_dir: Path = CONFIG_DIR) -> list[Upgrade]:
    trainer_newer = newer_base_images(TRAINER_IMAGES)
    sampler_newer = newer_base_images(SAMPLER_IMAGES)
    upgrades = []
    for path in sorted(config_dir.glob("[!_]*.py")):
        recipe = load(path)
        trainer = trainer_newer[recipe.trainer_image]
        sampler = sampler_newer[recipe.inference_image]
        if not trainer and not sampler:
            continue
        upgrades.append(
            Upgrade(
                config=recipe.name,
                path=str(path.relative_to(REPO_ROOT)),
                images_behind=max(len(trainer), len(sampler)),
                trainer_image=recipe.trainer_image,
                inference_image=recipe.inference_image,
                target_trainer_image=trainer[-1] if trainer else recipe.trainer_image,
                target_inference_image=sampler[-1]
                if sampler
                else recipe.inference_image,
                validations=validations(recipe.name),
            )
        )
    return upgrades


def select(
    upgrades: list[Upgrade], total: int, fraction: float, rng: random.Random
) -> list[Upgrade]:
    """Most images behind first, random among ties, ceil(fraction * total) at most."""
    order = sorted(upgrades, key=lambda u: (-u.images_behind, rng.random()))
    return order[: math.ceil(fraction * total)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    total = len(list(CONFIG_DIR.glob("[!_]*.py")))
    chosen = select(candidates(), total, args.fraction, random.Random(args.seed))
    print(json.dumps([asdict(upgrade) for upgrade in chosen], indent=2))


if __name__ == "__main__":
    main()
