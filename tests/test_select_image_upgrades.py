import random
import re

import pytest

from scripts.select_image_upgrades import (
    Upgrade,
    candidates,
    family,
    newer_base_images,
    select,
)
from spindle.providers.modal.deployment_apps import SAMPLER_IMAGES, TRAINER_IMAGES


@pytest.mark.parametrize("registry", [TRAINER_IMAGES, SAMPLER_IMAGES])
def test_registries_list_releases_in_upgrade_order(registry: dict[str, object]) -> None:
    releases: dict[str, list[tuple[int, ...]]] = {}
    for name in registry:
        if match := re.match(r"[a-z]+-v([\d.]+)-", name):
            version = tuple(int(part) for part in match[1].split("."))
            releases.setdefault(family(name)[0], []).append(version)
    assert all(versions == sorted(versions) for versions in releases.values())


def test_variants_count_as_current_only_after_the_newest_base_image() -> None:
    newer = newer_base_images(
        dict.fromkeys(
            [
                "sglang-v0.5.21-aaaaaaa",
                "sglang-v0.5.21-bbbbbbb-glm",
                "megatron-1111111",
                "sglang-v0.5.23-ccccccc",
                "sglang-v0.5.23-ccccccc-glm",
            ]
        )
    )
    assert newer == {
        "sglang-v0.5.21-aaaaaaa": ["sglang-v0.5.23-ccccccc"],
        "sglang-v0.5.21-bbbbbbb-glm": ["sglang-v0.5.23-ccccccc"],
        "megatron-1111111": [],
        "sglang-v0.5.23-ccccccc": [],
        "sglang-v0.5.23-ccccccc-glm": [],
    }


def test_select_takes_the_furthest_behind_then_a_random_share() -> None:
    upgrades = [
        Upgrade(f"c{i}", "", behind, "", "", "", "", [])
        for i, behind in enumerate([1, 2, 1, 1, 1, 3])
    ]
    picks = {
        tuple(u.config for u in select(upgrades, 10, 0.3, random.Random(seed)))
        for seed in range(20)
    }
    assert all(pick[:2] == ("c5", "c1") and len(pick) == 3 for pick in picks)
    assert len({pick[2] for pick in picks}) > 1


def test_selection_reads_every_bundled_config() -> None:
    for upgrade in candidates():
        assert upgrade.target_trainer_image in TRAINER_IMAGES
        assert upgrade.target_inference_image in SAMPLER_IMAGES
