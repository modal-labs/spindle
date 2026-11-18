"""Miles v0.1.2 release image with Spindle's Miles fork branch for v0.1.2."""

import modal

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image
from .miles_v011_image import BRIDGE_IMPORT_CACHE_INSTALL

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.1.2"


# Pinned by commit: Modal caches this layer by command, so fetching the branch
# by name would keep serving an older head.
# Editing this rebuilds every image below; add a new builder instead.
def image_at(miles_commit: str) -> modal.Image:
    return release_image.run_commands(
        f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {miles_commit}"
        f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
        # Fails the build if the fork branch did not replace the release checkout.
        f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
    ).add_local_python_source("spindle", ignore=ignore_config_source)


# Editing this rebuilds every image below; add a new builder instead.
def glm53_image_at(miles_commit: str) -> modal.Image:
    """Miles-pinned GLM model dependency with the release Bridge core preserved."""
    return release_image.run_commands(
        f"git -C {MILES_PATH} fetch --depth=1"
        f" {MILES_REPOSITORY} {miles_commit}"
        f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
        # Miles owns the provider and runtime pins. The installer exposes only GLM
        # model packages from the dependency, retaining the release Bridge core.
        f"pip install -r {MILES_PATH}/docker/requirements-glm53.txt",
        f"python {MILES_PATH}/docker/install_glm53_provider.py",
        BRIDGE_IMPORT_CACHE_INSTALL,
    ).add_local_python_source("spindle", ignore=ignore_config_source)


# Scores the GLM sparse-attention indexer in query blocks to bound its memory.
KPOOL_QUERY_BLOCKS_PATCH = (
    "--- a/megatron/bridge/models/glm5_next/kpool_indexer.py\n"
    "+++ b/megatron/bridge/models/glm5_next/kpool_indexer.py\n"
    "@@ -287,6 +287,9 @@\n"
    "         BLOCK=_SELECT_BLOCK,\n"
    "     )\n"
    "     return out\n"
    "+\n"
    "+\n"
    "+_QUERY_BLOCK = 32768\n"
    " \n"
    " \n"
    " def kpool_select_topk(\n"
    "@@ -311,7 +314,38 @@\n"
    "     pool_base = pool_cu_seqlens[seq_indices].to(torch.int32)\n"
    "     local_positions = (token_ids - seq_token_base).to(torch.int32)\n"
    '     eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")\n'
    "+\n"
    "+    replay = _indexer_replay_manager()\n"
    "+    if pooled_k.shape[0] > 0 and (replay is None or not replay.enabled):\n"
    "+        from megatron.bridge.models.glm5.tilelang.tilelang_indexer_fwd import indexer_fwd_interface\n"
    " \n"
    "+        # Full [tokens, pools] float32 scores grow quadratically (64 GiB at 256K tokens), and\n"
    "+        # selection is per query row, so score and select one block of queries at a time.\n"
    "+        blocks = []\n"
    "+        for start in range(0, num_tokens, _QUERY_BLOCK):\n"
    "+            rows = slice(start, start + _QUERY_BLOCK)\n"
    "+            with torch.no_grad():\n"
    "+                block_logits = indexer_fwd_interface(\n"
    "+                    index_q[rows],\n"
    "+                    pooled_k,\n"
    "+                    head_weights[rows],\n"
    "+                    pool_base[rows].to(torch.int32),\n"
    "+                    (pool_base[rows] + eligible_pools[rows]).to(torch.int32),\n"
    "+                    clean_logits=True,\n"
    "+                )\n"
    "+            blocks.append(\n"
    "+                _select_expand_tail(\n"
    "+                    block_logits,\n"
    "+                    seq_token_base[rows],\n"
    "+                    pool_base[rows],\n"
    "+                    local_positions[rows],\n"
    "+                    index_topk,\n"
    "+                    kpool,\n"
    "+                )\n"
    "+            )\n"
    "+            del block_logits\n"
    "+        return torch.cat(blocks).unsqueeze(1)\n"
    "+\n"
    "     if pooled_k.shape[0] > 0:\n"
    "         # TileLang is only needed for the scoring kernel; keep it out of module import so the\n"
    "         # packed-sequence helpers (pool_boundaries, ...) stay importable without it.\n"
    "@@ -329,7 +363,6 @@\n"
    "     else:\n"
    '         pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)\n'
    " \n"
    "-    replay = _indexer_replay_manager()\n"
    "     if replay is not None and replay.enabled:\n"
    "         topk_fn = replay.get_topk_fn(\n"
    "             _pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool),\n"
)
KPOOL_QUERY_BLOCKS_INSTALL = (
    "python - <<'PY'\n"
    "import subprocess\n"
    "subprocess.run(['patch', '-p1', '-d', '/opt/miles-glm53-bridge'], input="
    + repr(KPOOL_QUERY_BLOCKS_PATCH)
    + ", text=True, check=True)\nPY\n"
)


# Editing this rebuilds every image below; add a new builder instead.
def glm53_kpool_image_at(miles_commit: str) -> modal.Image:
    return release_image.run_commands(
        f"git -C {MILES_PATH} fetch --depth=1"
        f" {MILES_REPOSITORY} {miles_commit}"
        f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
        f"pip install -r {MILES_PATH}/docker/requirements-glm53.txt",
        f"python {MILES_PATH}/docker/install_glm53_provider.py",
        BRIDGE_IMPORT_CACHE_INSTALL,
        KPOOL_QUERY_BLOCKS_INSTALL,
    ).add_local_python_source("spindle", ignore=ignore_config_source)


# Each series lists fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
image_01_f2f8447 = image_at("f2f8447c33ac0755a0988c30d413d334dea0a26a")
# Branch head: the GLM-5.3 provider hooks, inert without the glm53 build steps, and per-adapter seeds through load_slot.
image_02_99e1ebe = image_at("99e1ebe8e86974e0b3c290f47f415ee47726f965")

# Merged modal-projects/miles#31 on spindle-miles-v0.1.2.
glm53_image_01_464efc9 = glm53_image_at("464efc9781bbf8edf2135f0857f5aa417685eae2")
# Merged modal-projects/miles#43: per-adapter seeds through load_slot.
glm53_image_02_99e1ebe = glm53_image_at("99e1ebe8e86974e0b3c290f47f415ee47726f965")
# Merged modal-projects/miles#43, plus the blocked GLM indexer.
glm53_kpool_image_01_99e1ebe = glm53_kpool_image_at(
    "99e1ebe8e86974e0b3c290f47f415ee47726f965"
)
