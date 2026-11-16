"""Miles-pinned GLM model dependency with the release Bridge core preserved."""

from .miles_image import BRIDGE_IMPORT_CACHE_PATCH
from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image

# Scores and selects pooled keys one query block at a time. The pinned GLM
# indexer otherwise allocates 64 GiB of float32 scores at 256K tokens.
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

# Merged modal-projects/miles#31 on spindle-miles-v0.1.2.
MILES_COMMIT = "464efc9781bbf8edf2135f0857f5aa417685eae2"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1"
    f" https://github.com/modal-projects/miles.git {MILES_COMMIT}"
    f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
    # Miles owns the provider and runtime pins. The installer exposes only GLM
    # model packages from the dependency, retaining the release Bridge core.
    f"pip install -r {MILES_PATH}/docker/requirements-glm53.txt",
    f"python {MILES_PATH}/docker/install_glm53_provider.py",
    "python - <<'PY'\n"
    "import subprocess\n"
    "subprocess.run(['patch', '-p1', '-d', '/opt/miles-glm53-bridge'], input="
    + repr(KPOOL_QUERY_BLOCKS_PATCH)
    + ", text=True, check=True)\nPY\n",
    "python - <<'PY'\n"
    "from importlib.metadata import distribution\n"
    "import subprocess\n"
    "root = distribution('megatron-bridge').locate_file('')\n"
    "subprocess.run(['patch', '-p1', '-d', str(root)], input="
    + repr(BRIDGE_IMPORT_CACHE_PATCH)
    + ", text=True, check=True)\nPY\n",
).add_local_python_source("spindle", ignore=ignore_config_source)
