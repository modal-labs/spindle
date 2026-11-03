# Capacity-validation server

A second team-wide Spindle deployment, like the [shared server](../shared-server/README.md), that runs only on B300 and only on one node: every trainer gets at most eight B300s, with no clustered launcher and no RDMA. Recipes run on a frontend named `cv` in the Modal environment `capacity-validation`, so its URL is `https://<workspace>-capacity-validation--cv.us-east.modal.run`. Like the shared server, it requires a Modal proxy token.

The environment is temporary, and this directory lives on a local branch rather than `main`. Unlike the bundled recipes, these B300 variants have not been performance tested: each recipe keeps its bundled parallelism, so most use fewer GPUs' worth of memory than B300 offers.

Each file in `recipes/` subclasses a bundled recipe, renames it with a `cv-` prefix, and sets `trainer_gpu` and `inference_gpu` to `"B300"`. Nemotron-3 Super uses its bundled B300 layout, which trains on four GPUs. [The test](../../tests/test_capacity_validation_server.py) checks that every recipe stays on one node of eight or fewer B300s.

`recipes/` holds the 13 shared-server base models that already train on one node. The rest train on two or three nodes today and need a single-node layout first:

| Base model | Trainer on the shared server | One node of 8× B300 |
| --- | --- | --- |
| Qwen3.5-397B-A17B | 2 × 8 H200 | Likely fits; needs EP8 |
| Nemotron-3-Ultra-550B | 2 × 8 B300 | Likely fits; needs EP8 |
| DeepSeek-V3.1 | 2 × 8 H200 | Tight; needs EP8 |
| DeepSeek-V4.1-Flash and -1017 | 2 × 8 B300 | Unmeasured; 8K is likelier than 32K |
| GLM-5.3 (256K) | 2 × 8 B300 | Unlikely at 256K |
| Kimi-K2.6 | 3 × 8 H200 | Does not fit in BF16 |

## Set up

The environment needs the three secrets the [README](../../README.md#secrets) describes, `spindle-api`, `spindle-proxy` and `huggingface-secret`, with the proxy token allowed for `capacity-validation`.

Volumes belong to one environment. To avoid downloading about 6 TB of base weights again, copy the `spindle` environment's weights and kernel cache, file by file, across as many CPU containers as Modal will run:

```bash
uv run modal run -e capacity-validation examples/capacity-validation-server/copy_volumes.py
```

A rerun skips files that are already there, so it resumes an interrupted copy. Checkpoints and the snapshot bulletin hold the shared server's own state, so they are not copied.

## Deploy

List every recipe, because the argument list is the full set the frontend serves:

```bash
uv run spindle deploy examples/capacity-validation-server/recipes/*.py
```

## Use

As on the shared server, with this environment's URL and API key, and a proxy token allowed for `capacity-validation`:

```bash
modal workspace proxy-tokens allow <wk-token-id> capacity-validation
export TINKER_BASE_URL=https://<workspace>-capacity-validation--cv.us-east.modal.run
python examples/shared-server/with_proxy_auth.py -m tinker_cookbook.recipes.chat_sl.train \
    model_name=Qwen/Qwen3-4B base_url=$TINKER_BASE_URL
```

Check a deploy with the shared server's smoke test:

```bash
uv run python examples/shared-server/smoke.py --base-model Qwen/Qwen3-4B --expect-commit "$(git rev-parse HEAD)"
```
