# Python deployment configs

A recipe subclasses `BaseConfig` and exports `config = Config()`. Settings are flat, untyped Python attributes. Backend options are ordinary dictionaries.

```python
from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "my-9b"
    model = "Qwen/Qwen3.5-9B-Base"
    max_context_length = 16384
    backend = "miles"
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_cpu = 16
    trainer_memory_mib = 65536
    trainer_max_clients_per_instance = 6
    inference_gpu = "H200"
    inference_max_replicas = 8
    miles_cfg = {
        "model_type": "qwen3.5-9B",
        "tensor_model_parallel_size": 4,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head",
        ],
    }
    sglang_cfg = {"max_running_requests": 32}


config = Config()
```

See the [9B LoRA recipe](../src/spindle/configs/qwen35_9b_lora_16k.py) and [4B FFT recipe](../src/spindle/configs/qwen35_4b_fft_64k.py) for complete examples. `spindle deploy` downloads weights to `/assets/<model id>` in the background; requests that arrive before the download finishes wait for it.

## Variants

Override ordinary attributes directly. Use dotted `overrides` to change individual backend options:

```python
from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "my-9b-more-memory"
    trainer_memory_mib = 98304
    overrides = {"miles_cfg.max_tokens_per_gpu": 8192}


config = Config()
```

Each parent's settings and overrides apply before its child's. Constructor fields and overrides apply last: `Config(trainer_gpu="H200", overrides={"sglang_cfg.max_running_requests": 16})`. Assigning a dictionary or list replaces that value; `trainer_env = {}` clears inherited environment settings. Instances own independent copies of mutable values and can also be edited directly.

`trainer_gpu` and `inference_gpu` take one GPU type or a list in preference order, such as `["H200", "B300"]`. Modal schedules the first type with capacity. GPU counts and parallelism settings apply to every entry, so list only types the recipe fits on.

When several clients share a trainer, it trains all of their queued batches together by default, and each client waits for the whole step to finish. For RL, where clients sample at different times, set `trainer_max_clients_per_batch = 1`. The trainer then runs one client at a time, so each client can go back to sampling as soon as its own step is done.

Sampler replicas scale between `inference_min_replicas` and `inference_max_replicas`. Each sampled sequence counts as one request, so `num_samples=4` counts four times. By default another replica starts once every replica holds `sglang_cfg.max_running_requests` requests. Set `inference_target_concurrency` lower to scale out earlier. Idle replicas stop after `inference_scaledown_window_s`, so replicas added in one sampling round stay up for the next.

`inference_scaleup_window_s` delays scale-out until load has stayed above the target that long. Leave it unset for training loops that sample in repeated rounds. See the next section for when to set it.

### Older Qwen3.5 LoRA checkpoints

The Qwen3.5-2B, Qwen3.5-4B and Qwen3.5-9B-Base LoRA recipes now also train the linear-attention layers. A checkpoint saved by these recipes before that change has no adapters for those layers, so loading it fails with `checkpoint LoRA target_modules do not match the deployment`.

To keep training or sampling from such a checkpoint, deploy a variant with the old target list and the old sampler image:

```python
from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen35-4b-lora-32k-attn-only"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    overrides = {
        **Parent.overrides,
        "miles_cfg.megatron_to_hf_mode": "bridge",
        "miles_cfg.lora_type": "lora",
        "miles_cfg.target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
    }


config = Config()
```

Use the same pattern for the other recipes by changing the parent and `name`. On a running server, add it with `spindle add`.

A checkpoint always restores on the config that saved it, so `create_training_client_from_state` still sends an old checkpoint to the updated recipe. Load it into the variant instead:

1. Create a training client with `base_model` set to the variant's name, and the same rank as the checkpoint.
2. Call `load_state` or `load_state_with_optimizer` with the old checkpoint path.
3. To sample from it, call `save_weights_for_sampler` on that client and use the new sampler path. Sampler paths saved before the change also route to the updated recipe.

To train the linear-attention layers, start a new run on the bundled recipe. An old checkpoint can't be extended with the new adapters.

### Older Qwen3.5-9B instruct 16K checkpoints

`qwen35-9b-instruct-lora-16k` used to train on eight H100s with tensor parallel size 8. It now trains on one B200. A checkpoint loads only on the trainer layout, checkpoint format (Miles v0.1.x or v0.2.0) and adapter type that saved it. Load a checkpoint saved with Spindle v0.1.7 or earlier into this variant, following the steps above:

```python
from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-instruct-lora-16k-h100"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    trainer_gpu = "H100"
    trainer_gpus_per_node = 8
    overrides = {
        **Parent.overrides,
        "miles_cfg.megatron_to_hf_mode": "bridge",
        "miles_cfg.lora_type": "lora",
        "miles_cfg.tensor_model_parallel_size": 8,
    }


config = Config()
```

## Spec decoding

We have certain configs with speculative decoding using public DFlash checkpoints, which we finds increases rollout tps by 2-3x depending on workload. Note that this throughput gain is dependent on accept length which can differ based on your training domain, # of steps in RL, and sampling params. 

The following LoRA recipes enable speculative decoding by default:

| Recipe | Method | Draft checkpoint |
| --- | --- | --- |
| `qwen35-4b-lora-32k` | Native MTP (`NEXTN`) | Reuses the target checkpoint |
| `qwen3-8b-lora-32k` | DFlash | `z-lab/Qwen3-8B-DFlash-b16` |
| `qwen35-9b-instruct-lora-16k` | DFlash | `modal-labs/Qwen3.5-9B-DFlash` |
| `qwen35-9b-instruct-lora-32k` | DFlash | `modal-labs/Qwen3.5-9B-DFlash` |
| `qwen38-27b-lora-32k` | DFlash | `incoai/Qwen3.8-27B-DFlash2` |
| `glm53-flash-lora-32k` | DFlash | `modal-labs/GLM-5.3-Flash-DFlash` |
| `nemotron35-lightning-30b-a3b-lora-64k` | Native MTP (`NEXTN`) | Reuses the target checkpoint |

Deploy a recipe by passing its Python path to `spindle deploy`, for example:

```sh
spindle deploy src/spindle/configs/qwen35_4b_lora_32k.py
```

For a separate drafter, set `inference_draft_model` and
`inference_draft_model_revision`, and select the algorithm in `sglang_cfg`.
Spindle downloads the draft alongside the target and supplies its local path to
SGLang. With `NEXTN` and no separate draft, Spindle supplies the resolved target
checkpoint instead. The GLM recipe retains its parent's BF16 target weights and
eight inference GPUs. Reference runs and benchmark artifacts are maintained
separately from these deployment recipes.

## Reopening a named sampling checkpoint on a fresh deployment

Named exports saved with this version persist their metadata on
`platform.storage.bulletin` (default: `spindle-snapshot-bulletin`). A path such as
`tinker://<run>/sampler_weights/eval` resolves to
`/bulletin/sampler_weights/<run>/eval/metadata.json`. This immutable metadata
identifies the base model, engine definition, and exact snapshot version; the
adapter weights remain in the existing bulletin snapshot directory.

```python
# Save once on the training deployment; wait for durable publication.
checkpoint = trainer.save_weights_for_sampler("eval").result()

# Later, connect to a fresh deployment using the same bulletin volume.
service = tinker.ServiceClient(base_url=new_url, api_key=api_key)
sampler = service.create_sampling_client(model_path=checkpoint.path)
```

The new deployment must use the same Modal workspace/environment and bulletin
volume, have access to the base weights, and serve the checkpoint's engine
definition. No original model session, trainer, or artifacts Dict entry is
required for LoRA sampling. New named exports write their metadata only to the
volume, and export retries read it there too. Expiration still applies when the
export has a TTL.
FFT publications retain their existing dependencies on the base and preceding
delta snapshots. Ephemeral latest clients retain their existing lifecycle.

Older exports without volume metadata continue to resolve through their original
artifacts Dict; retrying the original named export with the updated server writes
its durable metadata. A training checkpoint (`tinker://<run>/weights/<name>`,
created by `save_state`) still needs to be restored and exported for sampling.
The Tinker SDK interface is unchanged. Servers predating this storage format
cannot reopen new named exports after a server downgrade.

## Recommended configuration for eval jobs

The defaults suit training loops. An eval sends one short batch, and any replica it starts takes a few minutes to load, so it arrives after the work is done and then sits idle. For eval-only configs:

- Give one replica more slots, so it can take the whole batch.
- Wait 120 seconds before adding a replica.
- Cap replicas at 1 or 2.
- Stop idle replicas after 60 seconds.

```python
from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-eval"
    inference_max_replicas = 2
    inference_scaleup_window_s = 120
    inference_scaledown_window_s = 60
    overrides = {
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 128,
    }


config = Config()
```

On a running server, add this as its own config with `spindle add` and pass its name as `base_model`.

## Backend options

| Setting | Consumed by |
| --- | --- |
| `trainer_*`, `inference_*` | Modal GPU/CPU/memory allocation, scaling, timeouts and Spindle admission limits |
| `megatron_cfg` | Existing Megatron `EngineModelConfig`; provider, optimizer and distributed options use its native dictionaries |
| `miles_cfg` | Existing `MilesBackendConfig`; `cli_options` supplies additional Miles arguments |
| `sglang_cfg` | SGLang `ServerArgs` |

Modal and backend libraries validate their own options. Spindle checks integration requirements such as trainer slot capacity, supported training modes, and parallelism agreeing with allocated GPUs. It supplies managed model paths, context length and adapter settings; conflicting backend overrides are rejected.

`BaseConfig` does not enforce field types or reject arbitrary attributes. Extra backend options belong in the corresponding backend dictionary. A misspelled top-level attribute is ordinary Python data and may be unused.

Miles argument conversion lives in [miles_arguments.py](../src/spindle/backends/miles_arguments.py). SGLang receives `ServerArgs(**settings)` in its [worker entrypoint](../src/spindle/inference/sglang.py). Backend libraries validate native options when workers start; frontend config imports remain CPU-only.

### Miles options

Besides the LoRA settings, `miles_cfg` accepts `megatron_to_hf_mode` (`bridge`, the default, or `raw`), `lora_type` (`lora`, the default, or `canonical_lora`) and `dynamic_batch_size`.

For Miles native canonical multi-LoRA, set `trainer_image = "miles-v0.2.0-e814245"`, `miles_cfg.megatron_to_hf_mode = "raw"` and `miles_cfg.lora_type = "canonical_lora"`. Miles builds the model and attaches independent adapter factors to each Q/K/V and MLP gate/up projection. It still uses Megatron-Core; model loading in raw mode uses Miles' mbridge integration instead of Megatron-Bridge's model provider. The single-node dense Qwen2.5, Qwen3, Qwen3.5, Qwen3.8 and Nemotron-3 Nano LoRA recipes use this path. Nano keeps expert reductions in FP32 and enables deterministic backward and collective kernels.

Native slots support dense projections, the output head and grouped expert projections on registered Miles architectures. GDN `in_proj` and GLM-5.3 KDA slots require context parallel size 1. Miles rejects unsupported targets or model architectures; the other bundled recipes keep their existing adapter implementation.

### Images

`trainer_image` and `inference_image` select the trainer and sampler images. Each takes a built-in name or a `module:attribute` path to a `modal.Image`. The names are listed in [deployment_apps.py](../src/spindle/providers/modal/deployment_apps.py): `miles-v0.1.1-52a7a99`, `miles-v0.1.2-99e1ebe`, `miles-v0.1.2-99e1ebe-glm53`, `miles-v0.1.2-99e1ebe-kpool-glm53`, `miles-v0.2.0-8c5de26`, `miles-v0.2.0-e814245`, `megatron-2d1fa8d` and `megatron-2d1fa8d-fla06` for trainers, and `sglang-v0.5.21-a3e0e17`, `sglang-v0.5.23-c4182b2`, `sglang-v0.5.23-c4182b2-glm53` and `sglang-v0.5.23-e003cf9` for samplers. A name is the upstream release followed by the short commit of the Spindle fork that the image installs. A suffix such as `-glm53` marks extra model-specific build steps. The Megatron image has no release, so its name has only the Megatron-LM commit. `miles-v0.1.1-52a7a99`, `miles-v0.1.2-99e1ebe`, `miles-v0.1.2-99e1ebe-glm53`, `miles-v0.1.2-99e1ebe-kpool-glm53` and `miles-v0.2.0-8c5de26` support seeded LoRA initialization (`create_lora_training_client(seed=...)`), and `miles-v0.2.0-8c5de26` also adds Beam LoRA support. `miles-v0.1.1-fdd808c`, `miles-v0.1.2-f2f8447` (DeepSeek-V4.1) and `miles-v0.1.2-464efc9-glm53` (GLM-5.3 Flash) are the images Spindle 0.1.7 and earlier trained on. No bundled config uses them, and states saved on them restore on the newer image of the same release. Use a `module:attribute` path for an image you define in your own code.

Every bundled config sets both images itself rather than inheriting them, so upgrading SGLang or Miles for one config doesn't change any other. If you leave a field empty in your own config, it uses the backend default (`miles-v0.1.1-52a7a99`, `megatron-2d1fa8d` or `sglang-v0.5.23-c4182b2`), and that default can change in a later Spindle release. Set both fields to keep a config on the images it was tested with.

#### Changing an image

Never change a named image in place. That includes its fork commit, base image, packages and build steps. Configs that use the name were validated on its exact contents, and an edit would change them all at once without a test.

To change an image:

1. Add a new image under a new name, in a new module if the build differs. Name it after the release and the fork commit it installs, such as `sglang-v0.5.24-<commit>`. If only the build steps change, add a short suffix that says what changed.
2. Move configs to the new name one at a time. Record a reference run for each config you move.
3. Once no bundled config, default or reference run uses the old name, remove it and the module that builds it.

`miles-v0.2.0-8c5de26` supports dense native adapters; `miles-v0.2.0-e814245` also supports routed expert adapters and Nemotron-3 Nano. On the built-in v0.2.0 images, use `raw` with `canonical_lora` or `bridge` with `lora`. Spindle rejects Bridge with `canonical_lora`, which has not been validated here. The older Miles images support the fused `lora` layout. Multi-node trainers gather captures through the shared checkpoint volume.

A checkpoint restores on any trainer with the same checkpoint format and adapter layout as the one that saved it. The Miles v0.1.x and v0.2.0 images use different formats. Checkpoints without a recorded layout are treated as `lora`. To restore an older checkpoint after a recipe switched to native canonical adapters, use a Miles v0.1.x image and the `lora` layout. Restoring on a different Miles commit of the same format logs a `spindle_checkpoint_revision_mismatch` event.

## Prepare and launch

~~~text
load(config.py) → recipe: BaseConfig
  → DeploymentConfig.create(recipe)
  → deploy named trainer and inference apps if they are missing or changed
  → deploy the frontend with the current config set
~~~

The launcher consumes the saved settings. It does not reparse backend configuration or add another set of backend defaults. JSON decoding in a trainer reconstructs the saved config, without importing the author's config file.

| File | Responsibility |
| --- | --- |
| [configuration.py](../src/spindle/configuration.py) | BaseConfig defaults, inheritance and overrides |
| [deployments.py](../src/spindle/deployments.py) | Python recipe loader and DeploymentConfig |
| [backends/deployment.py](../src/spindle/backends/deployment.py) | Backend settings used when creating a DeploymentConfig |
| [megatron_runtime/common/settings.py](../src/spindle/backends/megatron_runtime/common/settings.py) | Shared Megatron ownership rules and constructor dictionaries |
| [deployment_apps.py](../src/spindle/providers/modal/deployment_apps.py) | Trainer, inference, and pool app builders |
| [deployment_configs.py](../src/spindle/providers/modal/deployment_configs.py) | Read saved configs and frontend platform settings |
| [deployment_cli.py](../src/spindle/deployment_cli.py) | Deploy trainer and inference apps by calling the builders, then deploy the frontend |

## Multi-node Miles

[qwen38_27b_lora_256k.py](../src/spindle/configs/qwen38_27b_lora_256k.py) configures two nodes with eight H200s per node, TP2 and CP8. The trainer app uses Modal's clustered launcher and RDMA. Every rank mounts the same volumes; rank 0 starts the engine, and the other ranks join Ray using the launcher merged in #39. The driver receives the Ray address. GPU counts cannot be independently overridden through Miles options.

This path has CPU construction/topology tests. This refactor has not been redeployed or tested on multiple GPU nodes.

## Deploy and update

~~~bash
spindle config init --preset qwen35-9b-lora-16k > my_model.py
spindle config validate my_model.py
spindle deploy my_model.py
~~~

Bundled configs deploy by name, e.g. `spindle deploy qwen35-9b-lora-16k gpt-oss-20b-lora-32k`, and `spindle deploy --all` deploys every bundled config. With no arguments in a terminal, `spindle deploy` asks which models to deploy; without a terminal it exits instead of deploying.

Validation loads the recipes and checks that they can share one frontend. Backend integration settings are checked when creating a DeploymentConfig at deploy time. Native backend options are checked at engine startup.

The checked-in [deploy_models.sh](../scripts/deploy_models.sh) lists the complete active config set. Add a config path there, then run it. The deployment command owns frontend selection and trainer/inference updates:

~~~bash
./scripts/deploy_models.sh --app my-spindle --env dev
./scripts/deploy_models.sh --refresh-trainer qwen35-9b-lora-16k
./scripts/deploy_models.sh --refresh-inference qwen35-9b-lora-16k
~~~

Frontend, region, environment, secret names, and volume names default in `BaseConfig.platform`. `platform.modal.region` defaults to `None`, which leaves GPU placement unpinned (Modal schedules trainers and inference pools in any region); set it only to restrict placement. The frontend server, which runs on CPU, is placed in that region, or in `us-east` when it is unset; this never affects where GPUs are scheduled. Set them in the same Python config using dotted overrides:

```python
from spindle.configs.gpt_oss_20b_lora_32k import Config as Parent


class Config(Parent):
    overrides = {
        "platform.frontend": "my-spindle",
        "platform.modal.environment": "dev",
        "platform.modal.region": "us-east",
        "platform.secrets.api": "my-api-secret",
        "platform.storage.checkpoints": "my-checkpoints",
    }


config = Config()
```

Set `"platform.modal.exclude_reserved_capacity": True` to keep multi-node trainers, and trainers with `trainer_rdma`, off GPUs reserved for your workspace. This leaves the reservation free for other jobs. Modal accepts the option only on clustered functions, so other trainers and inference pools are unaffected.

The frontend's URL is `https://<workspace>-<environment>--<frontend>-server.modal.run`. To choose the part after `--`, set `"platform.url_label"`; for example `"prod"` gives `https://<workspace>-<environment>--prod.modal.run`. Changing the label changes the URL clients use.

Set `"platform.requires_proxy_auth": True` when the environment blocks unauthenticated URLs, or to keep the server off the open internet. Modal then rejects any request without a [proxy token](https://modal.com/docs/guide/webhook-proxy-auth) allowed for the environment, before it reaches Spindle. Clients send the token as headers next to their Tinker API key:

```python
service = tinker.ServiceClient(
    default_headers_overrides={"Modal-Key": token_id, "Modal-Secret": token_secret}
)
```

Tinker SDKs older than 0.36 call this argument `default_headers`. Scripts that create their own `ServiceClient`, such as tinker-cookbook's, can run through [`with_proxy_auth.py`](../examples/shared-server/with_proxy_auth.py), which adds the headers to every client the script creates.

All configs deployed together must share platform settings. Use a shared parent config when deploying several models. Explicit `--app`, `--env`, and `--region` flags override the corresponding recipe settings for that invocation. Credentials remain in Modal secrets; recipes contain their names.

One frontend serves all models through Tinker’s `base_model`. If several recipes serve the same model, requests by model name go to the one with the longest context. To use a specific recipe, pass its name as `base_model`. `get_server_capabilities` lists every recipe and its context length.

`GET /api/v1/spindle/queues` describes each recipe's training queue. Each trainer holds up to `trainer_max_clients_per_instance` training clients, and a recipe runs at most `trainer_max_instances` trainers. When every slot is in use, a new client waits in line instead of failing. The client that has waited longest gets the next free slot. It keeps waiting until a slot frees, with no time limit.

```json
{"name": "qwen35_2b_lora_32k", "base_model": "Qwen/Qwen3.5-2B", "parameterization": "lora",
 "backlog": {"count": 1, "oldest_age_s": 42.0},
 "slots": {"used": 6, "free": 0, "per_trainer": 6},
 "trainers": {"running": 1, "starting": 1, "max": 4},
 "recent": {"window_s": 3600.0, "added": 4, "placed": 3, "released": 2,
            "wait_s": {"p50": 12.0, "p95": 80.0, "max": 95.0}}}
```

- `backlog` counts training clients waiting for a slot, and how long the oldest has waited.
- `slots` counts clients on running trainers and the open slots left on them.
- `trainers` counts trainers that are running, trainers that are still booting, and the most the recipe allows.
- `recent` covers the last `window_s` seconds. `added` counts clients that asked for a slot, `placed` counts clients that got one, and `released` counts slots that opened up. A slot opens when a client unloads its model or its session closes or goes idle. Models lost when a trainer stops are not counted, since no slot opens for anyone else. `wait_s` gives the median, 95th percentile and longest wait of the clients placed in the window, or `null` if none were placed.

A client can start right away when `slots.free` is greater than `backlog.count`. Otherwise it waits for a trainer to boot, or for another client to finish or go idle for `session_idle_timeout_s`. While it waits, the Tinker SDK shows its place in line, such as `all trainer slots for Qwen/Qwen3.5-2B are in use; waiting for one to free up (2nd in line, 1m 10s so far)`. A client that stops waiting, for example because its script exited, loses its place within 30 seconds. The queue response and each frontend's view of the line can be a few seconds old, so a new client can occasionally take a slot just as it frees up.

When no slot is free and no trainer is booting, `recent.released` shows how fast slots open up. For example, 4 slots released in the last hour means a slot opens about every 15 minutes. `backlog.count` shows how many other clients are waiting for those slots.

## The frontend registry

Every recipe a frontend serves is one `<app_id>/<recipe>.json` config file in the shared `spindle-deployments` Modal Volume, under a folder named after the frontend's app ID. `spindle deploy` deploys the trainer and inference apps that are missing or changed, then writes the listed configs to the registry and publishes the frontend with only its platform and lifecycle settings. The frontend reads the registry, so recipes not listed in a deploy keep serving. A deploy that changes platform settings must list every recipe the frontend serves, since unlisted ones would keep apps built for the old settings.

Upgrading a frontend that still ships a configs file needs no extra step: `spindle deploy` on the new version writes every recipe the old frontend serves to the registry before the swap, so nothing drops out and unchanged apps are not deployed again.

## Add or remove a config on a running frontend

Anyone with the frontend's API key can add and remove configs. There are two ways to add configurations: SDK or REST API.

### SDK

~~~bash
uv run spindle add qwen35-9b-lora-16k-h200 --base qwen35-9b-lora-16k \
  --trainer_gpu=H200 --miles_cfg.max_tokens_per_gpu=8192
~~~

`spindle add` waits until the config's apps are ready. To remove a config:

~~~bash
uv run spindle remove qwen35-9b-lora-16k-h200
~~~

If models still train on it, `spindle remove` lists them and prompts the user first; `--force` skips the interactive prompt.

### REST API

`POST /api/v1/spindle/deployments` writes a config into a running frontend's registry and deploys its trainer and inference apps in the background, without redeploying the frontend. Send a new `name` with either the deployed config to start from as `base` plus the settings to change as `overrides` (the same dotted keys as a config's `overrides`), or a full `recipe`:

~~~bash
curl -sS -i -X POST "$TINKER_BASE_URL/api/v1/spindle/deployments" \
  -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "qwen35-9b-lora-16k-h200",
       "base": "qwen35-9b-lora-16k",
       "overrides": {"trainer_gpu": "H200", "miles_cfg.max_tokens_per_gpu": 8192}}'
~~~

`DELETE /api/v1/spindle/deployments/{name}` removes a config and stops its trainer, inference, and sampler pool apps, and `GET /api/v1/spindle/deployments` lists them. If models still train on the config, `DELETE` returns HTTP 409 listing them; add `?force=true` to remove it anyway, which ends their training.

Configs deployed before this API are already registered and listed. To move a config kept in a Python file onto the API, send its fields as `recipe` (`vars(config)`).

Each listed config reports `apps` as `"deploying"`, `"ready"`, `"failed"` (with `apps_error`) or `"missing"`. Sending the same config again retries a failed deploy.

If you find a config that clearly beats the current baselines, please upstream it to Spindle as a bundled config.

## Update isolation

Trainer apps are named `{frontend}-trainer-{name}` and inference apps `{frontend}-inference-{name}`, where `{frontend}` is the frontend app name (`spindle` by default, or the `--app` value). An inference-only change reuses the trainer app. A trainer-only change reuses the inference app. Source-only upgrades use `--refresh-trainer` / `--refresh-inference`. Updating inference settings also redeploys existing pools for that recipe before publishing the new frontend configuration. An interrupted pool update can be retried with the same command. Changes to region, storage, or secrets refresh both worker apps.

Every app a deployment creates starts with its frontend name, so two deployments with different `--app` values can share one Modal environment and the same recipe names. Drain jobs before removing a recipe or making incompatible model, context, or adapter changes.

Trainers receive the frontend App ID at launch and use its shared registry for engine state and reconciliation. A startup failure pauses new starts for five minutes; the failure then expires, allowing subsequent requests and reconciliation to retry. `spindle deployment retry` clears it immediately.

The CLI reads the current configs straight from the frontend's registry folder to compare settings, and checks trainer and inference app existence directly with Modal. There is no retained history of earlier configs. Run deploy commands sequentially.

`--app`, `--env`, `--region`, secrets, and storage names are frontend/platform settings. They are not copied onto each recipe.

LoRA pools are named `{frontend}-lora-{name}`. FFT pools are named `{frontend}-fft-{session_digest}-latest` or `{frontend}-fft-{session_digest}-v{version}` to fit Modal’s 63-character limit. The digest identifies the recipe/client pair; it is not a config revision.

`spindle stop --app my-spindle --env dev` stops the frontend and every app whose name starts with `my-spindle-trainer-`, `my-spindle-inference-`, `my-spindle-lora-`, or `my-spindle-fft-`, ending any runs on them. Deployments made with `--app` on Spindle 0.1.5 or older named their worker and pool apps `spindle-...`; stop those with `modal app stop` after upgrading. It also deletes the stopped frontend's folder from the registry Volume, leaving other frontends' folders alone: the next `spindle deploy` under that name creates a new frontend app with an empty registry, so it serves only the configs it lists.

See [validation results](deployment-validation.md) for CPU coverage, live GPU checks, and their limits.


### Updating live LoRA adapters in place

To let active generations follow newer sampler weights, set
`inference_lora_update_in_place = True` and use
`inference_image = "spindle.providers.modal.live_lora_image:image_01_884f397"`.
For GLM Flash, use
`"spindle.providers.modal.live_lora_image:glm53_image_01_884f397"`.
This requires the adapter-update SGLang fork. It is off by default.

Clients from `save_weights_and_get_sampling_client()` use a minimum version.
When a request requires a newer version, the sidecar resolves the published
snapshot and calls `/update_lora_adapter` on one stable adapter per training
run. Active requests for that run then continue with the new weights. Requests
with older minimums can use the already loaded newer version. Publication alone
does not trigger an update: a request must require the new version.

Exact checkpoint pins remain immutable and use separate adapters. Use those
when a trajectory must stay on one policy. In-place updates retain sampled
tokens and their behavior logprobs, and rebuild only the affected requests' KV.
Other adapters keep their active KV, but upload bandwidth and replay compute
can still increase their latency. More frequent updates can cost more than
adapter consolidation saves. The sidecar reports the actual starting and ending
versions and retains SGLang's policy-version token spans in response metadata.

Use ordinary CUDA tensor-parallel serving. Pipeline/context parallelism,
attention DP, speculative decoding, sessions, beam search and hierarchical
cache are not supported by this endpoint. Rank, alpha, targets, tensor shapes
and dtypes must stay fixed. A failed preparation preserves the previous version;
a failed GPU rollback blocks that adapter until it is repaired.
