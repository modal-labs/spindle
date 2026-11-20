# Changelog

Notable changes to Spindle, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [semantic versioning](https://semver.org/). Before 1.0, a minor release (0.x.0) can change config defaults or deployment behavior; a patch release (0.x.y) only fixes bugs. Each release is tagged `vX.Y.Z` and published as a GitHub Release.

## 0.1.9 - 2026-11-19

New LoRA recipes train Kimi K3, Beam 501B-A23B, GLM-5.3 Flash at 256K context and DeepSeek-V4.1-Flash-1017 at 64K context. Miles LoRA training states now restore after a trainer image update that keeps the same checkpoint format, and removing a config stops its sampler pools.

### Added

- A guess-the-number reference workload trains through the stock Tinker SDK and records game reward, success rate, timings and sampling/training KL. Its typed workload settings include the tokenizer pins and total steps. An optional save/train-ahead/restore schedule tests checkpoint continuation with optimizer state. Plots distinguish retained and discarded updates. An evaluation-only mode repeats the same seeded step-0 evaluation and plots its distribution across image pairs.

- A two-node Qwen3.8-27B DAPO-Math reference definition uses the shared Tinker workload.
- DAPO-Math references validate checkpoint restoration by replaying an Adam momentum step and sampling the restored adapter after the measured workload.
- `glm53-flash-lora-256k`: GLM-5.3 Flash LoRA with 256K-token context on one 8×B300 node, with expert-routing replay. It trains on a new image, `miles-v0.1.2-99e1ebe-kpool-glm53`, which changes GLM's sparse-attention indexer so 256K sequences fit in GPU memory.
- `deepseek-v41-flash-1017-lora-64k`: DeepSeek-V4.1-Flash-1017 LoRA with 64K context. Like `deepseek-v41-flash-lora-64k`, it trains on B300 or B200.
- `kimi-k3-lora-32k` trains LoRA adapters on Moonshot AI Kimi K3 from the MXFP4 release weights, with a 2-node B300 trainer and a 1-node B300 sampler. Adapters train attention, the dense first layer and the shared experts; routed experts stay frozen.
- `beam-501b-a23b-lora-5k` trains `reflection/Beam-1.0-501B-A23B` with BF16 LoRA on two eight-H200 trainer nodes and samples on one eight-B300 node. Attention and MLP gates are separate LoRA targets, and training uses routing replay. It trains on `miles-v0.2.0-8c5de26` and samples on the new `sglang-v0.5.23-568ab74` image, which serves Beam LoRA including its attention output gate. Its trainers use local compiler caches, like `qwen38-27b-lora-256k`.
- The reference-run catalog has a Beam definition, `beam_blog_repro.toml`, that trains on DAPO-Math and evaluates on AIME25 to reproduce the published Beam LoRA run. The index holds an earlier 42-update execution and the published curve.
- Every reference-run output follows one format, validated by `reference_runs/catalog.py` and described in `reference_runs/README.md`. Index entries for executions outside Spindle record `source` instead of Spindle provenance.

### Changed

- Reference-run index entries now record the configured sampler image in `inference_image` and no longer infer `miles_commit` from the trainer image name.
- `gpt-oss-20b-lora-32k` trains on `miles-v0.2.0-e814245` and samples on `sglang-v0.5.23-e003cf9`. To restore a checkpoint saved on its previous Miles v0.1.1 image, set `trainer_image` back to `miles-v0.1.1-52a7a99`; Miles v0.2.0 uses a different checkpoint format.

- `qwen35-9b-instruct-lora-16k` trains on one B200 instead of eight H100s, and samples on B200 only, with the `modal-labs/Qwen3.5-9B-DFlash` drafter. It no longer falls back to H200 samplers. Older checkpoints also need the old eight-H100 layout; see "Older Qwen3.5-9B instruct 16K checkpoints" in docs/deployment-configs.md.
- `qwen38-27b-lora-256k` uses `miles-v0.2.0-8c5de26` and local trainer compiler caches to avoid corrupt generated modules observed with shared caches. Checkpoints saved with the previous `miles-v0.1.1-52a7a99` default cannot restore with the new image. Set `trainer_image` back to `miles-v0.1.1-52a7a99` to restore those checkpoints.
- Miles LoRA checkpoints restore on a trainer running a different Miles commit, as long as the checkpoint format, adapter layout, base model and LoRA settings match. Before, any commit change rejected the restore, so moving a config to a newer image stranded its saved training states. Training states saved with 0.1.7 or earlier now restore on the current v0.1.x images without setting `trainer_image`. Checkpoints saved on `miles-v0.1.x` images still don't restore on `miles-v0.2.0` images, which use a different checkpoint format.
- A checkpoint that a Miles v0.2.0 trainer refuses to load, for example because its expert adapters don't match, now fails only that request. Before, it stopped the trainer and every model on it.
- DeepSeek-V4.1-Flash samplers run SGLang v0.5.23 like every other sampler, with decoder SWA bounded replay turned off. On v0.5.23 that feature made LoRA samplers crash on long prompts, and its logprobs don't match a full prefill. The samplers switch over at their next deploy (#304).

### Fixed

- `qwen35-9b-lora-64k` and `qwen35-9b-lora-128k` start again. They split the model across four GPUs (tensor parallel size 4) instead of eight, because the Miles native adapters they use since 0.1.8 reject eight for Qwen3.5-9B. `qwen35-9b-lora-128k` and the shared-server recipe built on it train on four B200s instead of eight H200s.
- Removing a config with `spindle remove` or `DELETE /api/v1/spindle/deployments/{name}` also stops its LoRA and full fine-tuning sampler pools. Before, they kept running, and billing for warm replicas, until stopped by hand.
- Miles v0.2.0 trainers can gather checkpoint and sampler captures across nodes without racing checkpoint persistence.
- Reference plots leave gaps for missing measurements instead of joining timing points across a restart.
- Long-sequence training requests with expert-routing replay no longer fail with intermittent 503 errors.

## 0.1.8 - 2026-11-18

Qwen3.5 LoRA recipes now train the linear-attention layers, and the dense Qwen and Nemotron-3 Nano LoRA recipes move to Miles native adapters, so checkpoints they saved before 0.1.8 load only on their original trainer image. A new recipe trains Nemotron 3.5 Super VL 120B-A12B on text, sampled logprobs match the trainer at any temperature, multi-target scoring handles mixed candidate counts in one pass, and a rewritten guide shows how to profile and tune a deployment config.

### Added

- Miles LoRA clients accept the stock Tinker SDK’s `create_lora_training_client(seed=...)` to reproduce adapter initialization without changing other clients’ random streams.
- `nemotron35-super-vl-120b-a12b-lora-64k` trains LoRA adapters on NVIDIA Nemotron 3.5 Super VL 120B-A12B. Training and sampling take text only: image inputs are not accepted, and adapters apply to the language model.
- Sampling supports the Tinker SDK's `prompt_logprobs_last_n`. It returns prompt logprobs for only the last N prompt tokens, so the rest of the prompt can come from the prefix cache.
- Pretrained DFlash inference recipes for Qwen3-8B, Qwen3.5-9B, Qwen3.8-27B and GLM-5.3 Flash, plus Qwen3.5-4B native MTP using its cached target checkpoint.
- An opt-in `miles-v0.2.0` trainer image with native adapter checkpoints, for single-node trainers in bridge mode. It trains the same shared-`lora_A` adapter layout as the existing images, which remain the default. Checkpoints don't restore across Miles versions. See [Images](docs/deployment-configs.md#images).
- `spindle usage --group-by` takes several label keys separated by commas, such as `team_id,run_id`, and prints one column per key.
- `megatron_cfg.logprob_chunk_tokens` makes full fine-tuning trainers score log-probs that many tokens at a time, so memory no longer grows with row length. It needs tensor and pipeline parallel size 1.
- Reference runs automatically index successful executions under `reference_runs/outputs/`, recording server-reported Spindle commit and version. Process locks protect run IDs, output paths and index updates; scratch outputs remain uncatalogued.

### Changed

- The Qwen3.5-2B, Qwen3.5-4B and Qwen3.5-9B-Base LoRA recipes now also train the linear-attention layers, which make up three of every four layers. Checkpoints these recipes saved before this change can't be loaded into them. To keep using one, load it into a variant with the old targets, as described in [Older Qwen3.5 LoRA checkpoints](docs/deployment-configs.md#older-qwen35-lora-checkpoints).
- Single-node dense Qwen2.5, Qwen3, Qwen3.5, Qwen3.8 and Nemotron-3 Nano LoRA recipes use Miles native canonical adapters with independent projection factors and a sampler image supporting canonical GDN targets. Nano uses FP32 expert reductions and deterministic kernels so repeated scores and optimizer replay agree. Older checkpoints require their original trainer image and adapter layout.
- Miles multi-target scoring shares one vocabulary softmax across mixed candidate counts in supported scoring modes, preserving each datum's output shape and batch order.
- [Profiling](docs/profiling.md) now walks from measuring a step to choosing which config fields to change, for trainers and samplers.
- Reference runs use flat TOML definitions, a typed execution index, and flat JSONL outputs. A shared launcher selects the exact deployment for the existing DAPO-Math workload.
- The reference launcher accepts workload arguments only through the TOML definition's `script_args`; extra command-line arguments are rejected.
- Reference deployment overrides use explicit typed fields and reject undeclared settings.
- Indexed reference runs require a TOML definition in `reference_runs/configs/`. Temporary definitions can use scratch output paths.
- DAPO-Math reference definitions must specify a tokenizer model and pinned revision. Direct script callers must pass `--tokenizer MODEL REVISION`.
- The standard 32K LoRA recipes for Qwen3-8B, Qwen3.5-9B Instruct, Qwen3.8-27B and GLM-5.3 Flash use DFlash; Qwen3.5-4B uses native MTP. These replace the separate `with-spec` and `mtp` variants. Qwen3-8B samples on one B300 instead of two H100 GPUs, and Qwen3.5-4B uses B200 for training and sampling.
- `nemotron35-lightning-30b-a3b-lora-64k` trains on two B300s with one model copy per GPU, and samples on B200s with 128 requests per sampler. The previous H200 layout is now `nemotron35-lightning-30b-a3b-lora-64k-h200`; use it when B300s or B200s are not available. The 256K recipe and the shared-server recipe keep H200 trainers and samplers.
- `nemotron35-lightning-30b-a3b-lora-64k` samples on a B200 with the model's multi-token prediction layer, which speeds up decoding. `nemotron35-lightning-30b-a3b-lora-64k-h200` keeps an H200 sampler, which now runs 89 requests at once instead of 128.
- The 64K full fine-tuning recipes split each model across fewer GPUs. Qwen3.5-4B and Qwen3.5-9B train on four B300s or B200s, Qwen3.6-27B on eight B300s, and the 35B-A3B recipes on eight B300s or B200s. Training states saved on the old layouts can't be resumed with `load_state_with_optimizer`, but `load_state` still loads their weights.
- Full fine-tuning trainers compile their linear-attention and attention kernels once instead of on most steps, and split rows evenly across data-parallel GPUs, so steps are faster and steadier.
- MoE LoRA training on the default Miles images gives the expert token permute a scatter-add backward, which is faster than the BF16 index_add it replaces.
- Miles LoRA training builds the logits gradient once per micro-batch instead of once per row. Steps with many short rows run faster and hold less memory; results are unchanged.
- Every bundled config sets its own `trainer_image` and `inference_image`, so SGLang and Miles can be upgraded one config at a time. The images each config runs are unchanged. Configs that leave these fields empty still use the defaults. See [Images](docs/deployment-configs.md#images).
- Image names now include the fork commit, and a named image is never changed in place. The names `miles-v0.1.2` and `miles-v0.2.0` are rejected, so update `trainer_image` in your configs to `miles-v0.1.2-99e1ebe` or `miles-v0.2.0-8c5de26`. The defaults now have names too: `miles-v0.1.1-52a7a99` and `megatron-2d1fa8d` for trainers, and `sglang-v0.5.23-c4182b2` for samplers. The DeepSeek-V4.1 and GLM-5.3 Flash images are now `sglang-v0.5.21-a3e0e17`, `sglang-v0.5.23-c4182b2-glm53` and `miles-v0.1.2-99e1ebe-glm53`.
- Synthetic guess-number regressions select existing cookbook renderers for Qwen2.5-3B/7B-Instruct, Qwen3-4B and Qwen3.5-2B instead of skipping them when cookbook model metadata is missing.

### Removed

- `SPINDLE_EXCLUDE_RESERVED_CAPACITY` is no longer read. Set `platform.modal.exclude_reserved_capacity` in each config instead.

### Fixed

- Qwen3.5 and Qwen3.6 LoRA trainers no longer pause for several seconds when a step brings new row lengths. On a fresh `qwen35-4b-lora-32k` trainer, such steps took 7.5–26 s instead of about 2.5 s.
- Sampled-token logprobs are now the model's temperature-1 logprobs at any sampling temperature, matching what the trainer computes. RL scripts that sample at a temperature other than 1 get correct importance ratios for importance sampling, PPO, CISPO and DRO.
- LoRA trainers for the Nemotron recipes restart the Mamba state at each row packed into a micro-batch. Before, every row after the first started from the previous row's state, so its log-probs and gradients were wrong. The `miles-v0.2.0` image already did this.
- Miles LoRA trainers no longer rescale gradients kept from earlier `forward_backward` calls, including another client's on the same trainer. The `miles-v0.2.0` image already had this fix.
- When a `forward_backward` or `forward` call runs out of GPU memory on a Miles LoRA trainer, the request fails with an out-of-memory error and the trainer keeps serving, with earlier gradients kept. This covers one-GPU trainers and dense models that use only data parallelism. Other layouts still restart the trainer.
- On those trainers, when `forward_backward` requests from several clients run together and one runs out of GPU memory, only that request fails. The others run again one at a time and return their normal results. Before, every request that ran with it failed with the same error and lost its gradients.
- Miles LoRA trainers deployed with `SPINDLE_TORCH_PROFILE_STEP` keep serving requests while the profiled step's trace is written. Before, every client on the trainer waited for the write, which took from half a minute to several minutes. Traces no longer record tensor shapes or memory unless you set `SPINDLE_TORCH_PROFILE_SHAPES=1` or `SPINDLE_TORCH_PROFILE_MEMORY=1`.
- Miles LoRA trainers deployed with `SPINDLE_TORCH_PROFILE_STEP` no longer crash after the profiled step when the client sends the next `forward_backward` while sampler weights are still publishing. The trainer no longer writes `controller.trace.json.gz` or `controller.key_averages.txt`; the per-GPU `rank<N>` traces are unchanged.
- `create_lora_training_client` now fails right away with a 400 when the deployment can't train the requested LoRA: a rank above its maximum, a different `train_attn` or `train_mlp`, or `train_unembed=False` on a deployment that trains the unembedding. The error names the settings to use. Before, it came only after a trainer had started and loaded the model.
- Sampling requests with an option Spindle does not support, such as `prompt_alt_tokens_k` or `topk_sample_logprobs`, now fail right away with a 400 that names the option. Before, the server retried them three times before the client saw an error.
- Built-in Miles v0.2.0 images reject the unvalidated Bridge/canonical adapter combination before starting a trainer.
- Malformed `spindle add` fields such as `--=1` reach recipe validation instead of being parsed as abbreviated top-level options.
- A sample that fails because its sampler can't read the recipe registry is reported as lost, so the Tinker SDK resubmits it instead of ending the run.
- Speculative samplers use a corrected SGLang runtime that honors the earliest matching string, regex or EOS stop within an accepted token block.
- Draft checkpoint revisions use separate cache directories, and downloads sharing target weights run one at a time.
- Trainer backends use a local Unix socket so another process cannot take their port while model weights load and make startup fail.
- Miles LoRA images support seeded adapter initialization under `miles-v0.1.1-52a7a99`, `miles-v0.1.2-99e1ebe`, `miles-v0.1.2-99e1ebe-glm53` and `miles-v0.2.0-8c5de26`. The default Miles image and every bundled Miles config use these. A checkpoint restores only on the Miles revision that saved it. To restore a training state saved with 0.1.7 or earlier, set `trainer_image` to `miles-v0.1.1-fdd808c`, `miles-v0.1.2-f2f8447` for DeepSeek-V4.1, or `miles-v0.1.2-464efc9-glm53` for GLM-5.3 Flash.

## 0.1.7 - 2026-11-15

See what each run or team costs on a shared deployment: Spindle records the tokens every session uses, serves them through Tinker's `get_billing_usage`, and `spindle usage` turns them into dollars per run or label. See [Cost attribution](docs/cost-attribution.md).

### Added

- Token usage per session on shared trainers and samplers, stored on a `<frontend>-usage` Modal Volume and readable with Tinker's `get_billing_usage`. Label runs with `user_metadata`; `SPINDLE_USAGE_LABEL_KEYS` in the frontend's API secret lists the keys kept (default `run_id,team_id`). See [Cost attribution](docs/cost-attribution.md).
- `spindle usage` splits Modal app cost across runs or labels by tokens processed, reading costs from Modal's billing report. Idle and cold-start time of trainers and sampler pools isn't charged to a run but still costs you, so it's shown as separate lines; `--hide-overhead` hides them. If Modal's billing report doesn't cover the whole window yet, only the hours it covers are split.
- `reference_runs/`: a DAPO-Math GRPO workload on the stock Tinker SDK that records learning and timing metrics, with recorded results and a chart script for comparing deployments. Its deployment recipe fixes four H100 trainer GPUs and two single-H100 sampler replicas, and the client records that hardware contract.
- GLM-5.3 Flash LoRA recipes with a model-specific Miles image, including single-node B300 training with RDMA and expert-routing replay.
- `mis`, `score_centered`, and `score_centered_mis` train on recorded sampler supports in Miles LoRA and Megatron full-parameter backends. Replay requires explicit sampling temperature; omitted weights mask prompt positions in both backends.
- `platform.modal.exclude_reserved_capacity`: when `True`, multi-node and RDMA trainers stay off GPUs reserved for the workspace, so the reservation stays free for other jobs. The shared-server example turns it on (#255).

### Changed

- Samplers run SGLang v0.5.23 with PyTorch 2.14.1 and Transformers 5.19. A server's samplers switch over when it is redeployed with a new Spindle version. DeepSeek-V4.1-Flash samplers stay on SGLang v0.5.21, because their LoRA samplers crash on v0.5.23 after loading a trained adapter (#240, #270).
- `inference_target_concurrency` now defaults to `None`, which adds a sampler replica only when every replica holds `sglang_cfg.max_running_requests` requests. New `inference_scaleup_window_s` delays scale-out until load has stayed above the target that long.
- The Qwen3.5-9B 16K and Qwen3.5-4B 32K recipes sample on B200, falling back to H200 (9B) or H100 (4B) when no B200 is free. Batches finish faster; a B200 replica costs more per hour. The 4B sampler also runs 128 requests at once. Qwen3.5-2B and the 9B 64K and 128K recipes stay on Hopper (#254).
- The `spindle deploy` model picker starts with no model selected, so you no longer deploy Qwen3.6-35B-A3B unless you pick it (#281).
- Spindle now requires `tinker>=0.26.0`, the first SDK release whose `SupportedModel` has `trainable` and `sampleable`.
- Replay training prepares expert routes and top-p supports with arrays, reducing CPU work before training.

### Fixed

- A request a Modal-hosted frontend rejected with a 400, such as a misaligned `get_billing_usage` window, no longer makes the Tinker SDK retry it for 5 minutes. Modal's proxy compressed the error and dropped its `Content-Length`, which the SDK treats as a retryable proxy rejection; error responses are now sent uncompressed.
- Local sampling task runners can return an asyncio future as well as a coroutine.
- Retrieving a null or list result returns JSON when the client requests protobuf.
- Frontend containers reread the recipe list as soon as it changes, so a config just added or removed through another container is seen at once by name, by model ID, and in `GET /api/v1/spindle/deployments` and `get_server_capabilities`.
- `spindle stop` no longer aborts on apps that `spindle remove` is already stopping, and removing a config while its apps are still deploying no longer leaves those apps running.
- Sampling requests rejected with HTTP 429 or 503 can retry with the same request ID as soon as capacity is available, instead of replaying the rejection for 30 seconds.
- `get_server_capabilities` now sets `trainable` and `sampleable` on each model, which Tinker SDK 0.26 and later read. Before, both were `null`.
- `spindle remove` and `DELETE /api/v1/spindle/deployments/{name}` now also stop the config's sampler pools. Before, a config with `inference_min_replicas` above 0 kept its warm samplers running after removal.
- Sampling recovers when SGLang loads an adapter but its acknowledgement is lost. A later load error is accepted only when SGLang's registry confirms the same adapter name and checkpoint path.
- Sampling from a `qwen35_35b_a3b_fft_64k` or `qwen36_35b_a3b_fft_64k` full fine-tune no longer runs out of GPU memory while the sampler loads the trained weights (#260).

## 0.1.6 - 2026-11-06

Add and remove configs on a running frontend without redeploying it, wait in line for a trainer slot instead of failing, and see each recipe's training queue.

### Added

- `spindle deploy` registers every recipe in one per-frontend registry instead of baking a configs file into the frontend image, so a deploy adds or updates configs instead of replacing the set; upgrading a pre-registry frontend carries over every recipe it serves (#181).
- `POST /api/v1/spindle/deployments` registers a config on a running frontend and deploys its apps, and `DELETE /api/v1/spindle/deployments/{name}` removes one (#160).
- `spindle add NAME --base BASE --FIELD=VALUE ...` registers a config through that API and waits until its apps are ready; `spindle remove NAME` removes a config while checking whether there are any models still training on it live (#160).
- `qwen25_7b_instruct_lora_32k`: Qwen2.5-7B-Instruct LoRA with 32K context, trained on 4× H100 and sampled on one GPU per replica.
- `qwen25_3b_lora_32k`: Qwen2.5-3B LoRA with 32K context, with one model replica per GPU. The shared server serves both Qwen2.5 recipes (#206).
- `GET /api/v1/spindle/queues` describes each recipe's training queue: waiting clients, free trainer slots, and how many clients were added, placed and released in the last hour (#201, #205).
- `spindle stop` stops a deployment: its frontend, then every trainer, inference, and sampler app it created. Pass the same `--app` and `--env` as the deploy.

### Changed

- Every app a deployment creates now starts with its frontend name, so `spindle deploy --app my-spindle` names its trainers `my-spindle-trainer-<recipe>` and its sampler pools `my-spindle-lora-<recipe>`. Two deployments with different `--app` values can share one Modal environment and recipe names. Deployments without `--app` keep their `spindle-...` names. A deployment made with `--app` on an earlier release gets new worker apps on its next deploy; stop its old `spindle-...` apps with `modal app stop`.
- When every trainer for a recipe is full, creating a model waits for a slot instead of failing with `trainer capacity exhausted`. The Tinker SDK shows the request's place in line. When a slot frees up, the request that has waited longest gets it. Rarely, a brand-new request takes it first.
- Spindle requires Modal 1.6.3 or newer, the first release that builds images with [image builder version](https://modal.com/docs/guide/images#image-builder-updates) 2026.10. Multi-node trainers use `modal.clustered` instead of the deprecated `modal.experimental.clustered`.

### Fixed

- The `qwen35_4b_lora_32k` and `qwen35_2b_lora_32k` trainers no longer run out of GPU memory on batches of many short sequences. They pack up to 16K tokens per GPU instead of 32K; sequences up to the full 32K context still train (#203).
- Full fine-tune training clients and `sample_with_replay` work with Tinker SDK 0.38 (#212).

## 0.1.5 - 2026-11-02

Faster DeepSeek-V4.1-Flash training, 64K and 128K DeepSeek-V4.1-Flash recipes, and frontends that can require a Modal proxy token.

### Added

- `deepseek_v41_flash_lora_64k`: DeepSeek-V4.1-Flash LoRA with 64K context. It trains at TP8 on B300 or B200; H200 cannot hold a 64K training sequence for this model (#178).
- `deepseek_v41_flash_lora_128k`: DeepSeek-V4.1-Flash LoRA with 128K context (#172).
- LoRA trainers can split one long sequence across GPUs for models that need it, such as DeepSeek-V4.1 (Miles `allgather_cp` with `qkv_format: bshd`), including batches that train several LoRA adapters at once (#172).
- `platform.url_label` sets the part of the frontend URL after the double dash, so `"prod"` gives `https://<workspace>-<environment>--prod.modal.run`. `platform.requires_proxy_auth` makes Modal turn away frontend requests without a [proxy token](https://modal.com/docs/guide/webhook-proxy-auth). Deployments that set neither keep their URL and stay reachable by stock Tinker clients (#180).
- `examples/shared-server`: one deployment a team shares, with one LoRA recipe for each bundled base model on a `prod` frontend in its own environment, a smoke test, a script that adds the proxy token to any Tinker script, and a workflow that redeploys it from `main` (#139, #189, #192).
- Every Spindle deployment is tagged with its version, such as `v0.1.5`. The Tag column of each app's deployment history shows which Spindle release it runs (#179).

### Changed

- When a frontend serves several recipes for the same model, requests by model name go to the recipe with the longest context, instead of whichever was deployed first. Pass a recipe's name as `base_model` to pick that recipe. `get_server_capabilities` lists every recipe and its context length. Sampling a saved adapter uses the recipe that trained it (#171).
- Deployments now download base weights for up to four recipes at once, so a small model no longer waits behind a large one (#182).
- The DeepSeek-V4.1-Flash 8K and 32K LoRA trainers now train several sequences together instead of one at a time, so training steps take less time and use more GPU memory. On two 8× H200 nodes, a 32K GRPO training step drops from 128 to 57 seconds. Recipes based on them inherit this; set `miles_cfg.dynamic_batch_size: False` and `miles_cfg.cli_options.qkv_format: "bshd"` to keep the previous behavior (#193).

### Fixed

- Scripts on Tinker SDK 0.36 or newer, including tinker-cookbook, no longer fail with a 404 when they exit. Spindle now accepts the SDK's end-of-session call and frees that session's models right away (#190).
- Creating a model no longer fails with `trainer capacity exhausted` when a trainer finishes starting while the model is being placed (#185).

## 0.1.4 - 2026-10-29

Faster DeepSeek-V4.1-Flash sampling and sampler starts, and agent skills for working with Spindle.

### Added

- `deepseek_v41_flash_lora_32k_b300`: the 32K DeepSeek-V4.1-Flash recipe on B300, which holds a 32K row at TP2 (#163).
- `spindle skills install` installs agent skills that explain deploying Spindle, where it differs from Tinker, and how to run experiments on it (#140).

### Changed

- The DeepSeek-V4.1-Flash 32K recipes train at TP4 instead of TP8 on H200 and B200, so each rank runs half as many micro-batches per step. The 1017 32K recipe inherits this (#163).
- The DeepSeek-V4.1-Flash 32K recipes decode with the checkpoint's draft heads (SGLang's DSpark mode). A long sample that used to finish alone at 30 to 60 tokens per second now finishes several times faster, and samples keep the model's own distribution because the full model checks every draft token. The sampler takes four to seven minutes longer to start, which a 32K run earns back within about five steps. The 8K recipes are unchanged (#165).
- DeepSeek-V4.1-Flash samplers run 128 requests at once instead of 32, so a GRPO step's samples generate together instead of in four waves (#147).
- DeepSeek-V4.1-Flash samplers start sooner: they read the checkpoint with eight threads instead of one (#148), each GPU rank pins its own eighth of the Engram tables instead of all ranks pinning one shared copy, which saves 200 to 250 seconds on 8× H200 (#149), and they keep their compiled kernels on the shared kernel-cache volume, like trainers, which saves about five minutes once the cache is warm (#146, #167). Other recipes can keep SGLang's caches on the volume too by setting `SGLANG_CACHE_DIR` under `/root/.cache/kernel-cache` in `inference_env`.
- Sampler images no longer build SGLang's prompt-indexing extension at startup; it ships prebuilt (#146).
- Every sampler sets `SGLANG_ENABLE_LORA_CSGMV_SPLIT_K=1`. SGLang still applies split-K only to the BF16 rank-32 CSGMV shapes it measured, and never in deterministic mode. LoRA servers already set the flag (#123); set it to `0` in `inference_env` to turn it off. `qwen35_9b_instruct_lora_256k` set the flag under its old name, so split-K never ran there before (#151).

## 0.1.3 - 2026-10-28

Faster training for LoRA classifiers that score several candidate labels per position, and cache-aware routing for LoRA samplers.

### Added

- `qwen3_4b_lora_32k`: Qwen3-4B LoRA on 4× H100 with 32K context, one model replica per GPU (#142).
- Qwen3.8-27B LoRA recipe with B200 samplers, `qwen38_27b_lora_16k_b200`, which serves with the TensorRT-LLM attention backend and 64-token pages (#145).
- LoRA samplers can route each request to the replica with its longest cached prefix with `inference_lora_routing = "cache_wait"`, while keeping request load and pinned KV memory within bounds. The router runs in Rust inside each sampling container. It measures the pool's prefill rate from completed requests unless `inference_routing_prefill_tokens_per_second` sets it, and `inference_routing_balance_absolute`, `inference_routing_balance_relative` and `inference_routing_cache_threshold` tune it. Pools without routable replicas, including pools scaled to zero and pools whose SGLang cache publishes no observations, fall back to Modal's gateway routing (#121, #122, #123).

### Changed

- LoRA recipes use `cache_wait` routing by default when SGLang runs one data-parallel rank, and Modal session affinity otherwise. `inference_lora_routing = "affinity"` keeps session affinity. `inference_routing_balance_absolute` defaults to 40% of `max_running_requests`, and at least 4 (#123).
- LoRA samplers enable SGLang's CSGMV split-K kernel. For rank-32 adapters on models with 4,096-wide hidden layers, such as Qwen3.5-9B and Qwen3-8B, it speeds up the LoRA kernels for decode-sized batches by up to 2.9× on H200 and 2.6× on B200. Set `SGLANG_ENABLE_LORA_CSGMV_SPLIT_K=0` in `inference_env` to disable it (#123).
- The frontend server runs in `platform.modal.region`, or `us-east` when no region is set. Only the CPU frontend is placed; trainers and samplers stay unpinned unless the recipe sets a region. Before, frontend containers could start in any region, and one on another continent added 2 s or more to each training step (#152).
- `forward` requests use the same binary transport and protobuf results as `forward_backward`, and queued forwards are batched into one trainer call (#153).
- The Miles backend prepares labels and weights with numpy instead of per-value Python loops (#154), and passes training rows and log probabilities to its GPU workers as arrays instead of Python lists (#158). Large batches spend less time between the request and the GPU work.
- When every datum in a batch has the same number of candidate labels (`target_tokens` shaped `[sequence_length, K]`), the Miles backend scores all K labels from one softmax over the vocabulary instead of K (#157).

## 0.1.2 - 2026-10-27

DeepSeek-V4.1-Flash LoRA recipes for the original and 1017 releases, loaded from DeepSeek's own weights, and multi-LoRA trainer scheduling for clients that train independently.

### Added

- DeepSeek-V4.1-Flash-1017 LoRA recipes, `deepseek_v41_flash_1017_lora_8k` and `deepseek_v41_flash_1017_lora_32k`, which load the 1017 release (#129).
- DeepSeek-V4.1-Flash LoRA recipe with 32K context, `deepseek_v41_flash_lora_32k`. It splits each layer across eight GPUs instead of four (#128).
- `deepseek_v41_flash_lora_8k` falls back to B200 and H200 for the trainer and samplers (#128).
- `trainer_max_clients_per_batch` caps how many multi-LoRA clients a trainer trains together. With `1`, each client's step runs on its own, so clients that train independently no longer wait for each other (#124).
- Samplers serve SGLang's Prometheus metrics at `/metrics`, so Modal collects inference metrics for sampler deployments (#125).
- Example launchers for the Qwen3.8-27B DAPO run on 4 H200s and 12 B200s (#135) and the Qwen3.5-9B SWE-Gym run on B300s (#136).

### Changed

- DeepSeek-V4.1-Flash recipes load DeepSeek's FP8/FP4 release instead of a BF16 export. The trainer dequantizes it to BF16 as it loads, and samplers serve it with FP8 routed experts. Redeploying downloads the release once (#128).
- `deepseek_v41_flash_lora_8k` keeps one sampler running for the length of a run, so the first sample doesn't wait for the weights to load. Recipes that subclass it inherit this (#134).
- DeepSeek-V4.1-Flash samplers queue up to 128 requests instead of 8, so a GRPO step's group samples no longer get 503s and retries (#141).
- Multi-LoRA trainers run optimizer steps, checkpoint saves and sampler-weight captures ahead of queued `forward_backward` requests, and serve queued training requests oldest first (#124).
- A sampling request delivered again with the same request ID shares the in-flight work or the completed response instead of generating twice (#120).

### Fixed

- DeepSeek-V4.1-Flash trainers on B200 no longer hang in their first step (#128).
- Miles' gloo process group waits as long as `distributed_timeout_minutes`, so trainers on slow-loading hosts no longer time out after 30 minutes (#128).
- The Miles v0.1.2 trainer image, the Miles Megatron base image and the SGLang image are pinned to exact commits or digests, so a cached build cannot serve an older fork branch (#128, #131).
- The rollout image builds SGLang's radix-tree binding from the pinned fork revision, so the scheduler no longer crashes when routing observations are enabled (#137).
- A sampling request that waited longer than 30 seconds, or a future polled after its sample completed, no longer fails with `unknown key family: sampler_warm` (#130).
- Training runs named in sampler paths, `<model_id>:train:0`, resolve to the model's checkpoints, so tinker-cookbook's `run_inspect_evals` finds the base model (#130).

## 0.1.1 - 2026-10-23

The first versioned release. Deployments made before it report `0.1.0`.

### Added

- Spindle reports its version and source commit: `spindle --version`, `spindle_version` and `spindle_commit` (short SHA) tags on every Modal app Spindle creates, and the version and full SHA in `/api/v1/get_server_capabilities`.
- Spindle apps carry `_modal_source`, `spindle_role`, job type, framework and model tags for usage reporting (#99).
- `trainer_gpu` and `inference_gpu` accept a list of GPU types in preference order (#89). `qwen36-35b-a3b-lora-32k` falls back from H200 to B300 and B200.
- `spindle deploy` interactive mode (#91) and a check for required environment variables (#92).
- Support for Tinker SDK 0.25 through 0.33 (#3, #69).
- Multiple target tokens per input position (#104).
- Model recipes with multi-LoRA backend support (#35), a Qwen3.5 9B 256K config (#86), and output-head LoRA for Qwen3.6 35B (#103).
- DeepSeek-V4.1-Flash LoRA recipe, `deepseek_v41_flash_lora_8k`, trained and served on B300s from pinned BF16 weights (#73, #74, #75).
- `trainer_image` selects a named trainer image, and multi-LoRA Miles recipes can run on Miles v0.1.2. LoRA target modules accept wildcard checkpoint paths (#74).
- While a trainer waits for GPU capacity, `create_model` reports which GPUs it is waiting for and for how long, instead of "starting a trainer" (#113).
- Per-phase startup timing on trainer and sampler containers (#58).

### Changed

- **GPU placement is no longer pinned to `us-west` by default** (#4). `platform.modal.region` now defaults to `None`; set it only to restrict placement.
- SGLang v0.5.21 and a newer Stitch (#30).
- gpt-oss presets: train `lm_head` LoRA to match Tinker SDK defaults (#6), serve LoRA from BF16 weights (#7), and retuned trainer and sampler parallelism and admission limits (#36, #40, #45, #49).
- Sampling: multi-turn conversations route to one sampler replica (#46), and warm sampler pools persist between training steps (#37).
- Faster cold starts: TileLang kernels persist in the shared kernel cache (#82), the asset download is skipped once weights are on the volume (#83), and `spindle deploy` builds the sampler image up front (#95).
- `spindle deploy` starts each model's base-weight download in the background. `create_model` and `create_sampling_client` return immediately, report download progress while they wait, and work for checkpoints that take longer than a request timeout to download (#107).
- Deployment configs reach the frontend as a file instead of an environment variable (#94).
- Optimizers other than AdamW are rejected instead of silently running AdamW (#78, #101).
- A client that asks to train the unembedding, as Tinker clients do by default, keeps it frozen when the deployment doesn't target it, instead of being refused (#74).
- Lower request and training overhead (#29, #31, #41, #42).

### Fixed

- `spindle deploy` from macOS and other onboarding issues (#23).
- Miles CLI overrides applied after all argument groups are registered (#5).
- Miles LoRA A initialization matches PEFT and Tinker (#10).
- Grouped-expert LoRA checkpoint loading in the pinned Miles version (#16).
- `execute_sample` is non-preemptible (#15).
- `load_state` on multi-node LoRA trainers, which failed on missing checkpoint shards (#74).
