# Reference runs

A flat catalog of validation definitions and measured executions. The DAPO-Math
references train through the stock Tinker SDK.

```text
reference_runs/
  configs/dapo_qwen35_4b.toml   # Immutable validation definition
  configs/dapo_qwen38_27b_multinode.toml
  configs/beam_blog_repro.toml
  configs/guess_number_gpt_oss_20b_restore.toml
  scripts/dapo_math.py         # Dataset, grading and training settings
  scripts/guess_number.py      # Multi-turn number guessing through Tinker
  scripts/beam_blog_repro.py    # Beam LoRA training with AIME25 evaluation
  run.py                      # Prepare config or launch the script
  index.toml                  # Execution provenance and output paths
  outputs/*.jsonl             # Flat, incremental metric files
  plot.py                     # Compare indexed or new executions
```

## Definition

A definition selects a bundled deployment config and a script. The script path
is relative to the TOML file.

```toml
config = "qwen35-4b-lora-32k"
script = "../scripts/dapo_math.py"

[workload]
tokenizer_model = "Qwen/Qwen3.5-4B"
tokenizer_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
```

The existing H100 reference pins its hardware and client capacity in
`[config_overrides]`, preserving the old Python reference recipe. Its fields
are declared in `ConfigOverrides` in `run.py`; unknown fields and incorrect
types are rejected. Omitted fields keep their deployment recipe values.
Use `inference_gpu = "H100!"` to select exact H100 hardware,
`"sglang_cfg.tp_size"` to match sampler tensor parallelism to its GPU count, or the declared
`"miles_cfg.max_lora_slots"` key to set adapter capacity. Optional
`script_args = []` contains the workload's literal CLI arguments. The launcher
does not forward extra arguments from its own command line. Dataset, grading,
learning rate, batch size and seed stay in the Tinker script. Every definition
requires a typed `workload` block with `tokenizer_model` and a full
`tokenizer_revision`. It also accepts typed `total_steps`, `save_step` and
`restore_step` fields, forwarded as workload flags when present.
Existing `[tokenizer]` blocks migrate to `[workload]` with the fields renamed
to `tokenizer_model` and `tokenizer_revision`.
The launcher supplies `--tokenizer MODEL REVISION`, the output path, and
`--base-model` with the resolved deployment name. Scripts use that deployment
name in `create_lora_training_client` to select the exact recipe.
Script arguments cannot override these values or the typed step flags. Direct callers of
`dapo_math.py` must also pass `--tokenizer MODEL REVISION`; it has no default.

`dapo_qwen38_27b_multinode.toml` uses the same workload with a pinned Qwen3.8-27B
tokenizer in its definition. It inherits the committed `qwen38-27b-lora-256k`
topology: two eight-H200 trainer nodes, TP2/CP8, and two
four-H200 sampler replicas. The definition selects exact H200 hardware and one
training client/adapter slot. Trainer compiler caches stay on local disk because
shared-cache validation read a generated Python module containing null bytes.
The workload generates at most 2,048 response tokens; it does not validate 256K
sequences. Compare new
measurements against the same validation, rather than the 4B learning curve.

After the measured updates, one nonzero-gradient update creates a probe
checkpoint with active Adam momentum. The script applies a zero-gradient Adam
step, restores the probe checkpoint, and replays that momentum-only step.
This checks optimizer restoration without requiring deterministic backward
kernels. It compares token scores, restores the final measured checkpoint,
then publishes and samples its adapter. The `restore_validation` record reports
the comparison separately from update timings. A failed check fails the run
and leaves its metrics uncatalogued. The three probe steps do not enter the
learning curve. The `restore-probe`, `restore-expected` and `restore-replayed`
checkpoints retain their states.

`beam_blog_repro.toml` trains `reflection/Beam-1.0-501B-A23B` with BF16
LoRA on the bundled `beam-501b-a23b-lora-5k` recipe: two eight-H200 trainer
nodes and one eight-B300 sampler. Its script uses routing replay, CISPO with
one update of rollout prefetch and a 400-update cosine learning-rate schedule,
and evaluates on AIME25. It writes local evaluations, run identity, the replay
cache and plots to a work directory next to the output (`outputs/NAME/`), which
Git ignores. The launcher indexes the run when the script finishes all
planned updates. A run that fails or exceeds its update time budget exits with
an error after saving a checkpoint and is not indexed. Run the script directly
with `--resume` or `--stop-after` to continue after a restart, and add that run
to the index yourself; `--plot-only` compares a run with the published Beam
LoRA curve, indexed as `beam-blog-digitized`. Index entries for executions
outside Spindle record `source` instead of Spindle provenance. The indexed
execution predates the recipe's Miles v0.2.0 image and stopped at 42 of 100 planned updates.

Once published, a validation TOML file is immutable. Create a new definition
when changing its script selection, arguments or deployment overrides. Spindle
commits and observed runtime image versions identify executions and belong in
the index, not in the validation definition. Deployment recipes own image
selection; record the images used for each execution in the index.

## Guess-the-number regression

`configs/guess_number_gpt_oss_20b_restore.toml` runs `scripts/guess_number.py` on
`gpt-oss-20b-lora-32k`. It uses the recipe's selected images, including the
older Miles v0.1.1 trainer. It requests one H200 trainer instead of the
recipe's eight data-parallel replicas and uses one H200 sampler with tensor parallel size one. Image versions belong to each execution's index entry.

The task follows tinker-cookbook 0.6.1's multi-turn guess-number game: a hidden
integer from 0 through 1023, at most ten guesses, higher/lower feedback, reward
1 for a correct guess and -1 for each malformed answer. The script uses the
stock Tinker SDK and a pinned tokenizer's chat template, without the cookbook
training loop. It uses low reasoning effort for GPT-OSS. These measurements
are a separate workload from the cookbook regression, not interchangeable baselines.

The definition retains six GRPO updates. Each update uses eight hidden numbers and four independent trajectories
per number, rank-32 LoRA, a 64-token response cap and learning rate 3e-5.
Each trajectory's total reward is centered within its group and applied to
its sampled tokens. Prior conversation tokens receive zero advantage.
Eight held-out numbers are evaluated before training and after the last update.
Each policy is published for sampling, and the final checkpoint is saved.

The checkpoint schedule is:

```toml
[workload]
tokenizer_model = "openai/gpt-oss-20b"
tokenizer_revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
total_steps = 6
save_step = 2
restore_step = 4
```

The script saves weights and optimizer state after step 2, trains through step 4,
loads step 2 with `load_state_with_optimizer`, then trains steps 3 through 6.
It runs eight optimizer updates in total, retaining six. A fixed prompt's token
scores on published sampler snapshots must change before restore and match
the saved scores afterward. Training-forward scores are not used because
the training backend can return different scores without a weight change.
This checks restored weights; optimizer state is loaded through the SDK's
optimizer-restore API. The final checkpoint includes the continued optimizer state.
The schedule requires `0 < save_step < restore_step < total_steps`.
Direct callers use `--total-steps`, `--save-step` and `--restore-step`.

Discarded updates remain in JSONL with `discarded = true`. `attempt` counts all
optimizer updates; `step` is the policy's logical step and rewinds on restore.
`restore_validation` records the rewind and score comparison. Plots omit the
discarded branch and leave a timing gap at the first resumed update. Sampler
publication names stay unique across both passes. A restore failure fails the
run and prevents index registration.

Step 2 is already included in its checkpoint and is not trained twice. After
restore, logical steps 3 and 4 select the same hidden-number batches and sampling
seeds as their first pass. Their rollouts are generated again with the restored
policy. The script does not cache or replay sampled responses; seeded sampling
can still vary between executions. The first pass's weight and optimizer updates
are undone by loading the checkpoint. The held-out numbers stay disjoint from
all training batches.

In evaluation and update records, `responses` counts games, `reward` is mean
cumulative game reward, `response_tokens_mean` is per assistant turn, and
`truncated_fraction` counts assistant turns that hit the response cap.
`success_fraction` and `turns_mean` describe completed games.
`kl_sample_train_v2` is the mean squared sample/train logprob difference divided
by two, measured on all sampled tokens before the optimizer update.
A successful run confirms the API path works; these short runs do not establish learning.

Prepare and deploy this definition as below, then launch it with:

```bash
uv run python reference_runs/run.py reference_runs/configs/guess_number_gpt_oss_20b_restore.toml \
  --output reference_runs/outputs/guess-number-gpt-oss-20b-restore.jsonl
```

## Prepare and run

Prepare the selected recipe with its reference overrides, then deploy it
explicitly. This creates a new Python file and does not launch any resources:

```bash
uv run python reference_runs/run.py reference_runs/configs/dapo_qwen35_4b.toml \
  --write-config /tmp/dapo-reference.py
uv run spindle deploy /tmp/dapo-reference.py --app reference --env dev
```

Set `TINKER_BASE_URL` and `TINKER_API_KEY` for that deployment, then:

```bash
uv run python reference_runs/run.py reference_runs/configs/dapo_qwen35_4b.toml \
  --output reference_runs/outputs/my-run.jsonl
```

For a small smoke run, create a separate TOML definition with
`script_args = ["--smoke"]` at the top level and pass it to the launcher.

The launcher does not deploy, stop or clean up apps. For Modal proxy auth, use
`examples/shared-server/with_proxy_auth.py` to invoke `reference_runs/run.py`
with the same arguments. The output file must not exist; concurrent launches
cannot share it. Stopped executions cannot be resumed.

The definition uses the existing fixed-H100 deployment recipe, which inherits
settings from the committed Qwen3.5-4B config. Verify its resolved settings and
actual hardware before comparing executions. The old measurements validate
the historical deployment revisions in the index, not today's parent config.

## Metrics and index

Every reference script writes the same JSONL format, so `plot.py` can compare any
two executions. `read_output` in `catalog.py` validates each complete line with the
`Record`, `Evaluation` and `Update` models; tests check every indexed output and
each script's output against them. Records are appended as the workload
progresses, so a partial run remains useful; a final partial line is ignored.

Every record has `event` and `time` (Unix seconds). `evaluation` and `update`
records also require:

| Field | Meaning |
| --- | --- |
| `step` | Accepted optimizer updates before the evaluation, or the update number |
| `responses` | Sampled responses scored |
| `reward` | Mean reward; for exact-answer grading, the fraction answered correctly |
| `response_tokens_mean` | Mean sampled response length in tokens |
| `truncated_fraction` | Fraction of responses that hit the token limit |

`update` records add `seconds_since_previous_update` from the second accepted
update on. The interval includes publication, evaluation and checkpoint
overhead. Scripts may add any other fields and events, for example
`run`, `publication` and `checkpoint` in `dapo_math.py`, or per-problem accuracy
and replay diagnostics in `beam_blog_repro.py`. Add new fields under their own
names rather than reusing a standard name with a different meaning.

An execution outside Spindle, indexed with `source`, holds only `evaluation`
records with `step` and optional `reward` and `response_tokens_mean`, read from
its published chart.

`index.toml` maps stable run IDs to validation definitions and JSONL files.
`ReferenceRunIndex` and `ReferenceRun` in `catalog.py` validate its fields and unique IDs.
Successful `run.py --output ...` executions under this checkout's
`reference_runs/outputs/` directory register automatically. Scratch outputs, such
as `/tmp/run.jsonl`, stay uncatalogued, including with `--index`. Symlinks are
resolved before deciding whether an output belongs to the catalog. The output
filename without its extension becomes the run ID. `--index` selects another
catalog for these outputs. Indexed runs require a TOML definition directly in
this checkout's `reference_runs/configs/` directory, so its catalog name identifies
the definition. Temporary or external definitions can use scratch output paths.

The launcher records the validation name, relative output path, SDK version, and
server-reported `spindle_commit` and `spindle_version` from the stock Tinker SDK
`get_server_capabilities` request before starting the workload. The response
identifies the frontend answering that request. Missing metadata is omitted; the
launcher never substitutes its local source revision. The SDK request uses a raw
response to retain fields its typed capability model discards.
Image, hardware and region settings describe the resolved recipe; they do not
attest the remote trainer's actual source, image or GPU identity. Observed runtime
details can be added to the entry after verification. Historical entries may
omit the server version.

Concurrent runs reserve their IDs and output paths with process locks before
starting. Completion updates use a short index lock and atomic replacement.
Failed runs keep their partial metrics and are not indexed. Keep the persistent
hidden lock files in place. These locks coordinate processes using the same
filesystem; independent checkouts must merge their catalogs through Git.
Do not invent digests or hardware observations
for historical runs where they were not recorded. The Qwen3.5-4B JSONL files are
kept byte-for-byte, including their original hardware metadata. The Beam file's
field names were converted to this format; its index notes list the changes.

Only completed runs' compact metrics and index entries belong in Git. Keep
partial or in-progress metrics uncommitted. Raw generations, token
arrays, checkpoints, logs and generated pictures stay outside the catalog.

## Compare

Pass indexed IDs or a new JSONL path, with no network access required:

```bash
uv run python reference_runs/plot.py \
  reference_runs/outputs/my-run.jsonl dapo-qwen35-4b-miles-v011
```

The chart shows held-out accuracy, training reward, response length and seconds
between accepted updates. It supports partial files and writes
`reference_runs/reference-runs.png`, which is ignored by Git. Compare the same
workload settings and actual hardware, accounting for intentional deployment
changes recorded in the index. Plots do not automatically establish regression
thresholds or learning parity.

The workload uses 20 GRPO updates, 32 problems × 8 samples, rank-32 LoRA and
learning rate 4e-5. It evaluates 128 held-out problems × 2 samples every five
updates with a 2,048-token response cap. Rewards match the last boxed answer to
the label; advantages subtract the group's mean reward. Checkpoints are saved
every ten updates. `--smoke` uses two updates, four groups, eight held-out
problems and evaluation after every update. This workload tests checkpoint
saving, not restoration.
