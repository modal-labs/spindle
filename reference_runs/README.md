# Reference runs

A flat catalog of validation definitions and measured executions. The DAPO-Math
references train through the stock Tinker SDK.

```text
reference_runs/
  configs/dapo_qwen35_4b.toml   # Immutable validation definition
  configs/dapo_qwen38_27b_multinode.toml
  scripts/dapo_math.py         # Dataset, grading and training settings
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

[tokenizer]
model = "Qwen/Qwen3.5-4B"
revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
```

The existing H100 reference pins its hardware and client capacity in
`[config_overrides]`, preserving the old Python reference recipe. Its fields
are declared in `ConfigOverrides` in `run.py`; unknown fields and incorrect
types are rejected. Omitted fields keep their deployment recipe values.
Use `inference_gpu = "H100!"` to select exact H100 hardware, or the declared
`"miles_cfg.max_lora_slots"` key to set adapter capacity. Optional
`script_args = []` contains the workload's literal CLI arguments. The launcher
does not forward extra arguments from its own command line. Dataset, grading,
learning rate, batch size and seed stay in the Tinker script. Every definition
requires a typed `tokenizer` with its Hugging Face model ID and full commit revision.
The launcher supplies `--tokenizer MODEL REVISION`, the output path, and
`--base-model` with the resolved deployment name. Scripts use that deployment
name in `create_lora_training_client` to select the exact recipe.
Script arguments cannot override these values. Direct callers of
`dapo_math.py` must also pass `--tokenizer MODEL REVISION`; it has no default.

`dapo_qwen38_27b_multinode.toml` uses the same workload with a pinned Qwen3.8-27B
tokenizer in its definition. It inherits the committed `qwen38-27b-lora-256k`
topology: two eight-H200 trainer nodes, TP2/CP8, and two
four-H200 sampler replicas. The definition selects exact H200 hardware and one
training client/adapter slot.
The workload generates at most 2,048 response tokens; it does not validate 256K
sequences. Compare new
measurements against the same validation, rather than the 4B learning curve.

Once published, a validation TOML file is immutable. Create a new definition
when changing its script selection, arguments or deployment overrides. Spindle
commits and observed runtime image versions identify executions and belong in
the index, not in the validation definition. Deployment recipes own image
selection; record the images used for each execution in the index.

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

Each JSONL record has `time` (Unix seconds) and `event`. Records are appended as
the workload progresses so a partial run remains useful.

| Event | Contents |
| --- | --- |
| `run` | Effective workload settings, Adam parameters, SDK version, selected deployment name |
| `publication` | Accepted update number and elapsed seconds |
| `evaluation` | Update number, accuracy, response count, mean tokens, truncation fraction, seconds |
| `checkpoint` | Update number and save duration; no checkpoint URI |
| `update` | Accepted step, batch accuracy/tokens/truncation, client sampling/backward/optimizer times, server metrics |

Update intervals include publication, evaluation and checkpoint overhead. The
first update has no preceding interval. Missing server metrics stay absent.
The script rejects an explicitly unsuccessful optimizer result.

`index.toml` maps stable run IDs to validation definitions and JSONL files.
`ReferenceRunIndex` and `ReferenceRun` in `catalog.py` validate its fields and unique IDs.
Successful `run.py --output ...` executions under this checkout's
`reference_runs/outputs/` directory register automatically. Scratch outputs, such
as `/tmp/run.jsonl`, stay uncatalogued, including with `--index`. Symlinks are
resolved before deciding whether an output belongs to the catalog. The output
filename without its extension becomes the run ID. `--index` selects another
catalog for these outputs.

The launcher records the validation name, relative output path, SDK version, and
server-reported `spindle_commit` and `spindle_version` from the stock Tinker SDK
`get_server_capabilities` request before starting the workload. The response
identifies the frontend answering that request. Missing metadata is omitted; the
launcher never substitutes its local source revision. On Tinker 0.26, an HTTP
response hook retains these fields before the SDK discards unknown fields.
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
for historical runs where they were not recorded. Existing historical JSONL
files are kept byte-for-byte, including their original hardware metadata.

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
