# Reference runs

A flat catalog of validation definitions and measured executions. The existing
DAPO-Math reference trains Qwen3.5-4B through the stock Tinker SDK.

```text
reference_runs/
  configs/dapo_qwen35_4b.toml   # Immutable validation definition
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
```

The existing H100 reference pins its hardware and client capacity in
`[config_overrides]`, preserving the old Python reference recipe. It uses the
existing dotted config override mechanism,
for example `inference_gpu = "H100!"` to select exact H100 hardware. Optional
`script_args = []` contains literal CLI arguments. Dataset, grading, learning
rate, batch size and seed stay in the Tinker script. The launcher supplies
`--base-model` with the resolved deployment name and passes the output path;
scripts use that value directly in `create_lora_training_client` to select the
exact recipe. Script arguments cannot override these two values.

Once published, a validation TOML file is immutable. Create a new definition
when changing its script selection, arguments or deployment overrides. Spindle
commits and runtime image versions identify executions and belong in the index,
not in the validation definition. Deployment recipes still own default images.

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

# Small smoke run; the extra arguments go directly to the workload script.
uv run python reference_runs/run.py reference_runs/configs/dapo_qwen35_4b.toml \
  --output /tmp/dapo-smoke.jsonl -- --smoke
```

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
`ReferenceRunIndex` and `ReferenceRun` in `plot.py` validate its fields and unique IDs.
Record the tested Spindle commit, image identities, actual hardware and relevant
notes there. This provenance is manual for now; the script does not query a
Spindle-specific provenance API. Do not invent digests or hardware observations
for historical runs where they were not recorded. Existing historical JSONL
files are kept byte-for-byte, including their original hardware metadata.

Only compact metrics and index entries belong in Git. Raw generations, token
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
