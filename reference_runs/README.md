# Reference runs

Client workloads that train and sample through the Tinker API and record what
happened. Run one against a deployment before and after a backend change, then
compare the recorded metrics.

The scripts connect to a server that already exists. They do not deploy, stop or
clean up anything.

## DAPO-Math GRPO

[`dapo_math.py`](dapo_math.py) trains a LoRA adapter on
[DAPO-Math](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) with GRPO. It
uses only the stock Tinker SDK, so it runs against any Tinker-compatible service
that serves the model. The default model is `Qwen/Qwen3.5-4B`.

Each update samples 8 responses to each of 32 problems from the latest published
weights, then runs one `forward_backward` and one `optim_step`. The loop is
synchronous: it never samples for a later update while the trainer works.
[`config.json`](results/dapo-math-qwen35-4b/config.json) in a run directory holds
every setting and the rules for data selection, rewards and evaluation.

### Run it

Set `TINKER_BASE_URL` and `TINKER_API_KEY`. For a server behind Modal proxy auth,
also set `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET`.

```bash
# Fresh run: 30 updates, evaluation every 5, checkpoint every 10.
uv run python reference_runs/dapo_math.py --output runs/dapo

# Bounded smoke run: 2 small updates, then stop at a checkpoint.
uv run python reference_runs/dapo_math.py --smoke --output runs/smoke --stop-after 2

# Continue either run from its latest checkpoint in a new process.
uv run python reference_runs/dapo_math.py --output runs/smoke --resume
```

`--stop-after N` saves a checkpoint after update N and exits. `--resume` reads
the configuration from the run directory, so it takes no configuration flags.
`--help` lists a flag for every setting.

The script refuses to start a fresh run in a directory that already holds one,
and only one process can write to a run directory at a time.

### Restart rules

A checkpoint is the trainer state with optimizer state (`save_state`), the path
of the sampler weights published at that update, and the data cursor. On
`--resume` the script restores the trainer with
`create_training_client_from_state_with_optimizer`, opens a sampling client on
the recorded sampler path instead of publishing again, and continues at the
recorded cursor. The next update's behavior policy is therefore the same weights
the checkpoint named.

This shows that the SDK restore succeeds and that training continues. It does not
compare tensors, so it is not evidence that the restored weights or optimizer
moments equal the saved ones.

Each optimizer request has one of three outcomes:

| Outcome | Records | What the script does |
| --- | --- | --- |
| accepted | `optimizer_result` with `accepted: true`, then `update` | continues |
| rejected | `optimizer_result` with `accepted: false` | exits; the update is not counted |
| unknown | `optimizer_submitted` with no `optimizer_result` | exits; never sends the request again |

A service that reports no `update_successful:mean` metric is taken to apply
every request it answers.

After an unknown outcome, or after accepted updates that no checkpoint covers,
`--resume` stops and lists the affected updates. Pass `--discard-uncheckpointed`
to return to the latest checkpoint. The script records a `rollback` and trains
those updates again from the restored state.

Sampled batches are kept in `local/` inside the run directory. If a batch was
recorded for the next update, `--resume` trains on that batch. If its file is
missing or changed, the script stops. Pass `--resample-pending` to sample a new
batch; the script records `pending_rollout_discarded`.

### Records

`metrics.jsonl` has one JSON object per line. The script appends each record and
flushes it to disk before it continues, so an interrupted run leaves a readable
file. A reader ignores a last line with no newline. Every record has `time` (Unix
seconds) and `event`. A field is absent when the service did not report it;
nothing is filled in as zero.

| Event | Fields |
| --- | --- |
| `segment` | One per process: `client_revision`, `script_sha256`, `packages`, `deployment` (server-reported revisions, or `unknown`), `evaluation_ids` (dataset row indices), `prompts_sha256`, `resumed_from` |
| `created`, `restore` | `model_id`, `seconds`; `restore` adds the checkpoint `path`, `sampler_path` and `step` |
| `publication` | `step`, sampler `path`, `seconds` for `save_weights_for_sampler` and the new sampling client |
| `checkpoint` | `step`, state `path`, `sampler_path`, `next_cursor`, `seconds` |
| `evaluation` | `step`, `sampler_path`, `accuracy`, `problems`, `responses`, `correct_per_problem` (in `evaluation_ids` order), response-token statistics, `truncated_fraction`, `seconds` |
| `rollout` | `step`, `behavior_policy_step`, `sampler_path`, `cursor`, `sha256` of the local batch file, `reward`, response-token statistics, `truncated_fraction`, `seconds` |
| `forward_backward` | `step`, `seconds`, the service's `metrics` |
| `optimizer_submitted`, `optimizer_result` | `step`; the result adds `accepted`, `seconds` and the service's `metrics` |
| `update` | One per accepted update, described below |
| `rollback` | `to_step`, `discarded` (update to outcome) |
| `pending_rollout_discarded` | `step`, `reason` |
| `stopped`, `completed` | `step` |

An `update` record has:

- `step`, the count of accepted optimizer updates, and `elapsed_seconds` since
  the run's first record, including any time the client was not running.
- `seconds_since_previous_update`, the wall time since the previous accepted
  update in the same process. It includes publication, evaluation and
  checkpoints between the two. It is absent for the first update of a process.
- `sampling_seconds`, `forward_backward_seconds` and `optimizer_seconds`, timed
  at the client.
- `learning_rate`, `loss_sum` and `grad_norm`.
- `reward`, `responses`, `response_tokens`, `response_tokens_mean`,
  `response_tokens_max` and `truncated_fraction` for the batch.
- `behavior_policy_step`, `sampler_path` and `cursor`, which identify the batch.
- `sampled_log_ratio_mean`: the mean over sampled response tokens of the
  trainer's log probability minus the sampler's. It is a single-sample estimate
  of the negative per-token KL divergence from sampler to trainer, not an exact
  KL. `importance_weight_mean` is the mean of the exponentiated differences.

`rollback` removes earlier records for later updates from the run's history.
The batch sampled for the first update after the checkpoint stays.
`dapo_math.surviving` applies this rule; the chart script uses it.

### Compare runs

```bash
uv run --with matplotlib python reference_runs/plot.py \
  reference_runs/results/dapo-math-qwen35-4b runs/dapo
```

This draws held-out accuracy, training reward, response length and seconds
between accepted updates for each run, and writes `charts.png` into the first
run directory. It reads only `metrics.jsonl`, so it works offline and on a run
that has not finished. Generated images are ignored by Git.

Compare runs only when their `config.json` files match. When you add data from
outside this repository, record its source, say whether the values were reported
or read off a chart, and do not assume that its step axis counts optimizer
updates.

### What to check in

Check in `config.json` and `metrics.jsonl` from a run under `results/`, with a
short README that states the code revision, the deployment and what the run does
not show. `local/` holds sampled tokens and evaluation responses; it and the
charts stay out of Git.
