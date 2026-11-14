# Reference runs

Client workloads that train and sample through the Tinker API and record
metrics. Run one against a deployment before and after a backend change, then
compare the recorded numbers.

The client scripts connect to a server that already exists. They do not deploy,
stop or clean up anything.

## DAPO-Math GRPO

[`dapo_math.py`](dapo_math.py) trains a LoRA adapter on
[DAPO-Math](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) with GRPO. It
uses only the stock Tinker SDK. The settings are the `Config` dataclass at the
top of the script; the default model is `Qwen/Qwen3.5-4B`.

Each update samples 8 responses to each of 32 problems from the latest published
weights, then runs one `forward_backward` and one `optim_step`. A response earns
reward 1 when its last `\boxed{...}` equals the label. The advantage is the
reward minus the mean reward for that problem. Evaluation samples 2 responses to
each of 128 held-out problems, before training and every 5 updates. The script
saves trainer state with `save_state` every 10 updates.

Deploy [`dapo_math_config.py`](dapo_math_config.py) before running the client:

```bash
uv run spindle deploy reference_runs/dapo_math_config.py --app reference --env dev
```

The recipe fixes training at four H100 GPUs and sampling at two replicas with
one H100 each. It uses Modal's `H100!` selector to disable
[automatic H200 upgrades](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s).
It has no GPU fallback or sampler autoscaling, and admits one
training client. Use a dedicated deployment without other sampling clients.
The client requests `dapo-math-qwen35-4b-h100` through the stock Tinker SDK and
records this hardware contract. Point it at the deployment of this recipe;
metadata alone does not verify the hardware of an arbitrary server.

Set `TINKER_BASE_URL` and `TINKER_API_KEY`, then:

```bash
# 20 updates.
uv run python reference_runs/dapo_math.py --output reference_runs/runs/dapo

# Smoke run: 2 small updates.
uv run python reference_runs/dapo_math.py --smoke --output reference_runs/runs/smoke
```

For a server behind Modal proxy auth, run the script through
[`with_proxy_auth.py`](../examples/shared-server/with_proxy_auth.py):

```bash
uv run python examples/shared-server/with_proxy_auth.py \
  reference_runs/dapo_math.py --output reference_runs/runs/dapo
```

The output directory must not exist. A run that stops early cannot be continued;
start a new one.

## Two-node Qwen3.8-27B

[`dapo_math_27b.py`](dapo_math_27b.py) uses the same workload and training loop
with Qwen3.8-27B and the existing `qwen38-27b-lora-256k` recipe. Deploy it with:

```bash
uv run spindle deploy src/spindle/configs/qwen38_27b_lora_256k.py --app reference --env dev
uv run python reference_runs/dapo_math_27b.py --smoke --output reference_runs/runs/27b-smoke
uv run python reference_runs/dapo_math_27b.py --output reference_runs/runs/27b
```

This recipe requests two trainer nodes with eight H200 GPUs each (TP2, CP8)
and two sampler replicas with four GPUs each (TP4), selected from H200, B300
and B200. Use a dedicated deployment and verify the actual hardware.
The client records the requested
hardware, not a measurement of the allocation. The existing proxy-auth wrapper
works with this entry point too.

Only the model, tokenizer revision, recipe name and hardware record differ from
the 4B client. The full run still uses 20 updates, rank 32, learning rate 4e-5,
and a 2,048-token response limit, with thinking disabled. These are validation
settings, not tuned 27B hyperparameters. Short math prompts do not validate
256K context support. This client saves checkpoints but does not test restore.
Compare backend versions on this same workload and hardware; the recorded 4B
results are not a performance or learning baseline for 27B.

## Records

`metrics.jsonl` has one JSON object per line, appended as the run goes, so a run
that stops early still has its earlier records. Every record has `time` (Unix
seconds) and `event`.

| Event | Fields |
| --- | --- |
| `run` | `config`, `adam` (the optimizer parameters sent), `tinker_sdk`, `recipe`, `hardware` |
| `publication` | `step`, `seconds` to save sampler weights and open a sampling client |
| `evaluation` | `step`, `reward` (accuracy), `responses`, `response_tokens_mean`, `truncated_fraction`, `seconds` |
| `checkpoint` | `step`, `seconds` for `save_state` |
| `update` | One per accepted optimizer update, described below |

An `update` record has:

- `step`, the count of accepted optimizer updates. The script exits if the
  service reports `update_successful:mean` other than 1. The batch for update k
  is sampled from the weights after k - 1 updates, so its `reward` is one
  update behind the `evaluation` at the same step.
- `reward`, `responses`, `response_tokens_mean` and `truncated_fraction` for the
  sampled batch.
- `sampling_seconds`, `forward_backward_seconds` and `optimizer_seconds`, timed
  at the client.
- `seconds_since_previous_update`, which also covers publication, evaluation
  and checkpoints between the two updates. The first update has none.
- `forward_metrics` and `optimizer_metrics`, exactly as the service returned
  them. On Spindle these include `loss:sum` and `grad_norm:mean`. A metric the
  service does not report is absent.

## Compare runs

```bash
uv run python reference_runs/plot.py \
  reference_runs/results/dapo-math-qwen35-4b reference_runs/runs/dapo
```

This draws held-out accuracy, training reward, response length and seconds
between accepted updates for each run into `reference_runs/reference-runs.png`. It reads only
`metrics.jsonl`, so it works offline and on a run that has not finished.

Compare runs only when their `run` records have the same `config` and
`hardware`. Missing hardware metadata means the allocation is unknown. Check in a
run's `metrics.jsonl` under `results/` with a short README that states the code
revision, the deployment and what the run does not show. Charts stay out of Git.
