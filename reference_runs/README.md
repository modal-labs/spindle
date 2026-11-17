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

## Native LoRA RL and checkpoint validation

[`native_lora_rl.py`](native_lora_rl.py) checks a deployed native LoRA recipe
through the stock Tinker SDK. By default, two concurrent clients use ranks 8
and 16. Each runs five multiplication GRPO updates, with four groups of eight
responses and a 64-token generation cap. Reward is answer correctness minus
0.001 per generated token; advantages are centered within each group and the
batch loss is normalized over 32 responses.

```bash
uv run --with jinja2 python reference_runs/native_lora_rl.py \
  --recipe qwen3-4b-lora-32k --output reference_runs/runs/native-qwen3-4b
```

Set `TINKER_BASE_URL` and `TINKER_API_KEY` first. For a protected frontend, use
the same `with_proxy_auth.py` wrapper shown above. `--clients 1` selects a
single rank-8 client; `--steps` changes the number of RL updates. Add
`--train-unembed` when the deployed recipe includes the output head.

The client requires finite sampling logprobs, a nonzero gradient norm, and a
change in fixed-probe trainer logprobs. After both RL loops finish, each client
is verified separately so tenant batching cannot change the replay workload.
It saves optimizer state, takes another
update, restores the checkpoint, and repeats that update on the same batch.
Restored probe logprobs must match within 1e-5. The next-update difference is
recorded separately as `resumed_step_max_diff`; use `--require-exact-replay` to
require a difference no larger than 1e-5. A successful state restore does not
guarantee bitwise numerical execution of a new backward pass. Finally, the
client publishes the trained adapter and checks sampling and prompt scoring.
Per-client JSONL files record rewards, timings, optimizer metrics, and restore
differences. A `result.json` is written only after every client passes.

This is an execution and restore check, not a convergence benchmark or a test
of equality with the official Tinker service. Concurrent requests can be
combined by the trainer, but this script does not assert a particular packing
schedule. Like the other reference clients, it leaves deployment cleanup to
the operator.
