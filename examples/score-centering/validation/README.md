# Real RL GPU validation

PR #173 was validated end to end on commit
`a3156b51193a339e5c7ccc49d4ecd41d4f3ca4b7`. The recorded
[results](results.json) contain the per-update losses, gradient norms, optimizer
results, timings, and published-probability changes.

| Case | Model | Training | TP / CP / DP | Result |
| --- | --- | --- | --- | --- |
| `fft4b` | Qwen3.5-4B | Megatron full parameter | 1 / 1 / 8 | 10 updates passed |
| `m111` | Qwen3-4B | Miles v0.1.1, LoRA rank 16 | 1 / 1 / 8 | 10 updates passed |
| `m112cp2` | Qwen3-4B | Miles v0.1.2, LoRA rank 16 | 1 / 2 / 4 | 10 updates passed |

Each case used one node with eight B300 GPUs and RDMA enabled, plus one B300
sampler. Cases ran sequentially. The LoRA targets were attention projections and
dense MLP projections; they did not include the language-model head.

## What ran

The runner used the deployed Tinker HTTP API, normal Spindle scheduling, native
Megatron/Miles forward/backward and optimizer operations, and SGLang sampling.
No tokens, sampling probabilities, rewards, gradients, or optimizer responses
were mocked. Rewards came from parsing actual arithmetic answers and checking
`a * b + c`. Advantages used a fixed reward baseline of 0.5, normalized by the
batch's generated-token count.

Each update sampled 16 trajectories from eight prompts, with a 64-token response
cap and a 2,048-token context limit. Each case ran three `score_centered_mis`,
three `score_centered`, three `mis`, and one ordinary `ppo` update. Replay sampling
used temperature 1, top-k 64, and top-p 0.95. PPO used unrestricted sampling at
temperature 1. Explicit response-only weights alternated with inferred weights
within each training batch.

All 30 updates returned finite losses, positive gradient norms, and successful
optimizer results. Every update published weights. Each case changed the probe's
published log probabilities, saved a training checkpoint, reloaded optimizer state,
and sampled successfully afterward. All validation containers were stopped.

These are short-context dense-model smoke tests. They do not establish convergence,
long-context capacity, routed-expert replay correctness, production batch-size
performance, or correctness of the separate gradient-history change in PR #221.
Cold kernel compilation and sampler restarts are included in the recorded times;
these numbers are not a steady-state performance comparison.

## Reproduce

Run from the repository root with Python 3.12 and the usual Modal credentials,
`spindle-api` and `spindle-proxy` secrets. Choose a dedicated frontend name and
set `MODAL_PROFILE` and `MODAL_ENVIRONMENT` for your own workspace. The recipes
create separate checkpoint, bulletin, and result volumes for that frontend.

The recorded run used a local allocation-only patch because single-node RDMA
requires all eight B300 GPUs. The patch enables the clustered allocation and EFA;
it leaves the normal single-node Ray/backend startup branch unchanged. It is
included as a reproduction artifact and does not change Spindle's defaults.

```bash
set -e
uv sync --locked --python 3.12
export SPINDLE_VALIDATION_FRONTEND=score-centering-validation
validation_dir=examples/score-centering/validation

git apply "$validation_dir/rdma-allocation.patch"
cleanup_validation() {
    uv run spindle stop --app "$SPINDLE_VALIDATION_FRONTEND" --env "$MODAL_ENVIRONMENT"
    git apply -R "$validation_dir/rdma-allocation.patch"
}
trap cleanup_validation EXIT

for validation_case in fft4b m111 m112cp2; do
    uv run spindle deploy "$validation_dir/$validation_case.py"
    validation_url=$(uv run python -c 'import modal, os; print(modal.Function.from_name(os.environ["SPINDLE_VALIDATION_FRONTEND"], "server").get_web_url())')
    uv run modal run --detach "$validation_dir/run.py" --case "$validation_case" --url "$validation_url"
    uv run spindle stop --app "$SPINDLE_VALIDATION_FRONTEND" --env "$MODAL_ENVIRONMENT"
done

```

Stop and inspect any failed case before proceeding to the next one. Persisted
events are in `<frontend>-results`, under `<case>/events.jsonl`. Training
checkpoints remain in `<frontend>-checkpoints` after shutdown; the full-parameter
checkpoint from the recorded run was about 136 GB.

The published runner preserves the sampling and training workload from the
recorded driver. Packaging removed one-off administrative retry cleanup,
parameterized names, added the positive-gradient/success checks that were verified
from the recorded events, and moved imports to module scope. Recipes use a
600-second sampler idle window, matching the override used during validation.
The recorded FFT run initially used 60 seconds and restarted its sampler during
cold compilation. The packaged runner itself has not been rerun on GPUs; the
backend code and all GPU claims above refer to `a3156b5`.
