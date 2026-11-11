# Regression runs

`run.py` trains each chosen model for a few steps of reinforcement learning against a Spindle server, using tinker-cookbook's [guess-the-number recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/multiplayer_rl/guess_number) without changes. It is a multi-turn GRPO task: the model gets up to 10 turns to guess a hidden number, with "too high" or "too low" after each guess. A run exercises training, weight publishing, multi-turn sampling and the LoRA path the way a real RL job does.

Each run appends one line per model to [`results/guess_number.jsonl`](results/guess_number.jsonl), and the script prints each result next to the last saved one for the same model and settings. Commit that file, so the next large change has a baseline to compare against.

## Run it

Point it at a server the same way as any Tinker client: set `TINKER_BASE_URL` and `TINKER_API_KEY`, plus `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET` if the server requires a [proxy token](../examples/shared-server/README.md#use).

Run one model, or a few, when a change affects specific configs:

```bash
uv run --with tinker-cookbook==0.6.1 python regression/run.py --model Qwen/Qwen3-8B
```

Run every model the server serves before a change that touches all of them, such as an SGLang or Miles upgrade:

```bash
uv run --with tinker-cookbook==0.6.1 python regression/run.py --all
```

A model that tinker-cookbook has no renderer for is skipped. Models with a no-thinking renderer use it, because each turn is a short answer.

The defaults are 3 steps, 8 groups of 4 games each, and 4 models at a time. `--steps`, `--batch-size`, `--group-size` and `--parallel` change them. Results are only compared with earlier runs that used the same settings. `--wandb-project` also logs each run to Weights & Biases, and `--no-save` prints results without saving them.

## What to look at

| Field | Meaning |
| --- | --- |
| `kl_v2` | Mean of tinker-cookbook's `optim/kl_sample_train_v2`: how far the sampler's log probabilities are from the trainer's on the same tokens. It should stay near its earlier value. |
| `first_step_s`, `median_step_s` | Seconds for the first step, which includes cold starts and compilation, and the median of the later steps. |
| `reward_first`, `reward_last` | Mean reward on the first and last step. Over 3 steps this mostly shows that rollouts ran. |
| `ok` | All steps finished without an error. |

Each line also records the server's Spindle version and commit and the Tinker and tinker-cookbook versions.

Runs on a cold server spend most of their time starting trainers and samplers, so a full sweep takes hours on the largest models. KL from a few steps is noisy: compare against several earlier runs before calling a change a regression.
